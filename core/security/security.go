package security

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// SSRFBlockedError is returned when a URL, hostname, or IP is blocked by policy.
type SSRFBlockedError struct {
	Reason string
}

func (e SSRFBlockedError) Error() string {
	return "SSRF blocked: " + e.Reason
}

// SSRFPolicy controls URL and IP validation.
type SSRFPolicy struct {
	AllowedSchemes         map[string]bool
	BlockPrivateIPs        bool
	BlockLocalhost         bool
	BlockCloudMetadata     bool
	BlockKubernetesLocal   bool
	AllowedHosts           map[string]bool
	AdditionalBlockedCIDRs []*net.IPNet
}

// DNSResolver resolves a hostname for SSRF-safe transports.
type DNSResolver func(ctx context.Context, hostname string) ([]net.IP, error)

// SSRFSafeTransport validates every outgoing request against an SSRF policy,
// validates all resolved IPs, then pins the request to the first resolved IP
// while preserving the original Host header and HTTPS SNI name.
type SSRFSafeTransport struct {
	Policy   SSRFPolicy
	Base     *http.Transport
	Resolver DNSResolver
}

// DefaultSSRFPolicy matches the Python default: HTTP(S) allowed, private and
// metadata targets blocked.
func DefaultSSRFPolicy() SSRFPolicy {
	return SSRFPolicy{
		AllowedSchemes:       map[string]bool{"http": true, "https": true},
		BlockPrivateIPs:      true,
		BlockLocalhost:       true,
		BlockCloudMetadata:   true,
		BlockKubernetesLocal: true,
		AllowedHosts:         map[string]bool{},
	}
}

// ValidateURL validates scheme and hostname without doing DNS resolution.
func ValidateURL(rawURL string, policy SSRFPolicy) error {
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme == "" || parsed.Hostname() == "" {
		return SSRFBlockedError{Reason: "invalid URL"}
	}
	scheme := strings.ToLower(parsed.Scheme)
	if len(policy.AllowedSchemes) > 0 && !policy.AllowedSchemes[scheme] {
		return SSRFBlockedError{Reason: "scheme not allowed"}
	}
	return ValidateHostname(parsed.Hostname(), policy)
}

// ValidateHostname checks hostname block rules.
func ValidateHostname(hostname string, policy SSRFPolicy) error {
	host := strings.TrimSuffix(strings.ToLower(hostname), ".")
	if policy.AllowedHosts != nil && policy.AllowedHosts[host] {
		return nil
	}
	if policy.BlockLocalhost && (host == "localhost" || host == "localhost.localdomain" || host == "host.docker.internal") {
		return SSRFBlockedError{Reason: "localhost hostname"}
	}
	if policy.BlockCloudMetadata {
		switch host {
		case "metadata", "metadata.google.internal", "metadata.amazonaws.com", "instance-data":
			return SSRFBlockedError{Reason: "cloud metadata hostname"}
		}
	}
	if policy.BlockKubernetesLocal && strings.HasSuffix(host, ".svc.cluster.local") {
		return SSRFBlockedError{Reason: "kubernetes internal hostname"}
	}
	if ip := net.ParseIP(host); ip != nil {
		return ValidateResolvedIP(ip.String(), policy)
	}
	return nil
}

// ValidateResolvedIP checks a resolved IP address against private, metadata,
// and caller-provided blocked CIDR ranges.
func ValidateResolvedIP(ip string, policy SSRFPolicy) error {
	parsed := net.ParseIP(ip)
	if parsed == nil {
		return SSRFBlockedError{Reason: "invalid IP address"}
	}
	if policy.AllowedHosts != nil && policy.AllowedHosts[strings.ToLower(ip)] {
		return nil
	}
	if policy.BlockCloudMetadata && isCloudMetadataIP(parsed) {
		return SSRFBlockedError{Reason: "cloud metadata IP"}
	}
	if policy.BlockLocalhost && parsed.IsLoopback() {
		return SSRFBlockedError{Reason: "localhost IP"}
	}
	if policy.BlockPrivateIPs && isPrivateOrReserved(parsed) {
		return SSRFBlockedError{Reason: "private IP range"}
	}
	for _, cidr := range policy.AdditionalBlockedCIDRs {
		if cidr != nil && cidr.Contains(parsed) {
			return SSRFBlockedError{Reason: "blocked CIDR"}
		}
	}
	return nil
}

// ValidateSafeURL resolves a URL host and rejects it if any resolved IP is
// blocked. AllowedHosts bypass DNS/IP validation for explicit trusted hosts.
func ValidateSafeURL(rawURL string, policy SSRFPolicy) (string, error) {
	if err := ValidateURL(rawURL, policy); err != nil {
		return "", err
	}
	parsed, _ := url.Parse(rawURL)
	host := strings.ToLower(parsed.Hostname())
	if policy.AllowedHosts != nil && policy.AllowedHosts[host] {
		return rawURL, nil
	}
	ips, err := net.LookupIP(host)
	if err != nil {
		return "", fmt.Errorf("resolve %q: %w", host, err)
	}
	for _, ip := range ips {
		if err := ValidateResolvedIP(ip.String(), policy); err != nil {
			return "", err
		}
	}
	return rawURL, nil
}

// IsSafeURL is the non-throwing form of ValidateSafeURL.
func IsSafeURL(rawURL string, policy SSRFPolicy) bool {
	_, err := ValidateSafeURL(rawURL, policy)
	return err == nil
}

// NewSSRFSafeTransport creates an SSRF-validating HTTP round tripper.
func NewSSRFSafeTransport(policy SSRFPolicy, base *http.Transport, resolver DNSResolver) *SSRFSafeTransport {
	return &SSRFSafeTransport{
		Policy:   policy,
		Base:     base,
		Resolver: resolver,
	}
}

// RoundTrip implements http.RoundTripper.
func (t *SSRFSafeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req == nil || req.URL == nil {
		return nil, SSRFBlockedError{Reason: "invalid request"}
	}
	policy := t.Policy
	if policy.AllowedSchemes == nil {
		policy.AllowedSchemes = DefaultSSRFPolicy().AllowedSchemes
	}
	if err := ValidateURL(req.URL.String(), policy); err != nil {
		return nil, err
	}
	hostname := strings.ToLower(req.URL.Hostname())
	if policy.AllowedHosts != nil && policy.AllowedHosts[hostname] {
		return t.transportForRequest(req.URL.Scheme, hostname).RoundTrip(req)
	}
	resolver := t.Resolver
	if resolver == nil {
		resolver = defaultDNSResolver
	}
	ips, err := resolver(req.Context(), hostname)
	if err != nil {
		return nil, fmt.Errorf("resolve %q: %w", hostname, err)
	}
	if len(ips) == 0 {
		return nil, SSRFBlockedError{Reason: "DNS resolution returned no results"}
	}
	for _, ip := range ips {
		if err := ValidateResolvedIP(ip.String(), policy); err != nil {
			return nil, err
		}
	}
	pinned := req.Clone(req.Context())
	pinned.URL = cloneURL(req.URL)
	originalHost := req.URL.Host
	pinned.URL.Host = net.JoinHostPort(ips[0].String(), requestPort(req.URL))
	pinned.Host = originalHost
	return t.transportForRequest(req.URL.Scheme, hostname).RoundTrip(pinned)
}

func (t *SSRFSafeTransport) transportForRequest(scheme string, serverName string) *http.Transport {
	var base *http.Transport
	if t.Base != nil {
		base = t.Base.Clone()
	} else if transport, ok := http.DefaultTransport.(*http.Transport); ok {
		base = transport.Clone()
	} else {
		base = &http.Transport{}
	}
	if strings.EqualFold(scheme, "https") {
		if base.TLSClientConfig == nil {
			base.TLSClientConfig = cloneTLSConfig(nil)
		} else {
			base.TLSClientConfig = cloneTLSConfig(base.TLSClientConfig)
		}
		base.TLSClientConfig.ServerName = serverName
	}
	return base
}

func defaultDNSResolver(ctx context.Context, hostname string) ([]net.IP, error) {
	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, hostname)
	if err != nil {
		return nil, err
	}
	ips := make([]net.IP, 0, len(addrs))
	for _, addr := range addrs {
		ips = append(ips, addr.IP)
	}
	return ips, nil
}

func requestPort(u *url.URL) string {
	if port := u.Port(); port != "" {
		return port
	}
	if strings.EqualFold(u.Scheme, "https") {
		return "443"
	}
	return "80"
}

func cloneURL(u *url.URL) *url.URL {
	copied := *u
	return &copied
}

func cloneTLSConfig(cfg *tls.Config) *tls.Config {
	if cfg == nil {
		return &tls.Config{}
	}
	return cfg.Clone()
}

func isCloudMetadataIP(ip net.IP) bool {
	for _, value := range []string{
		"169.254.169.254",
		"169.254.170.2",
		"169.254.170.23",
		"100.100.100.200",
	} {
		if ip.Equal(net.ParseIP(value)) {
			return true
		}
	}
	_, linkLocal, _ := net.ParseCIDR("169.254.0.0/16")
	return linkLocal.Contains(ip)
}

func isPrivateOrReserved(ip net.IP) bool {
	if ip.IsPrivate() || ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsMulticast() || ip.IsUnspecified() {
		return true
	}
	ipv4Blocked := []string{
		"0.0.0.0/8", "100.64.0.0/10", "192.0.0.0/24", "192.0.2.0/24",
		"198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
		"224.0.0.0/4", "240.0.0.0/4",
	}
	ipv6Blocked := []string{
		"::1/128", "fc00::/7", "fe80::/10", "ff00::/8", "::ffff:0:0/96",
		"64:ff9b::/96",
	}
	blocked := ipv6Blocked
	if ip.To4() != nil {
		blocked = ipv4Blocked
	}
	for _, cidr := range blocked {
		_, network, _ := net.ParseCIDR(cidr)
		if network.Contains(ip) {
			return true
		}
	}
	return false
}
