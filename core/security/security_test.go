package security

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestValidateURLBlocksDangerousHosts(t *testing.T) {
	policy := DefaultSSRFPolicy()
	for _, raw := range []string{
		"http://localhost:8080",
		"http://127.0.0.1",
		"http://169.254.169.254/latest",
		"http://service.default.svc.cluster.local",
	} {
		if err := ValidateURL(raw, policy); err == nil {
			t.Fatalf("ValidateURL(%q) succeeded, want block", raw)
		}
	}
}

func TestValidateResolvedIPAllowedAndBlocked(t *testing.T) {
	policy := DefaultSSRFPolicy()
	if err := ValidateResolvedIP("8.8.8.8", policy); err != nil {
		t.Fatalf("public IP blocked: %v", err)
	}
	if err := ValidateResolvedIP("10.0.0.1", policy); err == nil {
		t.Fatal("private IP allowed")
	}
}

func TestAllowedHostBypassesHostname(t *testing.T) {
	policy := DefaultSSRFPolicy()
	policy.AllowedHosts["localhost"] = true
	if err := ValidateHostname("localhost", policy); err != nil {
		t.Fatalf("allowed host blocked: %v", err)
	}
}

func TestSSRFSafeTransportBlocksResolvedPrivateIP(t *testing.T) {
	transport := NewSSRFSafeTransport(DefaultSSRFPolicy(), nil, func(context.Context, string) ([]net.IP, error) {
		return []net.IP{net.ParseIP("127.0.0.1")}, nil
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "http://example.com/path", nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = transport.RoundTrip(req)
	if err == nil || !strings.Contains(err.Error(), "localhost IP") {
		t.Fatalf("error: %v", err)
	}
}

func TestSSRFSafeTransportPinsIPAndPreservesHost(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.Host, "example.com:") {
			t.Fatalf("host header: %q", r.Host)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatalf("server address: %v", err)
	}
	policy := SSRFPolicy{
		AllowedSchemes:       map[string]bool{"http": true},
		BlockPrivateIPs:      false,
		BlockLocalhost:       false,
		BlockCloudMetadata:   true,
		BlockKubernetesLocal: true,
		AllowedHosts:         map[string]bool{},
	}
	transport := NewSSRFSafeTransport(policy, nil, func(_ context.Context, host string) ([]net.IP, error) {
		if host != "example.com" {
			t.Fatalf("resolved host: %q", host)
		}
		return []net.IP{net.ParseIP("127.0.0.1")}, nil
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "http://example.com:"+port+"/path", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("round trip: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("status: %d", resp.StatusCode)
	}
}

func TestValidateURLInvalidAndScheme(t *testing.T) {
	policy := DefaultSSRFPolicy()
	for _, raw := range []string{
		"http://",          // missing hostname
		"example.com/path", // missing scheme
		"://bad",           // unparseable
	} {
		if err := ValidateURL(raw, policy); err == nil {
			t.Fatalf("ValidateURL(%q) succeeded, want invalid URL block", raw)
		}
	}
	if err := ValidateURL("ftp://example.com", policy); err == nil ||
		!strings.Contains(err.Error(), "scheme not allowed") {
		t.Fatalf("ftp scheme: %v", err)
	}

	openPolicy := SSRFPolicy{AllowedSchemes: map[string]bool{}}
	if err := ValidateURL("ftp://example.com", openPolicy); err != nil {
		t.Fatalf("empty AllowedSchemes should allow any scheme: %v", err)
	}
}

func TestValidateHostnameRules(t *testing.T) {
	policy := DefaultSSRFPolicy()
	for _, host := range []string{
		"metadata.google.internal",
		"METADATA",
		"instance-data",
		"api.svc.cluster.local",
		"LOCALHOST.", // trailing dot must still be treated as localhost
		"host.docker.internal",
	} {
		if err := ValidateHostname(host, policy); err == nil {
			t.Fatalf("ValidateHostname(%q) succeeded, want block", host)
		}
	}
	// Literal IP hostnames are routed through ValidateResolvedIP.
	if err := ValidateHostname("8.8.8.8", policy); err != nil {
		t.Fatalf("public IP hostname blocked: %v", err)
	}
	if err := ValidateHostname("10.1.2.3", policy); err == nil {
		t.Fatal("private IP hostname allowed")
	}
	// With kubernetes blocking disabled the suffix is allowed.
	relaxed := DefaultSSRFPolicy()
	relaxed.BlockKubernetesLocal = false
	if err := ValidateHostname("api.svc.cluster.local", relaxed); err != nil {
		t.Fatalf("cluster.local blocked with BlockKubernetesLocal=false: %v", err)
	}
}

func TestValidateResolvedIPRules(t *testing.T) {
	policy := DefaultSSRFPolicy()
	if err := ValidateResolvedIP("not-an-ip", policy); err == nil ||
		!strings.Contains(err.Error(), "invalid IP address") {
		t.Fatalf("invalid IP: %v", err)
	}
	if err := ValidateResolvedIP("169.254.169.254", policy); err == nil ||
		!strings.Contains(err.Error(), "cloud metadata IP") {
		t.Fatalf("metadata IP: %v", err)
	}

	bypass := DefaultSSRFPolicy()
	bypass.AllowedHosts["10.0.0.1"] = true
	if err := ValidateResolvedIP("10.0.0.1", bypass); err != nil {
		t.Fatalf("allowed IP blocked: %v", err)
	}

	_, blocked, err := net.ParseCIDR("8.8.8.0/24")
	if err != nil {
		t.Fatal(err)
	}
	cidrPolicy := SSRFPolicy{AdditionalBlockedCIDRs: []*net.IPNet{nil, blocked}}
	if err := ValidateResolvedIP("8.8.8.8", cidrPolicy); err == nil ||
		!strings.Contains(err.Error(), "blocked CIDR") {
		t.Fatalf("blocked CIDR: %v", err)
	}
	if err := ValidateResolvedIP("8.8.4.4", cidrPolicy); err != nil {
		t.Fatalf("IP outside blocked CIDR blocked: %v", err)
	}
}

func TestValidateSafeURLAndIsSafeURL(t *testing.T) {
	policy := DefaultSSRFPolicy()

	if _, err := ValidateSafeURL("http://", policy); err == nil {
		t.Fatal("invalid URL accepted")
	}

	policy.AllowedHosts["internal.trusted"] = true
	got, err := ValidateSafeURL("http://internal.trusted/path", policy)
	if err != nil || got != "http://internal.trusted/path" {
		t.Fatalf("allowed host bypass: got %q, err %v", got, err)
	}

	// IP literals are resolved locally without DNS, so these work offline.
	if _, err := ValidateSafeURL("http://8.8.8.8/path", policy); err != nil {
		t.Fatalf("public IP literal blocked: %v", err)
	}
	if _, err := ValidateSafeURL("http://127.0.0.1/path", policy); err == nil {
		t.Fatal("loopback IP literal allowed")
	}

	// An over-long hostname fails inside the resolver without any network.
	longHost := strings.Repeat("a", 250) + ".example.com"
	if _, err := ValidateSafeURL("http://"+longHost+"/", policy); err == nil ||
		!strings.Contains(err.Error(), "resolve") {
		t.Fatalf("unresolvable host: %v", err)
	}

	if !IsSafeURL("http://8.8.8.8/path", policy) {
		t.Fatal("IsSafeURL reported public IP literal unsafe")
	}
	if IsSafeURL("http://localhost/", policy) {
		t.Fatal("IsSafeURL reported localhost safe")
	}
}

func TestDefaultDNSResolver(t *testing.T) {
	// IP literals resolve without network access.
	ips, err := defaultDNSResolver(context.Background(), "127.0.0.1")
	if err != nil {
		t.Fatalf("resolve IP literal: %v", err)
	}
	if len(ips) == 0 || !ips[0].Equal(net.ParseIP("127.0.0.1")) {
		t.Fatalf("resolved IPs: %v", ips)
	}
	// An over-long hostname fails inside the resolver without any network.
	if _, err := defaultDNSResolver(context.Background(), strings.Repeat("a", 250)+".example.com"); err == nil {
		t.Fatal("unresolvable hostname succeeded")
	}
}

func TestRequestPort(t *testing.T) {
	cases := []struct {
		raw  string
		want string
	}{
		{"http://example.com", "80"},
		{"https://example.com", "443"},
		{"http://example.com:8080", "8080"},
		{"https://example.com:8443", "8443"},
	}
	for _, tc := range cases {
		req, err := http.NewRequest(http.MethodGet, tc.raw, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got := requestPort(req.URL); got != tc.want {
			t.Fatalf("requestPort(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

func TestCloneTLSConfig(t *testing.T) {
	fromNil := cloneTLSConfig(nil)
	if fromNil == nil || fromNil.ServerName != "" {
		t.Fatalf("cloneTLSConfig(nil) = %+v", fromNil)
	}
	original := &tls.Config{ServerName: "example.com", MinVersion: tls.VersionTLS12}
	cloned := cloneTLSConfig(original)
	if cloned == original {
		t.Fatal("cloneTLSConfig returned the same pointer")
	}
	if cloned.ServerName != "example.com" || cloned.MinVersion != tls.VersionTLS12 {
		t.Fatalf("clone lost fields: %+v", cloned)
	}
}

func TestTransportForRequest(t *testing.T) {
	// Base transport provided, plain HTTP: TLS config left alone.
	base := &http.Transport{}
	transport := NewSSRFSafeTransport(DefaultSSRFPolicy(), base, nil)
	got := transport.transportForRequest("http", "example.com")
	if got == base {
		t.Fatal("expected a clone of the base transport")
	}
	// http.Transport.Clone may materialize a default TLS config; the point is
	// that the plain-HTTP path never sets a ServerName.
	if got.TLSClientConfig != nil && got.TLSClientConfig.ServerName != "" {
		t.Fatalf("http scheme set ServerName: %q", got.TLSClientConfig.ServerName)
	}

	// Base transport provided, HTTPS: ServerName set on a cloned TLS config.
	baseTLS := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}
	transport = NewSSRFSafeTransport(DefaultSSRFPolicy(), baseTLS, nil)
	got = transport.transportForRequest("https", "secure.example.com")
	if got.TLSClientConfig == nil || got.TLSClientConfig.ServerName != "secure.example.com" {
		t.Fatalf("TLS config: %+v", got.TLSClientConfig)
	}
	if !got.TLSClientConfig.InsecureSkipVerify {
		t.Fatal("existing TLS settings not preserved")
	}
	if baseTLS.TLSClientConfig.ServerName != "" {
		t.Fatal("base transport TLS config was mutated")
	}

	// No base transport, HTTPS: falls back to a clone of http.DefaultTransport
	// and creates a fresh TLS config.
	transport = NewSSRFSafeTransport(DefaultSSRFPolicy(), nil, nil)
	got = transport.transportForRequest("https", "fallback.example.com")
	if got.TLSClientConfig == nil || got.TLSClientConfig.ServerName != "fallback.example.com" {
		t.Fatalf("TLS config: %+v", got.TLSClientConfig)
	}
}

type nonCloneableRoundTripper struct{}

func (nonCloneableRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("not a *http.Transport")
}

func TestTransportForRequestNonTransportDefault(t *testing.T) {
	original := http.DefaultTransport
	http.DefaultTransport = nonCloneableRoundTripper{}
	defer func() { http.DefaultTransport = original }()

	transport := NewSSRFSafeTransport(DefaultSSRFPolicy(), nil, nil)
	// HTTPS on a fresh transport with a nil TLS config exercises the
	// cloneTLSConfig(nil) branch.
	got := transport.transportForRequest("https", "example.com")
	if got == nil {
		t.Fatal("expected a fresh transport")
	}
	if got.TLSClientConfig == nil || got.TLSClientConfig.ServerName != "example.com" {
		t.Fatalf("TLS config: %+v", got.TLSClientConfig)
	}
}

func TestRoundTripInvalidRequest(t *testing.T) {
	transport := NewSSRFSafeTransport(DefaultSSRFPolicy(), nil, nil)
	if _, err := transport.RoundTrip(nil); err == nil {
		t.Fatal("nil request accepted")
	}
	if _, err := transport.RoundTrip(&http.Request{}); err == nil {
		t.Fatal("request without URL accepted")
	}
}

func TestRoundTripSchemeValidation(t *testing.T) {
	transport := NewSSRFSafeTransport(DefaultSSRFPolicy(), nil, func(context.Context, string) ([]net.IP, error) {
		t.Fatal("resolver should not be called for blocked schemes")
		return nil, nil
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "ftp://example.com/file", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := transport.RoundTrip(req); err == nil ||
		!strings.Contains(err.Error(), "scheme not allowed") {
		t.Fatalf("ftp scheme: %v", err)
	}

	// A nil AllowedSchemes map defaults to HTTP(S) only.
	nilSchemes := DefaultSSRFPolicy()
	nilSchemes.AllowedSchemes = nil
	transport = NewSSRFSafeTransport(nilSchemes, nil, nil)
	if _, err := transport.RoundTrip(req); err == nil ||
		!strings.Contains(err.Error(), "scheme not allowed") {
		t.Fatalf("nil AllowedSchemes ftp scheme: %v", err)
	}
}

func TestRoundTripResolverFailures(t *testing.T) {
	policy := DefaultSSRFPolicy()

	failing := NewSSRFSafeTransport(policy, nil, func(context.Context, string) ([]net.IP, error) {
		return nil, errors.New("dns exploded")
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "http://example.com/", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := failing.RoundTrip(req); err == nil ||
		!strings.Contains(err.Error(), "resolve") || !strings.Contains(err.Error(), "dns exploded") {
		t.Fatalf("resolver error: %v", err)
	}

	empty := NewSSRFSafeTransport(policy, nil, func(context.Context, string) ([]net.IP, error) {
		return nil, nil
	})
	if _, err := empty.RoundTrip(req); err == nil ||
		!strings.Contains(err.Error(), "no results") {
		t.Fatalf("empty resolution: %v", err)
	}
}

func TestRoundTripAllowedHostBypassesResolution(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	host := strings.TrimPrefix(server.URL, "http://")
	hostname, _, err := net.SplitHostPort(host)
	if err != nil {
		t.Fatal(err)
	}
	policy := DefaultSSRFPolicy()
	policy.AllowedHosts[hostname] = true
	transport := NewSSRFSafeTransport(policy, nil, func(context.Context, string) ([]net.IP, error) {
		t.Fatal("resolver called for allowed host")
		return nil, nil
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL+"/", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("round trip: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode)
	}
}

func TestRoundTripHTTPSPinsIPAndSetsServerName(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.Host, "secure.example.com:") {
			t.Fatalf("host header: %q", r.Host)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "https://"))
	if err != nil {
		t.Fatal(err)
	}
	policy := SSRFPolicy{
		AllowedSchemes: map[string]bool{"https": true},
		AllowedHosts:   map[string]bool{},
	}
	base := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}
	transport := NewSSRFSafeTransport(policy, base, func(_ context.Context, host string) ([]net.IP, error) {
		if host != "secure.example.com" {
			t.Fatalf("resolved host: %q", host)
		}
		return []net.IP{net.ParseIP("127.0.0.1")}, nil
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "https://secure.example.com:"+port+"/", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("round trip: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode)
	}
}

func TestIsPrivateOrReservedRanges(t *testing.T) {
	for _, ip := range []string{
		"10.0.0.1",         // RFC1918
		"100.64.0.1",       // CGNAT range, covered only by the CIDR list
		"192.0.2.1",        // documentation range
		"224.0.0.1",        // multicast
		"0.0.0.0",          // unspecified
		"fc00::1",          // IPv6 ULA, covered only by the CIDR list
		"64:ff9b::808:808", // NAT64 well-known prefix
		"fe80::1",          // link-local
	} {
		if !isPrivateOrReserved(net.ParseIP(ip)) {
			t.Fatalf("isPrivateOrReserved(%q) = false, want true", ip)
		}
	}
	for _, ip := range []string{
		"8.8.8.8",
		"2001:4860:4860::8888",
	} {
		if isPrivateOrReserved(net.ParseIP(ip)) {
			t.Fatalf("isPrivateOrReserved(%q) = true, want false", ip)
		}
	}
}

func TestRoundTripNilResolverUsesDefault(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Loopback-friendly policy; a nil Resolver must fall back to
	// defaultDNSResolver, which resolves the 127.0.0.1 literal locally.
	policy := SSRFPolicy{
		AllowedSchemes: map[string]bool{"http": true},
		AllowedHosts:   map[string]bool{},
	}
	transport := NewSSRFSafeTransport(policy, nil, nil)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL+"/", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("round trip: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode)
	}
}
