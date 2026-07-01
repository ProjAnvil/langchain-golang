package security

import (
	"context"
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
