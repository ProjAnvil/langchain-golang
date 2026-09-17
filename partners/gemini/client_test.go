package gemini

// Client-construction tests: provider headers and a custom HTTP client from
// the modelconfig reach the genai SDK client and from there the wire.

import (
	"net/http"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// TestNewClientSetsProviderHeaders: every configured header rides on the API
// request.
func TestNewClientSetsProviderHeaders(t *testing.T) {
	var gotHeader http.Header
	server := newGeminiTestServer(t, func(_ *testing.T, r *http.Request, _ map[string]any) {
		gotHeader = r.Header.Clone()
	}, `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("k"),
		modelconfig.WithHeader("X-Org", "projanvil"),
		modelconfig.WithHeader("X-Trace", "abc"),
	)
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if gotHeader.Get("X-Org") != "projanvil" || gotHeader.Get("X-Trace") != "abc" {
		t.Fatalf("provider headers = %v", gotHeader)
	}
}

// TestNewClientUsesConfiguredHTTPClient: a custom HTTP client (here one with
// its own transport) is used for the API call.
func TestNewClientUsesConfiguredHTTPClient(t *testing.T) {
	transport := &recordingTransport{roundTripper: http.DefaultTransport}
	server := newGeminiTestServer(t, func(*testing.T, *http.Request, map[string]any) {},
		`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("k"),
		modelconfig.WithHTTPClient(&http.Client{Transport: transport}),
	)
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if transport.calls != 1 {
		t.Fatalf("custom transport used %d times, want 1", transport.calls)
	}
}

type recordingTransport struct {
	roundTripper http.RoundTripper
	calls        int
}

func (t *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.calls++
	return t.roundTripper.RoundTrip(req)
}
