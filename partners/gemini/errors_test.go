package gemini

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// errorGeminiServer answers every generateContent call with the given status
// and JSON error body, shaped like the Gemini API's error envelope.
func errorGeminiServer(status int, message string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = fmt.Fprintf(w, `{"error":{"code":%d,"message":%q,"status":"RESOURCE_EXHAUSTED"}}`, status, message)
	}))
}

func errorTestModel(t *testing.T, status int, message string) ChatModel {
	t.Helper()
	server := errorGeminiServer(status, message)
	t.Cleanup(server.Close)
	return NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gemini-test"),
		modelconfig.WithAPIKey("test-key"),
	)
}

// TestInvokeErrorClassification asserts that genai APIError responses surface
// as standard model error kinds (langchain-core #39538 parity).
func TestInvokeErrorClassification(t *testing.T) {
	cases := []struct {
		name   string
		status int
		want   error
	}{
		{"rate limited", 429, lcerrors.ErrModelRateLimit},
		{"auth", 401, lcerrors.ErrModelAuth},
		{"not found", 404, lcerrors.ErrModelNotFound},
		{"server", 500, lcerrors.ErrModelServer},
		{"invalid request", 400, lcerrors.ErrModelInvalidRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := errorTestModel(t, tc.status, "boom")
			_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
			if err == nil {
				t.Fatal("expected error")
			}
			if !errors.Is(err, tc.want) {
				t.Fatalf("errors.Is(err, %v) = false, err=%v", tc.want, err)
			}
			pe, ok := errors.AsType[*lcerrors.ProviderError](err)
			if !ok {
				t.Fatalf("error does not carry *ProviderError: %T", err)
			}
			if pe.Provider != "gemini" {
				t.Fatalf("Provider = %q, want gemini", pe.Provider)
			}
		})
	}
}

// TestInvokeContextOverflowSniff asserts a 400 body carrying a context-window
// marker classifies as context overflow (Python ContextOverflowError parity).
func TestInvokeContextOverflowSniff(t *testing.T) {
	model := errorTestModel(t, 400, "request exceeds maximum context length of 8192 tokens")
	_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if !errors.Is(err, lcerrors.ErrModelContextOverflow) {
		t.Fatalf("expected ErrModelContextOverflow, got %v", err)
	}
}
