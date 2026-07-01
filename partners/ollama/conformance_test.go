package ollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/standardtests"
)

// newChatTestServer returns an httptest server that answers both non-streaming
// and streaming /api/chat requests with deterministic content and usage.
func newChatTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Stream {
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"ok"},"done":false}`+"\n")
			_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t2","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":2,"eval_count":1}`+"\n")
			return
		}
		_, _ = w.Write([]byte(`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop","prompt_eval_count":2,"eval_count":1}`))
	}))
}

func TestStandardChatModelBasics(t *testing.T) {
	standardtests.RunChatModelBasics(
		t,
		func(t testing.TB) language.ChatModel {
			server := newChatTestServer(t.(*testing.T))
			t.Cleanup(server.Close)
			model := NewChatModel(modelconfig.WithBaseURL(server.URL))
			return model
		},
		standardtests.ChatModelCapabilities{
			ToolCalling:    true,
			UsageMetadata:  true,
			Streaming:      true,
			ImageInputs:    true,
			StructuredOutput: true,
		},
	)
}

func TestStandardEmbeddingsBasics(t *testing.T) {
	standardtests.RunEmbeddingsBasics(
		t,
		func(t testing.TB) embeddings.Embeddings {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req embedRequest
				_ = json.NewDecoder(r.Body).Decode(&req)
				embeddings := make([][]float64, len(req.Input))
				for i := range req.Input {
					embeddings[i] = []float64{0.1, 0.2, 0.3}
				}
				resp, _ := json.Marshal(map[string]any{
					"model":      "nomic-embed-text",
					"embeddings": embeddings,
				})
				_, _ = w.Write(resp)
			}))
			t.Cleanup(server.Close)
			return NewEmbeddings(modelconfig.WithBaseURL(server.URL))
		},
	)
}
