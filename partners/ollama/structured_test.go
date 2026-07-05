package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
)

// TestInvokeStructured_JsonSchema mirrors Python's default
// with_structured_output(method="json_schema") (chat_models.py:1448): the
// request carries the JSON schema in the format field, and the model returns
// JSON text conforming to it.
func TestInvokeStructured_JsonSchema(t *testing.T) {
	var gotFormat any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotFormat = body["format"]
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "llama3",
			"done":    true,
			"message": map[string]any{"role": "assistant", "content": `{"answer":"ok"}`},
		})
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("llama3"),
	)

	sch := schema.Object(map[string]schema.Schema{"answer": schema.String("answer")}, "answer")
	resp, err := model.InvokeStructured(context.Background(), []messages.Message{messages.Human("hi")}, sch)
	if err != nil {
		t.Fatalf("InvokeStructured: %v", err)
	}

	// format field must carry the JSON schema (a map, not the string "json").
	if _, ok := gotFormat.(map[string]any); !ok {
		t.Fatalf("format should be the JSON schema map, got %#v", gotFormat)
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Content), &parsed); err != nil {
		t.Fatalf("Content not JSON: %v (content=%q)", err, resp.Content)
	}
	if parsed["answer"] != "ok" {
		t.Fatalf("structured content mismatch: %#v", parsed)
	}
}
