package anthropic

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

// TestInvokeStructured_FunctionCalling mirrors Python's default
// with_structured_output(method="function_calling") (chat_models.py:1938,2050):
// the model is forced to call a single tool whose input schema IS the requested
// schema, and InvokeStructured returns a message whose Content is the JSON of
// the tool_use input args.
func TestInvokeStructured_FunctionCalling(t *testing.T) {
	// The server captures the request and asserts tool_choice forces the
	// synthesized tool, then returns a tool_use content block.
	var gotToolChoice any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		gotToolChoice = body["tool_choice"]
		// Echo the model_provider tag so ResponseMetadata carries it.
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":    "msg_1",
			"type":  "message",
			"role":  "assistant",
			"model": "claude-sonnet-4-5",
			"content": []map[string]any{
				{
					"type":  "tool_use",
					"id":    "toolu_1",
					"name":  "weather_schema",
					"input": map[string]any{"condition": "sunny", "temperature": 72},
				},
			},
			"stop_reason": "tool_use",
			"usage":       map[string]any{"input_tokens": 10, "output_tokens": 5},
		})
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
		modelconfig.WithModel("claude-sonnet-4-5"),
	)

	sch := schema.Object(map[string]schema.Schema{
		"temperature": schema.String("temp"),
		"condition":   schema.String("cond"),
	}, "temperature", "condition")
	sch["title"] = "weather_schema"

	resp, err := model.InvokeStructured(context.Background(), []messages.Message{messages.Human("weather?")}, sch)
	if err != nil {
		t.Fatalf("InvokeStructured: %v", err)
	}

	// tool_choice must force the synthesized tool by name.
	tc, ok := gotToolChoice.(map[string]any)
	if !ok || tc["type"] != "tool" || tc["name"] != "weather_schema" {
		t.Fatalf("tool_choice mismatch: %#v", gotToolChoice)
	}

	// Content is the JSON of the tool_use input.
	var parsed map[string]any
	if err := json.Unmarshal([]byte(resp.Content), &parsed); err != nil {
		t.Fatalf("Content not JSON: %v (content=%q)", err, resp.Content)
	}
	if parsed["condition"] != "sunny" || parsed["temperature"] != float64(72) {
		t.Fatalf("structured content mismatch: %#v", parsed)
	}
}
