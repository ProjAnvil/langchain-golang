package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// Mirrors libs/partners/openai/tests/unit_tests/test_tools.py
// ::test_custom_tool (schema serialization, payload input replay, and
// custom_tool_call parsing from chat_models/base.py:4883-4891).

func newTestCustomTool(t *testing.T, opts ...CustomToolOption) CustomTool {
	t.Helper()
	tool, err := NewCustomTool("my_tool", "Do thing.", func(ctx context.Context, x string) (string, error) {
		return "a" + x, nil
	}, opts...)
	if err != nil {
		t.Fatal(err)
	}
	return tool
}

func TestCustomToolSchemaSerialization(t *testing.T) {
	format := map[string]any{"type": "grammar", "syntax": "lark", "definition": "..."}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got map[string]any
		_ = json.NewDecoder(r.Body).Decode(&got)
		toolsList, ok := got["tools"].([]any)
		if !ok || len(toolsList) != 1 {
			t.Errorf("tools = %v, want one entry", got["tools"])
			return
		}
		entry, _ := toolsList[0].(map[string]any)
		if entry["type"] != "custom" || entry["name"] != "my_tool" || entry["description"] != "Do thing." {
			t.Errorf("tool entry = %v, want {type:custom name:my_tool description:Do thing.}", entry)
		}
		gotFormat, ok := entry["format"].(map[string]any)
		if !ok || gotFormat["syntax"] != "lark" {
			t.Errorf("format = %v, want grammar/lark", entry["format"])
		}
		if _, hasParams := entry["parameters"]; hasParams {
			t.Errorf("custom tools must not send a parameters schema: %v", entry)
		}
		_, _ = w.Write([]byte(`{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	bound, err := model.BindTools([]coretools.Tool{newTestCustomTool(t, WithCustomToolFormat(format))})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bound.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
}

func TestCustomToolSchemaWithoutFormat(t *testing.T) {
	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	bound, err := model.BindTools([]coretools.Tool{newTestCustomTool(t)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bound.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	entry := got["tools"].([]any)[0].(map[string]any)
	if _, hasFormat := entry["format"]; hasFormat {
		t.Fatalf("format key must be omitted when unset: %v", entry)
	}
}

func TestCustomToolInvoke(t *testing.T) {
	// Python: my_tool.invoke({"args": {...}, "extras": {"type": "custom_tool_call"}})
	// runs the wrapped func on the freeform string. The Go ToolCall contract
	// carries the string under Args["__arg1"] (see parsing test below).
	result, err := newTestCustomTool(t).Invoke(context.Background(), map[string]any{"__arg1": "b"})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if result.Content != "ab" {
		t.Fatalf("content = %q, want ab", result.Content)
	}
	if _, err := newTestCustomTool(t).Invoke(context.Background(), map[string]any{"__arg1": 42}); err == nil {
		t.Fatal("expected error for non-string input")
	}
}

func TestCustomToolCallParsing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"resp_1","model":"gpt-test",
			"output":[{"type":"custom_tool_call","id":"ctc_abc123","call_id":"abc","name":"my_tool","input":"a"}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("Use the tool")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "abc" || tc.Name != "my_tool" || tc.Args["__arg1"] != "a" {
		t.Fatalf("tool call = %#v, want {id:abc name:my_tool args:{__arg1:a}}", tc)
	}
	if len(resp.ContentBlocks) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(resp.ContentBlocks))
	}
	block, ok := resp.ContentBlocks[0].(messages.NonStandardContentBlock)
	if !ok || block.Type != "custom_tool_call" {
		t.Fatalf("content block = %#v, want NonStandardContentBlock custom_tool_call", resp.ContentBlocks[0])
	}
	if block.Value["call_id"] != "abc" || block.Value["input"] != "a" {
		t.Fatalf("block value = %v, want raw item passthrough", block.Value)
	}
}

func TestCustomToolMessageReplay(t *testing.T) {
	// Mirrors the payload["input"] assertion in test_tools.py::test_custom_tool
	// (the Go Responses input items carry role/content without "type":"message"
	// — the port's existing shape — so the user item differs accordingly).
	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	}))
	defer server.Close()

	history := []messages.Message{
		messages.Human("Use the tool"),
		messages.AI("").WithContentBlocks([]messages.ContentBlock{
			messages.NonStandardContentBlock{
				Type: "custom_tool_call",
				Value: map[string]any{
					"type": "custom_tool_call", "id": "ctc_abc123",
					"call_id": "abc", "name": "my_tool", "input": "a",
				},
			},
		}),
		{
			Role:       messages.RoleTool,
			ToolCallID: "abc",
			ContentBlocks: []messages.ContentBlock{
				messages.NonStandardContentBlock{
					Type:  "custom_tool_call_output",
					Value: map[string]any{"type": "custom_tool_call_output", "output": "ab"},
				},
			},
		},
	}
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	if _, err := model.Invoke(context.Background(), history); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	input, ok := got["input"].([]any)
	if !ok || len(input) != 3 {
		t.Fatalf("input = %v, want 3 items", got["input"])
	}
	user, _ := input[0].(map[string]any)
	if user["role"] != "user" || user["content"] != "Use the tool" {
		t.Fatalf("input[0] = %v, want user message", user)
	}
	call, _ := input[1].(map[string]any)
	if call["type"] != "custom_tool_call" || call["id"] != "ctc_abc123" ||
		call["call_id"] != "abc" || call["name"] != "my_tool" || call["input"] != "a" {
		t.Fatalf("input[1] = %v, want custom_tool_call passthrough", call)
	}
	output, _ := input[2].(map[string]any)
	if output["type"] != "custom_tool_call_output" || output["call_id"] != "abc" || output["output"] != "ab" {
		t.Fatalf("input[2] = %v, want custom_tool_call_output", output)
	}
}
