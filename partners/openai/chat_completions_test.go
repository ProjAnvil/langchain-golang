package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestChatCompletionsInvoke(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-123",
			"model":"gpt-test",
			"choices":[{"message":{"role":"assistant","content":"hello back"}}],
			"usage":{"input_tokens":5,"output_tokens":6,"total_tokens":11}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()

	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hello")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if gotPath != "/chat/completions" {
		t.Fatalf("path = %q, want /chat/completions", gotPath)
	}
	if resp.Content != "hello back" {
		t.Fatalf("content = %q", resp.Content)
	}
	if gotBody["model"] != "gpt-test" {
		t.Fatalf("model = %v", gotBody["model"])
	}
	if msgs, ok := gotBody["messages"].([]any); !ok || len(msgs) != 1 {
		t.Fatalf("messages = %v", gotBody["messages"])
	}
}

func TestChatCompletionsToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-456",
			"model":"gpt-test",
			"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{
				"id":"call_1","type":"function",
				"function":{"name":"search","arguments":"{\"q\":\"hi\"}"}
			}]}}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()

	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("search hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_1" || tc.Name != "search" || tc.Args["q"] != "hi" {
		t.Fatalf("tool call = %#v", tc)
	}
}

func TestChatCompletionsToolResultSerialization(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()

	_, err := model.Invoke(context.Background(), []messages.Message{
		messages.Tool("toolcall-9", "result"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	msgs := gotBody["messages"].([]any)
	msg := msgs[0].(map[string]any)
	if msg["role"] != "tool" || msg["tool_call_id"] != "toolcall-9" {
		t.Fatalf("tool message = %#v", msg)
	}
}

func TestDefaultStillResponsesAPI(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if gotPath != "/responses" {
		t.Fatalf("path = %q, want /responses (default unchanged)", gotPath)
	}
}
