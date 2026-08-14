package messages

import (
	"reflect"
	"testing"
)

func TestConvertToMessagesStringToHuman(t *testing.T) {
	msgs, err := ConvertToMessages([]any{"hello"})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("len = %d, want 1", len(msgs))
	}
	if msgs[0].Role != RoleHuman {
		t.Fatalf("role = %q, want %q", msgs[0].Role, RoleHuman)
	}
	if msgs[0].Content != "hello" {
		t.Fatalf("content = %q, want %q", msgs[0].Content, "hello")
	}
}

func TestConvertToMessagesDictAssistant(t *testing.T) {
	msgs, err := ConvertToMessages([]any{map[string]any{"role": "assistant", "content": "hi"}})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("len = %d, want 1", len(msgs))
	}
	if msgs[0].Role != RoleAI {
		t.Fatalf("role = %q, want %q", msgs[0].Role, RoleAI)
	}
	if msgs[0].Content != "hi" {
		t.Fatalf("content = %q, want %q", msgs[0].Content, "hi")
	}
}

func TestConvertToMessagesDictContentBlocksRoundTrip(t *testing.T) {
	input := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "text", "text": "look at this"},
			map[string]any{"type": "image", "url": "https://example.com/img.png"},
		},
	}
	msgs, err := ConvertToMessages([]any{input})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("len = %d, want 1", len(msgs))
	}
	if msgs[0].Role != RoleHuman {
		t.Fatalf("role = %q, want %q", msgs[0].Role, RoleHuman)
	}
	if len(msgs[0].ContentBlocks) != 2 {
		t.Fatalf("content blocks len = %d, want 2", len(msgs[0].ContentBlocks))
	}
	tb, ok := msgs[0].ContentBlocks[0].(TextBlock)
	if !ok {
		t.Fatalf("block[0] type = %T, want TextBlock", msgs[0].ContentBlocks[0])
	}
	if tb.Text != "look at this" {
		t.Fatalf("block[0] text = %q, want %q", tb.Text, "look at this")
	}
	ib, ok := msgs[0].ContentBlocks[1].(ImageBlock)
	if !ok {
		t.Fatalf("block[1] type = %T, want ImageBlock", msgs[0].ContentBlocks[1])
	}
	if ib.URL != "https://example.com/img.png" {
		t.Fatalf("block[1] url = %q", ib.URL)
	}
}

func TestConvertToMessagesPassesThroughMessage(t *testing.T) {
	original := AI("hi")
	original.Name = "assistant-1"
	original.ID = "m1"
	original.ToolCalls = []ToolCall{{ID: "c1", Name: "search", Args: map[string]any{"q": "x"}}}

	msgs, err := ConvertToMessages([]any{original})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("len = %d, want 1", len(msgs))
	}
	if !reflect.DeepEqual(msgs[0], original) {
		t.Fatalf("message changed in pass-through:\n got %#v\nwant %#v", msgs[0], original)
	}
}

func TestConvertToMessagesUnsupportedInput(t *testing.T) {
	if _, err := ConvertToMessages([]any{42}); err == nil {
		t.Fatal("expected error for int input")
	}
	if _, err := ConvertToMessages([]any{map[string]any{"content": "missing role"}}); err == nil {
		t.Fatal("expected error for dict missing role")
	}
	if _, err := ConvertToMessages([]any{map[string]any{"role": "unknown-role", "content": "x"}}); err == nil {
		t.Fatal("expected error for unknown role")
	}
}

func TestConvertToOpenAIMessages(t *testing.T) {
	msgs := []Message{
		System("rules"),
		Human("hello"),
		AI("answer"),
		Tool("call-1", "result"),
	}
	out, err := ConvertToOpenAIMessages(msgs)
	if err != nil {
		t.Fatalf("ConvertToOpenAIMessages: %v", err)
	}
	wantRoles := []string{"system", "user", "assistant", "tool"}
	if len(out) != len(wantRoles) {
		t.Fatalf("len = %d, want %d", len(out), len(wantRoles))
	}
	for i, want := range wantRoles {
		if out[i]["role"] != want {
			t.Fatalf("out[%d] role = %v, want %q", i, out[i]["role"], want)
		}
	}
	if out[1]["content"] != "hello" {
		t.Fatalf("out[1] content = %v, want %q", out[1]["content"], "hello")
	}
	if out[3]["tool_call_id"] != "call-1" {
		t.Fatalf("out[3] tool_call_id = %v, want %q", out[3]["tool_call_id"], "call-1")
	}
}

func TestConvertToOpenAIMessagesToolCalls(t *testing.T) {
	msg := AI("")
	msg.ToolCalls = []ToolCall{
		{ID: "call_1", Name: "search", Args: map[string]any{"query": "langchain go"}},
	}
	out, err := ConvertToOpenAIMessages([]Message{msg})
	if err != nil {
		t.Fatalf("ConvertToOpenAIMessages: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	if out[0]["role"] != "assistant" {
		t.Fatalf("role = %v, want assistant", out[0]["role"])
	}
	toolCalls, ok := out[0]["tool_calls"].([]map[string]any)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("tool_calls = %#v, want 1 element", out[0]["tool_calls"])
	}
	first := toolCalls[0]
	if first["type"] != "function" || first["id"] != "call_1" {
		t.Fatalf("tool_call = %#v", first)
	}
	fn, ok := first["function"].(map[string]any)
	if !ok || fn["name"] != "search" {
		t.Fatalf("function = %#v", first["function"])
	}
	if fn["arguments"] != `{"query":"langchain go"}` {
		t.Fatalf("arguments = %v", fn["arguments"])
	}
}

func TestConvertToOpenAIMessagesContentBlocks(t *testing.T) {
	msg := Human("")
	msg.ContentBlocks = []ContentBlock{
		TextBlock{Text: "look at this"},
		ImageBlock{URL: "https://example.com/img.png"},
	}
	out, err := ConvertToOpenAIMessages([]Message{msg})
	if err != nil {
		t.Fatalf("ConvertToOpenAIMessages: %v", err)
	}
	content, ok := out[0]["content"].([]map[string]any)
	if !ok || len(content) != 2 {
		t.Fatalf("content = %#v, want 2 blocks", out[0]["content"])
	}
	if content[0]["type"] != "text" || content[0]["text"] != "look at this" {
		t.Fatalf("content[0] = %#v", content[0])
	}
	if content[1]["type"] != "image" || content[1]["url"] != "https://example.com/img.png" {
		t.Fatalf("content[1] = %#v", content[1])
	}
}
