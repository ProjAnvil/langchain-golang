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

// Dict-with-name / developer / tool_call_id / tool_calls scenarios mirroring
// Python's test_base.py `_convert_dict_to_message_*` tests.

func TestConvertToMessagesDictWithName(t *testing.T) {
	msgs, err := ConvertToMessages([]any{map[string]any{
		"role": "human", "content": "hi", "name": "alice",
	}})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if msgs[0].Name != "alice" {
		t.Fatalf("name = %q, want alice", msgs[0].Name)
	}
}

func TestConvertToMessagesDictDeveloperRole(t *testing.T) {
	msgs, err := ConvertToMessages([]any{map[string]any{
		"role": "developer", "content": "instructions",
	}})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if msgs[0].Role != RoleSystem {
		t.Fatalf("role = %q, want system (developer alias)", msgs[0].Role)
	}
}

func TestConvertToMessagesDictToolCallID(t *testing.T) {
	msgs, err := ConvertToMessages([]any{map[string]any{
		"role": "tool", "content": "result", "tool_call_id": "call-1",
	}})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if msgs[0].Role != RoleTool || msgs[0].ToolCallID != "call-1" {
		t.Fatalf("message = %#v", msgs[0])
	}
}

func TestConvertToMessagesDictAIToolCalls(t *testing.T) {
	msgs, err := ConvertToMessages([]any{map[string]any{
		"role": "ai", "content": "",
		"tool_calls": []any{map[string]any{
			"id": "c1", "type": "function",
			"function": map[string]any{"name": "search", "arguments": `{"q":"x"}`},
		}},
	}})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if len(msgs[0].ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(msgs[0].ToolCalls))
	}
	tc := msgs[0].ToolCalls[0]
	if tc.ID != "c1" || tc.Name != "search" || tc.Args["q"] != "x" {
		t.Fatalf("tool call = %#v", tc)
	}
}

func TestConvertToMessagesPointer(t *testing.T) {
	msg := AI("hi")
	msgs, err := ConvertToMessages([]any{&msg})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	if len(msgs) != 1 || msgs[0].Content != "hi" {
		t.Fatalf("pointer pass-through = %#v", msgs)
	}

	var nilMsg *Message
	if _, err := ConvertToMessages([]any{nilMsg}); err == nil {
		t.Fatal("expected error for nil *Message")
	}
}

func TestConvertToMessagesDictEdgeCases(t *testing.T) {
	// "type" is accepted as an alias for "role".
	msgs, err := ConvertToMessages([]any{map[string]any{"type": "user", "content": "hi"}})
	if err != nil {
		t.Fatalf("ConvertToMessages with type key: %v", err)
	}
	if msgs[0].Role != RoleHuman || msgs[0].Content != "hi" {
		t.Fatalf("message = %#v", msgs[0])
	}

	// Non-string role is rejected.
	if _, err := ConvertToMessages([]any{map[string]any{"role": 42, "content": "x"}}); err == nil {
		t.Fatal("expected error for non-string role")
	}

	// Missing content key is rejected.
	if _, err := ConvertToMessages([]any{map[string]any{"role": "human"}}); err == nil {
		t.Fatal("expected error for missing content")
	}

	// nil content becomes the empty string.
	msgs, err = ConvertToMessages([]any{map[string]any{"role": "human", "content": nil}})
	if err != nil {
		t.Fatalf("ConvertToMessages with nil content: %v", err)
	}
	if msgs[0].Content != "" {
		t.Fatalf("nil content = %q, want empty", msgs[0].Content)
	}

	// Unsupported content types are rejected.
	if _, err := ConvertToMessages([]any{map[string]any{"role": "human", "content": 42}}); err == nil {
		t.Fatal("expected error for int content")
	}

	// Unsupported elements inside a content list are rejected.
	if _, err := ConvertToMessages([]any{map[string]any{"role": "human", "content": []any{42}}}); err == nil {
		t.Fatal("expected error for int content block")
	}

	// Strings inside a content list become text blocks.
	msgs, err = ConvertToMessages([]any{map[string]any{"role": "human", "content": []any{"plain"}}})
	if err != nil {
		t.Fatalf("ConvertToMessages with string block: %v", err)
	}
	tb, ok := msgs[0].ContentBlocks[0].(TextBlock)
	if !ok || tb.Text != "plain" {
		t.Fatalf("string content block = %#v", msgs[0].ContentBlocks[0])
	}

	// id, response_metadata and additional_kwargs are carried over.
	msgs, err = ConvertToMessages([]any{map[string]any{
		"role": "ai", "content": "x", "id": "m1",
		"response_metadata": map[string]any{"model": "fake"},
		"additional_kwargs": map[string]any{"extra": true},
	}})
	if err != nil {
		t.Fatalf("ConvertToMessages with metadata: %v", err)
	}
	if msgs[0].ID != "m1" {
		t.Fatalf("id = %q, want m1", msgs[0].ID)
	}
	if msgs[0].ResponseMetadata["model"] != "fake" {
		t.Fatalf("response_metadata = %#v", msgs[0].ResponseMetadata)
	}
	if msgs[0].AdditionalKwargs["extra"] != true {
		t.Fatalf("additional_kwargs = %#v", msgs[0].AdditionalKwargs)
	}
}

func TestConvertToMessagesToolCallShapes(t *testing.T) {
	msgs, err := ConvertToMessages([]any{map[string]any{
		"role": "ai", "content": "",
		"tool_calls": []any{
			"not-a-map", // skipped
			map[string]any{"id": "c1", "name": "search", "args": map[string]any{"q": "x"}}, // LangChain shape
			map[string]any{"id": "c2", "type": "function", "function": map[string]any{ // invalid JSON args
				"name": "bad", "arguments": "{invalid"}},
			map[string]any{"id": "c3", "type": "function", "function": map[string]any{ // args already a map
				"name": "m", "arguments": map[string]any{"k": "v"}}},
		},
	}})
	if err != nil {
		t.Fatalf("ConvertToMessages: %v", err)
	}
	calls := msgs[0].ToolCalls
	if len(calls) != 3 {
		t.Fatalf("tool calls = %d, want 3 (non-map skipped)", len(calls))
	}
	if calls[0].ID != "c1" || calls[0].Name != "search" || calls[0].Args["q"] != "x" {
		t.Fatalf("langchain-shape tool call = %#v", calls[0])
	}
	if calls[1].ID != "c2" || calls[1].Name != "bad" || calls[1].Args != nil {
		t.Fatalf("invalid-JSON tool call = %#v, want nil args", calls[1])
	}
	if calls[2].ID != "c3" || calls[2].Name != "m" || calls[2].Args["k"] != "v" {
		t.Fatalf("map-args tool call = %#v", calls[2])
	}
}

func TestConvertToOpenAIMessagesEdgeCases(t *testing.T) {
	// Unknown roles are rejected.
	if _, err := ConvertToOpenAIMessages([]Message{{Role: "weird", Content: "x"}}); err == nil {
		t.Fatal("expected error for unknown role")
	}

	// The name field is emitted when set.
	named := Human("hi")
	named.Name = "alice"
	out, err := ConvertToOpenAIMessages([]Message{named})
	if err != nil {
		t.Fatalf("ConvertToOpenAIMessages: %v", err)
	}
	if out[0]["name"] != "alice" {
		t.Fatalf("name = %v, want alice", out[0]["name"])
	}
}
