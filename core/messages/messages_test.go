package messages

import (
	"encoding/json"
	"testing"
)

func TestMessageJSONRoundTrip(t *testing.T) {
	original := AI("use the search tool")
	original.ID = "msg_123"
	original.ToolCalls = []ToolCall{
		{
			ID:   "call_123",
			Name: "search",
			Args: map[string]any{"query": "langchain go"},
		},
	}
	original.UsageMetadata = UsageMetadata{
		InputTokens:  10,
		OutputTokens: 5,
		TotalTokens:  15,
	}
	original.ResponseMetadata = map[string]any{"model": "fake-chat"}

	data, err := MarshalJSONStable(original)
	if err != nil {
		t.Fatalf("marshal message: %v", err)
	}

	decoded, err := UnmarshalJSONStable(data)
	if err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}

	if decoded.Role != RoleAI {
		t.Fatalf("role mismatch: got %q", decoded.Role)
	}
	if decoded.ToolCalls[0].Name != "search" {
		t.Fatalf("tool call name mismatch: got %q", decoded.ToolCalls[0].Name)
	}
	if decoded.UsageMetadata.TotalTokens != 15 {
		t.Fatalf("usage mismatch: got %d", decoded.UsageMetadata.TotalTokens)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw json: %v", err)
	}
	if raw["role"] != string(RoleAI) {
		t.Fatalf("serialized role mismatch: got %v", raw["role"])
	}
}

func TestMessageUtilities(t *testing.T) {
	msgs := []Message{
		System("rules"),
		Human("hello"),
		Human("again"),
		AI("").WithContentBlocks([]ContentBlock{{"type": "text", "text": "answer"}}),
		Tool("call-1", "result"),
	}

	if got := Text(msgs[3]); got != "answer" {
		t.Fatalf("Text = %q, want answer", got)
	}
	if got := BufferString(msgs[:2]); got != "System: rules\nHuman: hello" {
		t.Fatalf("BufferString = %q", got)
	}
	filtered := Filter(msgs, FilterOptions{IncludeRoles: []Role{RoleHuman}})
	if len(filtered) != 2 {
		t.Fatalf("filtered len = %d, want 2", len(filtered))
	}
	merged := MergeRuns(msgs)
	if len(merged) != 4 || merged[1].Content != "hello\nagain" {
		t.Fatalf("unexpected merged messages: %#v", merged)
	}
	trimmed := Trim(msgs, len("result"), true)
	if len(trimmed) != 1 || trimmed[0].Role != RoleTool {
		t.Fatalf("unexpected trimmed messages: %#v", trimmed)
	}
}

func TestMessagesDictRoundTripAndClone(t *testing.T) {
	original := []Message{{Role: RoleAI, ContentBlocks: []ContentBlock{{"type": "text", "text": "x"}}}}
	dicts, err := MessagesToDict(original)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := MessagesFromDict(dicts)
	if err != nil {
		t.Fatal(err)
	}
	if Text(decoded[0]) != "x" {
		t.Fatalf("decoded text = %q", Text(decoded[0]))
	}
	clone := Clone(original[0])
	clone.ContentBlocks[0]["text"] = "changed"
	if Text(original[0]) != "x" {
		t.Fatal("clone mutation changed original")
	}
}
