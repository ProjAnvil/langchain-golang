package messages

import (
	"encoding/json"
	"testing"
)

func TestParseContentBlockServerToolCall(t *testing.T) {
	input := map[string]any{
		"type":  "server_tool_call",
		"id":    "stc_1",
		"name":  "web_search",
		"args":  map[string]any{"query": "langchain go"},
		"index": 2,
	}
	block := ParseContentBlock(input)
	stc, ok := block.(ServerToolCall)
	if !ok {
		t.Fatalf("expected ServerToolCall, got %T", block)
	}
	if stc.ID != "stc_1" {
		t.Fatalf("id mismatch: %#v", stc)
	}
	if stc.Name != "web_search" {
		t.Fatalf("name mismatch: %#v", stc)
	}
	if got := stc.Args["query"]; got != "langchain go" {
		t.Fatalf("args mismatch: %#v", stc.Args)
	}
	if stc.Index != 2 {
		t.Fatalf("index mismatch: %#v", stc)
	}

	out := BlockToMap(stc)
	if out["type"] != "server_tool_call" {
		t.Fatalf("type not preserved: %#v", out)
	}
	if out["id"] != "stc_1" || out["name"] != "web_search" {
		t.Fatalf("round-trip mismatch: %#v", out)
	}
}

func TestParseContentBlockServerToolCallChunk(t *testing.T) {
	input := map[string]any{
		"type":  "server_tool_call_chunk",
		"id":    "stc_2",
		"name":  "code_exec",
		"args":  "{\"code\": \"print(",
		"index": 3,
	}
	block := ParseContentBlock(input)
	chunk, ok := block.(ServerToolCallChunk)
	if !ok {
		t.Fatalf("expected ServerToolCallChunk, got %T", block)
	}
	if chunk.ID != "stc_2" || chunk.Name != "code_exec" {
		t.Fatalf("field mismatch: %#v", chunk)
	}
	if chunk.Args != "{\"code\": \"print(" {
		t.Fatalf("args mismatch: %#v", chunk)
	}
	if chunk.Index != 3 {
		t.Fatalf("index mismatch: %#v", chunk)
	}

	out := BlockToMap(chunk)
	if out["type"] != "server_tool_call_chunk" {
		t.Fatalf("type not preserved: %#v", out)
	}
	if out["id"] != "stc_2" || out["name"] != "code_exec" {
		t.Fatalf("round-trip mismatch: %#v", out)
	}
}

func TestParseContentBlockServerToolResult(t *testing.T) {
	input := map[string]any{
		"type":         "server_tool_result",
		"id":           "str_1",
		"tool_call_id": "stc_1",
		"status":       "success",
		"output":       "42",
		"index":        4,
	}
	block := ParseContentBlock(input)
	result, ok := block.(ServerToolResult)
	if !ok {
		t.Fatalf("expected ServerToolResult, got %T", block)
	}
	if result.ID != "str_1" {
		t.Fatalf("id mismatch: %#v", result)
	}
	if result.ToolCallID != "stc_1" {
		t.Fatalf("tool_call_id mismatch: %#v", result)
	}
	if result.Status != "success" {
		t.Fatalf("status mismatch: %#v", result)
	}
	if result.Output != "42" {
		t.Fatalf("output mismatch: %#v", result)
	}
	if result.Index != 4 {
		t.Fatalf("index mismatch: %#v", result)
	}

	out := BlockToMap(result)
	if out["type"] != "server_tool_result" {
		t.Fatalf("type not preserved: %#v", out)
	}
	if out["tool_call_id"] != "stc_1" || out["status"] != "success" {
		t.Fatalf("round-trip mismatch: %#v", out)
	}
}

func TestCloneBlockServerToolTypes(t *testing.T) {
	call := ServerToolCall{
		ID:     "stc_1",
		Name:   "web_search",
		Args:   map[string]any{"query": "go"},
		Index:  1,
		Extras: map[string]any{"trace": "t1"},
	}
	callClone := CloneBlock(call).(ServerToolCall)
	callClone.Args["query"] = "changed"
	callClone.Extras["trace"] = "changed"
	if call.Args["query"] != "go" || call.Extras["trace"] != "t1" {
		t.Fatal("ServerToolCall clone mutation changed original")
	}

	result := ServerToolResult{
		ID:         "str_1",
		ToolCallID: "stc_1",
		Status:     "success",
		Output:     map[string]any{"n": 1},
		Extras:     map[string]any{"trace": "t2"},
	}
	resultClone := CloneBlock(result).(ServerToolResult)
	resultClone.Output.(map[string]any)["n"] = 2
	resultClone.Extras["trace"] = "changed"
	if result.Output.(map[string]any)["n"] != 1 || result.Extras["trace"] != "t2" {
		t.Fatal("ServerToolResult clone mutation changed original")
	}
}

func TestMessageJSONServerToolCallRoundTrip(t *testing.T) {
	msg := Message{
		Role: RoleAI,
		ContentBlocks: []ContentBlock{
			ServerToolCall{ID: "stc_1", Name: "web_search", Args: map[string]any{"query": "go"}},
		},
	}

	data, err := MarshalJSONStable(msg)
	if err != nil {
		t.Fatalf("marshal message: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw json: %v", err)
	}
	blocks, ok := raw["content_blocks"].([]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("unexpected content_blocks: %#v", raw["content_blocks"])
	}
	first, ok := blocks[0].(map[string]any)
	if !ok {
		t.Fatalf("unexpected block shape: %#v", blocks[0])
	}
	if first["type"] != "server_tool_call" {
		t.Fatalf("type not preserved: %#v", first)
	}

	decoded, err := UnmarshalJSONStable(data)
	if err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	stc, ok := decoded.ContentBlocks[0].(ServerToolCall)
	if !ok {
		t.Fatalf("expected ServerToolCall, got %T", decoded.ContentBlocks[0])
	}
	if stc.ID != "stc_1" || stc.Name != "web_search" {
		t.Fatalf("round-trip mismatch: %#v", stc)
	}
	if got := stc.Args["query"]; got != "go" {
		t.Fatalf("args mismatch: %#v", stc.Args)
	}
}
