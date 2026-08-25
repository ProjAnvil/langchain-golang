package messages

import (
	"reflect"
	"testing"
)

// Mirrors default_tool_parser (messages/tool.py:349-383): the OpenAI
// "function" wire shape parses into ToolCall.
func TestDefaultToolParserOpenAIShape(t *testing.T) {
	raw := []map[string]any{
		{
			"id":   "call_1",
			"type": "function",
			"function": map[string]any{
				"name":      "search",
				"arguments": `{"q":"go"}`,
			},
		},
	}
	calls, invalid := DefaultToolParser(raw)
	if len(invalid) != 0 {
		t.Fatalf("invalid: %+v", invalid)
	}
	want := ToolCall{ID: "call_1", Name: "search", Args: map[string]any{"q": "go"}}
	if len(calls) != 1 || !reflect.DeepEqual(calls[0], want) {
		t.Fatalf("calls: %+v, want %+v", calls, want)
	}
}

// Entries without a "function" key are skipped entirely — including the
// LangChain shape {"id","name","args"} that the lenient ConvertToMessages
// parser (convert.go parseToolCalls) accepts. This contrast is the point of
// the exported strict parser.
func TestDefaultToolParserSkipsNonFunctionEntries(t *testing.T) {
	raw := []map[string]any{
		{},
		{"id": "call_x"},
		{"id": "call_lc", "name": "search", "args": map[string]any{"q": "go"}},
	}
	calls, invalid := DefaultToolParser(raw)
	if len(calls) != 0 || len(invalid) != 0 {
		t.Fatalf("calls=%+v invalid=%+v, want both empty", calls, invalid)
	}
}

// JSON parse failures become InvalidToolCall with the raw arguments string
// preserved and no error message (Python passes error=None).
func TestDefaultToolParserInvalidJSON(t *testing.T) {
	raw := []map[string]any{
		{
			"id": "call_2",
			"function": map[string]any{
				"name":      "search",
				"arguments": `{"q":`,
			},
		},
	}
	calls, invalid := DefaultToolParser(raw)
	if len(calls) != 0 {
		t.Fatalf("calls: %+v", calls)
	}
	want := InvalidToolCallBlock{ID: "call_2", Name: "search", Args: `{"q":`}
	if len(invalid) != 1 || !reflect.DeepEqual(invalid[0], want) {
		t.Fatalf("invalid: %+v, want %+v", invalid, want)
	}
}

// A missing/empty function name becomes "" (Python: name=function_name or "").
func TestDefaultToolParserEmptyName(t *testing.T) {
	raw := []map[string]any{
		{"function": map[string]any{"arguments": `{}`}},
	}
	calls, invalid := DefaultToolParser(raw)
	if len(invalid) != 0 {
		t.Fatalf("invalid: %+v", invalid)
	}
	want := ToolCall{Name: "", Args: map[string]any{}}
	if len(calls) != 1 || !reflect.DeepEqual(calls[0], want) {
		t.Fatalf("calls: %+v, want %+v", calls, want)
	}
}

// arguments "null" parses to the empty map (Python: args=function_args or {}).
func TestDefaultToolParserNullArguments(t *testing.T) {
	raw := []map[string]any{
		{"id": "call_3", "function": map[string]any{"name": "ping", "arguments": `null`}},
	}
	calls, invalid := DefaultToolParser(raw)
	if len(invalid) != 0 {
		t.Fatalf("invalid: %+v", invalid)
	}
	want := ToolCall{ID: "call_3", Name: "ping", Args: map[string]any{}}
	if len(calls) != 1 || !reflect.DeepEqual(calls[0], want) {
		t.Fatalf("calls: %+v, want %+v", calls, want)
	}
}

// Mixed input: valid calls and invalid calls are returned in their own
// lists, both in input order.
func TestDefaultToolParserMixed(t *testing.T) {
	raw := []map[string]any{
		{"id": "ok", "function": map[string]any{"name": "a", "arguments": `{}`}},
		{"id": "bad", "function": map[string]any{"name": "b", "arguments": `nope`}},
		{"id": "skipped"},
	}
	calls, invalid := DefaultToolParser(raw)
	if len(calls) != 1 || calls[0].ID != "ok" || calls[0].Name != "a" {
		t.Fatalf("calls: %+v", calls)
	}
	if len(invalid) != 1 || invalid[0].ID != "bad" || invalid[0].Name != "b" || invalid[0].Args != "nope" {
		t.Fatalf("invalid: %+v", invalid)
	}
}

// Mirrors default_tool_chunk_parser (messages/tool.py:386-412): every entry
// yields a chunk, function args stay raw strings, id/index pass through.
func TestDefaultToolChunkParser(t *testing.T) {
	raw := []map[string]any{
		{
			"id":    "call_1",
			"index": 0,
			"function": map[string]any{
				"name":      "search",
				"arguments": `{"q":`,
			},
		},
		{"index": 1},
	}
	chunks := DefaultToolChunkParser(raw)
	if len(chunks) != 2 {
		t.Fatalf("chunks: %+v", chunks)
	}
	wantFirst := ToolCallChunkBlock{ID: "call_1", Name: "search", Args: `{"q":`, Index: 0}
	if !reflect.DeepEqual(chunks[0], wantFirst) {
		t.Fatalf("chunk 0: %+v, want %+v", chunks[0], wantFirst)
	}
	// Entries without "function" yield zero name/args (Python: None).
	wantSecond := ToolCallChunkBlock{Index: 1}
	if !reflect.DeepEqual(chunks[1], wantSecond) {
		t.Fatalf("chunk 1: %+v, want %+v", chunks[1], wantSecond)
	}
}

// Empty input yields empty output for both parsers.
func TestDefaultToolParsersEmpty(t *testing.T) {
	calls, invalid := DefaultToolParser(nil)
	if len(calls) != 0 || len(invalid) != 0 {
		t.Fatalf("calls=%+v invalid=%+v", calls, invalid)
	}
	if chunks := DefaultToolChunkParser(nil); len(chunks) != 0 {
		t.Fatalf("chunks: %+v", chunks)
	}
}
