package outputparser

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestOutputFunctionsParser(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{
			"name":      "cookie",
			"arguments": `{"name":"chip","age":3}`,
		},
	}

	got, err := (OutputFunctionsParser{ArgsOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got != `{"name":"chip","age":3}` {
		t.Fatalf("got %#v", got)
	}

	parsed, err := (JSONOutputFunctionsParser{ArgsOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse json: %v", err)
	}
	if parsed.(map[string]any)["name"] != "chip" {
		t.Fatalf("parsed: %#v", parsed)
	}

	name, err := (JSONKeyOutputFunctionsParser{KeyName: "name"}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse key: %v", err)
	}
	if name != "chip" {
		t.Fatalf("name: %#v", name)
	}
}

func TestJSONOutputToolsParserFromMessageToolCalls(t *testing.T) {
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{
		{ID: "call-1", Name: "search", Args: map[string]any{"query": "go"}},
	}

	got, err := (JSONOutputToolsParser{ReturnID: true, FirstToolOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	call := got.(ParsedToolCall)
	if call.ID != "call-1" || call.Type != "search" || call.Args["query"] != "go" {
		t.Fatalf("call: %#v", call)
	}
}

func TestJSONOutputToolsParserFromAdditionalKwargs(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"tool_calls": []map[string]any{
			{
				"id": "call-1",
				"function": map[string]any{
					"name":      "search",
					"arguments": `{"query":"go"}`,
				},
			},
			{
				"id": "call-2",
				"function": map[string]any{
					"name":      "lookup",
					"arguments": ``,
				},
			},
		},
	}

	got, err := (JSONOutputToolsParser{ReturnID: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	calls := got.([]ParsedToolCall)
	if len(calls) != 2 {
		t.Fatalf("calls: %#v", calls)
	}
	if calls[0].Type != "search" || calls[0].Args["query"] != "go" {
		t.Fatalf("first call: %#v", calls[0])
	}
	if calls[1].Type != "lookup" || len(calls[1].Args) != 0 {
		t.Fatalf("second call: %#v", calls[1])
	}
}

func TestJSONOutputKeyToolsParser(t *testing.T) {
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{
		{Name: "search", Args: map[string]any{"query": "go"}},
		{Name: "lookup", Args: map[string]any{"id": "1"}},
	}

	got, err := (JSONOutputKeyToolsParser{KeyName: "lookup", FirstToolOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	args := got.(map[string]any)
	if args["id"] != "1" {
		t.Fatalf("args: %#v", args)
	}
}

func TestJSONOutputToolsParserInvalidArguments(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"tool_calls": []map[string]any{
			{
				"function": map[string]any{
					"name":      "search",
					"arguments": `{bad`,
				},
			},
		},
	}
	_, err := (JSONOutputToolsParser{}).ParseMessage(msg)
	if err == nil {
		t.Fatal("expected invalid JSON error")
	}
}

func TestJSONOutputToolsParserPartialArguments(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"tool_calls": []map[string]any{
			{
				"id": "call-1",
				"function": map[string]any{
					"name":      "search",
					"arguments": `{"query":"golang`,
				},
			},
			{
				"id": "call-2",
				"function": map[string]any{
					"name":      "skip",
					"arguments": `{"ok": tru`,
				},
			},
		},
	}

	got, err := (JSONOutputToolsParser{ReturnID: true, Partial: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse partial: %v", err)
	}
	calls := got.([]ParsedToolCall)
	if len(calls) != 1 {
		t.Fatalf("calls: %#v", calls)
	}
	if calls[0].ID != "call-1" || calls[0].Args["query"] != "golang" {
		t.Fatalf("call: %#v", calls[0])
	}
}

func TestJSONOutputFunctionsParserPartialArguments(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{
			"name":      "search",
			"arguments": `{"query":"go`,
		},
	}

	got, err := (JSONOutputFunctionsParser{ArgsOnly: true, Partial: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse partial: %v", err)
	}
	args := got.(map[string]any)
	if args["query"] != "go" {
		t.Fatalf("args: %#v", args)
	}

	key, err := (JSONKeyOutputFunctionsParser{KeyName: "missing", Partial: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse missing partial key: %v", err)
	}
	if key != nil {
		t.Fatalf("missing partial key got %#v", key)
	}

	key, err = (JSONKeyOutputFunctionsParser{KeyName: "query", Partial: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse partial key: %v", err)
	}
	if key != "go" {
		t.Fatalf("key: %#v", key)
	}
}

func TestOutputFunctionsParserMissingCall(t *testing.T) {
	msg := messages.AI("")
	_, err := (OutputFunctionsParser{}).ParseMessage(msg)
	if err == nil || !strings.Contains(err.Error(), "missing function_call") {
		t.Fatalf("err: %v", err)
	}
}

func TestOutputFunctionsParserCallVariants(t *testing.T) {
	// FunctionCall value is returned as-is.
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"function_call": FunctionCall{Name: "direct", Arguments: `{"a":1}`},
	}
	got, err := (OutputFunctionsParser{}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	call := got.(FunctionCall)
	if call.Name != "direct" || call.Arguments != `{"a":1}` {
		t.Fatalf("call: %#v", call)
	}

	// Map without arguments is rejected.
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{"name": "noargs"},
	}
	if _, err := (OutputFunctionsParser{}).ParseMessage(msg); err == nil ||
		!strings.Contains(err.Error(), "missing arguments") {
		t.Fatalf("map missing arguments: %v", err)
	}

	// Other shapes go through the JSON marshal round-trip.
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]string{"name": "viajson", "arguments": `{"b":2}`},
	}
	got, err = (OutputFunctionsParser{}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse via json: %v", err)
	}
	if got.(FunctionCall).Name != "viajson" {
		t.Fatalf("call: %#v", got)
	}

	// Round-trip shape without arguments is rejected.
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]string{"name": "noargs"},
	}
	if _, err := (OutputFunctionsParser{}).ParseMessage(msg); err == nil ||
		!strings.Contains(err.Error(), "missing arguments") {
		t.Fatalf("json missing arguments: %v", err)
	}

	// Unmarshalable values surface the marshal error.
	msg.AdditionalKwargs = map[string]any{
		"function_call": func() {},
	}
	if _, err := (OutputFunctionsParser{}).ParseMessage(msg); err == nil {
		t.Fatal("expected marshal error")
	}
}

func TestJSONOutputFunctionsParserVariants(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{
			"name":      "search",
			"arguments": `{"query":"go"}`,
		},
	}

	got, err := (JSONOutputFunctionsParser{}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	full := got.(map[string]any)
	if full["name"] != "search" || full["arguments"].(map[string]any)["query"] != "go" {
		t.Fatalf("full: %#v", full)
	}

	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{"name": "search", "arguments": `{bad`},
	}
	if _, err := (JSONOutputFunctionsParser{}).ParseMessage(msg); err == nil ||
		!strings.Contains(err.Error(), "parse function call arguments") {
		t.Fatalf("invalid args: %v", err)
	}

	// Partial mode with an unrecoverable fragment yields nil, nil.
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{"name": "search", "arguments": `tru`},
	}
	got, err = (JSONOutputFunctionsParser{Partial: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("partial: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil for incomplete partial, got %#v", got)
	}
}

func TestJSONKeyOutputFunctionsParserErrors(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{"name": "search", "arguments": `"just a string"`},
	}
	if _, err := (JSONKeyOutputFunctionsParser{KeyName: "k"}).ParseMessage(msg); err == nil ||
		!strings.Contains(err.Error(), "not a JSON object") {
		t.Fatalf("non-object args: %v", err)
	}

	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{"name": "search", "arguments": `{"a":1}`},
	}
	if _, err := (JSONKeyOutputFunctionsParser{KeyName: "missing"}).ParseMessage(msg); err == nil ||
		!strings.Contains(err.Error(), `missing key "missing"`) {
		t.Fatalf("missing key: %v", err)
	}

	// Partial mode with unparseable arguments yields nil, nil.
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{"name": "search", "arguments": `tru`},
	}
	got, err := (JSONKeyOutputFunctionsParser{KeyName: "k", Partial: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("partial: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil for incomplete partial, got %#v", got)
	}
}

func TestJSONOutputToolsParserFirstOnlyEmpty(t *testing.T) {
	msg := messages.AI("")
	got, err := (JSONOutputToolsParser{FirstToolOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil, got %#v", got)
	}
}

func TestJSONOutputKeyToolsParserVariants(t *testing.T) {
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{
		{ID: "call-1", Name: "search", Args: map[string]any{"query": "go"}},
		{ID: "call-2", Name: "search", Args: map[string]any{"query": "rust"}},
		{ID: "call-3", Name: "lookup", Args: map[string]any{"id": "1"}},
	}

	// FirstToolOnly with ReturnID returns the typed call.
	got, err := (JSONOutputKeyToolsParser{KeyName: "search", FirstToolOnly: true, ReturnID: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	call := got.(ParsedToolCall)
	if call.ID != "call-1" || call.Args["query"] != "go" {
		t.Fatalf("call: %#v", call)
	}

	// FirstToolOnly without a match yields nil.
	got, err = (JSONOutputKeyToolsParser{KeyName: "absent", FirstToolOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil, got %#v", got)
	}

	// ReturnID without FirstToolOnly returns typed calls.
	got, err = (JSONOutputKeyToolsParser{KeyName: "search", ReturnID: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	calls := got.([]ParsedToolCall)
	if len(calls) != 2 || calls[1].ID != "call-2" {
		t.Fatalf("calls: %#v", calls)
	}

	// Default returns just the args maps.
	got, err = (JSONOutputKeyToolsParser{KeyName: "search"}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	args := got.([]map[string]any)
	if len(args) != 2 || args[0]["query"] != "go" || args[1]["query"] != "rust" {
		t.Fatalf("args: %#v", args)
	}
}

func TestToolCallsFromMessageEdgeCases(t *testing.T) {
	// Unmarshalable raw value surfaces the marshal error.
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{"tool_calls": func() {}}
	if _, err := (JSONOutputToolsParser{}).ParseMessage(msg); err == nil {
		t.Fatal("expected marshal error")
	}

	// Raw value of the wrong shape fails to decode.
	msg.AdditionalKwargs = map[string]any{"tool_calls": "not a list"}
	if _, err := (JSONOutputToolsParser{}).ParseMessage(msg); err == nil ||
		!strings.Contains(err.Error(), "parse raw tool calls") {
		t.Fatalf("decode err: %v", err)
	}

	// Entries without a function payload are skipped.
	msg.AdditionalKwargs = map[string]any{
		"tool_calls": []map[string]any{{"id": "call-1"}},
	}
	got, err := (JSONOutputToolsParser{}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got.([]ParsedToolCall)) != 0 {
		t.Fatalf("calls: %#v", got)
	}

	// Partial arguments that decode to a non-object are skipped.
	msg.AdditionalKwargs = map[string]any{
		"tool_calls": []map[string]any{
			{
				"id":       "call-1",
				"function": map[string]any{"name": "search", "arguments": `[1, 2`},
			},
			{
				"id":       "call-2",
				"function": map[string]any{"name": "lookup", "arguments": `{"id":"1"}`},
			},
		},
	}
	got, err = (JSONOutputToolsParser{Partial: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse partial: %v", err)
	}
	calls := got.([]ParsedToolCall)
	if len(calls) != 1 || calls[0].Type != "lookup" {
		t.Fatalf("calls: %#v", calls)
	}
}

func TestCloneArgs(t *testing.T) {
	if got := cloneArgs(nil); got == nil || len(got) != 0 {
		t.Fatalf("nil args: %#v", got)
	}
	src := map[string]any{"a": 1}
	got := cloneArgs(src)
	got["b"] = 2
	if _, ok := src["b"]; ok {
		t.Fatal("cloneArgs should copy the map")
	}
}
