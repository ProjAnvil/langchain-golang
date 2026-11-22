package outputparser

import (
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

func rawTool(name string, arguments string, id string) map[string]any {
	call := map[string]any{
		"function": map[string]any{
			"name":      name,
			"arguments": arguments,
		},
	}
	if id != "" {
		call["id"] = id
	}
	return call
}

func TestParseToolCallValid(t *testing.T) {
	got, err := ParseToolCall(rawTool("search", `{"query":"go","limit":2}`, "call-1"), ParseToolCallOptions{ReturnID: true})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got == nil {
		t.Fatal("expected a parsed call")
	}
	if got.Type != "search" || got.ID != "call-1" {
		t.Fatalf("call: %#v", got)
	}
	if got.Args["query"] != "go" || got.Args["limit"].(float64) != 2 {
		t.Fatalf("args: %#v", got.Args)
	}

	noID, err := ParseToolCall(rawTool("search", `{"query":"go"}`, "call-1"), ParseToolCallOptions{})
	if err != nil {
		t.Fatalf("parse no id: %v", err)
	}
	if noID.ID != "" {
		t.Fatalf("expected empty id without ReturnID, got %q", noID.ID)
	}
}

func TestParseToolCallEmptyAndNullArguments(t *testing.T) {
	got, err := ParseToolCall(rawTool("ping", "", ""), ParseToolCallOptions{})
	if err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if len(got.Args) != 0 {
		t.Fatalf("empty args should map to {}: %#v", got.Args)
	}

	got, err = ParseToolCall(rawTool("ping", "null", ""), ParseToolCallOptions{})
	if err != nil {
		t.Fatalf("parse null: %v", err)
	}
	if len(got.Args) != 0 {
		t.Fatalf("null args should map to {}: %#v", got.Args)
	}
}

func TestParseToolCallNoFunctionAndMissingName(t *testing.T) {
	if got, err := ParseToolCall(map[string]any{"id": "call-1"}, ParseToolCallOptions{}); err != nil || got != nil {
		t.Fatalf("no function: got %#v err %v", got, err)
	}
	if got, err := ParseToolCall(map[string]any{"function": "not-a-map"}, ParseToolCallOptions{}); err != nil || got != nil {
		t.Fatalf("non-map function: got %#v err %v", got, err)
	}

	noName := map[string]any{"function": map[string]any{"arguments": `{"a":1}`}}
	got, err := ParseToolCall(noName, ParseToolCallOptions{})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Type != "" {
		t.Fatalf("missing name should default to empty, got %q", got.Type)
	}
}

func TestParseToolCallInvalidJSONIncludingYesNoTokens(t *testing.T) {
	for _, arguments := range []string{`{bad`, `YES`, `NO`} {
		t.Run(arguments, func(t *testing.T) {
			_, err := ParseToolCall(rawTool("search", arguments, ""), ParseToolCallOptions{})
			if err == nil {
				t.Fatal("expected invalid JSON error")
			}
			var ope *lcerrors.OutputParserException
			if !errors.As(err, &ope) {
				t.Fatalf("expected OutputParserException, got %T", err)
			}
			if !strings.Contains(err.Error(), "are not valid JSON") {
				t.Fatalf("error message: %v", err)
			}
		})
	}
}

func TestParseToolCallStrictMode(t *testing.T) {
	// A literal control character inside a JSON string is invalid under strict
	// JSON but tolerated when Strict is false (Python's default).
	arguments := "{\"text\":\"line1" + "\n" + "line2\"}"

	relaxed, err := ParseToolCall(rawTool("search", arguments, ""), ParseToolCallOptions{})
	if err != nil {
		t.Fatalf("non-strict parse: %v", err)
	}
	if relaxed.Args["text"] != "line1\nline2" {
		t.Fatalf("relaxed text: %#v", relaxed.Args["text"])
	}

	_, err = ParseToolCall(rawTool("search", arguments, ""), ParseToolCallOptions{Strict: true})
	if err == nil {
		t.Fatal("expected strict-mode error for control character in string")
	}
	var ope *lcerrors.OutputParserException
	if !errors.As(err, &ope) {
		t.Fatalf("expected OutputParserException, got %T", err)
	}
}

func TestParseToolCallPartial(t *testing.T) {
	got, err := ParseToolCall(rawTool("search", `{"query":"gol`, ""), ParseToolCallOptions{Partial: true})
	if err != nil {
		t.Fatalf("parse partial: %v", err)
	}
	if got == nil || got.Args["query"] != "gol" {
		t.Fatalf("partial call: %#v", got)
	}

	// Unrecoverable fragments and non-object fragments are skipped (nil, nil).
	if got, err := ParseToolCall(rawTool("search", `tru`, ""), ParseToolCallOptions{Partial: true}); err != nil || got != nil {
		t.Fatalf("unrecoverable partial: got %#v err %v", got, err)
	}
	if got, err := ParseToolCall(rawTool("search", `[1,2`, ""), ParseToolCallOptions{Partial: true}); err != nil || got != nil {
		t.Fatalf("non-object partial: got %#v err %v", got, err)
	}
}

func TestMakeInvalidToolCall(t *testing.T) {
	got := MakeInvalidToolCall(rawTool("search", `{bad`, "call-9"), "invalid json")
	if got.Name != "search" || got.Args != `{bad` || got.ID != "call-9" || got.Error != "invalid json" {
		t.Fatalf("invalid call: %#v", got)
	}
	empty := MakeInvalidToolCall(rawTool("search", `{}`, ""), "")
	if empty.Error != "" {
		t.Fatalf("empty error message should stay empty, got %#v", empty)
	}
	missing := MakeInvalidToolCall(map[string]any{"id": "call-1"}, "e")
	if missing.Name != "" || missing.Args != "" || missing.ID != "call-1" {
		t.Fatalf("missing function payload: %#v", missing)
	}
}

func TestParseToolCallsValidList(t *testing.T) {
	raw := []map[string]any{
		rawTool("search", `{"query":"go"}`, "call-1"),
		rawTool("lookup", `{"id":"1"}`, "call-2"),
	}
	got, err := ParseToolCalls(raw, ParseToolCallOptions{ReturnID: true})
	if err != nil {
		t.Fatalf("parse tool calls: %v", err)
	}
	if len(got) != 2 || got[0].Type != "search" || got[0].ID != "call-1" || got[1].Type != "lookup" {
		t.Fatalf("calls: %#v", got)
	}

	empty, err := ParseToolCalls(nil, ParseToolCallOptions{})
	if err != nil {
		t.Fatalf("empty list: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("empty list got %#v", empty)
	}
}

func TestParseToolCallsAggregatesMultipleInvalidCalls(t *testing.T) {
	raw := []map[string]any{
		rawTool("ok", `{"a":1}`, "call-0"),
		rawTool("search", `{bad`, "call-1"),
		rawTool("lookup", `YES`, "call-2"),
	}
	got, err := ParseToolCalls(raw, ParseToolCallOptions{})
	if err == nil {
		t.Fatal("expected aggregated error")
	}
	if got != nil {
		t.Fatalf("expected nil result on error, got %#v", got)
	}
	var ope *lcerrors.OutputParserException
	if !errors.As(err, &ope) {
		t.Fatalf("expected OutputParserException, got %T", err)
	}
	msg := err.Error()
	if strings.Count(msg, "are not valid JSON") != 2 {
		t.Fatalf("expected two invalid calls in message: %q", msg)
	}
	if !strings.Contains(msg, "Function search") || !strings.Contains(msg, "Function lookup") {
		t.Fatalf("aggregated message missing function names: %q", msg)
	}
	if !strings.Contains(msg, "\n\n") {
		t.Fatalf("expected blank-line join between exceptions: %q", msg)
	}
}

func searchSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"query": schema.String("search query"),
	}, "query")
}

func pydanticToolsMessage() messages.Message {
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{
		{ID: "call-1", Name: "search", Args: map[string]any{"query": "go"}},
		{ID: "call-2", Name: "search", Args: map[string]any{"query": "rust"}},
	}
	return msg
}

func TestPydanticToolsParserValid(t *testing.T) {
	parser := PydanticToolsParser{Tools: map[string]schema.Schema{"search": searchSchema()}, ReturnID: true}
	got, err := parser.ParseMessage(pydanticToolsMessage())
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	objects, ok := got.([]map[string]any)
	if !ok || len(objects) != 2 {
		t.Fatalf("objects: %#v", got)
	}
	if objects[0]["query"] != "go" || objects[1]["query"] != "rust" {
		t.Fatalf("object values: %#v", objects)
	}
}

func TestPydanticToolsParserFromAdditionalKwargs(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"tool_calls": []map[string]any{
			rawTool("search", `{"query":"go"}`, "call-1"),
		},
	}
	parser := PydanticToolsParser{Tools: map[string]schema.Schema{"search": searchSchema()}}
	got, err := parser.ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	objects := got.([]map[string]any)
	if len(objects) != 1 || objects[0]["query"] != "go" {
		t.Fatalf("objects: %#v", objects)
	}
}

func TestPydanticToolsParserUnknownTool(t *testing.T) {
	parser := PydanticToolsParser{Tools: map[string]schema.Schema{"search": searchSchema()}}
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{{Name: "missing", Args: map[string]any{}}}
	_, err := parser.ParseMessage(msg)
	if err == nil || !strings.Contains(err.Error(), "Unknown tool type") {
		t.Fatalf("unknown tool: %v", err)
	}
	if !strings.Contains(err.Error(), "Available tools") {
		t.Fatalf("missing available tools: %v", err)
	}
}

func TestPydanticToolsParserValidationFailure(t *testing.T) {
	parser := PydanticToolsParser{Tools: map[string]schema.Schema{"search": searchSchema()}}
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{{Name: "search", Args: map[string]any{}}}
	_, err := parser.ParseMessage(msg)
	if err == nil || !strings.Contains(err.Error(), "failed validation") {
		t.Fatalf("validation failure: %v", err)
	}
}

func TestPydanticToolsParserFirstToolOnly(t *testing.T) {
	parser := PydanticToolsParser{Tools: map[string]schema.Schema{"search": searchSchema()}, FirstToolOnly: true}
	got, err := parser.ParseMessage(pydanticToolsMessage())
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	object, ok := got.(map[string]any)
	if !ok || object["query"] != "go" {
		t.Fatalf("first only: %#v", got)
	}

	empty := PydanticToolsParser{Tools: map[string]schema.Schema{"search": searchSchema()}, FirstToolOnly: true}
	got, err = empty.ParseMessage(messages.AI(""))
	if err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil, got %#v", got)
	}
}

func TestPydanticToolsParserPartialSkipsIncomplete(t *testing.T) {
	strictSchema := schema.Object(map[string]schema.Schema{
		"query": schema.String("search query"),
		"limit": schema.Integer("result limit"),
	}, "query", "limit")
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"tool_calls": []map[string]any{
			rawTool("search", `{"query":"go`, "call-1"),
		},
	}
	parser := PydanticToolsParser{
		Tools:   map[string]schema.Schema{"search": strictSchema},
		Partial: true,
	}
	got, err := parser.ParseMessage(msg)
	if err != nil {
		t.Fatalf("partial parse: %v", err)
	}
	objects := got.([]map[string]any)
	if len(objects) != 0 {
		t.Fatalf("incomplete partial should be skipped, got %#v", objects)
	}
}

func TestPydanticToolsParserInvalidJSON(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"tool_calls": []map[string]any{rawTool("search", `{bad`, "call-1")},
	}
	parser := PydanticToolsParser{Tools: map[string]schema.Schema{"search": searchSchema()}}
	if _, err := parser.ParseMessage(msg); err == nil {
		t.Fatal("expected invalid JSON error")
	}
}

func TestPydanticToolsParserRefusal(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "no"}}
	parser := PydanticToolsParser{Tools: map[string]schema.Schema{"search": searchSchema()}}
	if _, err := parser.ParseMessage(msg); err == nil {
		t.Fatal("expected refusal error")
	}
}
