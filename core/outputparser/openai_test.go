package outputparser

import (
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
