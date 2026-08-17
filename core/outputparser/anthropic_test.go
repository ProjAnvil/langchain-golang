package outputparser

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestAnthropicToolsOutputParserFromContentBlocks(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.TextBlock{Text: "using a tool"},
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{
			"id":    "toolu_1",
			"name":  "search",
			"input": map[string]any{"query": "go"},
		}},
	}

	got, err := (AnthropicToolsOutputParser{}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	calls := got.([]AnthropicToolCall)
	if len(calls) != 1 {
		t.Fatalf("calls: %#v", calls)
	}
	if calls[0].ID != "toolu_1" || calls[0].Name != "search" || calls[0].Index != 1 {
		t.Fatalf("call: %#v", calls[0])
	}
	if calls[0].Args["query"] != "go" {
		t.Fatalf("args: %#v", calls[0].Args)
	}
}

func TestAnthropicToolsOutputParserArgsOnlyFirstToolOnly(t *testing.T) {
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{
		{ID: "toolu_1", Name: "search", Args: map[string]any{"query": "go"}},
		{ID: "toolu_2", Name: "lookup", Args: map[string]any{"id": "2"}},
	}
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{"id": "toolu_1", "name": "search", "input": map[string]any{"query": "go"}}},
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{"id": "toolu_2", "name": "lookup", "input": map[string]any{"id": "2"}}},
	}

	got, err := (AnthropicToolsOutputParser{ArgsOnly: true, FirstToolOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	args := got.(map[string]any)
	if args["query"] != "go" {
		t.Fatalf("args: %#v", args)
	}
}

func TestAnthropicToolsOutputParserNoToolCalls(t *testing.T) {
	msg := messages.AI("plain")

	got, err := (AnthropicToolsOutputParser{}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got.([]AnthropicToolCall)) != 0 {
		t.Fatalf("got %#v", got)
	}

	got, err = (AnthropicToolsOutputParser{FirstToolOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse first: %v", err)
	}
	if got != nil {
		t.Fatalf("got %#v", got)
	}
}

func TestAnthropicToolsOutputParserInvalidBlock(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{"id": "toolu_1", "input": map[string]any{"query": "go"}}},
	}

	_, err := (AnthropicToolsOutputParser{}).ParseMessage(msg)
	if err == nil {
		t.Fatal("expected missing name error")
	}
}

func TestAnthropicToolsOutputParserArgsOnlyList(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{
			"id": "toolu_1", "name": "search", "input": map[string]any{"query": "go"},
		}},
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{
			"id": "toolu_2", "name": "lookup", "input": map[string]any{"id": "2"},
		}},
	}

	got, err := (AnthropicToolsOutputParser{ArgsOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	args := got.([]map[string]any)
	if len(args) != 2 || args[0]["query"] != "go" || args[1]["id"] != "2" {
		t.Fatalf("args: %#v", args)
	}
}

func TestAnthropicToolsOutputParserFirstOnlyCall(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{
			"id": "toolu_1", "name": "search", "input": map[string]any{"query": "go"},
		}},
	}

	got, err := (AnthropicToolsOutputParser{FirstToolOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	call := got.(AnthropicToolCall)
	if call.Name != "search" || call.Args["query"] != "go" {
		t.Fatalf("call: %#v", call)
	}
}

func TestAnthropicToolsOutputParserArgsOnlyFirstEmpty(t *testing.T) {
	msg := messages.AI("plain")
	got, err := (AnthropicToolsOutputParser{ArgsOnly: true, FirstToolOnly: true}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil, got %#v", got)
	}
}

func TestAnthropicToolUseArgsEdgeCases(t *testing.T) {
	// Missing input yields empty args.
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{
			"id": "toolu_1", "name": "search",
		}},
	}
	got, err := (AnthropicToolsOutputParser{}).ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	calls := got.([]AnthropicToolCall)
	if len(calls) != 1 || len(calls[0].Args) != 0 {
		t.Fatalf("calls: %#v", calls)
	}

	// Non-object input is rejected.
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{
			"id": "toolu_1", "name": "search", "input": "not an object",
		}},
	}
	if _, err := (AnthropicToolsOutputParser{}).ParseMessage(msg); err == nil {
		t.Fatal("expected input type error")
	}
}
