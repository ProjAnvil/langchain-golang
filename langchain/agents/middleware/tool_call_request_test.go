package middleware

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

func mustTool(t *testing.T, name string) tools.Func {
	t.Helper()
	tool, err := tools.NewFunc(
		name,
		"A test tool.",
		schema.Object(map[string]schema.Schema{"x": schema.Integer("")}, "x"),
		func(context.Context, map[string]any) (tools.Result, error) {
			return tools.Result{Content: "ok"}, nil
		},
	)
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}
	return tool
}

func TestToolCallRequestOverrideToolCall(t *testing.T) {
	testTool := mustTool(t, "test_tool")
	originalCall := ToolCall{Name: "test_tool", Args: map[string]any{"x": 5}, ID: "1", Type: "tool_call"}
	modifiedCall := ToolCall{Name: "test_tool", Args: map[string]any{"x": 10}, ID: "1", Type: "tool_call"}

	original := ToolCallRequest{
		ToolCall: originalCall,
		Tool:     testTool,
		State:    map[string]any{"messages": []any{}},
		Runtime:  "runtime",
	}

	next := original.Override(WithToolCall(modifiedCall))
	if next.ToolCall.Args["x"] != 10 {
		t.Fatalf("tool call override mismatch: %#v", next.ToolCall.Args)
	}
	if original.ToolCall.Args["x"] != 5 {
		t.Fatalf("original tool call mutated: %#v", original.ToolCall.Args)
	}
	if next.Tool.Name() != original.Tool.Name() {
		t.Fatalf("tool should be preserved: got %q want %q", next.Tool.Name(), original.Tool.Name())
	}
	if next.State["messages"] == nil {
		t.Fatalf("state should be preserved: %#v", next.State)
	}
}

func TestToolCallRequestOverrideState(t *testing.T) {
	original := ToolCallRequest{
		ToolCall: ToolCall{Name: "test_tool", Args: map[string]any{"x": 5}, ID: "1", Type: "tool_call"},
		Tool:     mustTool(t, "test_tool"),
		State:    map[string]any{"messages": []any{"human"}},
		Runtime:  "runtime",
	}

	next := original.Override(WithToolCallState(map[string]any{"messages": []any{"human", "ai"}}))
	if len(next.State["messages"].([]any)) != 2 {
		t.Fatalf("state override mismatch: %#v", next.State)
	}
	if len(original.State["messages"].([]any)) != 1 {
		t.Fatalf("original state mutated: %#v", original.State)
	}
}

func TestToolCallRequestOverrideMultipleAttributes(t *testing.T) {
	testTool := mustTool(t, "test_tool")
	anotherTool := mustTool(t, "another_tool")
	original := ToolCallRequest{
		ToolCall: ToolCall{Name: "test_tool", Args: map[string]any{"x": 5}, ID: "1", Type: "tool_call"},
		Tool:     testTool,
		State:    map[string]any{"count": 1},
		Runtime:  "runtime",
	}

	next := original.Override(
		WithToolCall(ToolCall{Name: "another_tool", Args: map[string]any{"y": "hello"}, ID: "2", Type: "tool_call"}),
		WithTool(anotherTool),
		WithToolCallState(map[string]any{"count": 2}),
	)

	if next.ToolCall.Name != "another_tool" {
		t.Fatalf("tool call name mismatch: %q", next.ToolCall.Name)
	}
	if next.Tool.Name() != "another_tool" {
		t.Fatalf("tool mismatch: %q", next.Tool.Name())
	}
	if next.State["count"] != 2 {
		t.Fatalf("state mismatch: %#v", next.State)
	}
	if original.ToolCall.Name != "test_tool" || original.Tool.Name() != "test_tool" || original.State["count"] != 1 {
		t.Fatalf("original request mutated: %#v", original)
	}
}

func TestToolCallRequestOverrideWithCopyPattern(t *testing.T) {
	original := ToolCallRequest{
		ToolCall: ToolCall{Name: "test_tool", Args: map[string]any{"value": 5}, ID: "call_123", Type: "tool_call"},
		Tool:     mustTool(t, "test_tool"),
		State:    map[string]any{"messages": []any{}},
		Runtime:  "runtime",
	}

	modified := original.ToolCall.Clone()
	modified.Args = map[string]any{"value": 10}
	next := original.Override(WithToolCall(modified))

	if next.ToolCall.Args["value"] != 10 {
		t.Fatalf("modified value mismatch: %#v", next.ToolCall.Args)
	}
	if next.ToolCall.ID != "call_123" || next.ToolCall.Name != "test_tool" {
		t.Fatalf("metadata not preserved: %#v", next.ToolCall)
	}
	if original.ToolCall.Args["value"] != 5 {
		t.Fatalf("original tool call mutated: %#v", original.ToolCall.Args)
	}
}

func TestToolCallRequestOverrideChaining(t *testing.T) {
	original := ToolCallRequest{
		ToolCall: ToolCall{Name: "test_tool", Args: map[string]any{"x": 5}, ID: "1", Type: "tool_call"},
		Tool:     mustTool(t, "test_tool"),
		State:    map[string]any{"count": 1},
		Runtime:  "runtime",
	}

	final := original.
		Override(WithToolCall(ToolCall{Name: "test_tool", Args: map[string]any{"x": 10}, ID: "1", Type: "tool_call"})).
		Override(WithToolCallState(map[string]any{"count": 2})).
		Override(WithToolCall(ToolCall{Name: "test_tool", Args: map[string]any{"x": 15}, ID: "1", Type: "tool_call"}))

	if final.ToolCall.Args["x"] != 15 {
		t.Fatalf("final tool call mismatch: %#v", final.ToolCall.Args)
	}
	if final.State["count"] != 2 {
		t.Fatalf("final state mismatch: %#v", final.State)
	}
	if original.ToolCall.Args["x"] != 5 || original.State["count"] != 1 {
		t.Fatalf("original request mutated: %#v", original)
	}
}
