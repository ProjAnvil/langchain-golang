package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestToolCallLimitMiddlewareAllowsAndCountsToolCalls(t *testing.T) {
	limit := 2
	middleware, err := NewToolCallLimitMiddleware("", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new tool call limit middleware: %v", err)
	}

	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}, {ID: "2", Name: "calc"}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	threadCounts := update[ThreadToolCallCountKey].(map[string]int)
	runCounts := update[RunToolCallCountKey].(map[string]int)
	if threadCounts[allToolsCountKey] != 2 || runCounts[allToolsCountKey] != 2 {
		t.Fatalf("counts mismatch: thread=%#v run=%#v", threadCounts, runCounts)
	}
}

func TestToolCallLimitMiddlewareContinueBlocksExceededCalls(t *testing.T) {
	limit := 1
	middleware, err := NewToolCallLimitMiddleware("", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new tool call limit middleware: %v", err)
	}

	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}, {ID: "2", Name: "calc"}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	threadCounts := update[ThreadToolCallCountKey].(map[string]int)
	runCounts := update[RunToolCallCountKey].(map[string]int)
	if threadCounts[allToolsCountKey] != 1 || runCounts[allToolsCountKey] != 2 {
		t.Fatalf("counts mismatch: thread=%#v run=%#v", threadCounts, runCounts)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 1 || msgs[0].ToolCallID != "2" || msgs[0].ResponseMetadata["status"] != "error" {
		t.Fatalf("blocked messages mismatch: %#v", msgs)
	}
	if !strings.Contains(msgs[0].Content, "Do not make additional tool calls") {
		t.Fatalf("blocked content mismatch: %q", msgs[0].Content)
	}
}

func TestToolCallLimitMiddlewareSpecificToolEnd(t *testing.T) {
	limit := 0
	middleware, err := NewToolCallLimitMiddleware("search", &limit, nil, "end")
	if err != nil {
		t.Fatalf("new tool call limit middleware: %v", err)
	}

	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	if update["jump_to"] != "end" {
		t.Fatalf("expected jump_to end: %#v", update)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 2 || msgs[0].Role != messages.RoleTool || msgs[1].Role != messages.RoleAI {
		t.Fatalf("end messages mismatch: %#v", msgs)
	}
	if !strings.Contains(msgs[1].Content, "'search' tool call limit reached") {
		t.Fatalf("final content mismatch: %q", msgs[1].Content)
	}
}

func TestToolCallLimitMiddlewareEndRejectsOtherPendingTools(t *testing.T) {
	limit := 0
	middleware, err := NewToolCallLimitMiddleware("search", &limit, nil, "end")
	if err != nil {
		t.Fatalf("new tool call limit middleware: %v", err)
	}

	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}, {ID: "2", Name: "calc"}}
	_, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err == nil || !strings.Contains(err.Error(), "other tool calls pending") {
		t.Fatalf("expected pending tools error, got %v", err)
	}
}

func TestToolCallLimitMiddlewareErrorBehavior(t *testing.T) {
	limit := 0
	middleware, err := NewToolCallLimitMiddleware("", &limit, nil, "error")
	if err != nil {
		t.Fatalf("new tool call limit middleware: %v", err)
	}

	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	_, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	var limitErr ToolCallLimitExceededError
	if !errors.As(err, &limitErr) {
		t.Fatalf("expected ToolCallLimitExceededError, got %v", err)
	}
}

func TestToolCallLimitMiddlewareRequiresValidLimits(t *testing.T) {
	threadLimit := 1
	runLimit := 2
	if _, err := NewToolCallLimitMiddleware("", nil, nil, "continue"); err == nil {
		t.Fatal("expected missing limit error")
	}
	if _, err := NewToolCallLimitMiddleware("", &threadLimit, &runLimit, "continue"); err == nil {
		t.Fatal("expected run greater than thread error")
	}
}
