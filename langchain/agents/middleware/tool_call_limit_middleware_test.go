package middleware

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestToolCallLimitMiddlewareName(t *testing.T) {
	limit := 1
	generic, err := NewToolCallLimitMiddleware("", &limit, nil, "")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	if generic.Name() != "ToolCallLimitMiddleware" {
		t.Fatalf("generic name mismatch: %q", generic.Name())
	}
	specific, err := NewToolCallLimitMiddleware("search", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	if specific.Name() != "ToolCallLimitMiddleware[search]" {
		t.Fatalf("specific name mismatch: %q", specific.Name())
	}
}

func TestToolCallLimitMiddlewareValidationVariants(t *testing.T) {
	limit := 1
	if _, err := NewToolCallLimitMiddleware("", &limit, nil, "bogus"); err == nil ||
		!strings.Contains(err.Error(), "invalid exit_behavior") {
		t.Fatalf("expected invalid exit behavior error, got %v", err)
	}
	// Empty behavior defaults to continue.
	middleware, err := NewToolCallLimitMiddleware("", &limit, nil, "")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	if middleware.ExitBehavior != "continue" {
		t.Fatalf("default exit behavior mismatch: %q", middleware.ExitBehavior)
	}
}

func TestToolCallLimitMiddlewareAfterModelEdgeCases(t *testing.T) {
	limit := 1
	middleware, err := NewToolCallLimitMiddleware("", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}

	// Nil state: no AI message.
	update, err := middleware.AfterModel(context.Background(), nil)
	if err != nil || update != nil {
		t.Fatalf("expected nil update for nil state: %#v %v", update, err)
	}
	// Messages of the wrong type.
	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": "nope"})
	if err != nil || update != nil {
		t.Fatalf("expected nil update for wrong message type: %#v %v", update, err)
	}
	// No AI message.
	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{messages.Human("hi")}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without AI message: %#v %v", update, err)
	}
}

func TestToolCallLimitMiddlewareSpecificToolFiltering(t *testing.T) {
	limit := 1
	middleware, err := NewToolCallLimitMiddleware("search", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{
		{ID: "1", Name: "search"},
		{ID: "2", Name: "calc"},
		{ID: "3", Name: "search"},
	}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	threadCounts := update[ThreadToolCallCountKey].(map[string]int)
	if threadCounts["search"] != 1 {
		t.Fatalf("only matching calls should count: %#v", threadCounts)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 1 || msgs[0].ToolCallID != "3" {
		t.Fatalf("only the second search call should be blocked: %#v", msgs)
	}
	if !strings.Contains(msgs[0].Content, "Do not call 'search' again") {
		t.Fatalf("blocked content mismatch: %q", msgs[0].Content)
	}
}

func TestToolCallLimitMiddlewareAllAllowedNoMessages(t *testing.T) {
	limit := 5
	middleware, err := NewToolCallLimitMiddleware("", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	if _, ok := update["messages"]; ok {
		t.Fatalf("no artificial messages expected when nothing is blocked: %#v", update)
	}
	if update[ThreadToolCallCountKey].(map[string]int)[allToolsCountKey] != 1 {
		t.Fatalf("counts mismatch: %#v", update)
	}
}

func TestToolCallLimitMiddlewareRunLimitEndMessage(t *testing.T) {
	runLimit := 0
	middleware, err := NewToolCallLimitMiddleware("", nil, &runLimit, "end")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 2 || !strings.Contains(msgs[1].Content, "run limit exceeded (1/0 calls)") {
		t.Fatalf("final message should report the run limit: %#v", msgs)
	}
	if !strings.Contains(msgs[1].Content, "Tool call limit reached") {
		t.Fatalf("generic tool description expected: %q", msgs[1].Content)
	}
}

func TestToolCallLimitMiddlewareCountsFromAnyMap(t *testing.T) {
	limit := 3
	middleware, err := NewToolCallLimitMiddleware("", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	state := map[string]any{
		"messages": []messages.Message{ai},
		ThreadToolCallCountKey: map[string]any{
			allToolsCountKey: int64(1),
			"other":          float64(2),
			"bad":            "nope",
		},
		RunToolCallCountKey: map[string]int{allToolsCountKey: 1},
	}
	update, err := middleware.AfterModel(context.Background(), state)
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	threadCounts := update[ThreadToolCallCountKey].(map[string]int)
	if threadCounts[allToolsCountKey] != 2 || threadCounts["other"] != 2 || threadCounts["bad"] != 0 {
		t.Fatalf("counts from any-map mismatch: %#v", threadCounts)
	}
	if update[RunToolCallCountKey].(map[string]int)[allToolsCountKey] != 2 {
		t.Fatalf("run counts mismatch: %#v", update)
	}
}

func TestToolCallLimitExceededErrorMessage(t *testing.T) {
	threadLimit := 1
	runLimit := 2
	err := ToolCallLimitExceededError{
		ThreadCount: 2,
		RunCount:    3,
		ThreadLimit: &threadLimit,
		RunLimit:    &runLimit,
		ToolName:    "search",
	}
	got := err.Error()
	if !strings.Contains(got, "'search' tool call limit reached") ||
		!strings.Contains(got, "thread limit exceeded (2/1 calls)") ||
		!strings.Contains(got, "run limit exceeded (3/2 calls)") {
		t.Fatalf("error message mismatch: %q", got)
	}
}
