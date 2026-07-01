package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestModelCallLimitMiddlewareBeforeModelAllowsUnderLimit(t *testing.T) {
	limit := 2
	middleware, err := NewModelCallLimitMiddleware(&limit, nil, "end")
	if err != nil {
		t.Fatalf("new call limit middleware: %v", err)
	}

	command, err := middleware.BeforeModel(context.Background(), map[string]any{ThreadModelCallCountKey: 1})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if command != nil {
		t.Fatalf("expected nil command, got %#v", command)
	}
}

func TestModelCallLimitMiddlewareBeforeModelEndsWhenExceeded(t *testing.T) {
	threadLimit := 2
	runLimit := 1
	middleware, err := NewModelCallLimitMiddleware(&threadLimit, &runLimit, "end")
	if err != nil {
		t.Fatalf("new call limit middleware: %v", err)
	}

	command, err := middleware.BeforeModel(context.Background(), map[string]any{
		ThreadModelCallCountKey: 2,
		RunModelCallCountKey:    1,
	})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if command == nil {
		t.Fatal("expected command")
	}
	if command.Goto != "end" {
		t.Fatalf("goto mismatch: %q", command.Goto)
	}
	msgs, ok := command.Update["messages"].([]messages.Message)
	if !ok || len(msgs) != 1 {
		t.Fatalf("messages update mismatch: %#v", command.Update["messages"])
	}
	if !strings.Contains(msgs[0].Content, "thread limit (2/2)") || !strings.Contains(msgs[0].Content, "run limit (1/1)") {
		t.Fatalf("limit message mismatch: %q", msgs[0].Content)
	}
}

func TestModelCallLimitMiddlewareBeforeModelErrorsWhenExceeded(t *testing.T) {
	limit := 1
	middleware, err := NewModelCallLimitMiddleware(nil, &limit, "error")
	if err != nil {
		t.Fatalf("new call limit middleware: %v", err)
	}

	_, err = middleware.BeforeModel(context.Background(), map[string]any{RunModelCallCountKey: 1})
	var limitErr ModelCallLimitExceededError
	if !errors.As(err, &limitErr) {
		t.Fatalf("expected ModelCallLimitExceededError, got %v", err)
	}
	if limitErr.RunCount != 1 || limitErr.RunLimit == nil || *limitErr.RunLimit != 1 {
		t.Fatalf("limit error mismatch: %#v", limitErr)
	}
}

func TestModelCallLimitMiddlewareAfterModelIncrementsCounts(t *testing.T) {
	limit := 10
	middleware, err := NewModelCallLimitMiddleware(&limit, &limit, "end")
	if err != nil {
		t.Fatalf("new call limit middleware: %v", err)
	}

	update, err := middleware.AfterModel(context.Background(), map[string]any{
		ThreadModelCallCountKey: int64(2),
		RunModelCallCountKey:    float64(3),
	})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	if update[ThreadModelCallCountKey] != 3 {
		t.Fatalf("thread count mismatch: %#v", update[ThreadModelCallCountKey])
	}
	if update[RunModelCallCountKey] != 4 {
		t.Fatalf("run count mismatch: %#v", update[RunModelCallCountKey])
	}
}

func TestModelCallLimitMiddlewareRequiresALimit(t *testing.T) {
	_, err := NewModelCallLimitMiddleware(nil, nil, "end")
	if err == nil {
		t.Fatal("expected constructor error")
	}
}
