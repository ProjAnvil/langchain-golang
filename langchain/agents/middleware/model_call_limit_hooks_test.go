package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestModelCallLimitExceededErrorMessage(t *testing.T) {
	threadLimit := 3
	err := ModelCallLimitExceededError{ThreadCount: 3, RunCount: 1, ThreadLimit: &threadLimit}
	if got := err.Error(); !strings.Contains(got, "thread limit (3/3)") {
		t.Fatalf("error message mismatch: %q", got)
	}
}

func TestNewModelCallLimitMiddlewareValidation(t *testing.T) {
	if _, err := NewModelCallLimitMiddleware(nil, nil, "end"); err == nil {
		t.Fatal("expected missing limit error")
	}
	limit := 1
	if _, err := NewModelCallLimitMiddleware(&limit, nil, "bogus"); err == nil ||
		!strings.Contains(err.Error(), "invalid exit_behavior") {
		t.Fatalf("expected invalid exit behavior error, got %v", err)
	}
	// Empty behavior defaults to "end".
	middleware, err := NewModelCallLimitMiddleware(&limit, nil, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if middleware.ExitBehavior != "end" {
		t.Fatalf("default exit behavior mismatch: %q", middleware.ExitBehavior)
	}
}

func TestModelCallLimitMiddlewareBeforeModelEndBehavior(t *testing.T) {
	threadLimit := 2
	runLimit := 1
	middleware, err := NewModelCallLimitMiddleware(&threadLimit, &runLimit, "end")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}

	// Below the limits: no command.
	cmd, err := middleware.BeforeModel(context.Background(), map[string]any{
		ThreadModelCallCountKey: 1,
		RunModelCallCountKey:    0,
	})
	if err != nil || cmd != nil {
		t.Fatalf("expected no command below limits: %#v %v", cmd, err)
	}

	// Run limit exceeded: jump to end with an AI message.
	cmd, err = middleware.BeforeModel(context.Background(), map[string]any{
		ThreadModelCallCountKey: 1,
		RunModelCallCountKey:    1,
	})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if cmd == nil || cmd.Goto != "end" {
		t.Fatalf("expected end command: %#v", cmd)
	}
	msgs := cmd.Update["messages"].([]messages.Message)
	if len(msgs) != 1 || msgs[0].Role != messages.RoleAI ||
		!strings.Contains(msgs[0].Content, "run limit (1/1)") {
		t.Fatalf("end message mismatch: %#v", msgs)
	}
}

func TestModelCallLimitMiddlewareBeforeModelErrorBehavior(t *testing.T) {
	threadLimit := 1
	middleware, err := NewModelCallLimitMiddleware(&threadLimit, nil, "error")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	_, err = middleware.BeforeModel(context.Background(), map[string]any{
		ThreadModelCallCountKey: 1,
	})
	var limitErr ModelCallLimitExceededError
	if !errors.As(err, &limitErr) {
		t.Fatalf("expected ModelCallLimitExceededError, got %v", err)
	}
}

func TestModelCallLimitMiddlewareAfterModelIncrements(t *testing.T) {
	threadLimit := 10
	middleware, err := NewModelCallLimitMiddleware(&threadLimit, nil, "end")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	update, err := middleware.AfterModel(context.Background(), map[string]any{
		ThreadModelCallCountKey: int64(2),
		RunModelCallCountKey:    float64(3),
	})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	if update[ThreadModelCallCountKey] != 3 || update[RunModelCallCountKey] != 4 {
		t.Fatalf("increment mismatch: %#v", update)
	}

	// Nil state counts from zero.
	update, err = middleware.AfterModel(context.Background(), nil)
	if err != nil || update[ThreadModelCallCountKey] != 1 {
		t.Fatalf("nil state mismatch: %#v %v", update, err)
	}
}

func TestIntFromState(t *testing.T) {
	state := map[string]any{
		"int":     1,
		"int64":   int64(2),
		"float64": float64(3),
		"float32": float32(4),
		"string":  "nope",
	}
	cases := map[string]int{"int": 1, "int64": 2, "float64": 3, "float32": 4, "string": 0, "missing": 0}
	for key, want := range cases {
		if got := intFromState(state, key); got != want {
			t.Fatalf("intFromState(%q) = %d, want %d", key, got, want)
		}
	}
}

func TestStringsJoin(t *testing.T) {
	if got := stringsJoin(nil, ", "); got != "" {
		t.Fatalf("empty join mismatch: %q", got)
	}
	if got := stringsJoin([]string{"a", "b", "c"}, "|"); got != "a|b|c" {
		t.Fatalf("join mismatch: %q", got)
	}
}
