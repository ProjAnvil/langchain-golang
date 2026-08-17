package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestNewToolRetryMiddlewareDefaultsAndNormalization(t *testing.T) {
	middleware, err := NewToolRetryMiddleware(
		WithToolRetryOn(nil),
		WithToolRetrySleep(nil),
		WithToolRetryOnFailure(""),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	if middleware.RetryOn == nil || middleware.Sleep == nil {
		t.Fatal("nil RetryOn/Sleep should be replaced with defaults")
	}
	if middleware.OnFailure != "continue" {
		t.Fatalf("empty on_failure should default to continue, got %q", middleware.OnFailure)
	}
}

func TestNewToolRetryMiddlewareOnFailureAliases(t *testing.T) {
	middleware, err := NewToolRetryMiddleware(WithToolRetryOnFailure("return_message"))
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	if middleware.OnFailure != "continue" {
		t.Fatalf("return_message should normalize to continue, got %q", middleware.OnFailure)
	}

	middleware, err = NewToolRetryMiddleware(WithToolRetryOnFailure("raise"))
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	if middleware.OnFailure != "error" {
		t.Fatalf("raise should normalize to error, got %q", middleware.OnFailure)
	}

	if _, err := NewToolRetryMiddleware(WithToolRetryOnFailure("bogus")); err == nil ||
		!strings.Contains(err.Error(), "invalid on_failure") {
		t.Fatalf("expected invalid on_failure error, got %v", err)
	}
	if _, err := NewToolRetryMiddleware(WithToolRetryMaxRetries(-1)); err == nil {
		t.Fatal("expected negative max retries error")
	}
	if _, err := NewToolRetryMiddleware(WithToolRetryBackoff(0, 0, -1, false)); err == nil {
		t.Fatal("expected negative backoff factor error")
	}
}

func TestToolRetryMiddlewareUsesToolInstanceName(t *testing.T) {
	searchTool := mustTool(t, "search")
	retry, err := NewToolRetryMiddleware(
		WithToolRetryTools("search"),
		WithToolRetryMaxRetries(1),
		WithToolRetryBackoff(0, 0, 0, false),
		WithToolRetryOnFailure("error"),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	calls := 0
	// The ToolCall name does not match the filter, but the bound tool's name
	// does — the middleware must prefer the tool instance name.
	request := ToolCallRequest{
		ToolCall: ToolCall{Name: "renamed", ID: "call_1"},
		Tool:     searchTool,
	}
	_, err = retry.WrapToolCall(context.Background(), request, func(context.Context, ToolCallRequest) (messages.Message, error) {
		calls++
		return messages.Message{}, errors.New("down")
	})
	if err == nil {
		t.Fatal("expected handler error after retries")
	}
	if calls != 2 {
		t.Fatalf("expected retry keyed on tool instance name, got %d calls", calls)
	}
}

func TestToolRetryMiddlewareFailureFormatter(t *testing.T) {
	retry, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(0),
		WithToolRetryFailureFormatter(func(err error) string { return "custom: " + err.Error() }),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	response, err := retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("down")
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}
	if response.Content != "custom: down" {
		t.Fatalf("formatted content mismatch: %q", response.Content)
	}
}

func TestToolRetryMiddlewareDefaultFailureMessagePlural(t *testing.T) {
	retry, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(1),
		WithToolRetryBackoff(0, 0, 0, false),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	response, err := retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("down")
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}
	if !strings.Contains(response.Content, "failed after 2 attempts") {
		t.Fatalf("failure content mismatch: %q", response.Content)
	}
}

func TestToolRetryMiddlewareRetryOnPredicateStopsRetries(t *testing.T) {
	permanent := errors.New("permanent")
	retry, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(5),
		WithToolRetryOn(func(err error) bool { return !errors.Is(err, permanent) }),
		WithToolRetryOnFailure("error"),
		WithToolRetryBackoff(0, 0, 0, false),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	calls := 0
	_, err = retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		calls++
		return messages.Message{}, permanent
	})
	if !errors.Is(err, permanent) {
		t.Fatalf("expected permanent error, got %v", err)
	}
	if calls != 1 {
		t.Fatalf("predicate should stop retries after first failure, got %d calls", calls)
	}
}

func TestToolRetryMiddlewareSleepNotCalledForZeroDelay(t *testing.T) {
	slept := 0
	retry, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(1),
		WithToolRetryBackoff(0, 0, 0, false),
		WithToolRetrySleep(func(time.Duration) { slept++ }),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	_, _ = retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("down")
	})
	if slept != 0 {
		t.Fatalf("zero delay should not sleep, got %d sleeps", slept)
	}
}
