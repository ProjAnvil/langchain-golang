package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestNewModelRetryMiddlewareDefaultsAndNormalization(t *testing.T) {
	middleware, err := NewModelRetryMiddleware(
		WithModelRetryOn(nil),
		WithModelRetrySleep(nil),
		WithModelRetryOnFailure(""),
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

func TestNewModelRetryMiddlewareValidation(t *testing.T) {
	if _, err := NewModelRetryMiddleware(WithModelRetryOnFailure("bogus")); err == nil ||
		!strings.Contains(err.Error(), "invalid on_failure") {
		t.Fatalf("expected invalid on_failure error, got %v", err)
	}
	if _, err := NewModelRetryMiddleware(WithModelRetryMaxRetries(-1)); err == nil {
		t.Fatal("expected negative max retries error")
	}
	if _, err := NewModelRetryMiddleware(WithModelRetryBackoff(-time.Second, 0, 0, false)); err == nil {
		t.Fatal("expected negative initial delay error")
	}
}

func TestModelRetryMiddlewareRetryOnPredicateStopsRetries(t *testing.T) {
	permanent := errors.New("permanent")
	middleware, err := NewModelRetryMiddleware(
		WithModelRetryMaxRetries(3),
		WithModelRetryOn(func(err error) bool { return !errors.Is(err, permanent) }),
		WithModelRetryOnFailure("error"),
		WithModelRetryBackoff(0, 0, 0, false),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	calls := 0
	_, err = middleware.WrapModelCall(context.Background(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, permanent
	})
	if !errors.Is(err, permanent) {
		t.Fatalf("expected permanent error, got %v", err)
	}
	if calls != 1 {
		t.Fatalf("predicate should stop retries after the first failure, got %d calls", calls)
	}
}

func TestModelRetryMiddlewareFailureFormatter(t *testing.T) {
	middleware, err := NewModelRetryMiddleware(
		WithModelRetryMaxRetries(0),
		WithModelRetryFailureFormatter(func(err error) string { return "formatted: " + err.Error() }),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	response, err := middleware.WrapModelCall(context.Background(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, errors.New("down")
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if len(response.Result) != 1 || response.Result[0].Content != "formatted: down" {
		t.Fatalf("formatted response mismatch: %#v", response.Result)
	}
}

func TestModelRetryMiddlewareDefaultFailureMessagePlural(t *testing.T) {
	middleware, err := NewModelRetryMiddleware(
		WithModelRetryMaxRetries(1),
		WithModelRetryBackoff(0, 0, 0, false),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	response, err := middleware.WrapModelCall(context.Background(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, errors.New("down")
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if !strings.Contains(response.Result[0].Content, "failed after 2 attempts") {
		t.Fatalf("failure message mismatch: %q", response.Result[0].Content)
	}
	if response.Result[0].Role != messages.RoleAI {
		t.Fatalf("failure message should be an AI message: %#v", response.Result[0])
	}
}
