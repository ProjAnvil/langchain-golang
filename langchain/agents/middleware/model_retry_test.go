package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

func TestModelRetryMiddlewareRetriesUntilSuccess(t *testing.T) {
	calls := 0
	var slept []time.Duration
	retry, err := NewModelRetryMiddleware(
		WithModelRetryMaxRetries(2),
		WithModelRetryBackoff(time.Second, 10*time.Second, 2, false),
		WithModelRetrySleep(func(delay time.Duration) {
			slept = append(slept, delay)
		}),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	response, err := retry.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		if calls < 3 {
			return ModelResponse{}, errors.New("temporary")
		}
		return ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if response.Result[0].Content != "ok" {
		t.Fatalf("response mismatch: %#v", response.Result)
	}
	if calls != 3 {
		t.Fatalf("call count mismatch: got %d", calls)
	}
	if len(slept) != 2 || slept[0] != time.Second || slept[1] != 2*time.Second {
		t.Fatalf("sleep delays mismatch: %#v", slept)
	}
}

func TestModelRetryMiddlewareNonRetryableFailureContinues(t *testing.T) {
	retry, err := NewModelRetryMiddleware(
		WithModelRetryMaxRetries(5),
		WithModelRetryOn(func(error) bool { return false }),
		WithModelRetryBackoff(0, 0, 0, false),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	calls := 0
	response, err := retry.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, errors.New("no retry")
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if calls != 1 {
		t.Fatalf("call count mismatch: got %d", calls)
	}
	if !strings.Contains(response.Result[0].Content, "Model call failed after 1 attempt") {
		t.Fatalf("failure message mismatch: %q", response.Result[0].Content)
	}
}

func TestModelRetryMiddlewareOnFailureErrorReraises(t *testing.T) {
	wantErr := errors.New("permanent")
	retry, err := NewModelRetryMiddleware(
		WithModelRetryMaxRetries(0),
		WithModelRetryOnFailure("error"),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	_, err = retry.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected original error, got %v", err)
	}
}

func TestModelRetryMiddlewareCustomFailureFormatter(t *testing.T) {
	retry, err := NewModelRetryMiddleware(
		WithModelRetryMaxRetries(0),
		WithModelRetryFailureFormatter(func(error) string {
			return "custom unavailable"
		}),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	response, err := retry.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, errors.New("hidden")
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if response.Result[0].Content != "custom unavailable" {
		t.Fatalf("custom failure mismatch: %q", response.Result[0].Content)
	}
}

func TestModelRetryDefaultRetryOnClassifiedErrors(t *testing.T) {
	// A non-retryable classified model error (401) must NOT be retried under
	// the default predicate (Python langchain 1.4 default_retry_on, #39538):
	// exactly one handler call, OnFailure=continue swallows into an AIMessage.
	calls := 0
	mw, err := NewModelRetryMiddleware(WithModelRetrySleep(func(time.Duration) {}))
	if err != nil {
		t.Fatal(err)
	}
	authErr := lcerrors.NewProviderError("openai", "/responses", 401, "", 0)
	out, rerr := mw.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, authErr
	})
	if calls != 1 {
		t.Fatalf("handler called %d times, want 1 (non-retryable classified error)", calls)
	}
	if rerr != nil {
		t.Fatalf("OnFailure=continue must swallow the failure: %v", rerr)
	}
	if len(out.Result) != 1 {
		t.Fatalf("expected failure AIMessage, got %+v", out.Result)
	}

	// A retryable classified error (429) must be retried MaxRetries times.
	mw2, err := NewModelRetryMiddleware(
		WithModelRetryMaxRetries(2),
		WithModelRetrySleep(func(time.Duration) {}),
	)
	if err != nil {
		t.Fatal(err)
	}
	calls = 0
	_, _ = mw2.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, lcerrors.NewProviderError("openai", "/responses", 429, "", 0)
	})
	if calls != 3 {
		t.Fatalf("retryable classified error: handler called %d times, want 3 (1 + 2 retries)", calls)
	}

	// An unclassified error must keep retrying (Python: non-MLError -> True).
	calls = 0
	_, _ = mw2.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, errors.New("boom")
	})
	if calls != 3 {
		t.Fatalf("unclassified error: handler called %d times, want 3", calls)
	}

	// Context overflow is classified and non-retryable.
	calls = 0
	_, _ = mw.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, lcerrors.NewProviderError("openai", "/responses", 400, "maximum context length is 4096 tokens", 0)
	})
	if calls != 1 {
		t.Fatalf("context overflow: handler called %d times, want 1", calls)
	}
}
