package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/tools"
)

func TestToolRetryMiddlewareRetriesUntilSuccess(t *testing.T) {
	var slept []time.Duration
	retry, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(2),
		WithToolRetryBackoff(time.Second, 10*time.Second, 2, false),
		WithToolRetrySleep(func(delay time.Duration) {
			slept = append(slept, delay)
		}),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	calls := 0
	response, err := retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		calls++
		if calls < 3 {
			return messages.Message{}, errors.New("temporary")
		}
		return messages.Tool("call_1", "ok"), nil
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}
	if response.Content != "ok" {
		t.Fatalf("response mismatch: %#v", response)
	}
	if calls != 3 {
		t.Fatalf("call count mismatch: got %d", calls)
	}
	if len(slept) != 2 || slept[0] != time.Second || slept[1] != 2*time.Second {
		t.Fatalf("sleep delays mismatch: %#v", slept)
	}
}

func TestToolRetryMiddlewareSkipsUnmatchedTool(t *testing.T) {
	retry, err := NewToolRetryMiddleware(
		WithToolRetryTools("search"),
		WithToolRetryMaxRetries(5),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	calls := 0
	_, err = retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "calculator", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		calls++
		return messages.Message{}, errors.New("not retried")
	})
	if err == nil {
		t.Fatal("expected handler error")
	}
	if calls != 1 {
		t.Fatalf("call count mismatch: got %d", calls)
	}
}

func TestToolRetryMiddlewareToolInstances(t *testing.T) {
	searchTool, err := tools.NewSimple("search", "", func(context.Context, string) (tools.Result, error) {
		return tools.Result{}, nil
	})
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}

	retry, err := NewToolRetryMiddleware(
		WithToolRetryToolInstances(searchTool),
		WithToolRetryMaxRetries(2),
		WithToolRetryBackoff(0, 0, 0, false),
		WithToolRetryOnFailure("error"),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	calls := 0
	_, err = retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		calls++
		return messages.Message{}, errors.New("temporary")
	})
	if err == nil {
		t.Fatal("expected handler error after retries exhausted")
	}
	if calls != 3 {
		t.Fatalf("expected retries for tool matched by instance, got %d calls", calls)
	}

	calls = 0
	_, err = retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "calculator", ID: "call_2"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		calls++
		return messages.Message{}, errors.New("not retried")
	})
	if err == nil {
		t.Fatal("expected handler error")
	}
	if calls != 1 {
		t.Fatalf("expected no retries for unmatched tool, got %d calls", calls)
	}
}

func TestToolRetryMiddlewareFailureContinuesWithToolMessage(t *testing.T) {
	retry, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	response, err := retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("down")
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}
	if response.Role != messages.RoleTool || response.ToolCallID != "call_1" || response.Name != "search" {
		t.Fatalf("tool failure message metadata mismatch: %#v", response)
	}
	if response.ResponseMetadata["status"] != "error" {
		t.Fatalf("expected error status metadata: %#v", response.ResponseMetadata)
	}
	if !strings.Contains(response.Content, "Tool 'search' failed after 1 attempt") {
		t.Fatalf("failure content mismatch: %q", response.Content)
	}
}

func TestToolRetryMiddlewareOnFailureErrorReraises(t *testing.T) {
	wantErr := errors.New("permanent")
	retry, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(0),
		WithToolRetryOnFailure("error"),
	)
	if err != nil {
		t.Fatalf("new retry middleware: %v", err)
	}

	_, err = retry.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected original error, got %v", err)
	}
}
