package middleware

// Tests for ToolErrorMiddleware, the port of langchain 1.x's
// ToolErrorMiddleware (langchain/agents/middleware/tool_error.py, #38781).
// They mirror libs/langchain_v1/tests/unit_tests/agents/middleware/
// implementations/test_tool_error.py at the middleware-unit level; the
// through-create_agent mirrors live in langchain/agents/
// tool_error_middleware_test.go (Go cannot import the agents package from
// here without an import cycle).

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	agenttools "github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func TestNewToolErrorMiddlewareRequiresHandler(t *testing.T) {
	if _, err := NewToolErrorMiddleware(nil); err == nil || !strings.Contains(err.Error(), "on_error") {
		t.Fatalf("expected on_error validation error, got %v", err)
	}
	m, err := NewToolErrorMiddleware(func(error, ToolCallRequest) (any, bool) { return "handled", true })
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}
	if m.OnError == nil || m.ToolFilter != nil {
		t.Fatalf("unexpected middleware: %#v", m)
	}
}

func TestToolErrorMiddlewareHandledReturnsErrorToolMessage(t *testing.T) {
	m, err := NewToolErrorMiddleware(func(err error, request ToolCallRequest) (any, bool) {
		return "`" + request.ToolCall.Name + "` failed.", true
	})
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}

	toolErr := errors.New("secret detail")
	msg, err := m.WrapToolCall(t.Context(), ToolCallRequest{ToolCall: ToolCall{Name: "failing_tool", ID: "call_1"}},
		func(context.Context, ToolCallRequest) (messages.Message, error) {
			return messages.Message{}, toolErr
		})
	if err != nil {
		t.Fatalf("WrapToolCall() error = %v", err)
	}
	if msg.Role != messages.RoleTool || msg.ToolCallID != "call_1" || msg.Name != "failing_tool" {
		t.Fatalf("message metadata mismatch: %#v", msg)
	}
	if msg.ResponseMetadata["status"] != "error" {
		t.Fatalf("expected error status metadata: %#v", msg.ResponseMetadata)
	}
	if msg.Content != "`failing_tool` failed." {
		t.Fatalf("content mismatch: %q", msg.Content)
	}
	// on_error controls disclosure — the raw error message must not leak.
	if strings.Contains(msg.Content, "secret detail") {
		t.Fatalf("raw error leaked into content: %q", msg.Content)
	}
}

func TestToolErrorMiddlewareHandledContentBlocks(t *testing.T) {
	blocks := []messages.ContentBlock{messages.TextBlock{Text: "tool failed"}}
	m, err := NewToolErrorMiddleware(func(error, ToolCallRequest) (any, bool) {
		return blocks, true
	})
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}

	msg, err := m.WrapToolCall(t.Context(), ToolCallRequest{ToolCall: ToolCall{Name: "failing_tool", ID: "call_1"}},
		func(context.Context, ToolCallRequest) (messages.Message, error) {
			return messages.Message{}, errors.New("boom")
		})
	if err != nil {
		t.Fatalf("WrapToolCall() error = %v", err)
	}
	if msg.ResponseMetadata["status"] != "error" {
		t.Fatalf("expected error status metadata: %#v", msg.ResponseMetadata)
	}
	if len(msg.ContentBlocks) != 1 || !reflect.DeepEqual(msg.ContentBlocks[0], blocks[0]) {
		t.Fatalf("content blocks mismatch: %#v", msg.ContentBlocks)
	}
	if msg.Content != "" {
		t.Fatalf("expected empty string content alongside blocks, got %q", msg.Content)
	}
}

func TestToolErrorMiddlewareDeclineWrapsUnhandledToolError(t *testing.T) {
	toolErr := errors.New("propagate me")
	m, err := NewToolErrorMiddleware(func(error, ToolCallRequest) (any, bool) {
		return nil, false
	})
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}

	_, err = m.WrapToolCall(t.Context(), ToolCallRequest{ToolCall: ToolCall{Name: "failing_tool", ID: "call_1"}},
		func(context.Context, ToolCallRequest) (messages.Message, error) {
			return messages.Message{}, toolErr
		})
	if err == nil {
		t.Fatal("expected the declined error to be returned")
	}
	var unhandled *agenttools.UnhandledToolError
	if !errors.As(err, &unhandled) {
		t.Fatalf("expected *tools.UnhandledToolError, got %T: %v", err, err)
	}
	if !errors.Is(err, toolErr) {
		t.Fatalf("expected the original error inside the sentinel, got %v", err)
	}
}

func TestToolErrorMiddlewareToolFilter(t *testing.T) {
	var handledErrs int
	m, err := NewToolErrorMiddleware(
		func(error, ToolCallRequest) (any, bool) {
			handledErrs++
			return "handled", true
		},
		WithToolErrorTools("watched"),
	)
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}

	// A non-matching tool's error bypasses the middleware entirely: the
	// handler error propagates as-is (no conversion, no sentinel wrap).
	plain := errors.New("unwatched failure")
	_, err = m.WrapToolCall(t.Context(), ToolCallRequest{ToolCall: ToolCall{Name: "other", ID: "call_1"}},
		func(context.Context, ToolCallRequest) (messages.Message, error) {
			return messages.Message{}, plain
		})
	var unhandled *agenttools.UnhandledToolError
	if !errors.Is(err, plain) || errors.As(err, &unhandled) {
		t.Fatalf("expected the plain error to bypass the middleware, got %v", err)
	}

	// A matching tool's error is handled.
	msg, err := m.WrapToolCall(t.Context(), ToolCallRequest{ToolCall: ToolCall{Name: "watched", ID: "call_2"}},
		func(context.Context, ToolCallRequest) (messages.Message, error) {
			return messages.Message{}, plain
		})
	if err != nil || msg.Content != "handled" {
		t.Fatalf("expected handled error message, got (%#v, %v)", msg, err)
	}
	if handledErrs != 1 {
		t.Fatalf("OnError calls = %d, want 1", handledErrs)
	}
}

func TestToolErrorMiddlewareToolFilterByInstances(t *testing.T) {
	watched, err := coretools.NewSimple("watched", "", func(context.Context, string) (coretools.Result, error) {
		return coretools.Result{}, nil
	})
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}
	m, err := NewToolErrorMiddleware(
		func(error, ToolCallRequest) (any, bool) { return "handled", true },
		WithToolErrorToolInstances(watched),
	)
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}

	toolErr := errors.New("boom")
	msg, err := m.WrapToolCall(t.Context(), ToolCallRequest{
		ToolCall: ToolCall{Name: "watched", ID: "call_1"},
		Tool:     watched,
	}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, toolErr
	})
	if err != nil || msg.Content != "handled" {
		t.Fatalf("expected the instance-matched tool to be handled, got (%#v, %v)", msg, err)
	}
}

// TestToolErrorMiddlewareControlFlowPanicsPropagate mirrors Python's
// `except GraphBubbleUp: raise` clause (tool_error.py): control-flow signals
// never reach on_error. In the Go port interrupts are panics
// (*types.GraphInterrupt, recovered by the graph executor), so the middleware
// must not intercept them — the panic passes straight through WrapToolCall.
func TestToolErrorMiddlewareControlFlowPanicsPropagate(t *testing.T) {
	var handledErrs int
	m, err := NewToolErrorMiddleware(func(error, ToolCallRequest) (any, bool) {
		handledErrs++
		return "handled", true
	})
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected the GraphInterrupt panic to propagate")
		} else if _, ok := r.(*types.GraphInterrupt); !ok {
			t.Fatalf("expected *types.GraphInterrupt, got %T: %v", r, r)
		}
		if handledErrs != 0 {
			t.Fatalf("OnError must not see control-flow panics, saw %d", handledErrs)
		}
	}()

	_, _ = m.WrapToolCall(t.Context(), ToolCallRequest{ToolCall: ToolCall{Name: "failing_tool", ID: "call_1"}},
		func(context.Context, ToolCallRequest) (messages.Message, error) {
			panic(&types.GraphInterrupt{Interrupt: types.Interrupt{Value: "review"}})
		})
}

// TestToolErrorMiddlewareComposesWithToolRetry pins the composition guidance
// from the upstream docstring (tool_error.py): place ToolRetryMiddleware
// INNER and configure on_failure="error" so exhausted retries reach this
// middleware; with the default on_failure ("continue") the retry middleware
// converts failures itself and ToolErrorMiddleware never sees them.
func TestToolErrorMiddlewareComposesWithToolRetry(t *testing.T) {
	toolErr := errors.New("temporary")

	// Inner: retry with the default on_failure ("continue").
	retryContinue, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(1),
		WithToolRetryBackoff(0, 0, 0, false),
	)
	if err != nil {
		t.Fatalf("NewToolRetryMiddleware() error = %v", err)
	}
	handledContinue := 0
	outerContinue, err := NewToolErrorMiddleware(func(error, ToolCallRequest) (any, bool) {
		handledContinue++
		return "converted by error middleware", true
	})
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}

	calls := 0
	failing := func(context.Context, ToolCallRequest) (messages.Message, error) {
		calls++
		return messages.Message{}, toolErr
	}
	msg, err := outerContinue.WrapToolCall(t.Context(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}},
		func(ctx context.Context, r ToolCallRequest) (messages.Message, error) {
			return retryContinue.WrapToolCall(ctx, r, failing)
		})
	if err != nil {
		t.Fatalf("composition error = %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected 1 retry, got %d calls", calls)
	}
	if handledContinue != 0 {
		t.Fatalf("retry (on_failure=continue) must convert failures itself; OnError saw %d", handledContinue)
	}
	if msg.ResponseMetadata["status"] != "error" || strings.Contains(msg.Content, "error middleware") {
		t.Fatalf("expected the retry middleware's own failure message, got %#v", msg)
	}

	// Inner: retry with on_failure="error" — the exhausted-retry error
	// escapes the retry layer and reaches the (outer) error middleware.
	retryRaise, err := NewToolRetryMiddleware(
		WithToolRetryMaxRetries(1),
		WithToolRetryBackoff(0, 0, 0, false),
		WithToolRetryOnFailure("error"),
	)
	if err != nil {
		t.Fatalf("NewToolRetryMiddleware() error = %v", err)
	}
	handledRaise := 0
	outerRaise, err := NewToolErrorMiddleware(func(err error, _ ToolCallRequest) (any, bool) {
		handledRaise++
		if !errors.Is(err, toolErr) {
			t.Fatalf("OnError received %v, want the original tool error", err)
		}
		return "converted by error middleware", true
	})
	if err != nil {
		t.Fatalf("NewToolErrorMiddleware() error = %v", err)
	}

	calls = 0
	msg, err = outerRaise.WrapToolCall(t.Context(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "call_1"}},
		func(ctx context.Context, r ToolCallRequest) (messages.Message, error) {
			return retryRaise.WrapToolCall(ctx, r, failing)
		})
	if err != nil {
		t.Fatalf("composition error = %v", err)
	}
	if calls != 2 || handledRaise != 1 {
		t.Fatalf("expected 1 retry then 1 OnError call, got calls=%d handled=%d", calls, handledRaise)
	}
	if msg.Content != "converted by error middleware" || msg.ResponseMetadata["status"] != "error" {
		t.Fatalf("expected the error middleware's message, got %#v", msg)
	}
}
