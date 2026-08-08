package tools

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func echoTool(t *testing.T, name string, fn func(context.Context, map[string]any) (Result, error)) Tool {
	t.Helper()
	tool, err := NewFunc(name, "", schema.Object(nil), fn)
	if err != nil {
		t.Fatalf("NewFunc(%q) error = %v", name, err)
	}
	return tool
}

func aiMessageWithCalls(calls ...messages.ToolCall) messages.Message {
	msg := messages.AI("")
	msg.ToolCalls = calls
	return msg
}

func TestNewToolNodeValidation(t *testing.T) {
	if _, err := NewToolNode(nil); err == nil {
		t.Fatal("expected error for empty tool list")
	}
	if _, err := NewToolNode([]Tool{nil}); err == nil {
		t.Fatal("expected error for nil tool")
	}

	dup := echoTool(t, "same", func(context.Context, map[string]any) (Result, error) {
		return Result{}, nil
	})
	if _, err := NewToolNode([]Tool{dup, dup}); err == nil {
		t.Fatal("expected error for duplicate tool name")
	}
}

func TestPendingToolCallsAndHasPendingToolCalls(t *testing.T) {
	if HasPendingToolCalls(nil) {
		t.Fatal("expected no pending tool calls for empty message list")
	}

	msgs := []messages.Message{
		messages.Human("hi"),
		aiMessageWithCalls(messages.ToolCall{ID: "1", Name: "echo", Args: map[string]any{"x": "a"}}),
	}
	if !HasPendingToolCalls(msgs) {
		t.Fatal("expected pending tool calls")
	}
	calls := PendingToolCalls(msgs)
	if len(calls) != 1 || calls[0].Name != "echo" {
		t.Fatalf("PendingToolCalls() = %+v", calls)
	}

	// A later human message after the AI message shouldn't affect lookup
	// since we search from the end for the most recent AI message.
	msgs = append(msgs, messages.Tool("1", "result"), messages.AI("no calls here"))
	if HasPendingToolCalls(msgs) {
		t.Fatal("expected no pending tool calls once a later AI message has none")
	}
}

func TestToolNodeInvokeSuccess(t *testing.T) {
	echo := echoTool(t, "echo", func(_ context.Context, input map[string]any) (Result, error) {
		return Result{Content: fmt.Sprintf("got %v", input["x"])}, nil
	})
	node, err := NewToolNode([]Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	msgs := []messages.Message{
		aiMessageWithCalls(messages.ToolCall{ID: "1", Name: "echo", Args: map[string]any{"x": "hello"}}),
	}
	results, err := node.Invoke(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Role != messages.RoleTool || results[0].ToolCallID != "1" || results[0].Name != "echo" {
		t.Fatalf("unexpected result message: %+v", results[0])
	}
	if results[0].Content != "got hello" {
		t.Fatalf("Content = %q", results[0].Content)
	}
	if results[0].ResponseMetadata["status"] == "error" {
		t.Fatalf("expected no error status, got %+v", results[0].ResponseMetadata)
	}
}

func TestToolNodeInvokeNoPendingCalls(t *testing.T) {
	echo := echoTool(t, "echo", func(context.Context, map[string]any) (Result, error) {
		return Result{}, nil
	})
	node, err := NewToolNode([]Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}
	results, err := node.Invoke(context.Background(), []messages.Message{messages.Human("hi")}, nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected no results, got %+v", results)
	}
}

func TestToolNodeInvokeUnknownTool(t *testing.T) {
	echo := echoTool(t, "echo", func(context.Context, map[string]any) (Result, error) {
		return Result{}, nil
	})
	node, err := NewToolNode([]Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	msgs := []messages.Message{
		aiMessageWithCalls(messages.ToolCall{ID: "1", Name: "does-not-exist", Args: nil}),
	}
	results, err := node.Invoke(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ResponseMetadata["status"] != "error" {
		t.Fatalf("expected error status, got %+v", results[0].ResponseMetadata)
	}
	if results[0].Content == "" {
		t.Fatal("expected non-empty error content")
	}
}

func TestToolNodeInvokeErrorHandledByDefault(t *testing.T) {
	boom := echoTool(t, "boom", func(context.Context, map[string]any) (Result, error) {
		return Result{}, errors.New("kaboom")
	})
	node, err := NewToolNode([]Tool{boom})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	msgs := []messages.Message{
		aiMessageWithCalls(messages.ToolCall{ID: "1", Name: "boom"}),
	}
	results, err := node.Invoke(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v, want nil (error should be converted to a ToolMessage)", err)
	}
	if len(results) != 1 || results[0].ResponseMetadata["status"] != "error" {
		t.Fatalf("unexpected results: %+v", results)
	}
}

func TestToolNodeInvokeErrorPropagatesWhenUnhandled(t *testing.T) {
	sentinel := errors.New("kaboom")
	boom := echoTool(t, "boom", func(context.Context, map[string]any) (Result, error) {
		return Result{}, sentinel
	})
	node, err := NewToolNode([]Tool{boom}, WithHandleToolErrors(func(error) (string, bool) {
		return "", false
	}))
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	msgs := []messages.Message{
		aiMessageWithCalls(messages.ToolCall{ID: "1", Name: "boom"}),
	}
	_, err = node.Invoke(context.Background(), msgs, nil)
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected sentinel error to propagate, got %v", err)
	}
}

func TestToolNodeInvokeParallelPreservesOrder(t *testing.T) {
	var concurrent int32
	var maxConcurrent int32
	slow := echoTool(t, "slow", func(_ context.Context, input map[string]any) (Result, error) {
		n := atomic.AddInt32(&concurrent, 1)
		for {
			old := atomic.LoadInt32(&maxConcurrent)
			if n <= old || atomic.CompareAndSwapInt32(&maxConcurrent, old, n) {
				break
			}
		}
		defer atomic.AddInt32(&concurrent, -1)
		// Hold the slot briefly so concurrently dispatched calls actually
		// overlap; without this the assertion depends on scheduler luck.
		time.Sleep(10 * time.Millisecond)
		return Result{Content: fmt.Sprintf("%v", input["i"])}, nil
	})
	node, err := NewToolNode([]Tool{slow})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	var calls []messages.ToolCall
	for i := 0; i < 5; i++ {
		calls = append(calls, messages.ToolCall{ID: fmt.Sprintf("%d", i), Name: "slow", Args: map[string]any{"i": i}})
	}
	msgs := []messages.Message{aiMessageWithCalls(calls...)}

	results, err := node.Invoke(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}
	for i, result := range results {
		if result.ToolCallID != fmt.Sprintf("%d", i) {
			t.Fatalf("result[%d].ToolCallID = %q, want in-order match", i, result.ToolCallID)
		}
		if result.Content != fmt.Sprintf("%d", i) {
			t.Fatalf("result[%d].Content = %q, want %q", i, result.Content, fmt.Sprintf("%d", i))
		}
	}
	if atomic.LoadInt32(&maxConcurrent) < 2 {
		t.Fatalf("expected tool calls to run concurrently, max concurrent = %d", maxConcurrent)
	}
}

func TestToolNodeWithToolCallWrapper(t *testing.T) {
	echo := echoTool(t, "echo", func(_ context.Context, input map[string]any) (Result, error) {
		return Result{Content: fmt.Sprintf("%v", input["x"])}, nil
	})

	var wrapperCalls int32
	node, err := NewToolNode([]Tool{echo}, WithToolCallWrapper(
		func(ctx context.Context, req ToolCallRequest, next ToolHandler) (messages.Message, error) {
			atomic.AddInt32(&wrapperCalls, 1)
			modified := req.ToolCall
			modified.Args = map[string]any{"x": "wrapped"}
			return next(ctx, ToolCallRequest{ToolCall: modified, Tool: req.Tool, State: req.State})
		},
	))
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	msgs := []messages.Message{
		aiMessageWithCalls(messages.ToolCall{ID: "1", Name: "echo", Args: map[string]any{"x": "original"}}),
	}
	results, err := node.Invoke(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(results) != 1 || results[0].Content != "wrapped" {
		t.Fatalf("expected wrapper to modify args, got %+v", results)
	}
	if atomic.LoadInt32(&wrapperCalls) != 1 {
		t.Fatalf("expected wrapper to be called once, got %d", wrapperCalls)
	}
}

func TestToolNodeWithToolCallWrapperShortCircuit(t *testing.T) {
	echo := echoTool(t, "echo", func(context.Context, map[string]any) (Result, error) {
		t.Fatal("tool should not be invoked when wrapper short-circuits")
		return Result{}, nil
	})
	node, err := NewToolNode([]Tool{echo}, WithToolCallWrapper(
		func(_ context.Context, req ToolCallRequest, _ ToolHandler) (messages.Message, error) {
			msg := messages.Tool(req.ToolCall.ID, "cached")
			msg.Name = req.ToolCall.Name
			return msg, nil
		},
	))
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	msgs := []messages.Message{aiMessageWithCalls(messages.ToolCall{ID: "1", Name: "echo"})}
	results, err := node.Invoke(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(results) != 1 || results[0].Content != "cached" {
		t.Fatalf("expected cached short-circuit result, got %+v", results)
	}
}

func TestToolNodeAppendToolResults(t *testing.T) {
	echo := echoTool(t, "echo", func(context.Context, map[string]any) (Result, error) {
		return Result{Content: "ok"}, nil
	})
	node, err := NewToolNode([]Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	msgs := []messages.Message{
		messages.Human("hi"),
		aiMessageWithCalls(messages.ToolCall{ID: "1", Name: "echo"}),
	}
	out, err := node.AppendToolResults(context.Background(), msgs)
	if err != nil {
		t.Fatalf("AppendToolResults() error = %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(out))
	}
	if out[2].Role != messages.RoleTool || out[2].Content != "ok" {
		t.Fatalf("unexpected appended message: %+v", out[2])
	}

	// No pending calls: input returned unchanged.
	noCalls := []messages.Message{messages.Human("hi"), messages.AI("done")}
	out2, err := node.AppendToolResults(context.Background(), noCalls)
	if err != nil {
		t.Fatalf("AppendToolResults() error = %v", err)
	}
	if len(out2) != 2 {
		t.Fatalf("expected unchanged 2 messages, got %d", len(out2))
	}
}

func TestToolNodeToolsByName(t *testing.T) {
	echo := echoTool(t, "echo", func(context.Context, map[string]any) (Result, error) {
		return Result{}, nil
	})
	node, err := NewToolNode([]Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}
	byName := node.ToolsByName()
	if len(byName) != 1 || byName["echo"] == nil {
		t.Fatalf("ToolsByName() = %+v", byName)
	}
	// Mutating the returned map must not affect the node.
	delete(byName, "echo")
	if _, ok := node.ToolsByName()["echo"]; !ok {
		t.Fatal("expected ToolsByName() to return a defensive copy")
	}
}

func TestInvokeToolCallsFullSurfacesCommandArtifact(t *testing.T) {
	cmd := &types.Command{Update: map[string]any{"routed": true}, Goto: []any{"done"}}
	navigator := echoTool(t, "navigate", func(context.Context, map[string]any) (Result, error) {
		return Result{Content: "navigated", Artifact: cmd}, nil
	})
	plain := echoTool(t, "plain", func(context.Context, map[string]any) (Result, error) {
		// A non-Command artifact is ignored, matching InvokeToolCalls.
		return Result{Content: "plain", Artifact: 42}, nil
	})
	node, err := NewToolNode([]Tool{navigator, plain})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	calls := []messages.ToolCall{
		{ID: "1", Name: "navigate", Args: map[string]any{}},
		{ID: "2", Name: "plain", Args: map[string]any{}},
	}
	outcomes, err := node.InvokeToolCallsFull(context.Background(), calls, nil)
	if err != nil {
		t.Fatalf("InvokeToolCallsFull() error = %v", err)
	}
	if len(outcomes) != 2 {
		t.Fatalf("expected 2 outcomes, got %d", len(outcomes))
	}
	if outcomes[0].Message.Content != "navigated" || outcomes[0].Command != cmd {
		t.Fatalf("outcomes[0] = %+v, want the navigated message with its command", outcomes[0])
	}
	if outcomes[1].Message.Content != "plain" || outcomes[1].Command != nil {
		t.Fatalf("outcomes[1] = %+v, want the plain message with no command", outcomes[1])
	}

	// InvokeToolCalls discards commands but returns the same messages.
	msgs, err := node.InvokeToolCalls(context.Background(), calls, nil)
	if err != nil {
		t.Fatalf("InvokeToolCalls() error = %v", err)
	}
	if len(msgs) != 2 || msgs[0].Content != "navigated" || msgs[1].Content != "plain" {
		t.Fatalf("InvokeToolCalls() = %+v", msgs)
	}
}

func TestInvokeToolCallsFullWithWrapper(t *testing.T) {
	cmd := &types.Command{Goto: []any{"done"}}
	navigator := echoTool(t, "navigate", func(context.Context, map[string]any) (Result, error) {
		return Result{Content: "navigated", Artifact: cmd}, nil
	})

	// A wrapper that delegates to next still surfaces the tool's command.
	node, err := NewToolNode([]Tool{navigator}, WithToolCallWrapper(
		func(ctx context.Context, req ToolCallRequest, next ToolHandler) (messages.Message, error) {
			return next(ctx, req)
		},
	))
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}
	calls := []messages.ToolCall{{ID: "1", Name: "navigate", Args: map[string]any{}}}
	outcomes, err := node.InvokeToolCallsFull(context.Background(), calls, nil)
	if err != nil {
		t.Fatalf("InvokeToolCallsFull() error = %v", err)
	}
	if outcomes[0].Command != cmd {
		t.Fatalf("outcomes[0].Command = %v, want the command captured through the wrapper", outcomes[0].Command)
	}

	// A wrapper that short-circuits without calling next yields no command.
	node, err = NewToolNode([]Tool{navigator}, WithToolCallWrapper(
		func(_ context.Context, req ToolCallRequest, _ ToolHandler) (messages.Message, error) {
			return messages.Tool(req.ToolCall.ID, "short-circuited"), nil
		},
	))
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}
	outcomes, err = node.InvokeToolCallsFull(context.Background(), calls, nil)
	if err != nil {
		t.Fatalf("InvokeToolCallsFull() error = %v", err)
	}
	if outcomes[0].Message.Content != "short-circuited" || outcomes[0].Command != nil {
		t.Fatalf("outcomes[0] = %+v, want the short-circuit message with no command", outcomes[0])
	}
}
