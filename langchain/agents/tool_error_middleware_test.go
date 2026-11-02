package agents

// Through-CreateAgent mirrors of the upstream ToolErrorMiddleware tests
// (libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/
// test_tool_error.py, langchain #38781): the failing tool's error reaches
// OnError as a raw error (ToolNode applies its error conversion OUTSIDE the
// wrap chain, mirroring Python's _run_one), handled content becomes exactly
// one error ToolMessage, declining propagates the original error, and the
// no-middleware default conversion is unchanged by the layering move.

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// newFailingTool builds a tool that always fails, mirroring the upstream
// failing_tool: the error message carries a "secret detail" the OnError
// handler must NOT disclose unless it chooses to.
func newFailingTool(t *testing.T) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewSimple("failing_tool", "Tool that always fails.",
		func(_ context.Context, value string) (coretools.Result, error) {
			return coretools.Result{}, fmt.Errorf("secret detail: %s", value)
		})
	if err != nil {
		t.Fatalf("new failing tool: %v", err)
	}
	return tool
}

// failingToolModel returns a scripted model that requests one failing_tool
// call and then finishes, mirroring the upstream _model() helper.
func failingToolModel() *sequenceModel {
	return &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "1", Name: "failing_tool", Args: map[string]any{"value": "x"},
		}}},
		messages.AI("done"),
	}}
}

func toolMessagesOf(msgs []messages.Message) []messages.Message {
	var out []messages.Message
	for _, m := range msgs {
		if m.Role == messages.RoleTool {
			out = append(out, m)
		}
	}
	return out
}

// TestToolErrorMiddlewareHandledReturnsToolMessage mirrors
// test_tool_error_handled_returns_tool_message: OnError content is returned
// as an error ToolMessage and controls disclosure.
func TestToolErrorMiddlewareHandledReturnsToolMessage(t *testing.T) {
	mw, err := middleware.NewToolErrorMiddleware(func(err error, request middleware.ToolCallRequest) (any, bool) {
		return fmt.Sprintf("`%s` failed with %T.", request.ToolCall.Name, err), true
	})
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}

	agent, err := CreateAgent(failingToolModel(), []coretools.Tool{newFailingTool(t)},
		WithAgentMiddleware(mw))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	result, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	toolMsgs := toolMessagesOf(result)
	if len(toolMsgs) != 1 {
		t.Fatalf("expected exactly 1 ToolMessage, got %d: %+v", len(toolMsgs), result)
	}
	got := toolMsgs[0]
	if got.Name != "failing_tool" || got.ToolCallID != "1" {
		t.Fatalf("tool message metadata mismatch: %+v", got)
	}
	if got.ResponseMetadata["status"] != "error" {
		t.Fatalf("expected error status, got %#v", got.ResponseMetadata)
	}
	if !strings.Contains(got.Content, "`failing_tool` failed with") {
		t.Fatalf("content mismatch: %q", got.Content)
	}
	// on_error controls disclosure — the raw exception message is not leaked.
	if strings.Contains(got.Content, "secret detail") {
		t.Fatalf("raw error leaked into content: %q", got.Content)
	}
}

// TestToolErrorMiddlewareHandledContentBlocks is the content-blocks variant
// of the handled path (Python's OnError may return a list of content blocks).
func TestToolErrorMiddlewareHandledContentBlocks(t *testing.T) {
	blocks := []messages.ContentBlock{messages.TextBlock{Text: "failing_tool failed"}}
	mw, err := middleware.NewToolErrorMiddleware(func(error, middleware.ToolCallRequest) (any, bool) {
		return blocks, true
	})
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}

	agent, err := CreateAgent(failingToolModel(), []coretools.Tool{newFailingTool(t)},
		WithAgentMiddleware(mw))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	result, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	toolMsgs := toolMessagesOf(result)
	if len(toolMsgs) != 1 {
		t.Fatalf("expected exactly 1 ToolMessage, got %d", len(toolMsgs))
	}
	if toolMsgs[0].ResponseMetadata["status"] != "error" {
		t.Fatalf("expected error status, got %#v", toolMsgs[0].ResponseMetadata)
	}
	if len(toolMsgs[0].ContentBlocks) != 1 || !reflect.DeepEqual(toolMsgs[0].ContentBlocks[0], blocks[0]) {
		t.Fatalf("content blocks mismatch: %#v", toolMsgs[0].ContentBlocks)
	}
}

// TestToolErrorMiddlewareNonePropagates mirrors
// test_tool_error_none_propagates: returning not-handled from OnError lets
// the ORIGINAL error propagate out of agent Invoke (not converted).
func TestToolErrorMiddlewareNonePropagates(t *testing.T) {
	mw, err := middleware.NewToolErrorMiddleware(func(err error, _ middleware.ToolCallRequest) (any, bool) {
		var keyErr *tKeyError
		if errors.As(err, &keyErr) {
			return "handled", true
		}
		return nil, false
	})
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}

	agent, err := CreateAgent(failingToolModel(), []coretools.Tool{newFailingTool(t)},
		WithAgentMiddleware(mw))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	_, err = agent.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err == nil {
		t.Fatal("expected the original error to propagate out of Invoke")
	}
	if !strings.Contains(err.Error(), "secret detail") {
		t.Fatalf("expected the ORIGINAL tool error (with its detail), got: %v", err)
	}
}

// tKeyError is an error type the NonePropagates handler never sees from the
// failing tool (mirrors upstream's KeyError branch that stays unexercised).
type tKeyError struct{}

func (*tKeyError) Error() string { return "key error" }

// TestToolErrorMiddlewareToolFilter mirrors upstream's tools filter
// semantics through the agent: a non-matching tool's error bypasses the
// middleware entirely (OnError is never called). Under the Go catch-all
// default (see DIVERGENCES.md) the bypassed error is then converted by
// ToolNode's default handler, so the run still succeeds.
func TestToolErrorMiddlewareToolFilter(t *testing.T) {
	handled := 0
	mw, err := middleware.NewToolErrorMiddleware(
		func(error, middleware.ToolCallRequest) (any, bool) {
			handled++
			return "handled", true
		},
		middleware.WithToolErrorTools("other_tool"),
	)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}

	agent, err := CreateAgent(failingToolModel(), []coretools.Tool{newFailingTool(t)},
		WithAgentMiddleware(mw))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	result, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if handled != 0 {
		t.Fatalf("OnError must not see filtered-out tools, saw %d calls", handled)
	}
	toolMsgs := toolMessagesOf(result)
	if len(toolMsgs) != 1 || toolMsgs[0].ResponseMetadata["status"] != "error" {
		t.Fatalf("expected one default-converted error ToolMessage, got %+v", toolMsgs)
	}
	if !strings.Contains(toolMsgs[0].Content, "Please fix your mistakes.") {
		t.Fatalf("expected the default-handled content, got %q", toolMsgs[0].Content)
	}
}

// TestCreateAgentToolErrorDefaultUnchanged is the layering-move regression
// guard (behavior spec (f)): with no tool-error middleware at all, a failing
// tool's error is still converted into an error ToolMessage and the run
// succeeds — byte-identical observable behavior to before the move.
func TestCreateAgentToolErrorDefaultUnchanged(t *testing.T) {
	agent, err := CreateAgent(failingToolModel(), []coretools.Tool{newFailingTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	result, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v (default error handling must convert, not fail)", err)
	}
	toolMsgs := toolMessagesOf(result)
	if len(toolMsgs) != 1 || toolMsgs[0].ResponseMetadata["status"] != "error" {
		t.Fatalf("expected one error-status ToolMessage, got %+v", toolMsgs)
	}
	if !strings.Contains(toolMsgs[0].Content, "Please fix your mistakes.") {
		t.Fatalf("expected the default-handled content, got %q", toolMsgs[0].Content)
	}
	if len(result) < 4 || result[len(result)-1].Content != "done" {
		t.Fatalf("expected the run to continue to the final model response, got %+v", result)
	}
}
