package tools

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestToolNodeWrapInterruptPausesAndResumes locks the interrupt-escape
// forwarding: a graph interrupt raised inside the wrap chain (which runs on
// a ParallelMap worker goroutine) must pause the run through the graph
// executor's recover instead of crashing the process, and a resume answers
// it. This is the path an in-tool elicitation (partners/mcp) takes.
func TestToolNodeWrapInterruptPausesAndResumes(t *testing.T) {
	var runs int
	tool, toolErr := coretools.NewSimple("echo", "echo", func(context.Context, string) (coretools.Result, error) {
		runs++
		return coretools.Result{Content: "ran"}, nil
	})
	if toolErr != nil {
		t.Fatalf("tool: %v", toolErr)
	}
	node, nodeErr := NewToolNode([]coretools.Tool{tool}, WithToolCallWrapper(
		func(ctx context.Context, request ToolCallRequest, next ToolHandler) (messages.Message, error) {
			if answer := graphpkg.Interrupt(ctx, map[string]any{"q": 1}); answer != "go" {
				return messages.Message{}, context.Canceled
			}
			return next(ctx, request)
		}))
	if nodeErr != nil {
		t.Fatalf("node: %v", nodeErr)
	}
	aiMsg := messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c1", Name: "echo", Args: map[string]any{"input": "hi"}}}}
	g := graphpkg.NewStateGraph()
	g.AddNode("tools", func(rt runtime.Runtime, state map[string]any) (any, error) {
		t.Logf("tools node ran, runs=%d", runs)
		msgs, err := node.Invoke(rt, []messages.Message{aiMsg}, state)
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": append([]messages.Message{aiMsg}, msgs...)}, nil
	})
	g.AddEdge(types.START, "tools")
	g.AddEdge("tools", types.END)
	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	paused, err := cg.InvokeWithOptions(t.Context(), map[string]any{
		"messages": []messages.Message{aiMsg},
	}, graphpkg.Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("interrupts: %d, want 1 (pause)", len(paused.Interrupts))
	}
	if runs != 0 {
		t.Fatalf("tool must not run before the answer, runs=%d", runs)
	}
	final, err := cg.InvokeWithOptions(t.Context(), nil, graphpkg.Options{
		ThreadID: "t",
		Resume:   map[string]any{paused.Interrupts[0].ID: "go"},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if runs != 1 {
		t.Fatalf("tool must run exactly once after resume, runs=%d", runs)
	}
	if len(final.Interrupts) != 0 {
		t.Fatalf("resumed run must complete, interrupts=%d", len(final.Interrupts))
	}
}

// TestInterruptEscapeErrorMessage covers the escape error's text.
func TestInterruptEscapeErrorMessage(t *testing.T) {
	escape := &interruptEscape{}
	if escape.Error() == "" {
		t.Fatal("escape error message must not be empty")
	}
}

// TestUnhandledToolErrorMessages covers both Error() spellings.
func TestUnhandledToolErrorMessages(t *testing.T) {
	if (&UnhandledToolError{}).Error() != "tools: unhandled tool error" {
		t.Fatal("bare message mismatch")
	}
	wrapped := &UnhandledToolError{Err: context.Canceled}
	if wrapped.Error() != "tools: unhandled tool error: context canceled" {
		t.Fatalf("wrapped message: %q", wrapped.Error())
	}
}
