package graph

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/channels"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/checkpoint"
)

func TestLinearGraph(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("step1", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"count": state["count"].(int) + 1}, nil
	})
	g.AddNode("step2", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"count": state["count"].(int) + 10}, nil
	})
	g.AddEdge(agentruntime.START, "step1")
	g.AddEdge("step1", "step2")
	g.AddEdge("step2", agentruntime.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	result, err := cg.Invoke(context.Background(), map[string]any{"count": 0})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if result.Values["count"] != 11 {
		t.Fatalf("count = %v, want 11", result.Values["count"])
	}
	if len(result.Interrupts) != 0 {
		t.Fatalf("expected no interrupts, got %+v", result.Interrupts)
	}
}

// TestReActLoopShape mirrors the exact shape `create_agent` needs: a "model"
// node and a "tools" node, with a conditional edge routing back to "model"
// or ending, and a messages key accumulated via a reducer across loop
// iterations (channels.MessagesReducer is exercised directly in
// channels_test.go; this uses AppendSliceReducer with plain strings to keep
// the loop-shape assertions simple).
func TestReActLoopShape(t *testing.T) {
	g := NewStateGraph()
	g.AddReducer("messages", channels.AppendSliceReducer)

	calls := 0
	g.AddNode("model", func(_ context.Context, state map[string]any) (any, error) {
		calls++
		msgs, _ := state["messages"].([]string)
		if len(msgs) >= 2 {
			// no more tool calls requested: final answer.
			return map[string]any{"messages": []string{"final answer"}}, nil
		}
		return map[string]any{"messages": []string{fmt.Sprintf("call-tool-%d", calls)}}, nil
	})
	g.AddNode("tools", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"messages": []string{"tool-result"}}, nil
	})
	g.AddEdge(agentruntime.START, "model")
	g.AddConditionalEdges("model", func(_ context.Context, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]string)
		if len(msgs) > 0 && msgs[len(msgs)-1] == "final answer" {
			return To(agentruntime.END), nil
		}
		return To("tools"), nil
	})
	g.AddEdge("tools", "model")

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	result, err := cg.Invoke(context.Background(), map[string]any{"messages": []string{}})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	msgs := result.Values["messages"].([]string)
	// call-tool-1, tool-result, call-tool-2 (>=2 not yet true since len==2 checked
	// AFTER appending tool-result twice)... just assert loop terminated with the
	// final answer as the last message and multiple round trips occurred.
	if msgs[len(msgs)-1] != "final answer" {
		t.Fatalf("expected loop to terminate with final answer, got %+v", msgs)
	}
	if calls < 2 {
		t.Fatalf("expected at least 2 model calls (a loop), got %d", calls)
	}
}

func TestCommandGotoAndUpdate(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		return &agentruntime.Command{
			Update: map[string]any{"visited": "a"},
			Goto:   To("c"), // bypasses the static edge a->b
		}, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		t.Fatal("node b should not run when Command.Goto redirects to c")
		return nil, nil
	})
	g.AddNode("c", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"visited": state["visited"].(string) + ",c"}, nil
	})
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", agentruntime.END)
	g.AddEdge("c", agentruntime.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	result, err := cg.Invoke(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if result.Values["visited"] != "a,c" {
		t.Fatalf("visited = %v, want %q", result.Values["visited"], "a,c")
	}
}

// TestSendFanOut mirrors the map-reduce example from Python's Send docstring:
// a conditional edge fans out to the same node multiple times with different
// per-call input, and results are aggregated back via a reducer.
func TestSendFanOut(t *testing.T) {
	g := NewStateGraph()
	g.AddReducer("jokes", channels.AppendSliceReducer)

	var concurrentNow int32
	var maxConcurrent int32
	g.AddNode("generate_joke", func(_ context.Context, state map[string]any) (any, error) {
		n := atomic.AddInt32(&concurrentNow, 1)
		for {
			old := atomic.LoadInt32(&maxConcurrent)
			if n <= old || atomic.CompareAndSwapInt32(&maxConcurrent, old, n) {
				break
			}
		}
		defer atomic.AddInt32(&concurrentNow, -1)
		subject := state["subject"].(string)
		return map[string]any{"jokes": []string{"joke about " + subject}}, nil
	})
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) {
		return nil, nil
	})
	g.AddEdge(agentruntime.START, "start")
	g.AddConditionalEdges("start", func(_ context.Context, state map[string]any) ([]any, error) {
		subjects := state["subjects"].([]string)
		dests := make([]any, len(subjects))
		for i, s := range subjects {
			dests[i] = &agentruntime.Send{Node: "generate_joke", Arg: map[string]any{"subject": s}}
		}
		return dests, nil
	})
	g.AddEdge("generate_joke", agentruntime.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	result, err := cg.Invoke(context.Background(), map[string]any{"subjects": []string{"cats", "dogs"}})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	jokes, _ := result.Values["jokes"].([]string)
	sort.Strings(jokes)
	want := []string{"joke about cats", "joke about dogs"}
	if len(jokes) != 2 || jokes[0] != want[0] || jokes[1] != want[1] {
		t.Fatalf("jokes = %+v, want %+v", jokes, want)
	}
}

func TestInterruptAndResume(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("ask_human", func(ctx context.Context, state map[string]any) (any, error) {
		answer := Interrupt(ctx, "what is your name?")
		return map[string]any{"name": answer}, nil
	})
	g.AddEdge(agentruntime.START, "ask_human")
	g.AddEdge("ask_human", agentruntime.END)

	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %+v", first.Interrupts)
	}
	if first.Interrupts[0].Value != "what is your name?" {
		t.Fatalf("interrupt value = %v", first.Interrupts[0].Value)
	}
	if _, ok := first.Values["name"]; ok {
		t.Fatalf("expected no 'name' key before resume, got %+v", first.Values)
	}

	second, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t1", Resume: "Ada"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if second.Values["name"] != "Ada" {
		t.Fatalf("name = %v, want Ada", second.Values["name"])
	}

	// Checkpoint should be cleared after a completed run.
	if _, ok := saver.Get("t1"); ok {
		t.Fatal("expected checkpoint to be cleared after run completes")
	}
}

func TestResumeWithoutCheckpointerErrors(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("n", func(ctx context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(agentruntime.START, "n")
	g.AddEdge("n", agentruntime.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "x", Resume: "y"}); err == nil {
		t.Fatal("expected error resuming without a checkpointer")
	}
}

func TestCompileErrors(t *testing.T) {
	if _, err := NewStateGraph().Compile(); err == nil {
		t.Fatal("expected error when entry point is not set")
	}

	g := NewStateGraph()
	g.AddNode("a", func(context.Context, map[string]any) (any, error) { return nil, nil })
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", "missing")
	if _, err := g.Compile(); err == nil {
		t.Fatal("expected error for edge to unknown node")
	}

	dup := NewStateGraph()
	dup.AddNode("a", func(context.Context, map[string]any) (any, error) { return nil, nil })
	dup.AddNode("a", func(context.Context, map[string]any) (any, error) { return nil, nil })
	if _, err := dup.Compile(); err == nil {
		t.Fatal("expected error for duplicate node")
	}
}

func TestNodeWithNoOutgoingEdgeErrors(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", func(context.Context, map[string]any) (any, error) { return nil, nil })
	g.AddEdge(agentruntime.START, "a")
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.Invoke(context.Background(), map[string]any{}); err == nil {
		t.Fatal("expected runtime error for node with no outgoing edge")
	}
}

func TestUnsupportedCommandGraphErrors(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", func(context.Context, map[string]any) (any, error) {
		return &agentruntime.Command{Graph: agentruntime.ParentGraph}, nil
	})
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", agentruntime.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.Invoke(context.Background(), map[string]any{}); err == nil {
		t.Fatal("expected error for unsupported Command.Graph (subgraphs)")
	}
}

func TestRecursionLimitExceeded(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("loop", func(context.Context, map[string]any) (any, error) { return nil, nil })
	g.AddEdge(agentruntime.START, "loop")
	g.AddEdge("loop", "loop")
	cg, err := g.Compile(WithRecursionLimit(5))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(context.Background(), map[string]any{})
	if err == nil {
		t.Fatal("expected recursion limit error")
	}
}

func TestNodeErrorPropagates(t *testing.T) {
	sentinel := errors.New("boom")
	g := NewStateGraph()
	g.AddNode("a", func(context.Context, map[string]any) (any, error) { return nil, sentinel })
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", agentruntime.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(context.Background(), map[string]any{})
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected sentinel error, got %v", err)
	}
}

func TestInterruptOutsideGraphPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic calling Interrupt outside a graph run")
		}
	}()
	Interrupt(context.Background(), "value")
}

// TestCompiledGraph_InterruptBefore verifies that interrupt_before pauses the
// graph before the named node runs, that prior nodes' updates are visible in
// the paused Result, and that resuming (with Resume=nil, mirroring Python's
// `invoke(None, config)`) runs the paused node to completion. The resume must
// not re-run already-completed nodes (see TestInterruptBefore_ResumeDoesNotRerun).
func TestCompiledGraph_InterruptBefore(t *testing.T) {
	g := NewStateGraph()
	aRuns := 0
	bRuns := 0
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		aRuns++
		return map[string]any{"a_ran": true}, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		bRuns++
		return map[string]any{"b_ran": true}, nil
	})
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", agentruntime.END)
	saver := checkpoint.NewMemorySaver()
	compiled, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("b"))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	res, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(res.Interrupts) == 0 {
		t.Fatalf("expected run to pause before b")
	}
	if !res.Values["a_ran"].(bool) {
		t.Fatalf("a should have run")
	}
	if _, ran := res.Values["b_ran"]; ran {
		t.Fatalf("b should NOT have run yet")
	}
	if aRuns != 1 || bRuns != 0 {
		t.Fatalf("expected a=1 b=0 invocations, got a=%d b=%d", aRuns, bRuns)
	}

	// Resume.
	res2, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1", Resume: nil})
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if !res2.Values["b_ran"].(bool) {
		t.Fatalf("b should run after resume")
	}
	// Critical correctness check: a must NOT re-run on resume.
	if aRuns != 1 || bRuns != 1 {
		t.Fatalf("after resume expected a=1 b=1 invocations, got a=%d b=%d", aRuns, bRuns)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", res2.Interrupts)
	}
	// Checkpoint should be cleared after a completed run.
	if _, ok := saver.Get("t1"); ok {
		t.Fatal("expected checkpoint to be cleared after run completes")
	}
}

// TestCompiledGraph_InterruptAfter verifies that interrupt_after pauses after
// the named node runs (with its update visible) but before its successor, and
// that resuming runs the successor without re-running the paused-from node.
func TestCompiledGraph_InterruptAfter(t *testing.T) {
	g := NewStateGraph()
	aRuns := 0
	bRuns := 0
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		aRuns++
		return map[string]any{"a_ran": true}, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		bRuns++
		return map[string]any{"b_ran": true}, nil
	})
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", agentruntime.END)
	saver := checkpoint.NewMemorySaver()
	compiled, err := g.Compile(WithCheckpointer(saver), WithInterruptAfter("a"))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	res, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(res.Interrupts) == 0 {
		t.Fatalf("expected run to pause after a")
	}
	if !res.Values["a_ran"].(bool) {
		t.Fatalf("a should have run")
	}
	if _, ran := res.Values["b_ran"]; ran {
		t.Fatalf("b should NOT have run yet")
	}
	if aRuns != 1 || bRuns != 0 {
		t.Fatalf("expected a=1 b=0 invocations, got a=%d b=%d", aRuns, bRuns)
	}

	// Resume.
	res2, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1", Resume: nil})
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if !res2.Values["b_ran"].(bool) {
		t.Fatalf("b should run after resume")
	}
	// Critical correctness check: a must NOT re-run on resume.
	if aRuns != 1 || bRuns != 1 {
		t.Fatalf("after resume expected a=1 b=1 invocations, got a=%d b=%d", aRuns, bRuns)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", res2.Interrupts)
	}
}

// TestCompiledGraph_InterruptBeforeAndAfter verifies both options can be set
// simultaneously and pause at each configured boundary, with each resume
// advancing exactly one boundary.
func TestCompiledGraph_InterruptBeforeAndAfter(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"a_ran": true}, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"b_ran": true}, nil
	})
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", agentruntime.END)
	saver := checkpoint.NewMemorySaver()
	compiled, err := g.Compile(WithCheckpointer(saver),
		WithInterruptBefore("b"),
		WithInterruptAfter("a"),
	)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	// First run: a runs, then interrupt_after("a") fires before b is scheduled.
	res, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(res.Interrupts) == 0 {
		t.Fatalf("expected pause after a (interrupt_after fires first since a runs before b)")
	}
	if !res.Values["a_ran"].(bool) {
		t.Fatalf("a should have run")
	}

	// Resume: advances past interrupt_after("a"), then interrupt_before("b") fires.
	res2, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1", Resume: nil})
	if err != nil {
		t.Fatalf("Resume 1: %v", err)
	}
	if len(res2.Interrupts) == 0 {
		t.Fatalf("expected pause before b after first resume")
	}
	if _, ran := res2.Values["b_ran"]; ran {
		t.Fatalf("b should NOT have run yet (interrupt_before)")
	}

	// Resume again: b runs to completion.
	res3, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1", Resume: nil})
	if err != nil {
		t.Fatalf("Resume 2: %v", err)
	}
	if !res3.Values["b_ran"].(bool) {
		t.Fatalf("b should run after second resume")
	}
	if len(res3.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after final resume, got %+v", res3.Interrupts)
	}
}
