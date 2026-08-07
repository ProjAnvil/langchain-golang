package graph

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func TestLinearGraph(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("step1", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"count": state["count"].(int) + 1}, nil
	})
	g.AddNode("step2", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"count": state["count"].(int) + 10}, nil
	})
	g.AddEdge(types.START, "step1")
	g.AddEdge("step1", "step2")
	g.AddEdge("step2", types.END)

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
	g.AddEdge(types.START, "model")
	g.AddConditionalEdges("model", func(_ context.Context, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]string)
		if len(msgs) > 0 && msgs[len(msgs)-1] == "final answer" {
			return To(types.END), nil
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
		return &types.Command{
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
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	g.AddEdge("c", types.END)

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
	g.AddEdge(types.START, "start")
	g.AddConditionalEdges("start", func(_ context.Context, state map[string]any) ([]any, error) {
		subjects := state["subjects"].([]string)
		dests := make([]any, len(subjects))
		for i, s := range subjects {
			dests[i] = &types.Send{Node: "generate_joke", Arg: map[string]any{"subject": s}}
		}
		return dests, nil
	})
	g.AddEdge("generate_joke", types.END)

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
	g.AddEdge(types.START, "ask_human")
	g.AddEdge("ask_human", types.END)

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

	// D1: checkpoints survive completion — the final checkpoint is retained
	// with no scheduled tasks (empty Next).
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("expected final checkpoint to be retained, got tup=%+v err=%v", tup, err)
	}
	if len(tup.Checkpoint.Next) != 0 {
		t.Fatalf("expected empty Next in final checkpoint, got %+v", tup.Checkpoint.Next)
	}
}

func TestResumeWithoutCheckpointerErrors(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("n", func(ctx context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)
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
	g.AddEdge(types.START, "a")
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
	g.AddEdge(types.START, "a")
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
		return &types.Command{Graph: types.ParentGraph}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
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
	g.AddEdge(types.START, "loop")
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
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
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
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
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
	// D1: checkpoints survive completion — the final checkpoint is retained
	// with no scheduled tasks (empty Next).
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("expected final checkpoint to be retained, got tup=%+v err=%v", tup, err)
	}
	if len(tup.Checkpoint.Next) != 0 {
		t.Fatalf("expected empty Next in final checkpoint, got %+v", tup.Checkpoint.Next)
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
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
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
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
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

// TestVersionedCheckpointBookkeeping verifies the versioned executor's
// checkpoint history for a 3-superstep linear graph: one "input" checkpoint
// (step -1) plus one "loop" checkpoint per superstep, each carrying the
// channel versions written so far, the per-node versions-seen bookkeeping,
// and the planned tasks for the next superstep.
func TestVersionedCheckpointBookkeeping(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"k1": "v1"}, nil
	})
	g.AddNode("n2", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"k2": "v2"}, nil
	})
	g.AddNode("n3", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"k3": "v3"}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", "n3")
	g.AddEdge("n3", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	tuples, err := saver.List(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(tuples) != 4 {
		t.Fatalf("expected 4 checkpoints (input + 3 supersteps), got %d", len(tuples))
	}
	// List is newest-first: steps 2, 1, 0, -1.
	wantSteps := []int{2, 1, 0, -1}
	wantSources := []string{"loop", "loop", "loop", "input"}
	for i, tup := range tuples {
		if tup.Metadata.Step != wantSteps[i] || tup.Metadata.Source != wantSources[i] {
			t.Fatalf("checkpoint %d: metadata = %+v, want step %d source %q", i, tup.Metadata, wantSteps[i], wantSources[i])
		}
	}
	final := tuples[0]
	if len(final.Checkpoint.Next) != 0 {
		t.Fatalf("final checkpoint Next = %+v, want empty", final.Checkpoint.Next)
	}
	for _, k := range []string{"k0", "k1", "k2", "k3"} {
		if _, ok := final.Checkpoint.ChannelValues[k]; !ok {
			t.Fatalf("final checkpoint missing channel value %q: %+v", k, final.Checkpoint.ChannelValues)
		}
	}

	// Version bookkeeping: the input batch bumps the global version to 1
	// (k0@1); each superstep applies a single global bump, so the key written
	// in superstep i sits at version i+1 and earlier keys keep their version.
	wantVersions := []map[string]int64{
		{"k0": 1, "k1": 2, "k2": 3, "k3": 4},
		{"k0": 1, "k1": 2, "k2": 3},
		{"k0": 1, "k1": 2},
		{"k0": 1},
	}
	for i, tup := range tuples {
		if !reflect.DeepEqual(tup.Checkpoint.ChannelVersions, wantVersions[i]) {
			t.Fatalf("checkpoint %d (step %d): ChannelVersions = %+v, want %+v",
				i, tup.Metadata.Step, tup.Checkpoint.ChannelVersions, wantVersions[i])
		}
	}

	// The step-0 checkpoint plans n2 for the next superstep, with a populated
	// deterministic task ID.
	step0 := tuples[2]
	if len(step0.Checkpoint.Next) != 1 || step0.Checkpoint.Next[0].Node != "n2" {
		t.Fatalf("step-0 checkpoint Next = %+v, want a single n2 task", step0.Checkpoint.Next)
	}
	if step0.Checkpoint.Next[0].ID == "" {
		t.Fatal("planned task ID must be populated")
	}
	// VersionsSeen records n1's pre-write view at the step-0 boundary.
	if seen := step0.Checkpoint.VersionsSeen["n1"]; seen["k0"] != 1 {
		t.Fatalf("VersionsSeen[n1] = %+v, want k0@1", seen)
	}

	// Parent chain (D3): walking newest-first, each checkpoint's ParentConfig
	// must point at the previous checkpoint, and the first (input) checkpoint
	// must have no parent.
	for i, tup := range tuples {
		if i == len(tuples)-1 {
			if tup.ParentConfig != nil {
				t.Fatalf("first checkpoint (step %d): ParentConfig = %+v, want nil", tup.Metadata.Step, tup.ParentConfig)
			}
			continue
		}
		prev := tuples[i+1]
		if tup.ParentConfig == nil {
			t.Fatalf("checkpoint %d (step %d): ParentConfig is nil, want predecessor %q", i, tup.Metadata.Step, prev.Checkpoint.ID)
		}
		if tup.ParentConfig.CheckpointID != prev.Checkpoint.ID {
			t.Fatalf("checkpoint %d (step %d): ParentConfig.CheckpointID = %q, want %q",
				i, tup.Metadata.Step, tup.ParentConfig.CheckpointID, prev.Checkpoint.ID)
		}
		if tup.ParentConfig.ThreadID != "t1" {
			t.Fatalf("checkpoint %d (step %d): ParentConfig.ThreadID = %q, want %q",
				i, tup.Metadata.Step, tup.ParentConfig.ThreadID, "t1")
		}
	}
}

// TestSingleGlobalVersionPerSuperstep verifies that all channels written in
// the same superstep are bumped to one shared version, even when written by
// different concurrent tasks.
func TestSingleGlobalVersionPerSuperstep(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddReducer("ra", channels.AppendSliceReducer)
	g.AddReducer("rb", channels.AppendSliceReducer)
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("writer", func(_ context.Context, state map[string]any) (any, error) {
		key := state["key"].(string)
		return map[string]any{key: []string{"x"}}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddConditionalEdges("start", func(_ context.Context, _ map[string]any) ([]any, error) {
		return []any{
			&types.Send{Node: "writer", Arg: map[string]any{"key": "ra"}},
			&types.Send{Node: "writer", Arg: map[string]any{"key": "rb"}},
		}, nil
	})
	g.AddEdge("writer", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	latest, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || latest == nil {
		t.Fatalf("expected a final checkpoint, got tup=%+v err=%v", latest, err)
	}
	va, oka := latest.Checkpoint.ChannelVersions["ra"]
	vb, okb := latest.Checkpoint.ChannelVersions["rb"]
	if !oka || !okb {
		t.Fatalf("expected versions for ra and rb, got %+v", latest.Checkpoint.ChannelVersions)
	}
	if va != vb {
		t.Fatalf("channels written in one superstep must share a version: ra@%d rb@%d", va, vb)
	}
}

// TestLastValueDoubleWriteInOneSuperstepErrors verifies that two tasks
// writing the same unregistered (LastValue) key in one superstep surface an
// *channels.InvalidUpdateError instead of silently picking a winner.
func TestLastValueDoubleWriteInOneSuperstepErrors(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("worker", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"out": state["subject"]}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddConditionalEdges("start", func(_ context.Context, _ map[string]any) ([]any, error) {
		return []any{
			&types.Send{Node: "worker", Arg: map[string]any{"subject": "a"}},
			&types.Send{Node: "worker", Arg: map[string]any{"subject": "b"}},
		}, nil
	})
	g.AddEdge("worker", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	_, err = cg.Invoke(context.Background(), map[string]any{})
	if err == nil {
		t.Fatal("expected an error for two writes to one LastValue key in a single superstep")
	}
	var iu *channels.InvalidUpdateError
	if !errors.As(err, &iu) {
		t.Fatalf("expected *channels.InvalidUpdateError, got %v", err)
	}
}

// TestReducerFoldOrderDeterministic verifies that concurrent fan-out writes
// to a reducer key fold in deterministic active-task order across runs.
func TestReducerFoldOrderDeterministic(t *testing.T) {
	g := NewStateGraph()
	g.AddReducer("jokes", channels.AppendSliceReducer)
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("generate_joke", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"jokes": []string{"joke about " + state["subject"].(string)}}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddConditionalEdges("start", func(_ context.Context, _ map[string]any) ([]any, error) {
		dests := make([]any, 0, 4)
		for _, s := range []string{"a", "b", "c", "d"} {
			dests = append(dests, &types.Send{Node: "generate_joke", Arg: map[string]any{"subject": s}})
		}
		return dests, nil
	})
	g.AddEdge("generate_joke", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	want := []string{"joke about a", "joke about b", "joke about c", "joke about d"}
	for run := 0; run < 5; run++ {
		result, err := cg.Invoke(context.Background(), map[string]any{})
		if err != nil {
			t.Fatalf("run %d: Invoke() error = %v", run, err)
		}
		if got := result.Values["jokes"]; !reflect.DeepEqual(got, want) {
			t.Fatalf("run %d: jokes = %+v, want %+v", run, got, want)
		}
	}
}

// TestAddChannelExpiryBetweenSupersteps verifies that channels registered via
// AddChannel with expiring semantics (Ephemeral, non-accumulating Topic) keep
// their value for exactly the following superstep and are then cleared by the
// step-boundary empty Update — disappearing from both the node-visible state
// and the saved checkpoints.
func TestAddChannelExpiryBetweenSupersteps(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddChannel("tmp", channels.NewEphemeral(false))
	g.AddChannel("feed", channels.NewTopic(false))
	g.AddNode("n1", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"tmp": "t", "feed": "f", "keep": 1}, nil
	})
	n2Saw := map[string]bool{}
	g.AddNode("n2", func(_ context.Context, state map[string]any) (any, error) {
		_, n2Saw["tmp"] = state["tmp"]
		_, n2Saw["feed"] = state["feed"]
		return nil, nil
	})
	n3Saw := map[string]bool{}
	g.AddNode("n3", func(_ context.Context, state map[string]any) (any, error) {
		_, n3Saw["tmp"] = state["tmp"]
		_, n3Saw["feed"] = state["feed"]
		_, n3Saw["keep"] = state["keep"]
		return nil, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", "n3")
	g.AddEdge("n3", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	result, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	// The superstep right after the write still sees the values...
	if !n2Saw["tmp"] || !n2Saw["feed"] {
		t.Fatalf("n2 should see tmp and feed (written one superstep earlier), got %+v", n2Saw)
	}
	// ...and the one after that must not.
	if n3Saw["tmp"] || n3Saw["feed"] {
		t.Fatalf("n3 should NOT see expired tmp/feed, got %+v", n3Saw)
	}
	if !n3Saw["keep"] {
		t.Fatalf("n3 should still see the plain LastValue key, got %+v", n3Saw)
	}
	if _, ok := result.Values["tmp"]; ok {
		t.Fatalf("final state should not contain expired tmp, got %+v", result.Values)
	}

	// The step-0 checkpoint (after n1) carries the live ephemeral/topic
	// values; the step-1 checkpoint omits them.
	tuples, err := saver.List(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(tuples) != 4 {
		t.Fatalf("expected 4 checkpoints, got %d", len(tuples))
	}
	if _, ok := tuples[2].Checkpoint.ChannelValues["tmp"]; !ok {
		t.Fatalf("step-0 checkpoint should carry live tmp, got %+v", tuples[2].Checkpoint.ChannelValues)
	}
	if _, ok := tuples[1].Checkpoint.ChannelValues["tmp"]; ok {
		t.Fatalf("step-1 checkpoint should omit expired tmp, got %+v", tuples[1].Checkpoint.ChannelValues)
	}
	if _, ok := tuples[1].Checkpoint.ChannelValues["feed"]; ok {
		t.Fatalf("step-1 checkpoint should omit expired feed, got %+v", tuples[1].Checkpoint.ChannelValues)
	}
}

// TestCheckpointRetainedAfterCompletion pins D1: a completed run no longer
// deletes its thread's checkpoints; the final checkpoint is retained with an
// empty Next.
func TestCheckpointRetainedAfterCompletion(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"done": true}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("expected final checkpoint to be retained, got tup=%+v err=%v", tup, err)
	}
	if len(tup.Checkpoint.Next) != 0 {
		t.Fatalf("expected empty Next in final checkpoint, got %+v", tup.Checkpoint.Next)
	}
	if tup.Metadata.Source != "loop" {
		t.Fatalf("final checkpoint Source = %q, want %q", tup.Metadata.Source, "loop")
	}
	if tup.Checkpoint.ChannelValues["done"] != true {
		t.Fatalf("final checkpoint values = %+v, want done=true", tup.Checkpoint.ChannelValues)
	}
}

// TestNewTurnWithInputAfterCompletion pins D2: invoking with fresh (non-empty)
// input on a thread that already has checkpoints starts a NEW turn — the input
// is applied on top of the latest state and execution restarts from the entry
// point — rather than silently resuming.
func TestNewTurnWithInputAfterCompletion(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	aRuns := 0
	g.AddNode("a", func(_ context.Context, state map[string]any) (any, error) {
		aRuns++
		return map[string]any{"y": state["x"].(int) + 1}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 1}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	if first.Values["y"] != 2 || aRuns != 1 {
		t.Fatalf("turn 1: y = %v, aRuns = %d; want y=2, aRuns=1", first.Values["y"], aRuns)
	}

	second, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 10}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}
	// A resume would not re-run a; a new turn must.
	if aRuns != 2 {
		t.Fatalf("expected entry node to re-run on new turn, aRuns = %d", aRuns)
	}
	if second.Values["x"] != 10 || second.Values["y"] != 11 {
		t.Fatalf("turn 2 values = %+v, want x=10 y=11", second.Values)
	}

	// History keeps both turns: two "input" checkpoints, and the second
	// turn's loop checkpoint continues the step counter (restored step 0 -> 1).
	tuples, err := saver.List(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	inputCheckpoints := 0
	for _, tup := range tuples {
		if tup.Metadata.Source == "input" {
			inputCheckpoints++
		}
	}
	if inputCheckpoints != 2 {
		t.Fatalf("expected 2 input checkpoints (one per turn), got %d", inputCheckpoints)
	}
	if latest := tuples[0]; latest.Metadata.Step != 1 || latest.Metadata.Source != "loop" {
		t.Fatalf("latest checkpoint metadata = %+v, want step 1 source loop (step continues across turns)", latest.Metadata)
	}
}

// TestResumeSkipsCompletedSibling verifies pending-writes resume fidelity: in
// a superstep where sibling A completes and sibling B interrupts, resuming
// re-executes only B. A is not re-run (side-effect counter) and A's state
// update is applied exactly once, replayed from its persisted pending writes.
func TestResumeSkipsCompletedSibling(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddReducer("log", channels.AppendSliceReducer)
	var aRuns, bRuns int32
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&aRuns, 1)
		return map[string]any{"log": []string{"a"}}, nil
	})
	g.AddNode("b", func(ctx context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&bRuns, 1)
		Interrupt(ctx, "pause-b")
		return map[string]any{"log": []string{"b"}}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "a")
	g.AddEdge("start", "b")
	g.AddEdge("a", types.END)
	g.AddEdge("b", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "pause-b" {
		t.Fatalf("expected one interrupt (pause-b), got %+v", first.Interrupts)
	}
	if aRuns != 1 || bRuns != 1 {
		t.Fatalf("expected a=1 b=1 invocations at pause, got a=%d b=%d", aRuns, bRuns)
	}
	if _, ok := first.Values["log"]; ok {
		t.Fatalf("a's update must not be committed at the pause, got %+v", first.Values)
	}

	// The pause checkpoint plans the interrupted superstep's FULL task set
	// (both siblings), with distinct populated task IDs.
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	if len(tup.Checkpoint.Next) != 2 {
		t.Fatalf("pause checkpoint Next = %+v, want both siblings planned", tup.Checkpoint.Next)
	}
	if tup.Checkpoint.Next[0].ID == "" || tup.Checkpoint.Next[0].ID == tup.Checkpoint.Next[1].ID {
		t.Fatalf("planned task IDs must be populated and distinct, got %+v", tup.Checkpoint.Next)
	}

	second, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t1", Resume: "go"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if aRuns != 1 {
		t.Fatalf("completed sibling a must NOT re-run on resume, ran %d times", aRuns)
	}
	if bRuns != 2 {
		t.Fatalf("interrupted sibling b must re-run exactly once, ran %d times", bRuns)
	}
	if got, want := second.Values["log"], []string{"a", "b"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("log = %+v, want %+v (a's update applied exactly once)", got, want)
	}
}

// TestResumeRestoresCompletedTaskRouting pins D4: a completed sibling's
// Command.Goto destinations (a plain node name AND a *types.Send) persist as
// ReservedTasks writes — plain names normalized to types.Send — and both are
// dispatched on resume even though the completed sibling itself is not re-run.
func TestResumeRestoresCompletedTaskRouting(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	var aRuns, cRuns, dRuns int32
	var dSawX atomic.Bool
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&aRuns, 1)
		return &types.Command{
			Update: map[string]any{"from": "a"},
			Goto:   []any{"c", &types.Send{Node: "d", Arg: map[string]any{"x": 1}}},
		}, nil
	})
	g.AddNode("b", func(ctx context.Context, _ map[string]any) (any, error) {
		Interrupt(ctx, "pause-b")
		return nil, nil
	})
	g.AddNode("c", func(_ context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&cRuns, 1)
		return map[string]any{"c_ran": true}, nil
	})
	g.AddNode("d", func(_ context.Context, state map[string]any) (any, error) {
		atomic.AddInt32(&dRuns, 1)
		dSawX.Store(state["x"] == 1)
		return nil, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "a")
	g.AddEdge("start", "b")
	g.AddEdge("b", types.END)
	g.AddEdge("c", types.END)
	g.AddEdge("d", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 {
		t.Fatalf("expected one interrupt, got %+v", first.Interrupts)
	}

	// The completed sibling's routing must persist as ReservedTasks pending
	// writes, plain names normalized to types.Send (D4).
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	var sends []types.Send
	for _, w := range tup.PendingWrites {
		if w.Channel == checkpoint.ReservedTasks {
			s, ok := w.Value.(types.Send)
			if !ok {
				t.Fatalf("ReservedTasks write value = %T, want types.Send", w.Value)
			}
			sends = append(sends, s)
		}
	}
	wantSends := []types.Send{{Node: "c"}, {Node: "d", Arg: map[string]any{"x": 1}}}
	if !reflect.DeepEqual(sends, wantSends) {
		t.Fatalf("persisted ReservedTasks sends = %+v, want %+v", sends, wantSends)
	}

	second, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t1", Resume: "go"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if aRuns != 1 {
		t.Fatalf("completed sibling a must NOT re-run on resume, ran %d times", aRuns)
	}
	if cRuns != 1 || dRuns != 1 {
		t.Fatalf("a's goto destinations must run on resume: c=%d d=%d, want 1 each", cRuns, dRuns)
	}
	if !dSawX.Load() {
		t.Fatal("send-driven d invocation must receive its Send arg")
	}
	if second.Values["from"] != "a" || second.Values["c_ran"] != true {
		t.Fatalf("final values = %+v, want from=a and c_ran=true", second.Values)
	}
}

// TestInterruptBeforeResumesAllSiblings verifies that interrupt_before on a
// multi-successor superstep checkpoints the FULL planned task set, so resuming
// re-dispatches every sibling — not just the registered node.
func TestInterruptBeforeResumesAllSiblings(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	var bRuns, cRuns int32
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&bRuns, 1)
		return map[string]any{"b_ran": true}, nil
	})
	g.AddNode("c", func(_ context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&cRuns, 1)
		return map[string]any{"c_ran": true}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "b")
	g.AddEdge("start", "c")
	g.AddEdge("b", types.END)
	g.AddEdge("c", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("b"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].ID != interruptBeforeID+"b" {
		t.Fatalf("expected interrupt_before(b), got %+v", first.Interrupts)
	}
	if bRuns != 0 || cRuns != 0 {
		t.Fatalf("neither sibling should have run at the pause, got b=%d c=%d", bRuns, cRuns)
	}
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	if len(tup.Checkpoint.Next) != 2 {
		t.Fatalf("pause checkpoint Next = %+v, want the full sibling set [b c]", tup.Checkpoint.Next)
	}

	second, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if bRuns != 1 || cRuns != 1 {
		t.Fatalf("both siblings must run exactly once after resume, got b=%d c=%d", bRuns, cRuns)
	}
}

// TestSameNodeFanOutInterruptsResumeByTaskID pins D5: two tasks of the SAME
// node interrupting in one superstep are planned and resumed by task ID, not
// node name — both re-execute with their own Send arg and resume queue.
func TestSameNodeFanOutInterruptsResumeByTaskID(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	var workerRuns int32
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("worker", func(ctx context.Context, state map[string]any) (any, error) {
		atomic.AddInt32(&workerRuns, 1)
		k := state["k"].(string)
		v := Interrupt(ctx, k)
		return map[string]any{"out_" + k: v}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddConditionalEdges("start", func(_ context.Context, _ map[string]any) ([]any, error) {
		return []any{
			&types.Send{Node: "worker", Arg: map[string]any{"k": "x"}},
			&types.Send{Node: "worker", Arg: map[string]any{"k": "y"}},
		}, nil
	})
	g.AddEdge("worker", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 2 {
		t.Fatalf("expected 2 interrupts (one per fan-out task), got %+v", first.Interrupts)
	}
	if first.Interrupts[0].Value != "x" || first.Interrupts[1].Value != "y" {
		t.Fatalf("interrupt values = %+v, want one per task (x then y)", first.Interrupts)
	}

	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	if len(tup.Checkpoint.Next) != 2 || tup.Checkpoint.Next[0].ID == tup.Checkpoint.Next[1].ID {
		t.Fatalf("same-node tasks must be planned with distinct task IDs, got %+v", tup.Checkpoint.Next)
	}

	// A scalar resume with two pending interrupts is an error (Python parity):
	// resume with an interrupt-ID map instead. Both fan-out tasks interrupt
	// with the same ID ("worker-1": <node>-<counter>), so one map entry feeds
	// both.
	if _, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t1", Resume: "done"}); err == nil ||
		!strings.Contains(err.Error(), "interrupt ID") {
		t.Fatalf("scalar resume with 2 pending interrupts error = %v, want one requiring an interrupt-ID map", err)
	}

	second, err := cg.InvokeWithOptions(context.Background(), nil,
		Options{ThreadID: "t1", Resume: map[string]any{"worker-1": "done"}})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if workerRuns != 4 {
		t.Fatalf("worker must run 4 times total (2 initial + 2 resumed), got %d", workerRuns)
	}
	if second.Values["out_x"] != "done" || second.Values["out_y"] != "done" {
		t.Fatalf("values = %+v, want out_x=out_y=done (each task resumed with its own arg)", second.Values)
	}
}

// TestResumeByInterruptIDMap verifies resume values addressed by interrupt ID
// (map resume) keep working through the task-ID-keyed resume machinery.
func TestResumeByInterruptIDMap(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("ask", func(ctx context.Context, _ map[string]any) (any, error) {
		answer := Interrupt(ctx, "pick a color")
		return map[string]any{"answer": answer}, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].ID != "ask-1" {
		t.Fatalf("expected interrupt ask-1, got %+v", first.Interrupts)
	}

	second, err := cg.InvokeWithOptions(context.Background(), nil,
		Options{ThreadID: "t1", Resume: map[string]any{"ask-1": "blue"}})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if second.Values["answer"] != "blue" {
		t.Fatalf("answer = %v, want blue (map resume by interrupt ID)", second.Values["answer"])
	}
}

// TestResumeMapUnmatchedInterruptRepauses verifies that a map resume that
// addresses only some of the pending interrupts does NOT feed the unmatched
// ones a nil value: the unmatched interrupt re-fires (the run pauses again
// with the same interrupt), mirroring Python.
func TestResumeMapUnmatchedInterruptRepauses(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("askA", func(ctx context.Context, _ map[string]any) (any, error) {
		v := Interrupt(ctx, "a?")
		return map[string]any{"outA": v}, nil
	})
	g.AddNode("askB", func(ctx context.Context, _ map[string]any) (any, error) {
		v := Interrupt(ctx, "b?")
		return map[string]any{"outB": v}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "askA")
	g.AddEdge("start", "askB")
	g.AddEdge("askA", types.END)
	g.AddEdge("askB", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	first, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	ids := map[string]bool{}
	for _, intr := range first.Interrupts {
		ids[intr.ID] = true
	}
	if len(first.Interrupts) != 2 || !ids["askA-1"] || !ids["askB-1"] {
		t.Fatalf("expected interrupts askA-1 and askB-1, got %+v", first.Interrupts)
	}

	// Resume addressing only askA-1: askB must re-pause with the SAME
	// interrupt (not continue with a nil resume value).
	second, err := cg.InvokeWithOptions(ctx, nil,
		Options{ThreadID: "t1", Resume: map[string]any{"askA-1": "va"}})
	if err != nil {
		t.Fatalf("partial resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 1 || second.Interrupts[0].ID != "askB-1" || second.Interrupts[0].Value != "b?" {
		t.Fatalf("expected re-pause with interrupt askB-1, got %+v", second.Interrupts)
	}
	if _, ok := second.Values["outB"]; ok {
		t.Fatalf("outB must not be set while askB is still interrupted: %+v", second.Values)
	}

	// Resuming the remaining interrupt completes the run; the completed
	// sibling's write is replayed, not re-run.
	third, err := cg.InvokeWithOptions(ctx, nil,
		Options{ThreadID: "t1", Resume: map[string]any{"askB-1": "vb"}})
	if err != nil {
		t.Fatalf("final resume Invoke() error = %v", err)
	}
	if len(third.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after final resume, got %+v", third.Interrupts)
	}
	if third.Values["outA"] != "va" || third.Values["outB"] != "vb" {
		t.Fatalf("values = %+v, want outA=va outB=vb", third.Values)
	}
}
