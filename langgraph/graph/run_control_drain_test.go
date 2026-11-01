package graph

// Tests for cooperative draining (#9281): a caller-supplied RunControl
// plumbed through Options/Runtime that the run loop consumes at the superstep
// boundary — after the interrupt_before resolution, so breakpoints win over
// draining (Python 10f1fabb3). A drained run finishes the current superstep
// (writes applied, loop checkpoint saved with the next task set), then stops
// with a GraphDrainedError; a later nil-input resume continues from the
// boundary.

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// drainState is the total=False-style state the drain graphs run over.
type drainState = map[string]any

// drainTwoNodeGraph builds first -> second over a free-form state, recording
// which nodes ran. firstFn supplies node "first"'s body (the one requesting
// the drain); "second" returns {"second": "done"}.
func drainTwoNodeGraph(t *testing.T, firstFn func(runtime.Runtime, drainState) (any, error), opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("first", firstFn)
	g.AddNode("second", func(runtime.Runtime, drainState) (any, error) {
		return drainState{"second": "done"}, nil
	})
	g.AddEdge(types.START, "first")
	g.AddEdge("first", "second")
	g.AddEdge("second", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestRunControlDrainStopsAtSuperstepBoundary verifies the core drain
// contract: the drain-requesting superstep completes (its write is applied
// and checkpointed, next=[second]), "second" never runs, the run stops with
// a GraphDrainedError carrying the reason, and a nil-input resume continues
// from the boundary to completion (Python's
// test_run_control_request_drain_stops_future_steps +
// test_drain_with_exit_durability_persists_resume_checkpoint shape).
func TestRunControlDrainStopsAtSuperstepBoundary(t *testing.T) {
	for _, tc := range []struct {
		name       string
		durability Durability
	}{
		{"sync durability", ""},
		{"exit durability", DurabilityExit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			control := runtime.NewRunControl()
			cg := drainTwoNodeGraph(t,
				func(runtime.Runtime, drainState) (any, error) {
					control.RequestDrain("sigterm")
					return drainState{"first": "done"}, nil
				},
				WithCheckpointer(checkpoint.NewMemorySaver()),
			)

			opts := Options{ThreadID: "drain-boundary", Control: control}
			if tc.durability != "" {
				opts = opts.WithRunDurability(tc.durability)
			}
			res, err := cg.InvokeWithOptions(t.Context(), drainState{}, opts)
			drained, ok := errors.AsType[*types.GraphDrainedError](err)
			if !ok {
				t.Fatalf("InvokeWithOptions() error = %v, want a *types.GraphDrainedError", err)
			}
			if drained.Reason != "sigterm" {
				t.Errorf("drain reason = %q, want %q", drained.Reason, "sigterm")
			}
			snap, err := cg.GetState(t.Context(), checkpoint.Config{ThreadID: "drain-boundary"})
			if err != nil {
				t.Fatalf("GetState() error = %v", err)
			}
			if !reflect.DeepEqual(snap.Next, []string{"second"}) {
				t.Fatalf("checkpoint next = %v, want [second] (saved at the drain boundary)", snap.Next)
			}
			if got := snap.Values["first"]; got != "done" {
				t.Errorf("checkpoint values = %v, want the drained run's write persisted", snap.Values)
			}

			// Resume with a fresh control continues from the boundary.
			res, err = cg.InvokeWithOptions(t.Context(), nil, Options{ThreadID: "drain-boundary"})
			if err != nil {
				t.Fatalf("resume InvokeWithOptions() error = %v", err)
			}
			want := drainState{"first": "done", "second": "done"}
			if !reflect.DeepEqual(res.Values, want) {
				t.Fatalf("resumed state = %v, want %v", res.Values, want)
			}
		})
	}
}

// TestRunControlDrainViaRuntime verifies the Runtime plumbing: a node
// requests the drain through the run-scoped rt.Control (#9281 wires
// Runtime.Control to the run loop's control), with the default "shutdown"
// reason.
func TestRunControlDrainViaRuntime(t *testing.T) {
	var nodeControl *runtime.RunControl
	cg := drainTwoNodeGraph(t,
		func(rt runtime.Runtime, _ drainState) (any, error) {
			nodeControl = rt.Control
			rt.Control.RequestDrain("") // empty reason normalizes to "shutdown"
			return drainState{"first": "done"}, nil
		},
		WithCheckpointer(checkpoint.NewMemorySaver()),
	)

	_, err := cg.InvokeWithOptions(t.Context(), drainState{}, Options{ThreadID: "drain-runtime"})
	drained, ok := errors.AsType[*types.GraphDrainedError](err)
	if !ok {
		t.Fatalf("InvokeWithOptions() error = %v, want a *types.GraphDrainedError", err)
	}
	if drained.Reason != "shutdown" {
		t.Errorf("drain reason = %q, want the default %q", drained.Reason, "shutdown")
	}
	if nodeControl == nil || !nodeControl.DrainRequested() {
		t.Error("rt.Control is nil or not drained, want the run-scoped control the loop consumed")
	}
}

// TestRunControlDrainInTerminalStepFinishesNormally verifies that a drain
// requested during the LAST superstep never trips: with no next task the run
// completes normally (Python's tick checks "no tasks" before the drain).
func TestRunControlDrainInTerminalStepFinishesNormally(t *testing.T) {
	control := runtime.NewRunControl()
	g := NewStateGraph()
	g.AddNode("only", func(runtime.Runtime, drainState) (any, error) {
		control.RequestDrain("shutdown")
		return drainState{"value": "done"}, nil
	})
	g.AddEdge(types.START, "only")
	g.AddEdge("only", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(t.Context(), drainState{}, Options{Control: control})
	if err != nil {
		t.Fatalf("InvokeWithOptions() error = %v, want a normal completion", err)
	}
	if !reflect.DeepEqual(res.Values, drainState{"value": "done"}) {
		t.Fatalf("state = %v, want {value: done}", res.Values)
	}
	if !control.DrainRequested() {
		t.Error("control.DrainRequested() = false, want true")
	}
}

// TestRunControlDrainBeforeInterruptBeforeStopsAtBreakpoint verifies the
// precedence rule of 10f1fabb3: when a drain is requested during the step
// right before an interrupt_before breakpoint, the run stops AT the
// breakpoint (an interrupt pause, resumable) rather than draining — a resume
// passes the next step's breakpoints, so draining in front of one would skip
// it. The resume then runs the breakpointed node to completion.
func TestRunControlDrainBeforeInterruptBeforeStopsAtBreakpoint(t *testing.T) {
	var secondRuns []string
	control := runtime.NewRunControl()
	cg := drainTwoNodeGraph(t,
		func(rt runtime.Runtime, _ drainState) (any, error) {
			rt.Control.RequestDrain("rollout")
			return drainState{"first": "done"}, nil
		},
		WithCheckpointer(checkpoint.NewMemorySaver()),
		WithInterruptBefore("second"),
	)

	res, err := cg.InvokeWithOptions(t.Context(), drainState{}, Options{ThreadID: "drain-breakpoint", Control: control})
	if err != nil {
		t.Fatalf("InvokeWithOptions() error = %v, want the breakpoint pause, not a drain error", err)
	}
	if len(res.Interrupts) == 0 {
		t.Fatalf("interrupts = %v, want the interrupt_before pause to win over the drain", res.Interrupts)
	}
	if len(secondRuns) != 0 {
		t.Fatalf("second ran %v, want it paused at the breakpoint", secondRuns)
	}

	snap, err := cg.GetState(t.Context(), checkpoint.Config{ThreadID: "drain-breakpoint"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if !slices.Equal(snap.Next, []string{"second"}) {
		t.Fatalf("checkpoint next = %v, want [second] (paused at the breakpoint)", snap.Next)
	}

	// A resume (fresh control) runs the breakpointed node and completes.
	res, err = cg.InvokeWithOptions(t.Context(), nil, Options{ThreadID: "drain-breakpoint"})
	if err != nil {
		t.Fatalf("resume InvokeWithOptions() error = %v", err)
	}
	want := drainState{"first": "done", "second": "done"}
	if !reflect.DeepEqual(res.Values, want) {
		t.Fatalf("resumed state = %v, want %v", res.Values, want)
	}
}

// TestRunControlDrainFromSubgraph verifies a drain requested inside an
// AddSubgraph child stops the whole run (the drain bubbles like Python's
// GraphDrained) and that the parent thread resumes to completion: the child
// resumes from its own namespace's checkpoint and the parent continues after
// it (Python's test_drain_from_subgraph_can_resume_parent).
func TestRunControlDrainFromSubgraph(t *testing.T) {
	control := runtime.NewRunControl()
	var childRan, parentSecondRan []string

	child := NewStateGraph()
	child.AddNode("child_first", func(rt runtime.Runtime, _ drainState) (any, error) {
		childRan = append(childRan, "child_first")
		control.RequestDrain("sigterm")
		return drainState{"child_first": "done"}, nil
	})
	child.AddNode("child_second", func(runtime.Runtime, drainState) (any, error) {
		childRan = append(childRan, "child_second")
		return drainState{"child_second": "done"}, nil
	})
	child.AddEdge(types.START, "child_first")
	child.AddEdge("child_first", "child_second")
	child.AddEdge("child_second", types.END)
	compiledChild, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	parent := NewStateGraph()
	parent.AddSubgraph("child", compiledChild)
	parent.AddNode("parent_second", func(runtime.Runtime, drainState) (any, error) {
		parentSecondRan = append(parentSecondRan, "yes")
		return drainState{"parent_second": "done"}, nil
	})
	parent.AddEdge(types.START, "child")
	parent.AddEdge("child", "parent_second")
	parent.AddEdge("parent_second", types.END)
	cg, err := parent.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	_, err = cg.InvokeWithOptions(t.Context(), drainState{}, Options{ThreadID: "drain-subgraph", Control: control})
	drained, ok := errors.AsType[*types.GraphDrainedError](err)
	if !ok {
		t.Fatalf("InvokeWithOptions() error = %v, want the child's *types.GraphDrainedError to surface", err)
	}
	if drained.Reason != "sigterm" {
		t.Errorf("drain reason = %q, want %q", drained.Reason, "sigterm")
	}
	if !reflect.DeepEqual(childRan, []string{"child_first"}) {
		t.Fatalf("child ran %v, want [child_first] before the drain", childRan)
	}

	res, err := cg.InvokeWithOptions(t.Context(), nil, Options{ThreadID: "drain-subgraph"})
	if err != nil {
		t.Fatalf("resume InvokeWithOptions() error = %v", err)
	}
	want := drainState{"child_first": "done", "child_second": "done", "parent_second": "done"}
	if !reflect.DeepEqual(res.Values, want) {
		t.Fatalf("resumed state = %v, want %v", res.Values, want)
	}
	if !reflect.DeepEqual(childRan, []string{"child_first", "child_second"}) {
		t.Errorf("child ran %v on resume, want child_second only", childRan)
	}
	if len(parentSecondRan) != 1 {
		t.Errorf("parent_second ran %d times, want 1", len(parentSecondRan))
	}
}
