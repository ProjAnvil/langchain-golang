package graph

import (
	"context"
	"errors"
	"regexp"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// compileChild is a small helper building and compiling a one-node child
// graph whose entry node runs fn and then routes to END.
func compileChild(t *testing.T, nodeName string, fn NodeFunc) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode(nodeName, fn)
	g.AddEdge(types.START, nodeName)
	g.AddEdge(nodeName, types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}
	return cg
}

// TestSubgraphNodeSharesState verifies the base AddSubgraph contract: the
// subgraph node runs the child with the parent's state map as input and the
// child's final values merge back into the parent's state as the node's
// update, so shared keys flow in and out.
func TestSubgraphNodeSharesState(t *testing.T) {
	child := compileChild(t, "child_step", func(_ runtime.Runtime, state map[string]any) (any, error) {
		if state["value"] != 1 {
			t.Errorf("child saw value = %v, want 1 (parent state as input)", state["value"])
		}
		return map[string]any{"value": 2, "child_ran": true}, nil
	})

	g := NewStateGraph()
	g.AddSubgraph("sub", child)
	var afterSaw any
	g.AddNode("after", func(_ runtime.Runtime, state map[string]any) (any, error) {
		afterSaw = state["value"]
		return nil, nil
	})
	g.AddEdge(types.START, "sub")
	g.AddEdge("sub", "after")
	g.AddEdge("after", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(t.Context(), map[string]any{"value": 1})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["value"] != 2 {
		t.Fatalf("value = %v, want 2 (child final values merged back)", res.Values["value"])
	}
	if res.Values["child_ran"] != true {
		t.Fatalf("child_ran = %v, want true", res.Values["child_ran"])
	}
	if afterSaw != 2 {
		t.Fatalf("after node saw value = %v, want 2", afterSaw)
	}
}

// TestSubgraphParentCommandAppliesAtParent verifies D6: a node inside the
// child returning Command{Graph: ParentGraph} aborts the child's run; the
// AddSubgraph wrapper recovers the ParentCommandError and returns the command
// (Graph cleared) as the subgraph node's normal result, so the PARENT applies
// the update and routes the goto at its own level.
func TestSubgraphParentCommandAppliesAtParent(t *testing.T) {
	childNextRan := false
	child := NewStateGraph()
	child.AddNode("decide", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return &types.Command{
			Graph:  types.ParentGraph,
			Update: map[string]any{"k": "v"},
			Goto:   To("target"),
		}, nil
	})
	child.AddNode("child_next", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		childNextRan = true
		return nil, nil
	})
	child.AddEdge(types.START, "decide")
	child.AddEdge("decide", "child_next")
	child.AddEdge("child_next", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	targetRan := false
	g := NewStateGraph()
	g.AddSubgraph("sub", childCG)
	g.AddNode("target", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		targetRan = true
		return nil, nil
	})
	g.AddNode("fallback", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return nil, nil
	})
	g.AddEdge(types.START, "sub")
	g.AddEdge("sub", "fallback") // overridden by the command's Goto
	g.AddEdge("fallback", types.END)
	g.AddEdge("target", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if !targetRan {
		t.Fatal("parent did not route the bubbled command's Goto to \"target\"")
	}
	if childNextRan {
		t.Fatal("child_next ran; the parent-targeted command should have aborted the child's run")
	}
	if res.Values["k"] != "v" {
		t.Fatalf("k = %v, want \"v\" (parent applied the bubbled update)", res.Values["k"])
	}
}

// TestSubgraphGrandchildCommandAppliesAtChildLevel verifies the recursion
// half of D6: a grandchild node's Command{Graph: ParentGraph} is recovered by
// the child's AddSubgraph wrapper and applies at the CHILD level (update
// merged into child state, goto resolved against child nodes) — it must not
// reach the top graph, which has no "child_target" node.
func TestSubgraphGrandchildCommandAppliesAtChildLevel(t *testing.T) {
	grand := compileChild(t, "grand_step", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return &types.Command{
			Graph:  types.ParentGraph,
			Update: map[string]any{"level": "child"},
			Goto:   To("child_target"),
		}, nil
	})

	childTargetRan := false
	child := NewStateGraph()
	child.AddSubgraph("grand", grand)
	child.AddNode("child_target", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		childTargetRan = true
		return nil, nil
	})
	child.AddNode("child_fallback", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return nil, nil
	})
	child.AddEdge(types.START, "grand")
	child.AddEdge("grand", "child_fallback")
	child.AddEdge("child_fallback", types.END)
	child.AddEdge("child_target", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	top := NewStateGraph()
	top.AddSubgraph("child", childCG)
	top.AddEdge(types.START, "child")
	top.AddEdge("child", types.END)
	topCG, err := top.Compile()
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}
	res, err := topCG.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v (grandchild command must apply at the child level)", err)
	}
	if !childTargetRan {
		t.Fatal("child did not route the grandchild's command Goto to \"child_target\"")
	}
	if res.Values["level"] != "child" {
		t.Fatalf("level = %v, want \"child\" (merged up through both wrappers)", res.Values["level"])
	}
}

// TestSubgraphChildCommandReachesTopGraph verifies that a DIRECT node of the
// child graph returning Command{Graph: ParentGraph} bubbles one level further
// than a grandchild's: the child's run aborts, the top graph's AddSubgraph
// wrapper recovers it, and the top graph applies update+goto itself.
func TestSubgraphChildCommandReachesTopGraph(t *testing.T) {
	child := compileChild(t, "child_decide", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return &types.Command{
			Graph:  types.ParentGraph,
			Update: map[string]any{"from": "child"},
			Goto:   To("top_target"),
		}, nil
	})

	topTargetRan := false
	top := NewStateGraph()
	top.AddSubgraph("child", child)
	top.AddNode("top_target", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		topTargetRan = true
		return nil, nil
	})
	top.AddEdge(types.START, "child")
	top.AddEdge("child", types.END)
	top.AddEdge("top_target", types.END)
	topCG, err := top.Compile()
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}
	res, err := topCG.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if !topTargetRan {
		t.Fatal("top graph did not route the child-level command's Goto to \"top_target\"")
	}
	if res.Values["from"] != "child" {
		t.Fatalf("from = %v, want \"child\"", res.Values["from"])
	}
}

// TestSubgraphCheckpointsNamespaced verifies that when the parent graph has a
// checkpointer and ThreadID, child (and grandchild) runs checkpoint into the
// same thread under per-task namespaces rooted at <node>:<taskID> ("sub:<tid>"
// and "sub:<tid>/grand:<tid>" here; see taskCheckpointNS), while the parent's
// own checkpoints stay in the root namespace.
func TestSubgraphCheckpointsNamespaced(t *testing.T) {
	ctx := t.Context()

	grand := compileChild(t, "grand_step", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"grand_ran": true}, nil
	})
	child := NewStateGraph()
	child.AddSubgraph("grand", grand)
	child.AddEdge(types.START, "grand")
	child.AddEdge("grand", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	top.AddSubgraph("sub", childCG)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	topCG, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}
	res, err := topCG.InvokeWithOptions(ctx, map[string]any{"value": 1}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["grand_ran"] != true {
		t.Fatalf("grand_ran = %v, want true", res.Values["grand_ran"])
	}

	childNS := rootChildNamespaces(t, saver, "t1")
	if len(childNS) != 1 || !strings.HasPrefix(childNS[0], "sub:") {
		t.Fatalf("child namespaces = %v, want exactly one sub:<taskID> namespace", childNS)
	}
	// The grandchild namespace nests under the child's: discover it from the
	// child's own Parents record.
	childTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: childNS[0]}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(ns=%q) error = %v", childNS[0], err)
	}
	grandNS := ""
	for _, tup := range childTups {
		for ns := range tup.Metadata.Parents {
			if ns != "" {
				grandNS = ns
			}
		}
	}
	if !strings.HasPrefix(grandNS, childNS[0]+"/grand:") {
		t.Fatalf("grandchild namespace %q, want <childNS>/grand:<taskID>", grandNS)
	}

	for _, ns := range []string{"", childNS[0], grandNS} {
		tups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: ns}, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("List(ns=%q) error = %v", ns, err)
		}
		if len(tups) == 0 {
			t.Fatalf("List(ns=%q) returned no checkpoints", ns)
		}
		for _, tup := range tups {
			if tup.Config.CheckpointNS != ns {
				t.Fatalf("checkpoint %q stored under ns %q, want %q", tup.Checkpoint.ID, tup.Config.CheckpointNS, ns)
			}
		}
	}
}

// TestSubgraphWithoutParentCheckpointer verifies a subgraph still runs (with
// no checkpointing of its own) when the parent has no checkpointer.
func TestSubgraphWithoutParentCheckpointer(t *testing.T) {
	child := compileChild(t, "child_step", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"done": true}, nil
	})
	g := NewStateGraph()
	g.AddSubgraph("sub", child)
	g.AddEdge(types.START, "sub")
	g.AddEdge("sub", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["done"] != true {
		t.Fatalf("done = %v, want true", res.Values["done"])
	}
}

// TestCommandGraphUnsupportedValueErrors verifies that any non-empty
// Command.Graph other than types.ParentGraph remains an error.
func TestCommandGraphUnsupportedValueErrors(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("n", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return &types.Command{Graph: "bogus"}, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(t.Context(), nil)
	if err == nil {
		t.Fatal("Invoke() error = nil, want an error for Command.Graph \"bogus\"")
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Fatalf("error = %v, want it to name the unsupported Graph value", err)
	}
}

// TestTopLevelParentCommandDescriptiveError verifies that a
// Command{Graph: ParentGraph} surfaced from the TOP-level graph (no parent)
// reaches the caller as a descriptive error, still recognizable via
// errors.As as a *ParentCommandError carrying the command.
func TestTopLevelParentCommandDescriptiveError(t *testing.T) {
	cmd := &types.Command{
		Graph:  types.ParentGraph,
		Update: map[string]any{"k": "v"},
		Goto:   To("nowhere"),
	}
	g := NewStateGraph()
	g.AddNode("n", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return cmd, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(t.Context(), nil)
	if err == nil {
		t.Fatal("Invoke() error = nil, want a descriptive error for a top-level parent-targeted command")
	}
	if !strings.Contains(err.Error(), "parent") {
		t.Fatalf("error = %v, want a message describing the missing parent graph", err)
	}
	pce, ok := errors.AsType[*ParentCommandError](err)
	if !ok {
		t.Fatalf("error = %v, want it to unwrap to *ParentCommandError", err)
	}
	if pce.Command != cmd {
		t.Fatalf("ParentCommandError.Command = %v, want the original command %v", pce.Command, cmd)
	}
}

// TestSubgraphParentsPinTimeTravel verifies the Metadata.Parents wiring
// between a checkpointing parent and its subgraph under per-task namespacing:
// child checkpoints name the parent's position when the child ran
// (Parents[""]), parent checkpoints saved after the subgraph ran name the
// child's per-task namespace and position, and time-traveling the parent to
// such a checkpoint (Options.CheckpointID with fresh input) re-enters the
// child in a FRESH per-task namespace (the new task's ID differs from the
// recorded one — Python parity: fresh-input re-runs mint new task IDs, so the
// recorded pin does not apply), with the child state flowing through the
// parent checkpoint it resumes from.
func TestSubgraphParentsPinTimeTravel(t *testing.T) {
	ctx := t.Context()

	child := compileChild(t, "child_step", func(_ runtime.Runtime, state map[string]any) (any, error) {
		n, _ := state["child_n"].(int)
		return map[string]any{"child_n": n + 1}, nil
	})

	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("pre", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddSubgraph("sub", child)
	g.AddNode("post", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "pre")
	g.AddEdge("pre", "sub")
	g.AddEdge("sub", "post")
	g.AddEdge("post", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	list := func(ns string) []checkpoint.Tuple {
		t.Helper()
		tups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: ns}, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("List(ns=%q) error = %v", ns, err)
		}
		return tups
	}

	// Turn 1: a full run. Parent checkpoints (newest first): after post,
	// after sub, after pre, input; child checkpoints under its per-task ns:
	// loop, input.
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"value": 1}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	parentT1 := list("")
	if len(parentT1) != 4 {
		t.Fatalf("parent checkpoints after turn 1 = %d, want 4", len(parentT1))
	}
	childNSs := rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 1 || !strings.HasPrefix(childNSs[0], "sub:") {
		t.Fatalf("child namespaces after turn 1 = %v, want one sub:<taskID>", childNSs)
	}
	childNS1 := childNSs[0]
	childT1 := list(childNS1)
	if len(childT1) != 2 {
		t.Fatalf("child checkpoints after turn 1 = %d, want 2", len(childT1))
	}

	// Child checkpoints name the parent's position when the child ran: the
	// checkpoint saved after "pre" (parent step 0), the position the parent
	// held while the subgraph node executed.
	afterPreID := parentT1[2].Config.CheckpointID
	for _, tup := range childT1 {
		if got := tup.Metadata.Parents[""]; got != afterPreID {
			t.Fatalf("child checkpoint %q Parents[\"\"] = %q, want parent's pre-sub checkpoint %q",
				tup.Config.CheckpointID, got, afterPreID)
		}
	}
	// Parent checkpoints saved after the subgraph ran name the child's
	// position (keyed by the per-task namespace); the earlier ones have no
	// Parents.
	childPosT1 := childT1[0].Config.CheckpointID
	for i, want := range []string{childPosT1, childPosT1, "", ""} {
		got := parentT1[i].Metadata.Parents[childNS1]
		if got != want {
			t.Fatalf("parent checkpoint %q (step %d) Parents[%q] = %q, want %q",
				parentT1[i].Config.CheckpointID, parentT1[i].Metadata.Step, childNS1, got, want)
		}
	}

	// Turn 2: a new turn runs the subgraph task under a NEW per-task
	// namespace (fresh task ID), starting fresh from the parent state; the
	// child's own state flows in through the parent (child_n 1 -> 2).
	res, err := cg.InvokeWithOptions(ctx, map[string]any{"value": 2}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}
	if res.Values["child_n"] != 2 {
		t.Fatalf("turn 2 child_n = %v, want 2", res.Values["child_n"])
	}
	childNSs = rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 2 {
		t.Fatalf("child namespaces after turn 2 = %v, want 2 (one per turn's subgraph task)", childNSs)
	}

	// Time travel: pin the parent to its end-of-turn-1 checkpoint (whose
	// Metadata.Parents names the turn-1 child position) with fresh input. The
	// re-entered subgraph task mints a NEW task ID, so the recorded pin does
	// not apply (Python parity); the child runs in a fresh namespace starting
	// from the parent state recorded at the pinned checkpoint (child_n=1,
	// advanced to 2).
	endOfTurn1 := parentT1[0].Config.CheckpointID
	res, err = cg.InvokeWithOptions(ctx, map[string]any{"value": 3}, Options{ThreadID: "t1", CheckpointID: endOfTurn1})
	if err != nil {
		t.Fatalf("time-travel Invoke() error = %v", err)
	}
	if res.Values["child_n"] != 2 {
		t.Fatalf("time-travel child_n = %v, want 2 (child state restored via the pinned parent checkpoint)", res.Values["child_n"])
	}
	childNSs = rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 3 {
		t.Fatalf("child namespaces after time travel = %v, want 3 (a fresh namespace per re-entry)", childNSs)
	}
	// The fresh namespace's input checkpoint has NO parent within the child
	// namespace (fresh start), unlike the legacy shared-namespace fork.
	freshTups := list(childNSs[len(childNSs)-1])
	if len(freshTups) != 2 {
		t.Fatalf("fresh child namespace holds %d checkpoints, want 2", len(freshTups))
	}
	if in := freshTups[len(freshTups)-1]; in.Metadata.Source != "input" || in.ParentConfig != nil {
		t.Fatalf("fresh namespace input checkpoint = %+v, want a parentless fresh input", in)
	}
}

// TestSubgraphRepeatedExecutionDistinctNamespaces locks in the per-task
// namespacing for a subgraph node executed MULTIPLE times within one run
// (the loop case; see StateGraph.AddSubgraph): every execution is a distinct
// task with its own namespace and its own fresh input+loop history, so the
// executions never fork off one another's checkpoint history. The child's
// state still flows between executions through the PARENT state (child_n
// advances), because the child's final values merge back as the node's
// update. This replaces the old shared-namespace behavior where the second
// execution forked a new turn off the first's child checkpoint.
func TestSubgraphRepeatedExecutionDistinctNamespaces(t *testing.T) {
	ctx := t.Context()

	var childRuns int32
	child := compileChild(t, "child_step", func(_ runtime.Runtime, state map[string]any) (any, error) {
		childRuns++
		n, _ := state["child_n"].(int)
		return map[string]any{"child_n": n + 1}, nil
	})

	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("pre", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddSubgraph("sub", child)
	// again loops back to sub until visits reaches 2, so the subgraph node
	// executes twice per run.
	g.AddNode("again", func(_ runtime.Runtime, state map[string]any) (any, error) {
		v, _ := state["visits"].(int)
		return map[string]any{"visits": v + 1}, nil
	})
	g.AddEdge(types.START, "pre")
	g.AddEdge("pre", "sub")
	g.AddEdge("sub", "again")
	g.AddConditionalEdges("again", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		if state["visits"].(int) < 2 {
			return To("sub"), nil
		}
		return To(types.END), nil
	})
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	list := func(ns string) []checkpoint.Tuple {
		t.Helper()
		tups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: ns}, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("List(ns=%q) error = %v", ns, err)
		}
		return tups
	}

	// Turn 1: sub executes twice; child_n advances 1 -> 2 through the parent
	// state between the two executions.
	res, err := cg.InvokeWithOptions(ctx, map[string]any{"value": 1}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	if res.Values["child_n"] != 2 {
		t.Fatalf("turn 1 child_n = %v, want 2", res.Values["child_n"])
	}
	// Two distinct per-task namespaces, each with its own fresh 2-checkpoint
	// history (input + loop).
	childNSs := rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 2 {
		t.Fatalf("child namespaces after turn 1 = %v, want 2 (one per execution)", childNSs)
	}
	for _, ns := range childNSs {
		if !strings.HasPrefix(ns, "sub:") {
			t.Fatalf("child namespace %q does not carry the sub:<taskID> per-task format", ns)
		}
		tups := list(ns)
		if len(tups) != 2 {
			t.Fatalf("ns %q holds %d checkpoints, want 2 (a fresh input + loop history)", ns, len(tups))
		}
		// A fresh history: the input checkpoint has no parent (it does not
		// fork off the other execution's result).
		if in := tups[len(tups)-1]; in.Metadata.Source != "input" || in.ParentConfig != nil {
			t.Fatalf("ns %q input checkpoint = %+v, want a parentless fresh input", ns, in)
		}
	}
	// The end-of-run parent checkpoint names both executions' positions.
	endOfTurn1 := list("")[0]
	for _, ns := range childNSs {
		if endOfTurn1.Metadata.Parents[ns] == "" {
			t.Fatalf("end-of-turn-1 Parents does not name child namespace %q", ns)
		}
	}

	// Turn 2 (pinned to end-of-turn-1, loop reset): both executions again run
	// under fresh per-task namespaces — the recorded turn-1 positions name
	// turn-1 task IDs, which this turn's tasks do not reuse, so no pin
	// applies (Python parity; the legacy-format pin is covered by
	// TestSubgraphResumeLegacyNamespacePin).
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"visits": 0},
		Options{ThreadID: "t1", CheckpointID: endOfTurn1.Config.CheckpointID}); err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}
	if childRuns != 4 {
		t.Fatalf("child entry ran %d times total, want 4 (two executions per run)", childRuns)
	}
	childNSs = rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 4 {
		t.Fatalf("child namespaces after turn 2 = %v, want 4 (a fresh namespace per execution)", childNSs)
	}
}

// TestSubgraphInterruptPausesParent verifies the pause half of subgraph
// interrupt propagation (T15): a child graph that interrupts no longer errors
// the parent run — the wrapper propagates the child's interrupts upward, so
// the PARENT pauses with them, each interrupt carrying the interrupting task's
// full nested NS ("<sub>:<taskID>/<ask-node>:<taskID>"). The parent's pause
// checkpoint must (a) name the child's pause position in Metadata.Parents and
// (b) persist a ReservedInterrupt copy of the child's interrupts keyed by the
// subgraph task's planned ID (verbatim: same Value/ID/NS, and no
// ReservedResume prefix — the wrapper never consumes the parent-level queue).
// A completed sibling of the same superstep persists its writes as usual, the
// shape resume replays.
func TestSubgraphInterruptPausesParent(t *testing.T) {
	ctx := t.Context()

	var askRuns, preRuns, sibRuns atomic.Int32
	var resumedWith any
	child := NewStateGraph()
	child.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		askRuns.Add(1)
		v := Interrupt(rt, "pause-inside-child")
		resumedWith = v
		return map[string]any{"answer": v}, nil
	})
	child.AddEdge(types.START, "ask")
	child.AddEdge("ask", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	top.AddNode("pre", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		preRuns.Add(1)
		return nil, nil
	})
	top.AddSubgraph("sub", childCG)
	top.AddNode("sib", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		sibRuns.Add(1)
		return map[string]any{"sib": true}, nil
	})
	top.AddEdge(types.START, "pre")
	top.AddEdge("pre", "sub")
	top.AddEdge("pre", "sib")
	top.AddEdge("sub", types.END)
	top.AddEdge("sib", types.END)
	cg, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{"value": 1}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result (subgraph interrupts propagate to the parent)", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want exactly one propagated child interrupt", res.Interrupts)
	}
	intr := res.Interrupts[0]
	if intr.Value != "pause-inside-child" {
		t.Fatalf("interrupt Value = %v, want %q", intr.Value, "pause-inside-child")
	}
	if intr.ID != "ask-1" {
		t.Fatalf("interrupt ID = %q, want %q (verbatim child identity)", intr.ID, "ask-1")
	}
	// NS is the interrupting task's full nested namespace:
	// "<sub>:<subTaskID>/ask:<askTaskID>".
	childNS, askSeg, ok := strings.Cut(intr.NS, "/")
	if !ok || !strings.HasPrefix(childNS, "sub:") || !perTaskNSPattern.MatchString(childNS) {
		t.Fatalf("interrupt NS = %q, want <sub>:<taskID> child namespace prefix", intr.NS)
	}
	if want := "ask:"; !strings.HasPrefix(askSeg, want) || !perTaskNSPattern.MatchString(askSeg) {
		t.Fatalf("interrupt NS = %q, want the ask task segment <ask>:<taskID>", intr.NS)
	}
	if askRuns.Load() != 1 || preRuns.Load() != 1 || sibRuns.Load() != 1 {
		t.Fatalf("runs at pause: ask=%d pre=%d sib=%d, want 1 each", askRuns.Load(), preRuns.Load(), sibRuns.Load())
	}

	// The child namespace holds its own pause checkpoint (loop source), and the
	// parent's pause checkpoint names it in Metadata.Parents.
	childTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: childNS}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(child ns) error = %v", err)
	}
	if len(childTups) == 0 {
		t.Fatalf("child namespace %q holds no checkpoints", childNS)
	}
	childPause := childTups[0]
	if childPause.Metadata.Source != "loop" {
		t.Fatalf("child ns latest checkpoint source = %q, want the child's pause checkpoint (loop)", childPause.Metadata.Source)
	}
	pauseTup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil || pauseTup == nil {
		t.Fatalf("GetTuple(root latest) = (%v, %v), want the parent pause checkpoint", pauseTup, err)
	}
	if got := pauseTup.Metadata.Parents[childNS]; got != childPause.Config.CheckpointID {
		t.Fatalf("parent pause Parents[%q] = %q, want the child pause checkpoint %q", childNS, got, childPause.Config.CheckpointID)
	}

	// Pending writes: the subgraph task carries a verbatim ReservedInterrupt
	// copy (no ReservedResume — the wrapper consumes nothing at the parent
	// level), and the completed sibling persists its state write under its own
	// planned ID (the replay shape).
	var subTaskID, sibTaskID string
	for _, pt := range pauseTup.Checkpoint.Next {
		switch pt.Node {
		case "sub":
			subTaskID = pt.ID
		case "sib":
			sibTaskID = pt.ID
		}
	}
	if subTaskID == "" || sibTaskID == "" {
		t.Fatalf("pause checkpoint Next = %+v, want both sub and sib planned", pauseTup.Checkpoint.Next)
	}
	sawCopy, sawSib := false, false
	for _, w := range pauseTup.PendingWrites {
		switch w.TaskID {
		case subTaskID:
			if w.Channel == checkpoint.ReservedResume {
				t.Fatalf("subgraph task persisted a ReservedResume write (%+v); the wrapper consumes nothing at the parent level", w)
			}
			if w.Channel != checkpoint.ReservedInterrupt {
				t.Fatalf("subgraph task persisted a %q channel write, want ReservedInterrupt only", w.Channel)
			}
			cp, ok := w.Value.(types.Interrupt)
			if !ok || cp != intr {
				t.Fatalf("subgraph task interrupt copy = %+v, want the verbatim interrupt %+v", w.Value, intr)
			}
			sawCopy = true
		case sibTaskID:
			if w.Channel != "sib" || w.Value != true {
				t.Fatalf("completed sibling persisted (%q, %v), want the sib channel write", w.Channel, w.Value)
			}
			sawSib = true
		}
	}
	if !sawCopy {
		t.Fatalf("pause checkpoint has no ReservedInterrupt copy for the subgraph task: %+v", pauseTup.PendingWrites)
	}
	if !sawSib {
		t.Fatalf("pause checkpoint has no completed-sibling write for sib: %+v", pauseTup.PendingWrites)
	}

	// Resume half (§6.1): a scalar Resume is forwarded into the child, which
	// pins to its pause checkpoint (via the parent pause checkpoint's
	// Metadata.Parents) and continues. The completed sibling replays its
	// persisted write instead of re-running; only the interrupted child node
	// re-executes; the parent merges the child's final values and completes.
	res2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "42"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v, want completion", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	if resumedWith != "42" {
		t.Fatalf("child Interrupt() returned %v, want the forwarded resume value %q", resumedWith, "42")
	}
	if got := res2.Values["answer"]; got != "42" {
		t.Fatalf("resumed Values[answer] = %v, want 42 (child values merged into the parent)", got)
	}
	if got := res2.Values["sib"]; got != true {
		t.Fatalf("resumed Values[sib] = %v, want true (replayed sibling write)", got)
	}
	if askRuns.Load() != 2 {
		t.Fatalf("ask runs after resume = %d, want 2 (re-executed once to consume the resume value)", askRuns.Load())
	}
	if preRuns.Load() != 1 || sibRuns.Load() != 1 {
		t.Fatalf("runs after resume: pre=%d sib=%d, want 1 each (completed work replays, never re-runs)", preRuns.Load(), sibRuns.Load())
	}
}

// interruptingSubgraphFixture builds the §6.3/§6.4 shape: one subgraph node
// fanned into by two Sends, whose child interrupts per task, so a pause
// aggregates two interrupts with distinct per-task child namespaces. The
// child echoes the resume value back keyed by the Send's "who" arg, so the
// test can observe which interrupt each resume value reached.
func interruptingSubgraphFixture(t *testing.T) (*CompiledGraph, func(res any) (Result, error)) {
	t.Helper()
	child := NewStateGraph()
	child.AddNode("ask", func(rt runtime.Runtime, state map[string]any) (any, error) {
		who, _ := state["who"].(string)
		v := Interrupt(rt, "q-for-"+who)
		return map[string]any{"answered_" + who: v}, nil
	})
	child.AddEdge(types.START, "ask")
	child.AddEdge("ask", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	top := NewStateGraph()
	top.AddSubgraph("sub", childCG)
	top.SetConditionalEntryPoint(func(runtime.Runtime, map[string]any) ([]any, error) {
		return []any{
			&types.Send{Node: "sub", Arg: map[string]any{"who": "a"}},
			&types.Send{Node: "sub", Arg: map[string]any{"who": "b"}},
		}, nil
	})
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg, func(resume any) (Result, error) {
		return cg.InvokeWithOptions(t.Context(), nil, Options{ThreadID: "t1", Resume: resume})
	}
}

// answeredSet collects the fixture's echoed resume values into a set keyed by
// who, tolerating the unordered mapping of interrupt NS to Send arg.
func answeredSet(t *testing.T, res Result) map[string]any {
	t.Helper()
	out := map[string]any{}
	for _, who := range []string{"a", "b"} {
		v, ok := res.Values["answered_"+who]
		if !ok {
			t.Fatalf("Values = %v, want answered_%s (both interrupts answered)", res.Values, who)
		}
		out[who] = v
	}
	return out
}

// TestSubgraphNestedInterruptResumeScalar (§6.2): an interrupt raised two
// subgraph levels down carries the full nested NS
// "<sub>:<t1>/<mid>:<t2>/<ask>:<t3>", and ONE scalar resume penetrates both
// levels — each wrapper forwards the payload one level down until it reaches
// the interrupting task, with no map needed at any level.
func TestSubgraphNestedInterruptResumeScalar(t *testing.T) {
	ctx := t.Context()

	ask := NewStateGraph()
	ask.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "deep-question")
		return map[string]any{"deep_answer": v}, nil
	})
	ask.AddEdge(types.START, "ask")
	ask.AddEdge("ask", types.END)
	askCG, err := ask.Compile()
	if err != nil {
		t.Fatalf("grandchild Compile() error = %v", err)
	}

	mid := NewStateGraph()
	mid.AddSubgraph("inner", askCG)
	mid.AddEdge(types.START, "inner")
	mid.AddEdge("inner", types.END)
	midCG, err := mid.Compile()
	if err != nil {
		t.Fatalf("middle Compile() error = %v", err)
	}

	top := NewStateGraph()
	top.AddSubgraph("sub", midCG)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want the single grandchild interrupt", res.Interrupts)
	}
	intr := res.Interrupts[0]
	segs := strings.Split(intr.NS, "/")
	if len(segs) != 3 {
		t.Fatalf("interrupt NS = %q, want three nested task segments", intr.NS)
	}
	for i, prefix := range []string{"sub:", "inner:", "ask:"} {
		if !strings.HasPrefix(segs[i], prefix) || !perTaskNSPattern.MatchString(segs[i]) {
			t.Fatalf("interrupt NS segment %d = %q, want <%s><taskID>", i, segs[i], prefix)
		}
	}

	res2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "deep-answer"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v, want completion through both levels", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	if got := res2.Values["deep_answer"]; got != "deep-answer" {
		t.Fatalf("Values[deep_answer] = %v, want the grandchild's merged answer", got)
	}
}

// TestSubgraphParallelInterruptsMapResume (§6.3): two Sends fan into one
// subgraph node and both child tasks interrupt: the parent pauses with both
// interrupts aggregated; a scalar resume is rejected (>1 pending); a map
// keyed by NS restores both; answering only one re-pauses with exactly the
// other remaining.
func TestSubgraphParallelInterruptsMapResume(t *testing.T) {
	cg, resume := interruptingSubgraphFixture(t)
	ctx := t.Context()

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 2 {
		t.Fatalf("Interrupts = %+v, want both child tasks' interrupts aggregated", res.Interrupts)
	}
	nsA, nsB := res.Interrupts[0].NS, res.Interrupts[1].NS
	if nsA == nsB {
		t.Fatalf("interrupt NSes %q and %q must be distinct (per-task child namespaces)", nsA, nsB)
	}

	if _, err := resume("x"); err == nil || !strings.Contains(err.Error(), "requires a map") {
		t.Fatalf("scalar resume error = %v, want the >1-pending map requirement", err)
	}

	// Partial answer: only the first interrupt's NS is addressed; the second
	// child re-pauses, so the parent pauses again with exactly that one.
	res2, err := resume(map[string]any{nsA: "va"})
	if err != nil {
		t.Fatalf("partial map resume error = %v, want a re-pause", err)
	}
	if len(res2.Interrupts) != 1 || res2.Interrupts[0].NS != nsB {
		t.Fatalf("re-pause Interrupts = %+v, want exactly the unanswered %q", res2.Interrupts, nsB)
	}

	res3, err := resume(map[string]any{nsB: "vb"})
	if err != nil {
		t.Fatalf("final resume error = %v, want completion", err)
	}
	if len(res3.Interrupts) != 0 {
		t.Fatalf("final Interrupts = %+v, want none", res3.Interrupts)
	}
	got := answeredSet(t, res3)
	if !(got["a"] == "va" && got["b"] == "vb" || got["a"] == "vb" && got["b"] == "va") {
		t.Fatalf("answered values = %v, want the two resume values dispatched to their own tasks", got)
	}
}

// TestSubgraphResumeGraphAddressing (§6.4): Options.Graph strictly addresses a
// resume. Without Resume it errors; types.ParentGraph errors; an NS matching
// no pending interrupt errors listing the available NSes; a map carrying
// unknown keys re-pauses silently; and a scalar paired with a hitting Graph
// is allowed even with two interrupts pending (the scalar relaxation),
// leaving the unaddressed one paused.
func TestSubgraphResumeGraphAddressing(t *testing.T) {
	cg, resume := interruptingSubgraphFixture(t)
	ctx := t.Context()

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 2 {
		t.Fatalf("Interrupts = %+v, want two pending", res.Interrupts)
	}
	nsA, nsB := res.Interrupts[0].NS, res.Interrupts[1].NS

	if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Graph: nsA}); err == nil || !strings.Contains(err.Error(), "requires Options.Resume") {
		t.Fatalf("Graph without Resume error = %v, want the requirement error", err)
	}
	if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "x", Graph: types.ParentGraph}); err == nil || !strings.Contains(err.Error(), types.ParentGraph) {
		t.Fatalf("Graph=ParentGraph error = %v, want the ParentGraph rejection", err)
	}
	_, err = cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "x", Graph: "nope"})
	if err == nil || !strings.Contains(err.Error(), "nope") || !strings.Contains(err.Error(), nsA) || !strings.Contains(err.Error(), nsB) {
		t.Fatalf("Graph=nope error = %v, want it to name the bogus namespace and list %q and %q", err, nsA, nsB)
	}

	// A map whose keys match nothing is not an error: both interrupts re-fire.
	resMap, err := resume(map[string]any{"bogus": "v"})
	if err != nil {
		t.Fatalf("unknown-key map resume error = %v, want a silent re-pause", err)
	}
	if len(resMap.Interrupts) != 2 {
		t.Fatalf("unknown-key map re-pause Interrupts = %+v, want both re-fired", resMap.Interrupts)
	}

	// Scalar relaxation: Graph nsA + scalar answers exactly the hit
	// interrupt despite the second one pending.
	resScalar, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "va", Graph: nsA})
	if err != nil {
		t.Fatalf("Graph-addressed scalar resume error = %v, want the relaxation to allow it", err)
	}
	if len(resScalar.Interrupts) != 1 || resScalar.Interrupts[0].NS != nsB {
		t.Fatalf("relaxed scalar re-pause Interrupts = %+v, want exactly the unaddressed %q", resScalar.Interrupts, nsB)
	}

	res3, err := resume(map[string]any{nsB: "vb"})
	if err != nil {
		t.Fatalf("final resume error = %v, want completion", err)
	}
	got := answeredSet(t, res3)
	if !(got["a"] == "va" && got["b"] == "vb" || got["a"] == "vb" && got["b"] == "va") {
		t.Fatalf("answered values = %v, want each value at its own task", got)
	}
}

// TestSubgraphChildMultiInterruptMapDispatch (§6.5): a Send fan-out INSIDE
// the child interrupts twice in one superstep. The parent's pause carries
// both copies under the single subgraph task's planned ID, and a map resume
// forwarded verbatim lets the CHILD's planResume dispatch each value to its
// own inner task by NS.
func TestSubgraphChildMultiInterruptMapDispatch(t *testing.T) {
	ctx := t.Context()

	child := NewStateGraph()
	child.AddNode("ask", func(rt runtime.Runtime, state map[string]any) (any, error) {
		who, _ := state["who"].(string)
		v := Interrupt(rt, "q-for-"+who)
		return map[string]any{"answered_" + who: v}, nil
	})
	child.SetConditionalEntryPoint(func(runtime.Runtime, map[string]any) ([]any, error) {
		return []any{
			&types.Send{Node: "ask", Arg: map[string]any{"who": "a"}},
			&types.Send{Node: "ask", Arg: map[string]any{"who": "b"}},
		}, nil
	})
	child.AddEdge("ask", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	top.AddSubgraph("sub", childCG)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 2 {
		t.Fatalf("Interrupts = %+v, want both inner tasks' interrupts propagated", res.Interrupts)
	}
	nsA, nsB := res.Interrupts[0].NS, res.Interrupts[1].NS
	if nsA == nsB {
		t.Fatalf("interrupt NSes %q and %q must be distinct (per-task segments inside one child)", nsA, nsB)
	}
	// Both NSes nest under the single subgraph task's namespace.
	for _, ns := range []string{nsA, nsB} {
		childNS, askSeg, ok := strings.Cut(ns, "/")
		if !ok || !strings.HasPrefix(childNS, "sub:") || !perTaskNSPattern.MatchString(childNS) {
			t.Fatalf("interrupt NS = %q, want the shared <sub>:<taskID> child namespace prefix", ns)
		}
		if !strings.HasPrefix(askSeg, "ask:") || !perTaskNSPattern.MatchString(askSeg) {
			t.Fatalf("interrupt NS = %q, want a distinct <ask>:<taskID> segment per inner task", ns)
		}
	}
	// The parent's pause checkpoint carries BOTH copies under the one
	// subgraph task's planned ID.
	pauseTup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil || pauseTup == nil {
		t.Fatalf("GetTuple(root latest) = (%v, %v), want the parent pause checkpoint", pauseTup, err)
	}
	var subTaskID string
	for _, pt := range pauseTup.Checkpoint.Next {
		if pt.Node == "sub" {
			subTaskID = pt.ID
		}
	}
	copies := 0
	for _, w := range pauseTup.PendingWrites {
		if w.TaskID == subTaskID && w.Channel == checkpoint.ReservedInterrupt {
			// Both copies travel as ONE packed write (the saver keeps a
			// single reserved __interrupt__ slot per task, Python's
			// WRITES_IDX_MAP -3).
			copies += len(interruptsFromWrite(w.Value))
		}
	}
	if copies != 2 {
		t.Fatalf("subgraph task carries %d interrupt copies, want 2 (one per inner task)", copies)
	}

	res2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: map[string]any{nsA: "va", nsB: "vb"}})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v, want completion", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	got := answeredSet(t, res2)
	if !(got["a"] == "va" && got["b"] == "vb" || got["a"] == "vb" && got["b"] == "va") {
		t.Fatalf("answered values = %v, want the map dispatched per inner task by NS", got)
	}
}

// TestSubgraphBoundaryInterruptBeforePropagates (§6.6, before half): a
// child's internal interrupt_before boundary pauses the child, the wrapper
// propagates the boundary interrupt to the parent, and a nil-input resume
// lets the child run its gated node (the child's OWN skip applies) while the
// parent's same-named interrupt_before registration is NOT skipped by the
// foreign boundary interrupt (interruptOwnedBy): the parent pauses at its
// own gate on the following superstep, and a second resume finishes.
func TestSubgraphBoundaryInterruptBeforePropagates(t *testing.T) {
	ctx := t.Context()

	child := NewStateGraph()
	child.AddNode("gate", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"child_gate": true}, nil
	})
	child.AddEdge(types.START, "gate")
	child.AddEdge("gate", types.END)
	childCG, err := child.Compile(WithInterruptBefore("gate"))
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	top := NewStateGraph()
	top.AddSubgraph("sub", childCG)
	top.AddNode("gate", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"parent_gate": true}, nil
	})
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", "gate")
	top.AddEdge("gate", types.END)
	cg, err := top.Compile(WithCheckpointer(checkpoint.NewMemorySaver()), WithInterruptBefore("gate"))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want the propagated child boundary interrupt", res.Interrupts)
	}
	intr := res.Interrupts[0]
	if intr.ID != interruptBeforeID+"gate" {
		t.Fatalf("interrupt ID = %q, want the child's boundary identity", intr.ID)
	}
	if ns := intr.NS; !strings.HasSuffix(ns, "/gate") || strings.Count(ns, "/") != 1 {
		t.Fatalf("interrupt NS = %q, want the child's node-level namespace <sub>:<taskID>/gate", ns)
	}

	// First resume: the child skips ITS gate check (the boundary interrupt is
	// owned by the child's namespace) and completes; the parent's gate
	// registration is NOT skipped (the interrupt is foreign to the root), so
	// the parent pauses at its own gate.
	res2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v, want a paused Result", err)
	}
	if len(res2.Interrupts) != 1 {
		t.Fatalf("resume Interrupts = %+v, want the parent's own gate boundary interrupt", res2.Interrupts)
	}
	if res2.Interrupts[0].ID != interruptBeforeID+"gate" || res2.Interrupts[0].NS != "gate" {
		t.Fatalf("resume Interrupts = %+v, want the parent-level boundary interrupt (NS \"gate\")", res2.Interrupts)
	}
	if res2.Values["child_gate"] != true {
		t.Fatalf("resumed Values[child_gate] = %v, want true (child completed on its resume)", res2.Values["child_gate"])
	}

	// Second resume: now the parent's skip applies and the graph finishes.
	res3, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("second resume error = %v, want completion", err)
	}
	if len(res3.Interrupts) != 0 {
		t.Fatalf("second resume Interrupts = %+v, want none", res3.Interrupts)
	}
	if res3.Values["parent_gate"] != true || res3.Values["child_gate"] != true {
		t.Fatalf("final Values = %v, want both gates run", res3.Values)
	}
}

// TestSubgraphParentInterruptBeforeSubgraphNodeResume covers the
// forceResume degrade path (§3.3): the parent pauses via interrupt_before on
// the subgraph node itself, so on resume the sub task re-dispatches under a
// resuming parent while its child namespace holds NO checkpoint yet (the
// child never ran). The forwarded forceResume must degrade to a fresh-input
// child run instead of erroring "no checkpoint found".
func TestSubgraphParentInterruptBeforeSubgraphNodeResume(t *testing.T) {
	ctx := t.Context()

	child := NewStateGraph()
	child.AddNode("work", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"worked": true}, nil
	})
	child.AddEdge(types.START, "work")
	child.AddEdge("work", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	top := NewStateGraph()
	top.AddSubgraph("sub", childCG)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(checkpoint.NewMemorySaver()), WithInterruptBefore("sub"))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].ID != interruptBeforeID+"sub" {
		t.Fatalf("Interrupts = %+v, want the parent's interrupt_before pause on the subgraph node", res.Interrupts)
	}

	res2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("resume error = %v, want the forceResume to degrade into a fresh child run", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	if res2.Values["worked"] != true {
		t.Fatalf("resumed Values = %v, want the child to have run", res2.Values)
	}
}

// TestSubgraphBoundaryInterruptAfterResumes (§6.6, after half): a child's
// interrupt_after boundary pauses through the wrapper; the nil-input resume
// finds nothing left to run inside the child (its gated node already
// committed into the pause checkpoint), so the child completes and the parent
// finishes with the child's values merged.
func TestSubgraphBoundaryInterruptAfterResumes(t *testing.T) {
	ctx := t.Context()

	child := NewStateGraph()
	child.AddNode("gate", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"child_gate": true}, nil
	})
	child.AddEdge(types.START, "gate")
	child.AddEdge("gate", types.END)
	childCG, err := child.Compile(WithInterruptAfter("gate"))
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	top := NewStateGraph()
	top.AddSubgraph("sub", childCG)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].ID != interruptAfterID+"gate" {
		t.Fatalf("Interrupts = %+v, want the propagated child interrupt_after boundary", res.Interrupts)
	}

	res2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("resume error = %v, want completion", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	if res2.Values["child_gate"] != true {
		t.Fatalf("resumed Values = %v, want the child's committed gate value", res2.Values)
	}
}

// TestSubgraphChildRunErrorWraps verifies that a plain child run failure is
// wrapped with the subgraph node's name.
func TestSubgraphChildRunErrorWraps(t *testing.T) {
	want := errors.New("child boom")
	child := NewStateGraph()
	child.AddNode("inner", func(runtime.Runtime, map[string]any) (any, error) { return nil, want })
	child.AddEdge(types.START, "inner")
	child.AddEdge("inner", types.END)
	compiledChild, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	g := NewStateGraph()
	g.AddSubgraph("sub", compiledChild)
	g.AddEdge(types.START, "sub")
	g.AddEdge("sub", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(t.Context(), map[string]any{})
	if !errors.Is(err, want) || !strings.Contains(err.Error(), `subgraph "sub"`) {
		t.Fatalf("Invoke() error = %v, want it to wrap %v naming subgraph %q", err, want, "sub")
	}
}

// perTaskNSPattern matches one checkpoint-namespace segment of the per-task
// subgraph format "<node>:<16-hex task id>" (see taskCheckpointNS).
var perTaskNSPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*:[0-9a-f]{16}$`)

// rootChildNamespaces returns the child checkpoint namespaces recorded in the
// Metadata.Parents of the thread's ROOT-namespace checkpoints (union across
// the whole history, so namespaces from earlier turns/re-entries are
// included), skipping the empty parent-namespace entry.
func rootChildNamespaces(t *testing.T, saver *checkpoint.MemorySaver, threadID string) []string {
	t.Helper()
	tups, err := saver.List(t.Context(), checkpoint.Config{ThreadID: threadID}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(root) error = %v", err)
	}
	if len(tups) == 0 {
		t.Fatal("no root checkpoints")
	}
	union := map[string]bool{}
	for _, tup := range tups {
		for ns := range tup.Metadata.Parents {
			if ns != "" {
				union[ns] = true
			}
		}
	}
	out := make([]string, 0, len(union))
	for ns := range union {
		out = append(out, ns)
	}
	slices.Sort(out)
	return out
}

// TestSubgraphCheckpointsPerTaskNamespace verifies the Python-parity namespace
// format: a subgraph task checkpoints under <parentNS>/<node>:<taskID> (one
// namespace per subgraph TASK, not per node), and a grandchild under
// <childNS>/<grandnode>:<taskID>. The root namespace stays clean.
func TestSubgraphCheckpointsPerTaskNamespace(t *testing.T) {
	ctx := t.Context()

	grand := compileChild(t, "grand_step", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"grand_ran": true}, nil
	})
	child := NewStateGraph()
	child.AddSubgraph("grand", grand)
	child.AddEdge(types.START, "grand")
	child.AddEdge("grand", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	top.AddSubgraph("sub", childCG)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	topCG, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}
	res, err := topCG.InvokeWithOptions(ctx, map[string]any{"value": 1}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["grand_ran"] != true {
		t.Fatalf("grand_ran = %v, want true", res.Values["grand_ran"])
	}

	// The child namespace is "sub:<taskID>" and holds its own input + loop
	// checkpoints.
	childNSs := rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 1 {
		t.Fatalf("root Parents recorded %d child namespaces (%v), want exactly 1", len(childNSs), childNSs)
	}
	childNS := childNSs[0]
	seg, ok := strings.CutPrefix(childNS, "sub:")
	if !ok || !perTaskNSPattern.MatchString(childNS) {
		t.Fatalf("child namespace %q does not match per-task format sub:<taskID>", childNS)
	}
	if len(seg) != 16 {
		t.Fatalf("task ID suffix %q is %d chars, want 16 hex", seg, len(seg))
	}
	childTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: childNS}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(ns=%q) error = %v", childNS, err)
	}
	if len(childTups) != 2 {
		t.Fatalf("List(ns=%q) returned %d checkpoints, want 2 (input + loop)", childNS, len(childTups))
	}

	// The grandchild namespace nests under the child's per-task namespace:
	// "<childNS>/grand:<taskID>".
	grandNSs := make([]string, 0, 1)
	for ns := range childTups[0].Metadata.Parents {
		if ns != "" {
			grandNSs = append(grandNSs, ns)
		}
	}
	if len(grandNSs) != 1 {
		t.Fatalf("child Parents recorded %d grandchild namespaces (%v), want 1", len(grandNSs), grandNSs)
	}
	grandNS := grandNSs[0]
	if !strings.HasPrefix(grandNS, childNS+"/grand:") || !perTaskNSPattern.MatchString(strings.TrimPrefix(grandNS, childNS+"/")) {
		t.Fatalf("grandchild namespace %q does not match <childNS>/grand:<taskID>", grandNS)
	}
	grandTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: grandNS}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(ns=%q) error = %v", grandNS, err)
	}
	if len(grandTups) != 2 {
		t.Fatalf("List(ns=%q) returned %d checkpoints, want 2 (input + loop)", grandNS, len(grandTups))
	}

	// The old node-only namespace holds nothing.
	if tups, _ := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: "sub"}, checkpoint.ListOptions{}); len(tups) != 0 {
		t.Fatalf("legacy node-only ns \"sub\" holds %d checkpoints, want 0", len(tups))
	}
	// The root namespace holds only parent checkpoints.
	rootTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(root) error = %v", err)
	}
	for _, tup := range rootTups {
		if tup.Config.CheckpointNS != "" {
			t.Fatalf("checkpoint %q stored under ns %q, want root", tup.Checkpoint.ID, tup.Config.CheckpointNS)
		}
	}
}

// TestSubgraphParallelSendDistinctNamespaces reproduces the namespace-collision
// bug of node-only namespacing: two Sends fan into the SAME subgraph node in
// one superstep, which must produce two independent per-task namespaces (Python
// gives every subgraph task its own ns). Under node-only namespacing both
// tasks share one ns, interleaving their checkpoint histories.
func TestSubgraphParallelSendDistinctNamespaces(t *testing.T) {
	ctx := t.Context()

	// collectReducer appends each update to a []any so two Send tasks can
	// both write the same key in one superstep. The first write seeds a
	// BinaryOperator channel as the raw scalar (no op applied), so the
	// reducer must tolerate a non-slice existing value.
	collect := func(existing, update any) (any, error) {
		base, _ := existing.([]any)
		if base == nil && existing != nil {
			base = []any{existing}
		}
		return append(base, update), nil
	}

	// The child records the Send argument it received in its own state.
	child := compileChild(t, "child_step", func(_ runtime.Runtime, state map[string]any) (any, error) {
		item, _ := state["item"].(string)
		return map[string]any{"items": item}, nil
	})

	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	// Both "item" (the Send args echoed back in each child's final values)
	// and "items" (each child's output) receive one write per task in the
	// fan-in superstep, so both need collecting reducers.
	top.AddReducer("item", collect)
	top.AddReducer("items", collect)
	top.AddSubgraph("fan", child)
	top.SetConditionalEntryPoint(func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return []any{
			&types.Send{Node: "fan", Arg: map[string]any{"item": "a"}},
			&types.Send{Node: "fan", Arg: map[string]any{"item": "b"}},
		}, nil
	})
	top.AddEdge("fan", types.END)
	cg, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	items, _ := res.Values["items"].([]any)
	if len(items) != 2 {
		t.Fatalf("items = %v, want both Send args merged", res.Values["items"])
	}
	got := map[string]bool{}
	for _, it := range items {
		s, _ := it.(string)
		got[s] = true
	}
	if !got["a"] || !got["b"] {
		t.Fatalf("items = %v, want a and b", items)
	}

	// Two distinct per-task namespaces, each with its own complete history
	// (input + loop checkpoint). The parent's final checkpoint names both.
	childNSs := rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 2 {
		t.Fatalf("root Parents recorded %d child namespaces (%v), want 2 (one per Send task)", len(childNSs), childNSs)
	}
	seen := map[string]bool{}
	perNSItem := map[string]string{}
	for _, ns := range childNSs {
		if !strings.HasPrefix(ns, "fan:") || !perTaskNSPattern.MatchString(ns) {
			t.Fatalf("child namespace %q does not carry the fan:<taskID> per-task format", ns)
		}
		if seen[ns] {
			t.Fatalf("duplicate child namespace %q", ns)
		}
		seen[ns] = true
		tups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: ns}, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("List(ns=%q) error = %v", ns, err)
		}
		if len(tups) != 2 {
			t.Fatalf("ns %q holds %d checkpoints, want 2 (its own input + loop history)", ns, len(tups))
		}
		// Every checkpoint in this namespace carries the SAME item value (the
		// Send arg this task ran with), proving the two runs did not share
		// channel state through one namespace.
		for _, tup := range tups {
			item, _ := tup.Checkpoint.ChannelValues["item"].(string)
			if item == "" {
				t.Fatalf("ns %q checkpoint %q missing the item channel value", ns, tup.Checkpoint.ID)
			}
			if prev := perNSItem[ns]; prev != "" && prev != item {
				t.Fatalf("ns %q mixes item values %q and %q (shared channel state)", ns, prev, item)
			}
			perNSItem[ns] = item
		}
	}
	// The two namespaces ran with the two different Send args.
	gotArgs := map[string]bool{}
	for _, item := range perNSItem {
		gotArgs[item] = true
	}
	if !gotArgs["a"] || !gotArgs["b"] || len(gotArgs) != 2 {
		t.Fatalf("per-namespace item values = %v, want both a and b across the two namespaces", perNSItem)
	}
	// The node-only namespace must not exist as a shared bucket.
	if tups, _ := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: "fan"}, checkpoint.ListOptions{}); len(tups) != 0 {
		t.Fatalf("legacy node-only ns \"fan\" holds %d checkpoints, want 0 (no shared namespace)", len(tups))
	}
}

// TestSubgraphResumeLegacyNamespacePin verifies read-side compatibility with
// checkpoints written before per-task namespacing: Metadata.Parents keys of
// the node-only form ("<parentNS>/<name>") still pin the re-entered subgraph
// to the recorded child position (loaded from the legacy namespace), while
// the resumed run WRITES its new checkpoints under the new per-task
// namespace. Legacy data is not migrated.
func TestSubgraphResumeLegacyNamespacePin(t *testing.T) {
	ctx := t.Context()

	child := compileChild(t, "child_step", func(_ runtime.Runtime, state map[string]any) (any, error) {
		n, _ := state["child_n"].(int)
		return map[string]any{"child_n": n + 1}, nil
	})
	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	top.AddSubgraph("sub", child)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	// Hand-craft a legacy thread exactly as pre-per-task code wrote it: the
	// child history lives in the node-only namespace "sub", and the parent's
	// end checkpoint records Parents["sub"] = the child's final position.
	// The zero-prefixed IDs sort below every NewID value (13-digit UnixMilli
	// prefix), so the memory saver's ID-ordered List still ranks the NEW
	// run's checkpoints newest.
	legacyChildInput := checkpoint.Checkpoint{
		V:               1,
		ID:              "0000000000000-000000-0000000000000001",
		ChannelValues:   map[string]any{"child_n": 1},
		ChannelVersions: map[string]int64{"child_n": 1},
	}
	if _, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: "sub"}, legacyChildInput, checkpoint.Metadata{Source: "input", Step: -1}, nil); err != nil {
		t.Fatal(err)
	}
	legacyChildLoop := checkpoint.Checkpoint{
		V:               1,
		ID:              "0000000000000-000000-0000000000000002",
		ChannelValues:   map[string]any{"child_n": 2},
		ChannelVersions: map[string]int64{"child_n": 2},
	}
	if _, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: "sub", CheckpointID: "0000000000000-000000-0000000000000001"}, legacyChildLoop, checkpoint.Metadata{Source: "loop", Step: 0}, nil); err != nil {
		t.Fatal(err)
	}
	legacyParent := checkpoint.Checkpoint{
		V:               1,
		ID:              "0000000000000-000000-0000000000000003",
		ChannelValues:   map[string]any{"child_n": 2},
		ChannelVersions: map[string]int64{"child_n": 2},
	}
	if _, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t1"}, legacyParent,
		checkpoint.Metadata{Source: "loop", Step: 1, Parents: map[string]string{"sub": "0000000000000-000000-0000000000000002"}}, nil); err != nil {
		t.Fatal(err)
	}

	// Re-enter the parent pinned to its legacy end checkpoint with fresh
	// input. The subgraph task gets a NEW per-task namespace, but the legacy
	// Parents key must still pin the child to the recorded legacy position
	// (child state child_n=2), so the child advances 2 -> 3.
	res, err := cg.InvokeWithOptions(ctx, map[string]any{"child_n": 2},
		Options{ThreadID: "t1", CheckpointID: "0000000000000-000000-0000000000000003"})
	if err != nil {
		t.Fatalf("time-travel Invoke() error = %v", err)
	}
	if got, _ := res.Values["child_n"].(int); got != 3 {
		t.Fatalf("child_n = %v, want 3 (resumed from the pinned legacy child position 2)", res.Values["child_n"])
	}

	// The new child history lands in a per-task namespace, forked off the
	// pinned legacy checkpoint...
	childNSs := rootChildNamespaces(t, saver, "t1")
	var newNS string
	for _, ns := range childNSs {
		if strings.HasPrefix(ns, "sub:") {
			newNS = ns
		}
	}
	if newNS == "" {
		t.Fatalf("no per-task child namespace recorded after resume (Parents = %v)", childNSs)
	}
	newTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: newNS}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(ns=%q) error = %v", newNS, err)
	}
	if len(newTups) != 2 {
		t.Fatalf("new ns %q holds %d checkpoints, want 2 (input + loop)", newNS, len(newTups))
	}
	forked := false
	for _, tup := range newTups {
		if tup.Metadata.Source == "input" && tup.ParentConfig != nil && tup.ParentConfig.CheckpointID == "0000000000000-000000-0000000000000002" {
			forked = true
		}
	}
	if !forked {
		t.Fatalf("new-ns input checkpoint did not fork off the pinned legacy checkpoint legacy-c-loop")
	}
	// ...and the new run's final child state advanced from the legacy position.
	if got, _ := newTups[0].Checkpoint.ChannelValues["child_n"].(int); got != 3 {
		t.Fatalf("new ns latest child_n = %v, want 3", newTups[0].Checkpoint.ChannelValues["child_n"])
	}

	// The legacy namespace is untouched (no migration, no new writes).
	legacyTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: "sub"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(legacy ns) error = %v", err)
	}
	if len(legacyTups) != 2 {
		t.Fatalf("legacy ns holds %d checkpoints after resume, want 2 (untouched)", len(legacyTups))
	}
}

// failSecondNSTupleSaver fails the second GetTuple call that carries a
// non-empty checkpoint namespace: the first is the child run's own load, the
// second is the wrapper's post-run position lookup.
type failSecondNSTupleSaver struct {
	checkpoint.Saver
	calls atomic.Int64
}

func (s *failSecondNSTupleSaver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	if cfg.CheckpointNS != "" && s.calls.Add(1) == 2 {
		return nil, errSaverBoom
	}
	return s.Saver.GetTuple(ctx, cfg)
}

// TestSubgraphChildCheckpointLookupError verifies that a failure loading the
// child's final checkpoint position (for Metadata.Parents bookkeeping)
// surfaces as a subgraph error.
func TestSubgraphChildCheckpointLookupError(t *testing.T) {
	child := NewStateGraph()
	child.AddNode("inner", func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"inner": true}, nil
	})
	child.AddEdge(types.START, "inner")
	child.AddEdge("inner", types.END)
	compiledChild, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	saver := &failSecondNSTupleSaver{Saver: checkpoint.NewMemorySaver()}
	g := NewStateGraph()
	g.AddSubgraph("sub", compiledChild)
	g.AddEdge(types.START, "sub")
	g.AddEdge("sub", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t"})
	if !errors.Is(err, errSaverBoom) || !strings.Contains(err.Error(), `subgraph "sub"`) {
		t.Fatalf("InvokeWithOptions() error = %v, want it to wrap %v naming subgraph %q", err, errSaverBoom, "sub")
	}
}
