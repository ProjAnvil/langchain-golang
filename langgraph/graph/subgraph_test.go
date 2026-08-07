package graph

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
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
	child := compileChild(t, "child_step", func(_ context.Context, state map[string]any) (any, error) {
		if state["value"] != 1 {
			t.Errorf("child saw value = %v, want 1 (parent state as input)", state["value"])
		}
		return map[string]any{"value": 2, "child_ran": true}, nil
	})

	g := NewStateGraph()
	g.AddSubgraph("sub", child)
	var afterSaw any
	g.AddNode("after", func(_ context.Context, state map[string]any) (any, error) {
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
	res, err := cg.Invoke(context.Background(), map[string]any{"value": 1})
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
	child.AddNode("decide", func(_ context.Context, _ map[string]any) (any, error) {
		return &types.Command{
			Graph:  types.ParentGraph,
			Update: map[string]any{"k": "v"},
			Goto:   To("target"),
		}, nil
	})
	child.AddNode("child_next", func(_ context.Context, _ map[string]any) (any, error) {
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
	g.AddNode("target", func(_ context.Context, _ map[string]any) (any, error) {
		targetRan = true
		return nil, nil
	})
	g.AddNode("fallback", func(_ context.Context, _ map[string]any) (any, error) {
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
	res, err := cg.Invoke(context.Background(), nil)
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
	grand := compileChild(t, "grand_step", func(_ context.Context, _ map[string]any) (any, error) {
		return &types.Command{
			Graph:  types.ParentGraph,
			Update: map[string]any{"level": "child"},
			Goto:   To("child_target"),
		}, nil
	})

	childTargetRan := false
	child := NewStateGraph()
	child.AddSubgraph("grand", grand)
	child.AddNode("child_target", func(_ context.Context, _ map[string]any) (any, error) {
		childTargetRan = true
		return nil, nil
	})
	child.AddNode("child_fallback", func(_ context.Context, _ map[string]any) (any, error) {
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
	res, err := topCG.Invoke(context.Background(), nil)
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
	child := compileChild(t, "child_decide", func(_ context.Context, _ map[string]any) (any, error) {
		return &types.Command{
			Graph:  types.ParentGraph,
			Update: map[string]any{"from": "child"},
			Goto:   To("top_target"),
		}, nil
	})

	topTargetRan := false
	top := NewStateGraph()
	top.AddSubgraph("child", child)
	top.AddNode("top_target", func(_ context.Context, _ map[string]any) (any, error) {
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
	res, err := topCG.Invoke(context.Background(), nil)
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
// same thread under CheckpointNS = <parentNS>/<name> ("sub", "sub/grand"
// here), while the parent's own checkpoints stay in the root namespace.
func TestSubgraphCheckpointsNamespaced(t *testing.T) {
	ctx := context.Background()

	grand := compileChild(t, "grand_step", func(_ context.Context, _ map[string]any) (any, error) {
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

	for _, ns := range []string{"", "sub", "sub/grand"} {
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
	child := compileChild(t, "child_step", func(_ context.Context, _ map[string]any) (any, error) {
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
	res, err := cg.Invoke(context.Background(), nil)
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
	g.AddNode("n", func(_ context.Context, _ map[string]any) (any, error) {
		return &types.Command{Graph: "bogus"}, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(context.Background(), nil)
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
	g.AddNode("n", func(_ context.Context, _ map[string]any) (any, error) {
		return cmd, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(context.Background(), nil)
	if err == nil {
		t.Fatal("Invoke() error = nil, want a descriptive error for a top-level parent-targeted command")
	}
	if !strings.Contains(err.Error(), "parent") {
		t.Fatalf("error = %v, want a message describing the missing parent graph", err)
	}
	var pce *ParentCommandError
	if !errors.As(err, &pce) {
		t.Fatalf("error = %v, want it to unwrap to *ParentCommandError", err)
	}
	if pce.Command != cmd {
		t.Fatalf("ParentCommandError.Command = %v, want the original command %v", pce.Command, cmd)
	}
}

// TestSubgraphParentsPinTimeTravel verifies the Metadata.Parents wiring
// between a checkpointing parent and its subgraph: child checkpoints name the
// parent's position when the child ran (Parents[""]), parent checkpoints saved
// after the subgraph ran name the child's position (Parents["sub"]), and
// time-traveling the parent to such a checkpoint (Options.CheckpointID with
// fresh input) re-enters the child pinned to that recorded child checkpoint
// instead of the namespace's latest.
func TestSubgraphParentsPinTimeTravel(t *testing.T) {
	ctx := context.Background()

	child := compileChild(t, "child_step", func(_ context.Context, state map[string]any) (any, error) {
		n, _ := state["child_n"].(int)
		return map[string]any{"child_n": n + 1}, nil
	})

	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("pre", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddSubgraph("sub", child)
	g.AddNode("post", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
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
	// after sub, after pre, input; child checkpoints: loop, input.
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"value": 1}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	parentT1 := list("")
	childT1 := list("sub")
	if len(parentT1) != 4 {
		t.Fatalf("parent checkpoints after turn 1 = %d, want 4", len(parentT1))
	}
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
	// position; the earlier ones have no Parents.
	childPosT1 := childT1[0].Config.CheckpointID
	for i, want := range []string{childPosT1, childPosT1, "", ""} {
		got := parentT1[i].Metadata.Parents["sub"]
		if got != want {
			t.Fatalf("parent checkpoint %q (step %d) Parents[\"sub\"] = %q, want %q",
				parentT1[i].Config.CheckpointID, parentT1[i].Metadata.Step, got, want)
		}
	}

	// Turn 2: a new turn forks the child off its turn-1 position, advancing
	// child_n to 2; the child namespace's latest is now past the position
	// recorded at the end of turn 1.
	res, err := cg.InvokeWithOptions(ctx, map[string]any{"value": 2}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}
	if res.Values["child_n"] != 2 {
		t.Fatalf("turn 2 child_n = %v, want 2", res.Values["child_n"])
	}
	childT2 := list("sub")
	if len(childT2) != 4 {
		t.Fatalf("child checkpoints after turn 2 = %d, want 4", len(childT2))
	}
	latestChild := childT2[0].Config.CheckpointID
	if latestChild == childPosT1 {
		t.Fatal("child latest checkpoint did not advance in turn 2")
	}

	// Time travel: pin the parent to its end-of-turn-1 checkpoint (whose
	// Metadata.Parents names the child's turn-1 position) with fresh input,
	// re-entering the subgraph. The child must resume from the recorded
	// checkpoint, NOT its namespace's latest: child_n replays 1 -> 2 and the
	// new child checkpoints fork off the recorded position.
	endOfTurn1 := parentT1[0].Config.CheckpointID
	res, err = cg.InvokeWithOptions(ctx, map[string]any{"value": 3}, Options{ThreadID: "t1", CheckpointID: endOfTurn1})
	if err != nil {
		t.Fatalf("time-travel Invoke() error = %v", err)
	}
	if res.Values["child_n"] != 2 {
		t.Fatalf("time-travel child_n = %v, want 2 (re-entered child resumed from the recorded checkpoint, not latest)", res.Values["child_n"])
	}
	childTT := list("sub")
	if len(childTT) != 6 {
		t.Fatalf("child checkpoints after time travel = %d, want 6", len(childTT))
	}
	forkBase := childTT[1].ParentConfig // the forked turn's input checkpoint
	if forkBase == nil || forkBase.CheckpointID != childPosT1 {
		t.Fatalf("re-entered child forked off %v, want recorded checkpoint %q (not latest %q)",
			forkBase, childPosT1, latestChild)
	}
}
