package graph

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// snapshotLinearGraph builds the linear n1 -> n2 -> n3 graph used by the
// state-inspection tests: each node writes its own key and bumps its call
// counter, so a run produces four checkpoints (input + 3 supersteps).
func snapshotLinearGraph(t *testing.T, saver checkpoint.Saver, calls map[string]int) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	for _, name := range []string{"n1", "n2", "n3"} {
		node := name
		g.AddNode(node, func(_ context.Context, _ map[string]any) (any, error) {
			calls[node]++
			return map[string]any{"k" + node[1:]: "v" + node[1:]}, nil
		})
	}
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", "n3")
	g.AddEdge("n3", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestGetState(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := snapshotLinearGraph(t, saver, map[string]int{})
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	tuples, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil || len(tuples) != 4 {
		t.Fatalf("expected 4 checkpoints, got %d (err=%v)", len(tuples), err)
	}

	// Latest snapshot: the completed run's final loop checkpoint.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	for _, k := range []string{"k0", "k1", "k2", "k3"} {
		if _, ok := snap.Values[k]; !ok {
			t.Fatalf("GetState().Values missing %q: %+v", k, snap.Values)
		}
	}
	if len(snap.Next) != 0 {
		t.Fatalf("GetState().Next = %+v, want empty for a completed run", snap.Next)
	}
	if snap.Metadata.Step != 2 || snap.Metadata.Source != "loop" {
		t.Fatalf("GetState().Metadata = %+v, want step 2 source loop", snap.Metadata)
	}
	if snap.CreatedAt.IsZero() {
		t.Fatal("GetState().CreatedAt must be set")
	}
	if snap.Config.ThreadID != "t1" || snap.Config.CheckpointID != tuples[0].Checkpoint.ID {
		t.Fatalf("GetState().Config = %+v, want thread t1 checkpoint %q", snap.Config, tuples[0].Checkpoint.ID)
	}
	if snap.ParentConfig == nil || snap.ParentConfig.CheckpointID != tuples[1].Checkpoint.ID {
		t.Fatalf("GetState().ParentConfig = %+v, want checkpoint %q", snap.ParentConfig, tuples[1].Checkpoint.ID)
	}
	if len(snap.Interrupts) != 0 {
		t.Fatalf("GetState().Interrupts = %+v, want none", snap.Interrupts)
	}

	// Pinned to the step-0 checkpoint: state as of n1, n2 planned next, and
	// the input checkpoint as parent (D3).
	pinned, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: tuples[2].Checkpoint.ID})
	if err != nil {
		t.Fatalf("GetState() pinned error = %v", err)
	}
	if _, ok := pinned.Values["k1"]; !ok {
		t.Fatalf("pinned Values missing k1: %+v", pinned.Values)
	}
	if _, ok := pinned.Values["k2"]; ok {
		t.Fatalf("pinned Values must not contain k2 yet: %+v", pinned.Values)
	}
	if len(pinned.Next) != 1 || pinned.Next[0] != "n2" {
		t.Fatalf("pinned Next = %+v, want [n2]", pinned.Next)
	}
	if pinned.Metadata.Step != 0 {
		t.Fatalf("pinned Metadata.Step = %d, want 0", pinned.Metadata.Step)
	}
	if pinned.ParentConfig == nil || pinned.ParentConfig.CheckpointID != tuples[3].Checkpoint.ID {
		t.Fatalf("pinned ParentConfig = %+v, want checkpoint %q", pinned.ParentConfig, tuples[3].Checkpoint.ID)
	}

	if _, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "no-such-thread"}); err == nil {
		t.Fatal("GetState() for an unknown thread must error")
	}
}

func TestGetStateHistory(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := snapshotLinearGraph(t, saver, map[string]int{})
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	snaps, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	if len(snaps) != 4 {
		t.Fatalf("GetStateHistory() returned %d snapshots, want 4", len(snaps))
	}
	wantSteps := []int{2, 1, 0, -1}
	for i, snap := range snaps {
		if snap.Metadata.Step != wantSteps[i] {
			t.Fatalf("snapshot %d: Step = %d, want %d (newest-first)", i, snap.Metadata.Step, wantSteps[i])
		}
		if snap.Config.CheckpointID == "" {
			t.Fatalf("snapshot %d: Config.CheckpointID must be populated", i)
		}
		if i > 0 && snaps[i-1].Config.CheckpointID <= snap.Config.CheckpointID {
			t.Fatalf("snapshots not newest-first by ID: %q then %q", snaps[i-1].Config.CheckpointID, snap.Config.CheckpointID)
		}
	}

	limited, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{Limit: 2})
	if err != nil {
		t.Fatalf("GetStateHistory() with Limit error = %v", err)
	}
	if len(limited) != 2 || limited[0].Metadata.Step != 2 {
		t.Fatalf("GetStateHistory(Limit:2) = %d snapshots, first step %d; want 2 starting at step 2",
			len(limited), limited[0].Metadata.Step)
	}
}

// TestGetStateInterrupts verifies that a paused run's snapshot surfaces the
// pending interrupts recorded as ReservedInterrupt pending writes.
func TestGetStateInterrupts(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"k1": "v1"}, nil
	})
	g.AddNode("n2", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"k2": "v2"}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("n2"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	result, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(result.Interrupts) != 1 {
		t.Fatalf("expected a pause, got %+v", result.Interrupts)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if len(snap.Interrupts) != 1 || snap.Interrupts[0].ID != interruptBeforeID+"n2" {
		t.Fatalf("GetState().Interrupts = %+v, want the interrupt-before-n2 interrupt", snap.Interrupts)
	}
	if len(snap.Next) != 1 || snap.Next[0] != "n2" {
		t.Fatalf("GetState().Next = %+v, want [n2]", snap.Next)
	}
	if _, ok := snap.Values["k1"]; !ok {
		t.Fatalf("GetState().Values missing n1's write: %+v", snap.Values)
	}
}

func TestUpdateState(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	calls := map[string]int{}
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		calls["a"]++
		return nil, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		calls["b"]++
		return map[string]any{"via": "b"}, nil
	})
	g.AddNode("c", func(_ context.Context, _ map[string]any) (any, error) {
		calls["c"]++
		return map[string]any{"via": "c"}, nil
	})
	g.AddEdge(types.START, "a")
	// a's router reads the "route" key, so re-resolving a's successors after
	// an update must reflect the updated state.
	g.AddConditionalEdges("a", func(_ context.Context, state map[string]any) ([]any, error) {
		if state["route"] == "c" {
			return To("c"), nil
		}
		return To("b"), nil
	})
	g.AddEdge("b", types.END)
	g.AddEdge("c", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"route": "b"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if calls["a"] != 1 || calls["b"] != 1 || calls["c"] != 0 {
		t.Fatalf("calls = %+v, want a=1 b=1 c=0", calls)
	}

	tuples, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	var step0 checkpoint.Tuple
	for _, tup := range tuples {
		if tup.Metadata.Step == 0 {
			step0 = tup
		}
	}
	if step0.Checkpoint.ID == "" {
		t.Fatal("no step-0 checkpoint found")
	}

	newCfg, err := cg.UpdateState(ctx,
		checkpoint.Config{ThreadID: "t1", CheckpointID: step0.Checkpoint.ID},
		map[string]any{"route": "c"}, "a")
	if err != nil {
		t.Fatalf("UpdateState() error = %v", err)
	}
	if newCfg.CheckpointID == "" || newCfg.CheckpointID == step0.Checkpoint.ID {
		t.Fatalf("UpdateState() returned Config %+v, want a fresh checkpoint ID", newCfg)
	}

	snap, err := cg.GetState(ctx, newCfg)
	if err != nil {
		t.Fatalf("GetState() of update checkpoint error = %v", err)
	}
	if snap.Metadata.Source != "update" || snap.Metadata.Step != 1 {
		t.Fatalf("update checkpoint Metadata = %+v, want source update step 1 (S6: base step 0 + 1)", snap.Metadata)
	}
	if snap.Values["route"] != "c" {
		t.Fatalf("update checkpoint Values[route] = %v, want c", snap.Values["route"])
	}
	if len(snap.Next) != 1 || snap.Next[0] != "c" {
		t.Fatalf("update checkpoint Next = %+v, want [c] (a's re-resolved successor)", snap.Next)
	}
	if snap.ParentConfig == nil || snap.ParentConfig.CheckpointID != step0.Checkpoint.ID {
		t.Fatalf("update checkpoint ParentConfig = %+v, want %q", snap.ParentConfig, step0.Checkpoint.ID)
	}

	// The update checkpoint is the thread's latest, so a nil-input invoke
	// resumes from it and follows the re-resolved branch.
	result, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("resume after UpdateState error = %v", err)
	}
	if result.Values["via"] != "c" {
		t.Fatalf("resume Values[via] = %v, want c", result.Values["via"])
	}
	if calls["a"] != 1 || calls["b"] != 1 || calls["c"] != 1 {
		t.Fatalf("calls after resume = %+v, want a=1 b=1 c=1", calls)
	}
}

func TestUpdateStateAsNodeErrors(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	ctx := context.Background()

	g := NewStateGraph()
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	cfg := checkpoint.Config{ThreadID: "t1"}

	if _, err := cg.UpdateState(ctx, cfg, map[string]any{"x": 1}, "bogus"); err == nil ||
		!strings.Contains(err.Error(), "bogus") {
		t.Fatalf("UpdateState() with unknown asNode error = %v, want one naming the node", err)
	}
	if _, err := cg.UpdateState(ctx, cfg, map[string]any{"x": 1}, ""); err == nil {
		t.Fatal("UpdateState() with empty asNode on a multi-node graph must error")
	}

	// A single-node graph unambiguously attributes the update to its one node.
	single := NewStateGraph()
	single.AddNode("only", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	single.AddEdge(types.START, "only")
	single.AddEdge("only", types.END)
	scg, err := single.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := scg.InvokeWithOptions(ctx, map[string]any{"x": "start"}, Options{ThreadID: "t2"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	newCfg, err := scg.UpdateState(ctx, checkpoint.Config{ThreadID: "t2"}, map[string]any{"x": "updated"}, "")
	if err != nil {
		t.Fatalf("UpdateState() with inferred asNode error = %v", err)
	}
	snap, err := scg.GetState(ctx, newCfg)
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if snap.Values["x"] != "updated" {
		t.Fatalf("Values[x] = %v, want updated", snap.Values["x"])
	}
	if len(snap.Next) != 0 {
		t.Fatalf("Next = %+v, want empty (only -> END)", snap.Next)
	}
}

// TestTimeTravelCheckpointID verifies that Options.CheckpointID pins the
// run's starting checkpoint: a nil-input invoke resumes from the pinned
// historical checkpoint, and the new checkpoints fork off it (D3).
func TestTimeTravelCheckpointID(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	calls := map[string]int{}
	cg := snapshotLinearGraph(t, saver, calls)
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if calls["n1"] != 1 || calls["n2"] != 1 || calls["n3"] != 1 {
		t.Fatalf("calls = %+v, want each node run once", calls)
	}

	tuples, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil || len(tuples) != 4 {
		t.Fatalf("expected 4 checkpoints, got %d (err=%v)", len(tuples), err)
	}
	pinned := tuples[2] // step 0: state as of n1, n2 planned next
	if pinned.Metadata.Step != 0 {
		t.Fatalf("tuples[2].Metadata.Step = %d, want 0", pinned.Metadata.Step)
	}

	result, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", CheckpointID: pinned.Checkpoint.ID})
	if err != nil {
		t.Fatalf("time-travel Invoke() error = %v", err)
	}
	for _, k := range []string{"k0", "k1", "k2", "k3"} {
		if _, ok := result.Values[k]; !ok {
			t.Fatalf("time-travel result missing %q: %+v", k, result.Values)
		}
	}
	if calls["n1"] != 1 || calls["n2"] != 2 || calls["n3"] != 2 {
		t.Fatalf("calls after time travel = %+v, want n1=1 n2=2 n3=2 (resume from pinned, not a re-run)", calls)
	}

	// Fork links (D3): walking the new branch's ParentConfig chain must reach
	// the pinned checkpoint, and the forked checkpoints must be fresh IDs.
	latest, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if latest.Config.CheckpointID == tuples[0].Checkpoint.ID {
		t.Fatal("time travel must produce new checkpoint IDs, not overwrite the original branch")
	}
	if latest.ParentConfig == nil {
		t.Fatal("forked final checkpoint has no ParentConfig")
	}
	mid, err := cg.GetState(ctx, *latest.ParentConfig)
	if err != nil {
		t.Fatalf("GetState() of fork parent error = %v", err)
	}
	if mid.ParentConfig == nil || mid.ParentConfig.CheckpointID != pinned.Checkpoint.ID {
		t.Fatalf("fork branch ParentConfig chain = %+v -> %+v, want it to reach pinned %q",
			latest.Config.CheckpointID, mid.ParentConfig, pinned.Checkpoint.ID)
	}

	if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", CheckpointID: "does-not-exist"}); err == nil {
		t.Fatal("pinning a nonexistent checkpoint must error")
	}
}

// TestNewTurnFromPinnedCheckpoint verifies D2 with Options.CheckpointID: a
// non-empty input against a pinned historical checkpoint starts a NEW turn on
// top of the pinned state (the entry node re-runs) instead of resuming, and
// per S6 the turn's input checkpoint saves with Step = the pinned checkpoint's
// step (only a thread's FIRST input checkpoint is -1), forking off the pinned
// checkpoint (D3).
func TestNewTurnFromPinnedCheckpoint(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	calls := map[string]int{}
	cg := snapshotLinearGraph(t, saver, calls)
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	tuples, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil || len(tuples) != 4 {
		t.Fatalf("expected 4 checkpoints, got %d (err=%v)", len(tuples), err)
	}
	pinned := tuples[2] // step 0: state as of n1, n2 planned next
	if pinned.Metadata.Step != 0 {
		t.Fatalf("tuples[2].Metadata.Step = %d, want 0", pinned.Metadata.Step)
	}

	// New turn from the pinned checkpoint: the input applies on top of the
	// pinned state (k1 present, k2/k3 not yet) and execution restarts from
	// the entry point.
	result, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "w0"},
		Options{ThreadID: "t1", CheckpointID: pinned.Checkpoint.ID})
	if err != nil {
		t.Fatalf("new-turn Invoke() error = %v", err)
	}
	if result.Values["k0"] != "w0" {
		t.Fatalf("k0 = %v, want w0 (turn-2 input applied on top of the pinned state)", result.Values["k0"])
	}
	for _, k := range []string{"k1", "k2", "k3"} {
		if _, ok := result.Values[k]; !ok {
			t.Fatalf("new-turn result missing %q: %+v", k, result.Values)
		}
	}
	if calls["n1"] != 2 || calls["n2"] != 2 || calls["n3"] != 2 {
		t.Fatalf("calls = %+v, want each node run twice (a new turn re-runs from the entry point)", calls)
	}

	// S6: the new turn's input checkpoint continues the step counter from the
	// pinned (restored) checkpoint — Step 0, not -1 — and forks off it.
	tuples2, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	var forkedInput *checkpoint.Tuple
	for i := range tuples2 {
		tup := &tuples2[i]
		if tup.Metadata.Source == "input" && tup.ParentConfig != nil &&
			tup.ParentConfig.CheckpointID == pinned.Checkpoint.ID {
			forkedInput = tup
		}
	}
	if forkedInput == nil {
		t.Fatal("no input checkpoint forks off the pinned checkpoint")
	}
	if forkedInput.Metadata.Step != pinned.Metadata.Step {
		t.Fatalf("new-turn input checkpoint Step = %d, want %d (the pinned checkpoint's step, S6)",
			forkedInput.Metadata.Step, pinned.Metadata.Step)
	}
}

// TestStateAPIsRequireCheckpointer verifies GetState, GetStateHistory and
// UpdateState all fail clearly without a checkpointer.
func TestStateAPIsRequireCheckpointer(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	cfg := checkpoint.Config{ThreadID: "t1"}

	if _, err := cg.GetState(ctx, cfg); err == nil || !strings.Contains(err.Error(), "checkpointer") {
		t.Fatalf("GetState() error = %v, want a checkpointer error", err)
	}
	if _, err := cg.GetStateHistory(ctx, cfg, checkpoint.ListOptions{}); err == nil ||
		!strings.Contains(err.Error(), "checkpointer") {
		t.Fatalf("GetStateHistory() error = %v, want a checkpointer error", err)
	}
	if _, err := cg.UpdateState(ctx, cfg, map[string]any{"x": 1}, "a"); err == nil ||
		!strings.Contains(err.Error(), "checkpointer") {
		t.Fatalf("UpdateState() error = %v, want a checkpointer error", err)
	}
}

// TestUpdateStateSubgraphNamespace verifies that UpdateState against a
// subgraph-namespace checkpoint saves the update checkpoint into that same
// namespace (not the root namespace) and carries Metadata.Parents forward
// from the checkpoint it builds on.
func TestUpdateStateSubgraphNamespace(t *testing.T) {
	ctx := context.Background()
	saver := checkpoint.NewMemorySaver()

	child := compileChild(t, "child_step", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"x": "child"}, nil
	})
	top := NewStateGraph()
	top.AddSubgraph("sub", child)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"x": "start"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	childCfg := checkpoint.Config{ThreadID: "t1", CheckpointNS: "sub"}
	childTup, err := saver.GetTuple(ctx, childCfg)
	if err != nil || childTup == nil {
		t.Fatalf("expected a child-namespace checkpoint, got tup=%+v err=%v", childTup, err)
	}
	if childTup.Metadata.Parents[""] == "" {
		t.Fatalf("child checkpoint must name the parent's position in Parents[\"\"], got %+v", childTup.Metadata.Parents)
	}
	rootBefore, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}

	newCfg, err := cg.UpdateState(ctx, childCfg, map[string]any{"x": "updated"}, "sub")
	if err != nil {
		t.Fatalf("UpdateState() error = %v", err)
	}
	if newCfg.CheckpointNS != "sub" {
		t.Fatalf("UpdateState() returned Config with namespace %q, want %q (update must land in the subgraph namespace)",
			newCfg.CheckpointNS, "sub")
	}

	snap, err := cg.GetState(ctx, newCfg)
	if err != nil {
		t.Fatalf("GetState() of update checkpoint error = %v", err)
	}
	if snap.Config.CheckpointNS != "sub" {
		t.Fatalf("update checkpoint stored under namespace %q, want %q", snap.Config.CheckpointNS, "sub")
	}
	if snap.Metadata.Source != "update" {
		t.Fatalf("update checkpoint Metadata.Source = %q, want update", snap.Metadata.Source)
	}
	if snap.Values["x"] != "updated" {
		t.Fatalf("update checkpoint Values[x] = %v, want updated", snap.Values["x"])
	}
	if snap.Metadata.Parents[""] != childTup.Metadata.Parents[""] {
		t.Fatalf("update checkpoint Parents = %+v, want Parents carried forward from %+v",
			snap.Metadata.Parents, childTup.Metadata.Parents)
	}

	// The root namespace gained no checkpoint from the subgraph update.
	rootAfter, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(rootAfter) != len(rootBefore) {
		t.Fatalf("root namespace has %d checkpoints after subgraph UpdateState, want %d (unchanged)",
			len(rootAfter), len(rootBefore))
	}
}
