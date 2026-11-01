package graph

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// forkBranchGraph builds START -> a -> b over two channels: "log", a
// DeltaChannel with snapshotFrequency=100 (never cadence-snapshotted, so its
// value lives only as per-checkpoint pending writes reconstructed by the
// ancestor walk), and "plain", a plain accumulating reducer channel every
// checkpoint stores in full. Each node appends its turn-tagged token to both
// channels, so every turn contributes an input delta write plus one node
// delta write per superstep.
func forkBranchGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddReducer("plain", channels.AppendSliceReducer)
	node := func(name string) func(runtime.Runtime, map[string]any) (any, error) {
		return func(_ runtime.Runtime, state map[string]any) (any, error) {
			token := fmt.Sprintf("%s%v", name, state["turn"])
			return map[string]any{"log": []string{token}, "plain": []string{token}}, nil
		}
	}
	g.AddNode("a", node("a"))
	g.AddNode("b", node("b"))
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// mustGetLog reads the reconstructed "log" delta channel at cfg.
func mustGetLog(t *testing.T, cg *CompiledGraph, cfg checkpoint.Config) []string {
	t.Helper()
	snap, err := cg.GetState(t.Context(), cfg)
	if err != nil {
		t.Fatalf("GetState(%s) error = %v", cfg.CheckpointID, err)
	}
	got, _ := snap.Values["log"].([]string)
	return got
}

// TestExitDurabilityPinnedForkKeepsNewerBranchClean pins #9269: an
// exit-durability run pinned via Options.CheckpointID at a checkpoint the
// thread already moved past must not store its delta writes on that shared
// checkpoint — the newer branch's ancestor walk would replay them as its own
// state (durable cross-branch contamination). Mirrors Python's
// test_exit_runs_from_a_checkpoint_the_thread_moved_past_leave_its_newer_branch_alone
// (test_delta_channel_exit_mode.py): the pinned run instead force-snapshots
// the delta channels it wrote into its own final checkpoint.
func TestExitDurabilityPinnedForkKeepsNewerBranchClean(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	cg := forkBranchGraph(t, WithCheckpointer(saver)) // turns 1-2 run sync

	invoke := func(turn int, opts Options) (Result, error) {
		token := fmt.Sprintf("in%d", turn)
		return cg.InvokeWithOptions(ctx, map[string]any{
			"turn":  turn,
			"log":   []string{token},
			"plain": []string{token},
		}, opts)
	}

	// Turn 1 runs to completion; its final checkpoint is A.
	if _, err := invoke(1, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	snapA, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 1 error = %v", err)
	}
	cpA := snapA.Config.CheckpointID

	// Turn 2 continues the thread past A: the newer branch B.
	if _, err := invoke(2, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}
	snapB, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 2 error = %v", err)
	}
	cpB := snapB.Config.CheckpointID

	// Turn 3: an exit-durability run PINNED at A while B is the latest.
	// A second pinned run (Python's exit_runs=2 parametrization) repeats the
	// fork from the same superseded A, so both fork finals descend from it.
	for turn := 3; turn <= 4; turn++ {
		want := []string{"in1", "a1", "b1", fmt.Sprintf("in%d", turn), fmt.Sprintf("a%d", turn), fmt.Sprintf("b%d", turn)}
		res, err := invoke(turn, Options{ThreadID: "t", CheckpointID: cpA}.WithRunDurability(DurabilityExit))
		if err != nil {
			t.Fatalf("pinned exit Invoke() (turn %d) error = %v", turn, err)
		}
		if got, _ := res.Values["log"].([]string); !reflect.DeepEqual(got, want) {
			t.Fatalf("pinned run %d live log = %v, want %v", turn, got, want)
		}

		// (a) The newer branch B must NOT read the pinned run's writes: its
		// ancestor walk must not replay anything the forked run stored.
		if got := mustGetLog(t, cg, checkpoint.Config{ThreadID: "t", CheckpointID: cpB}); !reflect.DeepEqual(got, []string{"in1", "a1", "b1", "in2", "a2", "b2"}) {
			t.Fatalf("turn %d: B-branch log = %v, want [in1 a1 b1 in2 a2 b2] (pinned run's writes leaked into the newer branch)", turn, got)
		}

		// (b) The pinned run's own branch still reads its writes — via the
		// forced snapshot embedded in its final checkpoint (the thread's
		// latest), not via writes anchored on A.
		if got := mustGetLog(t, cg, checkpoint.Config{ThreadID: "t"}); !reflect.DeepEqual(got, want) {
			t.Fatalf("turn %d: forked branch log = %v, want %v", turn, got, want)
		}
		tupF, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
		if err != nil || tupF == nil {
			t.Fatalf("turn %d: GetTuple(latest) = (%v, %v)", turn, tupF, err)
		}
		if _, ok := channels.UnwrapDeltaSnapshot(tupF.Checkpoint.ChannelValues["log"]); !ok {
			t.Fatalf("turn %d: final checkpoint log = %#v, want a forced delta snapshot blob (writes anchored on A instead)", turn, tupF.Checkpoint.ChannelValues["log"])
		}
	}

	// (c) Python's cross-check: every checkpoint in the thread's history
	// reconstructs the delta channel to exactly the plain channel's value.
	hist, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	if len(hist) < 6 {
		t.Fatalf("history has %d checkpoints, want at least 6 (two sync turns + the exit final)", len(hist))
	}
	for _, s := range hist {
		log, _ := s.Values["log"].([]string)
		plain, _ := s.Values["plain"].([]string)
		if !reflect.DeepEqual(log, plain) {
			t.Fatalf("checkpoint %s: log = %v, plain = %v (delta reconstruction diverged from plain state)", s.Config.CheckpointID, log, plain)
		}
	}
}
