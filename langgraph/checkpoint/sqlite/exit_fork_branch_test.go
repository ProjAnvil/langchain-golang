package sqlite_test

import (
	"fmt"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// appendStrings folds []string updates into a []string base (a plain
// accumulating reducer for the "plain" cross-check channel).
func appendStrings(existing any, update any) (any, error) {
	base, _ := existing.([]string)
	add, _ := update.([]string)
	out := make([]string, 0, len(base)+len(add))
	out = append(out, base...)
	out = append(out, add...)
	return out, nil
}

// forkedExitGraph builds START -> a -> b over "log" (a DeltaChannel with
// snapshotFrequency=100, so every value lives only as per-checkpoint pending
// writes reconstructed by the ancestor walk) and "plain" (a plain accumulating
// reducer channel every checkpoint stores in full). Each node appends its
// turn-tagged token to both channels.
func forkedExitGraph(t *testing.T, saver checkpoint.Saver) *graph.CompiledGraph {
	t.Helper()
	g := graph.NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringTokenReducer, func() any { return []string{} }, 100))
	g.AddReducer("plain", appendStrings)
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
	cg, err := g.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestExitDurabilityPinnedForkKeepsNewerBranchClean is the SQL-saver variant
// of the #9269 fork test (see langgraph/graph/exit_fork_branch_test.go): an
// exit-durability run pinned at a checkpoint the thread moved past must not
// anchor its delta writes on that shared checkpoint, where the newer branch's
// ancestor walk would replay them. The pinned run's own branch reads its
// writes from the forced snapshot in its final checkpoint.
func TestExitDurabilityPinnedForkKeepsNewerBranchClean(t *testing.T) {
	saver, err := sqlite.New(filepath.Join(t.TempDir(), "checkpoints.db"), serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { saver.Close() })
	cg := forkedExitGraph(t, saver)
	ctx := t.Context()

	invoke := func(turn int, opts graph.Options) (graph.Result, error) {
		token := fmt.Sprintf("in%d", turn)
		return cg.InvokeWithOptions(ctx, map[string]any{
			"turn":  turn,
			"log":   []string{token},
			"plain": []string{token},
		}, opts)
	}

	if _, err := invoke(1, graph.Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	snapA, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 1 error = %v", err)
	}
	cpA := snapA.Config.CheckpointID

	if _, err := invoke(2, graph.Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}
	snapB, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 2 error = %v", err)
	}
	cpB := snapB.Config.CheckpointID

	res, err := invoke(3, graph.Options{ThreadID: "t", CheckpointID: cpA}.WithRunDurability(graph.DurabilityExit))
	if err != nil {
		t.Fatalf("pinned exit Invoke() error = %v", err)
	}
	if got, _ := res.Values["log"].([]string); !reflect.DeepEqual(got, []string{"in1", "a1", "b1", "in3", "a3", "b3"}) {
		t.Fatalf("pinned run live log = %v, want [in1 a1 b1 in3 a3 b3]", got)
	}

	// (a) The newer branch B reads only its own history.
	snapB2, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: cpB})
	if err != nil {
		t.Fatalf("GetState(B) error = %v", err)
	}
	if got, _ := snapB2.Values["log"].([]string); !reflect.DeepEqual(got, []string{"in1", "a1", "b1", "in2", "a2", "b2"}) {
		t.Fatalf("B-branch log = %v, want [in1 a1 b1 in2 a2 b2] (pinned run's writes leaked into the newer branch)", got)
	}

	// (b) The pinned run's branch reads its writes via the forced snapshot in
	// its final checkpoint, and the pinned checkpoint A carries no log writes
	// from the forked run (only turn 1's own b1 write remains).
	if got := mustForkLog(t, cg, checkpoint.Config{ThreadID: "t"}); !reflect.DeepEqual(got, []string{"in1", "a1", "b1", "in3", "a3", "b3"}) {
		t.Fatalf("forked branch log = %v, want [in1 a1 b1 in3 a3 b3]", got)
	}
	tupF, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tupF == nil {
		t.Fatalf("GetTuple(latest) = (%v, %v)", tupF, err)
	}
	if _, ok := channels.UnwrapDeltaSnapshot(tupF.Checkpoint.ChannelValues["log"]); !ok {
		t.Fatalf("final checkpoint log = %#v, want a forced delta snapshot blob (writes anchored on A instead)", tupF.Checkpoint.ChannelValues["log"])
	}
	tupA, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: cpA})
	if err != nil || tupA == nil {
		t.Fatalf("GetTuple(A) = (%v, %v)", tupA, err)
	}
	forkTokens := map[string]bool{"in3": true, "a3": true, "b3": true}
	for _, w := range tupA.PendingWrites {
		if w.Channel != "log" {
			continue
		}
		for _, tok := range w.Value.([]string) {
			if forkTokens[tok] {
				t.Fatalf("pinned checkpoint A carries the forked run's %q write: pending log writes = %+v", tok, tupA.PendingWrites)
			}
		}
	}

	// (c) Every checkpoint in the history: log == plain.
	hist, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	for _, s := range hist {
		log, _ := s.Values["log"].([]string)
		plain, _ := s.Values["plain"].([]string)
		if !reflect.DeepEqual(log, plain) {
			t.Fatalf("checkpoint %s: log = %v, plain = %v (delta reconstruction diverged from plain state)", s.Config.CheckpointID, log, plain)
		}
	}
}

func mustForkLog(t *testing.T, cg *graph.CompiledGraph, cfg checkpoint.Config) []string {
	t.Helper()
	snap, err := cg.GetState(t.Context(), cfg)
	if err != nil {
		t.Fatalf("GetState(%s) error = %v", cfg.CheckpointID, err)
	}
	got, _ := snap.Values["log"].([]string)
	return got
}

// TestSyncDeltaSnapshotPersistsThroughSqlite pins the enabler for the fork
// fix's forced snapshots: the delta snapshot blob a checkpoint stores (here
// via the ordinary cadence, snapshotFrequency=1) must be serializable by
// serde-based savers. Before the blob became JSON-native
// (DeltaChannel.SnapshotBlob returning the map form), every SQL/Redis saver
// rejected it: "serde: unregistered type channels.deltaSnapshot".
func TestSyncDeltaSnapshotPersistsThroughSqlite(t *testing.T) {
	saver, err := sqlite.New(filepath.Join(t.TempDir(), "checkpoints.db"), serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { saver.Close() })
	g := graph.NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringTokenReducer, func() any { return []string{} }, 1))
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"x"}}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(t.Context(), map[string]any{"log": []string{"in"}}, graph.Options{ThreadID: "t"}); err != nil {
		t.Fatalf("sync Invoke() error = %v", err)
	}
	snap, err := cg.GetState(t.Context(), checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got, _ := snap.Values["log"].([]string); !reflect.DeepEqual(got, []string{"in", "x"}) {
		t.Fatalf("log = %v, want [in x] (cadence snapshot blob lost)", got)
	}
}
