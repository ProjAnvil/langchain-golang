package graph

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// pauseCadenceGraph builds START -> w1 -> w2 -> (a || b) with a delta `log`
// (snapshotFrequency=2): the first loop checkpoint cadence-snapshots (updates
// {2,2}, counters reset), and the PAUSE save two supersteps later fires again
// via the supersteps bound ({0,2}) — from PRE-superstep state, while `b` — a
// parallel COMPLETED sibling of the interrupting `a` — has its write anchored
// on that same pause checkpoint. The resumed run's checkpoints stay below
// cadence, so every read after the resume reconstructs `log` by walking back
// THROUGH the pause checkpoint.
func pauseCadenceGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 2))
	g.AddNode("w1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"w1"}}, nil
	})
	g.AddNode("w2", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"w2"}}, nil
	})
	g.AddNode("a", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(rt, "stop")
		return nil, nil
	})
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"b"}}, nil
	})
	g.AddEdge(types.START, "w1")
	g.AddEdge("w1", "w2")
	g.AddEdge("w2", "a")
	g.AddEdge("w2", "b")
	g.AddEdge("a", types.END)
	g.AddEdge("b", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestPauseCadenceSnapshotDoesNotStrandSiblingWrites pins the pause-path
// cadence corner (#9235 review, Important 1): the pause checkpoint is saved
// from PRE-superstep state, so a cadence blob on it does NOT fold in the
// completed sibling writes anchored on it — the ancestor walk must not stop
// at that blob and skip them, or every read below the pause comes up short
// (Python collects an ancestor's pending writes before seeding, and reads
// correctly; pre-walk-reorder Go did too). The pause save now suppresses the
// snapshot for channels with anchored pause writes, keeping the checkpoint
// sentinel so the walk collects the writes.
func TestPauseCadenceSnapshotDoesNotStrandSiblingWrites(t *testing.T) {
	for _, d := range []Durability{DurabilitySync, DurabilityAsync, DurabilityExit} {
		t.Run(string(d), func(t *testing.T) {
			ctx := t.Context()
			saver := checkpoint.NewMemorySaver()
			cg := pauseCadenceGraph(t, WithCheckpointer(saver), WithDurability(d))
			cfg := Options{ThreadID: "t"}

			// Turn 1 pauses inside `a` while `b` completes.
			res, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []string{"in"}}, cfg)
			if err != nil {
				t.Fatalf("pause Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 1 {
				t.Fatalf("interrupts = %+v, want one pause", res.Interrupts)
			}

			// The pause checkpoint must NOT carry a log blob: its value would
			// predate b's anchored write. (Earlier checkpoints' cadence blobs
			// are legitimate — their values fold in their own anchored
			// writes.)
			pauseTup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
			if err != nil || pauseTup == nil {
				t.Fatalf("[%s] GetTuple(latest) = (%v, %v), want the pause checkpoint", d, pauseTup, err)
			}
			if _, isBlob := channels.UnwrapDeltaSnapshot(pauseTup.Checkpoint.ChannelValues["log"]); isBlob {
				t.Fatalf("[%s] the pause checkpoint %s carries a log blob — the pause save must suppress it for channels with anchored pause writes (the blob predates b's anchored write)",
					d, pauseTup.Checkpoint.ID)
			}

			// Resume: `a` re-runs, `b` replays; the run completes. The resumed
			// run's checkpoints stay below cadence, so GetState walks back
			// through the pause checkpoint.
			if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "go"}); err != nil {
				t.Fatalf("resume Invoke() error = %v", err)
			}
			want := []string{"in", "w1", "w2", "b"}
			snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
			if err != nil {
				t.Fatalf("GetState() error = %v", err)
			}
			if got, _ := snap.Values["log"].([]string); !reflect.DeepEqual(got, want) {
				t.Fatalf("[%s] log after resume = %v, want %v (the pause blob stranded b's anchored write)", d, got, want)
			}

			// Every persisted checkpoint in the history reads consistently.
			hist, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{})
			if err != nil {
				t.Fatalf("GetStateHistory() error = %v", err)
			}
			for _, s := range hist {
				got, _ := s.Values["log"].([]string)
				if len(got) == 0 {
					t.Fatalf("[%s] checkpoint %s (step %d) reads an empty log", d, s.Config.CheckpointID, s.Metadata.Step)
				}
				if got[0] != "in" {
					t.Fatalf("[%s] checkpoint %s (step %d) log = %v — reconstruction lost the thread's prefix", d, s.Config.CheckpointID, s.Metadata.Step, got)
				}
			}
		})
	}
}
