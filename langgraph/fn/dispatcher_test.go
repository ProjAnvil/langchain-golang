package fn

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func TestNextCallIdx(t *testing.T) {
	d := newDispatcher(nil)
	for want := 0; want < 3; want++ {
		if got := d.nextCallIdx(""); got != want {
			t.Fatalf("nextCallIdx(\"\") = %d, want %d", got, want)
		}
	}
	// A different parent path counts independently.
	if got := d.nextCallIdx("a@0"); got != 0 {
		t.Fatalf("nextCallIdx(\"a@0\") = %d, want 0", got)
	}
	if got := d.nextCallIdx(""); got != 3 {
		t.Fatalf("nextCallIdx(\"\") = %d, want 3", got)
	}
	if got := d.nextCallIdx("a@0"); got != 1 {
		t.Fatalf("nextCallIdx(\"a@0\") = %d, want 1", got)
	}
}

func TestDispatcherContext(t *testing.T) {
	if d := dispatcherFromContext(context.Background()); d != nil {
		t.Fatalf("dispatcherFromContext on bare ctx = %v, want nil", d)
	}
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	if got := dispatcherFromContext(ctx); got != d {
		t.Fatalf("dispatcherFromContext = %p, want %p", got, d)
	}
}

func TestRecordSealSnapshot(t *testing.T) {
	d := newDispatcher(nil)
	d.record(taskResult{name: "a", callIdx: 0, value: 1})
	d.record(taskResult{name: "b", callIdx: 1, value: 2})

	snap := d.snapshotResults()
	if len(snap) != 2 || snap[0].name != "a" || snap[1].name != "b" {
		t.Fatalf("snapshotResults = %+v, want [a b] in record order", snap)
	}
	// The snapshot is a copy: mutating it must not touch the live table.
	snap[0].name = "mutated"
	if d.results[0].name != "a" {
		t.Fatalf("snapshot aliases the live table: results[0].name = %q", d.results[0].name)
	}

	// After seal, late completions are dropped.
	d.seal()
	d.record(taskResult{name: "late", callIdx: 2, value: 3})
	if got := d.snapshotResults(); len(got) != 2 {
		t.Fatalf("snapshotResults after sealed record = %d results, want 2", len(got))
	}
}

// replayTuple builds the paused-run tuple shared by the loadReplay gate
// tests: one fn result write (task "a") and one interrupt write (filtered).
func replayTuple(source string, next []checkpoint.PlannedTask) *checkpoint.Tuple {
	return &checkpoint.Tuple{
		Config: checkpoint.Config{ThreadID: "t1"},
		Checkpoint: checkpoint.Checkpoint{
			ID:   "cp1",
			Next: next,
		},
		Metadata: checkpoint.Metadata{Source: source, Step: 3},
		PendingWrites: []checkpoint.Write{
			{TaskID: "a", Channel: checkpoint.ReservedReturn, Value: 1},
			{TaskID: "b", Channel: checkpoint.ReservedInterrupt, Value: types.Interrupt{}},
		},
	}
}

func TestLoadReplayGate1Resume(t *testing.T) {
	next := []checkpoint.PlannedTask{{ID: "x", Node: "entrypoint"}}
	d := newDispatcher(nil)
	d.loadReplay(replayTuple("loop", next), graph.Options{Resume: "v"})

	if d.replay == nil {
		t.Fatal("replay table is nil, want gate 1 hit")
	}
	if len(d.replay) != 1 {
		t.Fatalf("replay table = %v, want only task a (__interrupt__ filtered)", d.replay)
	}
	w, ok := d.replay["a"]
	if !ok || w.Channel != checkpoint.ReservedReturn || w.Value != 1 {
		t.Fatalf("replay[a] = %+v (ok=%v), want the __return__ write", w, ok)
	}
	if d.cpID != "cp1" || d.step != 3 {
		t.Fatalf("cpID/step = %q/%d, want cp1/3 from the tuple", d.cpID, d.step)
	}
}

func TestLoadReplayGate1RequiresNext(t *testing.T) {
	d := newDispatcher(nil)
	d.loadReplay(replayTuple("loop", nil), graph.Options{Resume: "v"})
	if d.replay != nil {
		t.Fatalf("replay table = %v, want nil (Resume set but Next empty)", d.replay)
	}
}

func TestLoadReplayFreshLoopRun(t *testing.T) {
	next := []checkpoint.PlannedTask{{ID: "x", Node: "entrypoint"}}
	d := newDispatcher(nil)
	d.loadReplay(replayTuple("loop", next), graph.Options{})
	if d.replay != nil {
		t.Fatalf("replay table = %v, want nil (fresh loop run, no Resume)", d.replay)
	}
}

func TestLoadReplayGate2InputWithFnWrites(t *testing.T) {
	d := newDispatcher(nil)
	d.loadReplay(replayTuple("input", nil), graph.Options{})
	if d.replay == nil {
		t.Fatal("replay table is nil, want gate 2 hit (input source + fn writes)")
	}
	if _, ok := d.replay["a"]; !ok || len(d.replay) != 1 {
		t.Fatalf("replay table = %v, want only task a", d.replay)
	}
	if d.cpID != "cp1" || d.step != 3 {
		t.Fatalf("cpID/step = %q/%d, want cp1/3 from the tuple", d.cpID, d.step)
	}
}

func TestLoadReplayGate2RequiresFnWrites(t *testing.T) {
	tup := replayTuple("input", nil)
	tup.PendingWrites = []checkpoint.Write{
		{TaskID: "b", Channel: checkpoint.ReservedInterrupt, Value: types.Interrupt{}},
	}
	d := newDispatcher(nil)
	d.loadReplay(tup, graph.Options{})
	if d.replay != nil {
		t.Fatalf("replay table = %v, want nil (input source but no fn writes)", d.replay)
	}
}

// Savers round-tripping through JSON serde decode the __fn_consumed__ count
// as float64; loadReplay must accept it alongside the memory saver's int.
func TestLoadReplayConsumedFloat64(t *testing.T) {
	tup := replayTuple("input", nil)
	tup.PendingWrites = []checkpoint.Write{
		{TaskID: "a", Channel: checkpoint.ReservedReturn, Value: 5},
		{TaskID: "a", Channel: checkpoint.ReservedFnConsumed, Value: float64(2)},
	}
	d := newDispatcher(nil)
	d.loadReplay(tup, graph.Options{})
	if d.replay == nil {
		t.Fatal("replay table is nil, want gate 2 hit")
	}
	if got := d.replayConsumed("a"); got != 2 {
		t.Fatalf("replayConsumed(\"a\") = %d, want 2 (float64 serde decode)", got)
	}
}
