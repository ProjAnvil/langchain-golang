package graph

import (
	"reflect"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// deltaToPlainApps builds two compiled graphs over one saver that differ only
// in the `items` field's channel kind: a DeltaChannel (the delta app) and a
// plain BinaryOperator reducer (the plain app) — the Go shape of Python's
// test_delta_channel_to_plain.py DeltaState/PlainState pair. The node appends
// one item per turn.
func deltaToPlainApps(t *testing.T, saver checkpoint.Saver, freq int) (delta, plain *CompiledGraph) {
	t.Helper()
	build := func(register func(*StateGraph)) *CompiledGraph {
		g := NewStateGraph()
		register(g)
		g.AddNode("chat", func(_ runtime.Runtime, state map[string]any) (any, error) {
			n := len(state["items"].([]int))
			return map[string]any{"items": []int{n + 1000}}, nil
		})
		g.AddEdge(types.START, "chat")
		g.AddEdge("chat", types.END)
		cg, err := g.Compile(WithCheckpointer(saver))
		if err != nil {
			t.Fatalf("Compile() error = %v", err)
		}
		return cg
	}
	delta = build(func(g *StateGraph) {
		g.AddChannel("items", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, freq))
	})
	plain = build(func(g *StateGraph) {
		g.AddReducer("items", channels.AppendSliceReducer)
	})
	return delta, plain
}

// deltaWrittenThread runs four turns of the delta app on a fresh thread,
// asserting the thread really has a snapshot blob to read through (Python:
// "the thread should have a snapshot to read through").
func deltaWrittenThread(t *testing.T, delta *CompiledGraph) {
	t.Helper()
	ctx := t.Context()
	cfg := Options{ThreadID: "1"}
	for i := range 4 {
		if _, err := delta.InvokeWithOptions(ctx, map[string]any{"items": []int{i}}, cfg); err != nil {
			t.Fatalf("delta invoke %d: %v", i, err)
		}
	}
	history, err := delta.GetStateHistory(ctx, checkpoint.Config{ThreadID: "1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory: %v", err)
	}
	for _, snap := range history {
		if len(snap.Values) > 0 {
			if _, ok := snap.Values["items"].([]int); !ok {
				t.Fatalf("delta app left items unreconstructed at checkpoint %s", snap.Config.CheckpointID)
			}
		}
	}
	// Blob presence is asserted on the raw tuples, not the projected values.
	hasBlob := false
	tuples, err := delta.checkpointer.List(ctx, checkpoint.Config{ThreadID: "1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	for _, tup := range tuples {
		if _, ok := channels.UnwrapDeltaSnapshot(tup.Checkpoint.ChannelValues["items"]); ok {
			hasBlob = true
		}
	}
	if !hasBlob {
		t.Fatal("the thread should have a snapshot blob to read through")
	}
}

// TestPlainReducerReadsDeltaWrittenThread: switching a field from a
// DeltaChannel to a plain reducer must keep the thread's value — every
// checkpoint written under the DeltaChannel reads the same through the plain
// app as through the delta app (#9302; the walk only rebuilt fields whose
// current channel is a DeltaChannel, so the plain app read them empty).
func TestPlainReducerReadsDeltaWrittenThread(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	delta, plain := deltaToPlainApps(t, saver, 3)
	deltaWrittenThread(t, delta)
	ctx := t.Context()

	deltaHistory, err := delta.GetStateHistory(ctx, checkpoint.Config{ThreadID: "1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("delta GetStateHistory: %v", err)
	}
	plainHistory, err := plain.GetStateHistory(ctx, checkpoint.Config{ThreadID: "1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("plain GetStateHistory: %v", err)
	}
	if len(plainHistory) != len(deltaHistory) {
		t.Fatalf("history lengths differ: plain %d, delta %d", len(plainHistory), len(deltaHistory))
	}
	for i := range deltaHistory {
		want, _ := deltaHistory[i].Values["items"].([]int)
		got, _ := plainHistory[i].Values["items"].([]int)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("checkpoint %s (step %d): plain app reads items=%v, delta app reads %v — a delta-written field must keep its value after the switch (#9302)",
				deltaHistory[i].Config.CheckpointID, deltaHistory[i].Metadata.Step, got, want)
		}
	}
}

// TestPlainReducerContinuesDeltaThread: the plain app can CONTINUE a
// delta-written thread — the accumulated value survives, the new turn's
// writes land on top of it, the head checkpoint stores the full value (later
// reads are plain reads), and the delta app reads the same thread back.
func TestPlainReducerContinuesDeltaThread(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	delta, plain := deltaToPlainApps(t, saver, 3)
	deltaWrittenThread(t, delta)
	ctx := t.Context()

	before, err := delta.GetState(ctx, checkpoint.Config{ThreadID: "1"})
	if err != nil {
		t.Fatalf("delta GetState: %v", err)
	}
	prior, _ := before.Values["items"].([]int)

	res, err := plain.InvokeWithOptions(ctx, map[string]any{"items": []int{99}}, Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("plain invoke: %v", err)
	}
	after, _ := res.Values["items"].([]int)
	// prior + the new input write (99) + the node's write (1000 + the
	// post-input length).
	want := append(append([]int{}, prior...), 99, 1000+len(prior)+1)
	if !reflect.DeepEqual(after, want) {
		t.Fatalf("plain continue: items = %v, want %v (prior %v + input 99 + node write)", after, want, prior)
	}

	// The head checkpoint stores the FULL value (a plain reducer snapshots at
	// every version), so later reads are plain reads and the delta app reads
	// the same thread back.
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "1"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: %v, %v", tup, err)
	}
	if got, ok := tup.Checkpoint.ChannelValues["items"].([]int); !ok || !reflect.DeepEqual(got, after) {
		t.Fatalf("head checkpoint items = %v (%T), want the full value %v", tup.Checkpoint.ChannelValues["items"], tup.Checkpoint.ChannelValues["items"], after)
	}
	deltaSnap, err := delta.GetState(ctx, checkpoint.Config{ThreadID: "1"})
	if err != nil {
		t.Fatalf("delta GetState after plain run: %v", err)
	}
	if got, _ := deltaSnap.Values["items"].([]int); !reflect.DeepEqual(got, after) {
		t.Fatalf("delta app reads items = %v after the plain run, want %v", got, after)
	}
}

// TestPlainReducerReplaysDeltaHistoryOneWriteAtATime: the plain app rebuilds
// an ex-delta channel from the ancestor walk with ONE reducer call per write
// — the writes span many supersteps, and a single batched call would reject a
// second Overwrite and drop the writes after it (#9302, Python's _replay).
// Hand-built chain: cp0 stores the plain seed [1]; cp1's writes are [2],
// Overwrite([3]), Overwrite([4]), [5] — the last Overwrite resets to [4],
// then [5] folds in: [4 5]. (Two Overwrites in the batch prove the
// one-call-per-write split: a single batched Update would reject the second.)
func TestPlainReducerReplaysDeltaHistoryOneWriteAtATime(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()

	mkCp := func(id string, values map[string]any, versions map[string]int64) checkpoint.Checkpoint {
		return checkpoint.Checkpoint{
			V: 1, ID: id, TS: time.Now(),
			ChannelValues:   values,
			ChannelVersions: versions,
			VersionsSeen:    map[string]map[string]int64{},
		}
	}
	if _, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t"},
		mkCp("cp0", map[string]any{"items": []int{1}}, map[string]int64{"items": 1}),
		checkpoint.Metadata{Source: "loop", Step: 0}, nil); err != nil {
		t.Fatalf("Put cp0: %v", err)
	}
	if _, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: "cp0"},
		mkCp("cp1", map[string]any{}, map[string]int64{"items": 2}),
		checkpoint.Metadata{Source: "loop", Step: 1}, nil); err != nil {
		t.Fatalf("Put cp1: %v", err)
	}
	// cp1 has a version for items but no stored value: the ex-delta sentinel.
	if err := saver.PutWrites(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: "cp1"}, []checkpoint.Write{
		{Channel: "items", Value: []int{2}},
		{Channel: "items", Value: channels.NewOverwrite([]int{3})},
		{Channel: "items", Value: channels.NewOverwrite([]int{4})},
		{Channel: "items", Value: []int{5}},
	}, "task-1", ""); err != nil {
		t.Fatalf("PutWrites: %v", err)
	}

	g := NewStateGraph()
	g.AddReducer("items", channels.AppendSliceReducer)
	g.AddNode("n", noopNode)
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: "cp1"})
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if got, ok := snap.Values["items"].([]int); !ok || !reflect.DeepEqual(got, []int{4, 5}) {
		t.Fatalf("items = %v (%T), want [4 5] (last Overwrite wins, later writes fold in; one reducer call per write)",
			snap.Values["items"], snap.Values["items"])
	}
}

// TestPlainThreadReadsNeverWalkHistory: a thread written entirely under a
// plain reducer never walks the ancestor chain — every read resolves from
// stored values (regression guard for the #9302 walk inclusion).
func TestPlainThreadReadsNeverWalkHistory(t *testing.T) {
	inner := checkpoint.NewMemorySaver()
	saver := &getTupleCountingSaver{Saver: inner}
	_, plain := deltaToPlainApps(t, saver, 3)
	ctx := t.Context()
	cfg := Options{ThreadID: "1"}

	if _, err := plain.InvokeWithOptions(ctx, map[string]any{"items": []int{0}}, cfg); err != nil {
		t.Fatalf("invoke 0: %v", err)
	}
	if _, err := plain.UpdateState(ctx, checkpoint.Config{ThreadID: "1"}, map[string]any{"items": []int{50}}, "chat"); err != nil {
		t.Fatalf("UpdateState: %v", err)
	}
	if _, err := plain.InvokeWithOptions(ctx, map[string]any{"items": []int{1}}, cfg); err != nil {
		t.Fatalf("invoke 1: %v", err)
	}
	saver.getTuples = 0
	if _, err := plain.GetState(ctx, checkpoint.Config{ThreadID: "1"}); err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if _, err := plain.GetStateHistory(ctx, checkpoint.Config{ThreadID: "1"}, checkpoint.ListOptions{}); err != nil {
		t.Fatalf("GetStateHistory: %v", err)
	}
	if saver.getTuples != 1 {
		t.Fatalf("plain-thread reads issued %d ancestor GetTuple calls, want exactly 1 (the GetState read itself; plain values resolve from stored state)",
			saver.getTuples)
	}
}
