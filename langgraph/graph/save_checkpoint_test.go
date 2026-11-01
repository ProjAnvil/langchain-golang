package graph

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// deltaGraphFreq builds n1 -> n2 where both nodes append to the "items" key
// backed by a DeltaChannel at the given snapshotFrequency, and "msg" is a plain
// last-value key. The BatchReducer is passed directly to NewDeltaChannel (C4),
// not wrapped via BatchFromReducer.
func deltaGraphFreq(t *testing.T, freq int, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("items", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, freq))
	g.AddNode("n1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"items": []int{1, 2}, "msg": "hello"}, nil
	})
	g.AddNode("n2", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"items": []int{3}}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestAdvanceDeltaCounters covers the pure counter-advancement helper against
// the Python reference (langgraph/pregel/_loop.py:1111-1124): every delta
// channel's superstep counter bumps unconditionally; the update counter bumps
// only when the channel was written this superstep; non-delta channels and nil
// inputs are handled.
func TestAdvanceDeltaCounters(t *testing.T) {
	delta := channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 3)
	lastValue := channels.NewLastValue()

	tests := []struct {
		name    string
		chs     map[string]channels.Channel
		prev    map[string][2]int
		updated map[string]bool
		want    map[string][2]int
	}{
		{
			name:    "fresh no prev no updates — s bumps u stays",
			chs:     map[string]channels.Channel{"items": delta},
			prev:    nil,
			updated: nil,
			want:    map[string][2]int{"items": {0, 1}},
		},
		{
			name:    "resume advances s and u when updated",
			chs:     map[string]channels.Channel{"items": delta},
			prev:    map[string][2]int{"items": {2, 3}},
			updated: map[string]bool{"items": true},
			want:    map[string][2]int{"items": {3, 4}},
		},
		{
			name:    "resume advances s only when not updated",
			chs:     map[string]channels.Channel{"items": delta},
			prev:    map[string][2]int{"items": {2, 3}},
			updated: map[string]bool{},
			want:    map[string][2]int{"items": {2, 4}},
		},
		{
			name:    "missing prev entry treated as zero",
			chs:     map[string]channels.Channel{"items": delta},
			prev:    map[string][2]int{"other": {9, 9}},
			updated: map[string]bool{"items": true},
			want:    map[string][2]int{"items": {1, 1}},
		},
		{
			name:    "non-delta channels skipped",
			chs:     map[string]channels.Channel{"items": delta, "msg": lastValue},
			prev:    map[string][2]int{"items": {1, 1}, "msg": {5, 5}},
			updated: map[string]bool{"items": true, "msg": true},
			want:    map[string][2]int{"items": {2, 2}},
		},
		{
			name: "no delta channels returns nil",
			chs:  map[string]channels.Channel{"msg": lastValue},
			prev: map[string][2]int{"msg": {5, 5}},
			want: nil,
		},
		{
			name: "nil channels returns nil",
			chs:  nil,
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := advanceDeltaCounters(tt.chs, tt.prev, tt.updated)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("advanceDeltaCounters = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestNonZeroCounters covers the zero-entry filter (langgraph/pregel/_loop.py:1158).
func TestNonZeroCounters(t *testing.T) {
	tests := []struct {
		name     string
		counters map[string][2]int
		want     map[string][2]int
	}{
		{"drops zero keeps rest", map[string][2]int{"a": {0, 0}, "b": {1, 2}}, map[string][2]int{"b": {1, 2}}},
		{"all zero returns nil", map[string][2]int{"a": {0, 0}}, nil},
		{"nil returns nil", nil, nil},
		{"empty returns nil", map[string][2]int{}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := nonZeroCounters(tt.counters)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("nonZeroCounters = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSaveCheckpointSnapshotsDeltaAtCadence verifies saveCheckpoint writes a
// delta snapshot blob into ChannelValues when a delta channel's update count
// reaches its snapshotFrequency. With freq=2 and two supersteps both writing
// "items", the final (step-1) checkpoint crosses the cadence (updates==2) and
// must carry the blob. The intermediate (step-0) checkpoint is below cadence
// and carries a non-zero counter in metadata instead.
func TestSaveCheckpointSnapshotsDeltaAtCadence(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := deltaGraphFreq(t, 2, WithCheckpointer(saver))
	ctx := t.Context()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	// History is newest-first; element 0 is the step-1 checkpoint.
	history, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(history) < 2 {
		t.Fatalf("expected at least 2 checkpoints, got %d", len(history))
	}

	// Latest checkpoint: cadence fired → items present as a snapshot blob and
	// the (just-reset) counter is zero so metadata has no counters entry.
	latest := history[0]
	blob, ok := latest.Checkpoint.ChannelValues["items"]
	if !ok {
		t.Fatal("latest checkpoint missing 'items' (cadence should have written a snapshot blob)")
	}
	if val, isSnap := channels.UnwrapDeltaSnapshot(blob); !isSnap {
		t.Fatalf("latest checkpoint 'items' = %T, want a delta snapshot blob", blob)
	} else if got, want := val.([]int), []int{1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("latest snapshot blob value = %v, want %v", got, want)
	}
	if latest.Metadata.CountersSinceDeltaSnapshot != nil {
		t.Fatalf("latest metadata counters = %v, want nil (just snapshotted)", latest.Metadata.CountersSinceDeltaSnapshot)
	}

	// Find the step-0 loop checkpoint (cadence not yet fired): its metadata must
	// carry the advanced counter {items:{1,2}} — every checkpoint save
	// (input + loop) advances the superstep counter, Python-style — and it
	// must OMIT items from ChannelValues (sentinel-only storage).
	var step0 *checkpoint.Tuple
	for i := range history {
		if history[i].Metadata.Source == "loop" && history[i].Metadata.Step == 0 {
			step0 = &history[i]
			break
		}
	}
	if step0 == nil {
		t.Fatalf("did not find step-0 loop checkpoint in history of %d", len(history))
	}
	if got, want := step0.Metadata.CountersSinceDeltaSnapshot, map[string][2]int{"items": {1, 2}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("step-0 metadata counters = %v, want %v", got, want)
	}
	if _, present := step0.Checkpoint.ChannelValues["items"]; present {
		t.Fatal("step-0 checkpoint has 'items' in ChannelValues, want it omitted (below cadence)")
	}
}

// TestSaveCheckpointPersistsDeltaCountersAcrossResume verifies the counters
// seeded from a loaded checkpoint (S3) continue advancing on a later run.
//
// A plain new turn now force-snapshots items at its input checkpoint: the
// loaded latest carries the last superstep's pending items write (Go anchors
// task writes on the checkpoint that observed them), and a non-resuming run
// materializes such channels instead of replaying the loaded writes (Python
// _loop.py:1011-1015) — which resets the counter. So the seeded-and-advanced
// counter is pinned via a load with NO pending writes instead: run 1's own
// input checkpoint (step -1; its empty-input fresh run persisted no
// NullTaskID write). A run pinned there (time travel) with an items-writing
// input seeds deltaCounters from that checkpoint's metadata, and the write
// re-creates the sentinel channel so the seeded counter is actually advanced:
// the loaded {items:{0,1}} bumps by one superstep and one update -> {1,2}.
func TestSaveCheckpointPersistsDeltaCountersAcrossResume(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := deltaGraphFreq(t, 4, WithCheckpointer(saver)) // freq=4: cadence never fires here
	ctx := t.Context()

	// First run completes (empty input, so its input checkpoint carries no
	// pending writes and counters {items:{0,1}}).
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	history, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	var firstInput *checkpoint.Tuple
	for i := range history {
		if history[i].Metadata.Source == "input" && history[i].Metadata.Step == -1 {
			firstInput = &history[i]
			break
		}
	}
	if firstInput == nil {
		t.Fatalf("did not find run 1 input checkpoint (source=input, step=-1) in history of %d", len(history))
	}

	// The pinned run's input write re-creates the (sentinel) channel and
	// applies the seeded counter.
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{99}},
		Options{ThreadID: "t1", CheckpointID: firstInput.Checkpoint.ID}); err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}

	history, err = saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	// The pinned run's input checkpoint is the input-source checkpoint whose
	// parent is the pinned one. Its seeded-then-advanced counter must be the
	// loaded checkpoint's counters advanced by one superstep and one update
	// (the items input write).
	var forkInput *checkpoint.Tuple
	for i := range history {
		if history[i].Metadata.Source == "input" && history[i].ParentConfig != nil &&
			history[i].ParentConfig.CheckpointID == firstInput.Checkpoint.ID {
			forkInput = &history[i]
			break
		}
	}
	if forkInput == nil {
		t.Fatalf("did not find the pinned run's input checkpoint in history of %d", len(history))
	}
	want := map[string][2]int{"items": {1, 2}}
	if got := forkInput.Metadata.CountersSinceDeltaSnapshot; !reflect.DeepEqual(got, want) {
		t.Fatalf("pinned run's input checkpoint counters = %v, want %v (seeded %v then advanced once)",
			got, want, firstInput.Metadata.CountersSinceDeltaSnapshot)
	}
}

// TestUpdateStateForcesDeltaSnapshot verifies UpdateState (Source=="update")
// force-snapshots every available delta channel even when the per-channel
// cadence has NOT fired (updates < snapshotFrequency).
//
// With freq=2 the Invoke run snapshots items at its step-1 checkpoint (cadence
// 2>=2), so items=[1,2,3] survives. The UpdateState then restores items=[1,2,3]
// and appends [99] -> [1,2,3,99]. The update checkpoint's OWN cadence (updates=1
// < 2) would NOT fire, so the blob can only come from the forced snapshot gated
// on Source=="update" — this isolates the forced path from the predicate.
func TestUpdateStateForcesDeltaSnapshot(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := deltaGraphFreq(t, 2, WithCheckpointer(saver)) // freq=2: snapshots at step-1, update cadence does not fire
	ctx := t.Context()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	// UpdateState writes to the delta channel; the update checkpoint's cadence
	// (updates=1 < 2) does NOT fire, so the blob can only come from the forced
	// snapshot gated on Source=="update".
	cfg, err := cg.UpdateState(ctx, checkpoint.Config{ThreadID: "t1"},
		map[string]any{"items": []int{99}}, "n2")
	if err != nil {
		t.Fatalf("UpdateState() error = %v", err)
	}
	tup, err := saver.GetTuple(ctx, cfg)
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if tup == nil {
		t.Fatal("GetTuple() returned nil for the update checkpoint")
	}
	blob, ok := tup.Checkpoint.ChannelValues["items"]
	if !ok {
		t.Fatal("update checkpoint missing 'items' (Source==update should force a snapshot blob)")
	}
	val, isSnap := channels.UnwrapDeltaSnapshot(blob)
	if !isSnap {
		t.Fatalf("update checkpoint 'items' = %T, want a delta snapshot blob (forced)", blob)
	}
	if got, want := val.([]int), []int{1, 2, 3, 99}; !reflect.DeepEqual(got, want) {
		t.Fatalf("update checkpoint items value = %v, want %v", got, want)
	}
}
