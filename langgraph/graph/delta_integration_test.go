package graph

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// intBatchReducer is a BatchReducer that appends int-slice updates to the
// existing value, mirroring operator.add for lists in a batch form.
func intBatchReducer(existing any, updates []any) (any, error) {
	base, _ := existing.([]int)
	out := make([]int, len(base))
	copy(out, base)
	for _, u := range updates {
		add, _ := u.([]int)
		out = append(out, add...)
	}
	return out, nil
}

// deltaGraph builds n1 -> n2 where both nodes append to the "items" key backed
// by a DeltaChannel, and "msg" is a plain last-value key. snapshotFrequency=1
// (always snapshot) is used because the current Go executor does not persist
// per-task writes in the normal flow, so sentinel-only storage between
// snapshots would lose data without ancestor-write replay support.
func deltaGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("items", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 1))
	g.AddNode("n1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"items": []int{1, 2}, "msg": "hello"}, nil
	})
	g.AddNode("n2", func(_ runtime.Runtime, state map[string]any) (any, error) {
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

// TestStreamDeltaMode (baseline 9): delta stream mode emits incremental state
// diffs — the first chunk is the full initial state, subsequent chunks carry
// only the keys that changed.
func TestStreamDeltaMode(t *testing.T) {
	cg := deltaGraph(t)
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{},
		StreamOptions{Modes: []StreamMode{StreamDelta}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("got 0 delta chunks, want at least 1")
	}
	for _, c := range chunks {
		if c.Mode != StreamDelta {
			t.Fatalf("chunk mode = %q, want delta", c.Mode)
		}
	}
	// The first delta chunk should include the initial writes (msg + items from n1).
	// n2 only changes items, so its delta chunk should only carry items.
	lastChunk := chunks[len(chunks)-1]
	payload, ok := lastChunk.Payload.(map[string]any)
	if !ok {
		t.Fatalf("last delta payload is %T, want map[string]any", lastChunk.Payload)
	}
	// The final delta chunk should include items (changed by n2) but not msg
	// (unchanged by n2). This verifies the incremental diff.
	if _, hasMsg := payload["msg"]; hasMsg {
		t.Fatalf("last delta chunk includes unchanged key 'msg': %+v", payload)
	}
	if _, hasItems := payload["items"]; !hasItems {
		t.Fatalf("last delta chunk missing changed key 'items': %+v", payload)
	}
}

// TestStreamDeltaValuesCoexist: delta and values modes can be requested
// together; each emits independently.
func TestStreamDeltaValuesCoexist(t *testing.T) {
	cg := deltaGraph(t)
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{},
		StreamOptions{Modes: []StreamMode{StreamValues, StreamDelta}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	var valuesCount, deltaCount int
	for _, c := range chunks {
		switch c.Mode {
		case StreamValues:
			valuesCount++
		case StreamDelta:
			deltaCount++
		}
	}
	if valuesCount == 0 {
		t.Fatal("got 0 values chunks, want at least 1")
	}
	if deltaCount == 0 {
		t.Fatal("got 0 delta chunks, want at least 1")
	}
}

// TestDeltaChannelSnapshotReconstruction (baselines 1, 4, 7, 8): a graph using
// a DeltaChannel persists state correctly; GetState reconstructs the delta
// channel value from the checkpoint (which forces a snapshot on first write).
func TestDeltaChannelSnapshotReconstruction(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := deltaGraph(t, WithCheckpointer(saver))
	ctx := context.Background()

	result, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	items, ok := result.Values["items"]
	if !ok {
		t.Fatal("result missing 'items' key")
	}
	if got, want := items, []int{1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("result items = %v, want %v", got, want)
	}

	// GetState must reconstruct the delta channel value.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	gotItems, ok := snap.Values["items"]
	if !ok {
		t.Fatal("GetState().Values missing 'items'")
	}
	if !reflect.DeepEqual(gotItems, []int{1, 2, 3}) {
		t.Fatalf("GetState().Values['items'] = %v, want [1 2 3]", gotItems)
	}
}

// TestDeltaChannelGetStateHistory: history traversal reconstructs delta
// channel values at each checkpoint.
func TestDeltaChannelGetStateHistory(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := deltaGraph(t, WithCheckpointer(saver))
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	history, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	if len(history) == 0 {
		t.Fatal("GetStateHistory() returned 0 snapshots")
	}

	// The latest snapshot should have the full accumulated items.
	latest := history[0]
	if got, ok := latest.Values["items"]; ok {
		if !reflect.DeepEqual(got, []int{1, 2, 3}) {
			t.Fatalf("latest history items = %v, want [1 2 3]", got)
		}
	}
}

// TestDeltaChannelUpdateState (baseline 7): updateState works with delta
// channels; metadata survives the reconstruction.
func TestDeltaChannelUpdateState(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := deltaGraph(t, WithCheckpointer(saver))
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	// UpdateState with a delta-channel key.
	cfg, err := cg.UpdateState(ctx, checkpoint.Config{ThreadID: "t1"},
		map[string]any{"items": []int{99}}, "n2")
	if err != nil {
		t.Fatalf("UpdateState() error = %v", err)
	}
	if cfg.CheckpointID == "" {
		t.Fatal("UpdateState() returned empty CheckpointID")
	}

	// GetState must reflect the update.
	snap, err := cg.GetState(ctx, cfg)
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	// The update checkpoint's source should be "update".
	if snap.Metadata.Source != "update" {
		t.Fatalf("Metadata.Source = %q, want 'update'", snap.Metadata.Source)
	}
}

// TestDeltaChannelSnapshotUnwrapping: verifies that delta snapshot blobs in
// ChannelValues are unwrapped to their plain values in the StateSnapshot.
func TestDeltaChannelSnapshotUnwrapping(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := deltaGraph(t, WithCheckpointer(saver))
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	// The value should be a plain []int, not a deltaSnapshot blob.
	items := snap.Values["items"]
	if _, isMap := items.(map[string]any); isMap {
		t.Fatalf("items is a map (unwrapped blob?), want []int: %+v", items)
	}
	got, ok := items.([]int)
	if !ok {
		t.Fatalf("items is %T, want []int", items)
	}
	if !reflect.DeepEqual(got, []int{1, 2, 3}) {
		t.Fatalf("items = %v, want [1 2 3]", got)
	}
}
