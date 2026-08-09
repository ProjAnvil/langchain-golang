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

// TestDeltaChannelPerTaskWritesReconstruction verifies that with
// snapshotFrequency > 1 (cadence never firing in two supersteps), GetState
// reconstructs the delta channel purely from per-task writes persisted in the
// normal commit flow — no checkpoint carries a snapshot blob, so the value can
// only come from the ancestor-write walk over each task's PutWrites.
//
// This is the load-bearing assertion for the per-task writes added in Task 4:
// before that change the normal flow persisted no per-task writes, and with the
// DeltaChannel.Checkpoint sentinel flip (no forced first snapshot) the value
// would be empty.
func TestDeltaChannelPerTaskWritesReconstruction(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	// freq=10: cadence never fires across the two supersteps (n1, n2).
	cg := deltaGraphFreq(t, 10, WithCheckpointer(saver))
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	// No checkpoint should carry a snapshot blob for "items" (cadence 10 never
	// fired), proving reconstruction must come from per-task writes.
	history, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(history) == 0 {
		t.Fatal("expected checkpoints, got none")
	}
	for i, h := range history {
		if blob, ok := h.Checkpoint.ChannelValues["items"]; ok {
			t.Fatalf("checkpoint %d (source=%s step=%d) unexpectedly carries an items blob %v — cadence should not have fired", i, h.Metadata.Source, h.Metadata.Step, blob)
		}
	}

	// GetState must reconstruct items=[1,2,3] via the per-task-write ancestor walk.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	got, ok := snap.Values["items"].([]int)
	if !ok {
		t.Fatalf("GetState().Values['items'] = %v (%T), want []int{1,2,3}", snap.Values["items"], snap.Values["items"])
	}
	if want := []int{1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("GetState().Values['items'] = %v, want %v", got, want)
	}
	// The non-delta channel ("msg", a LastValue) is stored in ChannelValues and
	// needs no reconstruction; assert it round-trips too.
	if got, want := snap.Values["msg"], "hello"; got != want {
		t.Fatalf("GetState().Values['msg'] = %v, want %q", got, want)
	}
}

// TestDeltaChannelInputWritesReconstruction verifies that delta-channel INPUT
// writes are persisted (NullTaskID) so GetState reconstructs them across turns
// even when every checkpoint stores the channel as a sentinel (cadence never
// fires). A single-node graph appends a constant on each turn; the accumulated
// value (input writes + node writes) is rebuilt by the ancestor walk.
func TestDeltaChannelInputWritesReconstruction(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	// BatchReducer passed directly to NewDeltaChannel (C4); freq=10 so cadence
	// never fires across the turns below.
	g.AddChannel("items", channels.NewDeltaChannel(
		func(existing any, updates []any) (any, error) {
			base, _ := existing.([]int)
			out := make([]int, len(base))
			copy(out, base)
			for _, u := range updates {
				out = append(out, u.([]int)...)
			}
			return out, nil
		},
		func() any { return []int{} },
		10,
	))
	g.AddNode("n1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"items": []int{10}}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	// Turn 1 (fresh): input writes items=[1]; n1 appends [10] -> [1,10].
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{1}}, Options{ThreadID: "t2"}); err != nil {
		t.Fatalf("invoke turn 1: %v", err)
	}
	// Turn 2 (new turn): input writes items=[2]; n1 appends [10] -> [2,10].
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{2}}, Options{ThreadID: "t2"}); err != nil {
		t.Fatalf("invoke turn 2: %v", err)
	}

	// GetState reconstructs the full accumulated history across both turns:
	// turn-1 input [1], turn-1 node [10], turn-2 input [2], turn-2 node [10].
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t2"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	got, ok := snap.Values["items"].([]int)
	if !ok {
		t.Fatalf("GetState().Values['items'] = %v (%T), want []int{1,10,2,10}", snap.Values["items"], snap.Values["items"])
	}
	if want := []int{1, 10, 2, 10}; !reflect.DeepEqual(got, want) {
		t.Fatalf("GetState().Values['items'] = %v, want %v", got, want)
	}
}
