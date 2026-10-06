package graph

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// appendInts is the BatchReducer used by the hydration tests' delta channel.
func appendInts(existing any, updates []any) (any, error) {
	base, _ := existing.([]int)
	out := make([]int, len(base))
	copy(out, base)
	for _, u := range updates {
		out = append(out, u.([]int)...)
	}
	return out, nil
}

// hydrationGraph builds a one-node graph with an items delta channel (freq=10,
// cadence never fires). The node records the value it OBSERVES on entry and
// appends [10], so a test can assert exactly what the run state contained.
func hydrationGraph(t *testing.T, saver checkpoint.Saver, observed *[]int) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("items", channels.NewDeltaChannel(appendInts, func() any { return []int{} }, 10))
	g.AddNode("n1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		if v, ok := state["items"].([]int); ok {
			*observed = append([]int(nil), v...)
		} else {
			*observed = nil
		}
		return map[string]any{"items": []int{10}}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestDeltaChannelNewTurnRunPathHydration verifies the run path (new turn)
// hydrates sentinel delta channels from the ancestor history before applying
// the input, so the node observes the ACCUMULATED value ([1,10,2]) instead
// of starting from empty ([2]) — Python langgraph resume semantics (#8548,
// #9170).
func TestDeltaChannelNewTurnRunPathHydration(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var observed []int
	cg := hydrationGraph(t, saver, &observed)
	ctx := t.Context()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{1}}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("invoke turn 1: %v", err)
	}
	if want := []int{1}; !reflect.DeepEqual(observed, want) {
		t.Fatalf("turn 1 observed %v, want %v", observed, want)
	}

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{2}}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("invoke turn 2: %v", err)
	}
	if want := []int{1, 10, 2}; !reflect.DeepEqual(observed, want) {
		t.Fatalf("turn 2 observed %v, want %v (accumulated across turns)", observed, want)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if want := []int{1, 10, 2, 10}; !reflect.DeepEqual(snap.Values["items"], any(want)) {
		t.Fatalf("final items = %v, want %v", snap.Values["items"], want)
	}
}

// TestDeltaChannelResumeRunPathHydration verifies resume (nil input after an
// interrupt) hydrates the accumulated delta value into the run state before
// the resumed node executes.
func TestDeltaChannelResumeRunPathHydration(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	observed := []int{}
	g := NewStateGraph()
	g.AddChannel("items", channels.NewDeltaChannel(appendInts, func() any { return []int{} }, 10))
	g.AddNode("ask", func(_ runtime.Runtime, state map[string]any) (any, error) {
		if v, ok := state["items"].([]int); ok {
			observed = append([]int(nil), v...)
		}
		return map[string]any{"items": []int{10}}, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("ask"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := t.Context()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{1}}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("invoke turn 1: %v", err)
	}
	// Second turn interrupted BEFORE the node runs: input [2] applied on top
	// of the accumulated [1,10]; resume must re-enter the node seeing [1,10,2].
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{2}}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("invoke turn 2: %v", err)
	}
	if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("resume: %v", err)
	}
	// The resumed node observes the hydrated accumulated input ([1,2] — the
	// node itself never ran before the interrupt, so no [10] appends exist
	// yet), appends [10], and the final state is [1,2,10].
	if want := []int{1, 2}; !reflect.DeepEqual(observed, want) {
		t.Fatalf("resumed node observed %v, want %v", observed, want)
	}
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if want := []int{1, 2, 10}; !reflect.DeepEqual(snap.Values["items"], any(want)) {
		t.Fatalf("final items = %v, want %v", snap.Values["items"], want)
	}
}

// TestUpdateStatePreservesUntouchedDeltaChannel verifies an update that does
// not touch a sentinel delta channel keeps the channel's accumulated value in
// the update checkpoint (it must not vanish), mirroring Python #9165/#9142.
func TestUpdateStatePreservesUntouchedDeltaChannel(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var observed []int
	cg := hydrationGraph(t, saver, &observed)
	ctx := t.Context()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{1}}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	// Update a DIFFERENT key: the delta channel must survive with [1,10].
	cfg, err := cg.UpdateState(ctx, checkpoint.Config{ThreadID: "t1"}, map[string]any{"other": "x"}, "n1")
	if err != nil {
		t.Fatalf("UpdateState: %v", err)
	}
	snap, err := cg.GetState(ctx, cfg)
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if want := []int{1, 10}; !reflect.DeepEqual(snap.Values["items"], any(want)) {
		t.Fatalf("update checkpoint items = %v, want %v", snap.Values["items"], want)
	}
}

// TestUpdateStateDeltaKeyIncludesAccumulatedValue verifies an update that
// writes a delta key forces a snapshot containing the accumulated ancestor
// value PLUS the update — not the update alone (Python #9165/#9142).
func TestUpdateStateDeltaKeyIncludesAccumulatedValue(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var observed []int
	cg := hydrationGraph(t, saver, &observed)
	ctx := t.Context()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{1}}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	// [1] + node [10] accumulated. An update writing items=[7] must snapshot
	// [1,10,7].
	cfg, err := cg.UpdateState(ctx, checkpoint.Config{ThreadID: "t1"}, map[string]any{"items": []int{7}}, "n1")
	if err != nil {
		t.Fatalf("UpdateState: %v", err)
	}
	snap, err := cg.GetState(ctx, cfg)
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if want := []int{1, 10, 7}; !reflect.DeepEqual(snap.Values["items"], any(want)) {
		t.Fatalf("update checkpoint items = %v, want %v", snap.Values["items"], want)
	}
}
