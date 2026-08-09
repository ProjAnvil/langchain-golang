package graph

import (
	"context"
	"reflect"
	_runtime "runtime"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestAsyncDurabilityEndToEnd verifies that a graph compiled with
// DurabilityAsync persists checkpoints correctly after invoke returns.
func TestAsyncDurabilityEndToEnd(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"x": 42}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityAsync))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatal(err)
	}

	// Checkpoint should be persisted after invoke returns (flush in defer)
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatal("expected checkpoint after async invoke")
	}
}

// TestExitDurabilityEndToEnd verifies that a graph compiled with
// DurabilityExit persists a single final checkpoint after invoke returns.
func TestExitDurabilityEndToEnd(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"x": 99}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityExit))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatal(err)
	}

	// Final checkpoint should exist after invoke returns
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatal("expected final checkpoint after exit-mode invoke")
	}
}

// TestAsyncDurabilityNoGoroutineLeak verifies no goroutine leaks across
// multiple async-mode invokes.
func TestAsyncDurabilityNoGoroutineLeak(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityAsync))
	if err != nil {
		t.Fatal(err)
	}

	before := _runtime.NumGoroutine()
	for i := 0; i < 10; i++ {
		_, err = cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
		if err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(100 * time.Millisecond) // allow cleanup
	after := _runtime.NumGoroutine()
	if after > before {
		t.Errorf("goroutine leak: before=%d after=%d", before, after)
	}
}

// stringBatchReducer is a BatchReducer that concatenates existing (a []string)
// with each update in the batch, mirroring intBatchReducer for string tokens.
// NewDeltaChannel takes a BatchReducer directly, NOT BatchFromReducer(Reducer).
func stringBatchReducer(existing any, updates []any) (any, error) {
	base, _ := existing.([]string)
	out := make([]string, len(base))
	copy(out, base)
	for _, u := range updates {
		add, _ := u.([]string)
		out = append(out, add...)
	}
	return out, nil
}

// exitDeltaGraph builds a single-node graph whose "msgs" channel is a
// DeltaChannel with snapshotFrequency=100, so it NEVER snapshots mid-loop and
// forces state reconstruction via the checkpoint ancestor-write walk. The
// "appender" node appends a constant token on every run, so each turn
// contributes both an input delta write and a node delta write.
func exitDeltaGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("msgs", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddNode("appender", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"msgs": []string{"node"}}, nil
	})
	g.AddEdge(types.START, "appender")
	g.AddEdge("appender", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestExitDurabilityDeltaChannelMultiTurn verifies the flushExit anchor fix: in
// exit mode, a second turn's accumulated delta writes must anchor on the
// PERSISTED parent checkpoint (initialCfg) — not the not-yet-persisted final
// checkpoint (flushCfg) — so they survive and GetState can reconstruct BOTH
// turns via the ancestor-write walk.
//
// snapshotFrequency=100 ensures "msgs" is never force-snapshotted, so every
// value lives only as per-checkpoint delta writes — the exact path the anchor
// bug corrupted. Before the fix, turn 2's flushExit PutWrites'd against a
// non-existent checkpoint (the in-memory-only flushCfg), which errored and
// aborted before the final checkpoint was saved; GetState then returned only
// turn 1's state. With the error-surfacing fix (named return in run), that
// PutWrites failure is now returned from Invoke itself.
func TestExitDurabilityDeltaChannelMultiTurn(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := exitDeltaGraph(t, WithCheckpointer(saver), WithDurability(DurabilityExit))
	ctx := context.Background()

	// Turn 1: input delta write ["t1","turn"]; node appends ["node"].
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"msgs": []string{"t1", "turn"}}, Options{ThreadID: "multi"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}

	// Turn 2: input delta write ["t2","turn"]; node appends ["node"] again.
	// A new turn loads the turn-1 checkpoint as its persisted parent, so this
	// flush is the one that exercises the persisted-parent anchor.
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"msgs": []string{"t2", "turn"}}, Options{ThreadID: "multi"}); err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}

	// GetState reconstructs "msgs" by walking the checkpoint parent chain and
	// replaying ancestor delta writes. Both turns must be present.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "multi"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	msgs, ok := snap.Values["msgs"]
	if !ok {
		t.Fatal("GetState().Values missing 'msgs'")
	}
	got, ok := msgs.([]string)
	if !ok {
		t.Fatalf("msgs is %T, want []string", msgs)
	}
	// Replay order: turn-1 input, turn-1 node, turn-2 input, turn-2 node.
	// Missing turn-2 tokens means the anchor bug lost the second turn's writes.
	want := []string{"t1", "turn", "node", "t2", "turn", "node"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("msgs = %v, want %v (turn-2 writes lost = anchor bug)", got, want)
	}
}
