package graph

import (
	"context"
	_runtime "runtime"
	"testing"
	"time"

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
