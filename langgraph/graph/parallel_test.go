package graph

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestSuperstepMaxConcurrencyBoundsParallelNodes runs a wide single-superstep
// fan-out and asserts Options.MaxConcurrency actually caps how many nodes
// execute at once (Python parity: Pregel's max_concurrency executor cap).
func TestSuperstepMaxConcurrencyBoundsParallelNodes(t *testing.T) {
	const branches = 12
	const limit = 3

	var inFlight, peak atomic.Int32
	g := NewStateGraph()
	g.AddNode("seed", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{}, nil
	})
	g.AddEdge(types.START, "seed")
	for b := 0; b < branches; b++ {
		name := "leaf" + string(rune('A'+b))
		g.AddNode(name, func(_ runtime.Runtime, _ map[string]any) (any, error) {
			cur := inFlight.Add(1)
			for {
				p := peak.Load()
				if cur <= p || peak.CompareAndSwap(p, cur) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			inFlight.Add(-1)
			return map[string]any{}, nil
		})
		g.AddEdge("seed", name)
		g.AddEdge(name, types.END)
	}
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	if _, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{MaxConcurrency: limit}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if got := peak.Load(); got > limit {
		t.Fatalf("peak in-flight nodes = %d, want <= %d", got, limit)
	}
	if got := peak.Load(); got < 2 {
		t.Fatalf("peak in-flight nodes = %d, want parallelism > 1 (bound must not serialize)", got)
	}
}

// TestSuperstepUnboundedByDefault asserts the default bound still allows a
// modest fan-out to run fully parallel (the pool is a cap, not a throttle).
func TestSuperstepUnboundedByDefault(t *testing.T) {
	var inFlight, peak atomic.Int32
	g := NewStateGraph()
	g.AddNode("seed", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{}, nil
	})
	g.AddEdge(types.START, "seed")
	for b := 0; b < 6; b++ {
		name := "leaf" + string(rune('A'+b))
		g.AddNode(name, func(_ runtime.Runtime, _ map[string]any) (any, error) {
			cur := inFlight.Add(1)
			for {
				p := peak.Load()
				if cur <= p || peak.CompareAndSwap(p, cur) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			inFlight.Add(-1)
			return map[string]any{}, nil
		})
		g.AddEdge("seed", name)
		g.AddEdge(name, types.END)
	}
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	if _, err := compiled.InvokeWithOptions(context.Background(), map[string]any{}, Options{}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if got := peak.Load(); got != 6 {
		t.Fatalf("peak in-flight nodes = %d, want 6 (small fan-out runs fully parallel)", got)
	}
}
