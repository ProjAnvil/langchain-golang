package graph

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// getTupleCountingSaver wraps a Saver counting GetTuple calls, to prove the
// never-written delta-channel short-circuit avoids the ancestor walk
// (Python langgraph #9141).
type getTupleCountingSaver struct {
	checkpoint.Saver
	getTuples int
}

func (c *getTupleCountingSaver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	c.getTuples++
	return c.Saver.GetTuple(ctx, cfg)
}

// TestReconstructSkipsNeverWrittenDeltaChannels builds a graph whose only
// delta channel is NEVER written, across several supersteps, then GetState:
// with the ChannelVersions short-circuit the ancestor walk never starts
// (exactly one GetTuple — the GetState read itself); without it, the walk
// exhausts the whole chain (Python langgraph #9141).
func TestReconstructSkipsNeverWrittenDeltaChannels(t *testing.T) {
	inner := checkpoint.NewMemorySaver()
	saver := &getTupleCountingSaver{Saver: inner}
	g := NewStateGraph()
	g.AddChannel("phantom", channels.NewDeltaChannel(appendInts, func() any { return []int{} }, 100))
	g.AddNode("n1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"plain": "v"}, nil
	})
	g.AddNode("n2", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"plain": "v2"}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := t.Context()
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	history, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil || len(history) < 3 {
		t.Fatalf("expected >=3 checkpoints, got %d (err=%v)", len(history), err)
	}

	saver.getTuples = 0
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if _, ok := snap.Values["phantom"]; ok {
		t.Fatalf("phantom channel must stay absent, got %v", snap.Values["phantom"])
	}
	if saver.getTuples != 1 {
		t.Fatalf("GetState issued %d GetTuple calls, want 1 (the read itself; never-written channel must not walk history of %d)",
			saver.getTuples, len(history))
	}
}
