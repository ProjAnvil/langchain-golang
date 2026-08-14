package graph

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// bulkUpdateGraph builds the small a -> b -> END graph used by the
// bulk-update tests. asNode "a" is a registered node so updates can be
// attributed to it explicitly.
func bulkUpdateGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestBulkUpdateStateMultipleSupersteps verifies that two supersteps, each
// with one update, step the checkpoint forward twice and that the final state
// reflects both updates.
func TestBulkUpdateStateMultipleSupersteps(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := bulkUpdateGraph(t, saver)
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	base, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}

	finalCfg, err := cg.BulkUpdateState(ctx, checkpoint.Config{ThreadID: "t1"}, [][]BulkUpdate{
		{{Values: map[string]any{"k1": "v1"}, AsNode: "a"}},
		{{Values: map[string]any{"k2": "v2"}, AsNode: "a"}},
	})
	if err != nil {
		t.Fatalf("BulkUpdateState() error = %v", err)
	}
	if finalCfg.CheckpointID == "" || finalCfg.CheckpointID == base.Config.CheckpointID {
		t.Fatalf("BulkUpdateState() returned Config %+v, want a fresh checkpoint ID", finalCfg)
	}

	snap, err := cg.GetState(ctx, finalCfg)
	if err != nil {
		t.Fatalf("GetState() of final bulk checkpoint error = %v", err)
	}
	if snap.Values["k1"] != "v1" {
		t.Fatalf("final Values[k1] = %v, want v1", snap.Values["k1"])
	}
	if snap.Values["k2"] != "v2" {
		t.Fatalf("final Values[k2] = %v, want v2", snap.Values["k2"])
	}
	if snap.Metadata.Source != "update" {
		t.Fatalf("final Metadata.Source = %q, want update", snap.Metadata.Source)
	}
	if snap.Metadata.Step != base.Metadata.Step+2 {
		t.Fatalf("final Metadata.Step = %d, want %d (base %d stepped forward twice)",
			snap.Metadata.Step, base.Metadata.Step+2, base.Metadata.Step)
	}
}

// TestBulkUpdateStateMultipleUpdatesInSuperstep verifies that a single
// superstep with two updates applies them sequentially (two checkpoint
// writes), and that the reserved TaskID field is accepted without affecting
// the result.
func TestBulkUpdateStateMultipleUpdatesInSuperstep(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := bulkUpdateGraph(t, saver)
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	base, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}

	finalCfg, err := cg.BulkUpdateState(ctx, checkpoint.Config{ThreadID: "t1"}, [][]BulkUpdate{
		{
			{Values: map[string]any{"x": "first"}, AsNode: "a"},
			{Values: map[string]any{"y": "second"}, AsNode: "a", TaskID: "reserved-not-used"},
		},
	})
	if err != nil {
		t.Fatalf("BulkUpdateState() error = %v", err)
	}

	snap, err := cg.GetState(ctx, finalCfg)
	if err != nil {
		t.Fatalf("GetState() of final bulk checkpoint error = %v", err)
	}
	if snap.Values["x"] != "first" {
		t.Fatalf("final Values[x] = %v, want first", snap.Values["x"])
	}
	if snap.Values["y"] != "second" {
		t.Fatalf("final Values[y] = %v, want second", snap.Values["y"])
	}
	if snap.Metadata.Step != base.Metadata.Step+2 {
		t.Fatalf("final Metadata.Step = %d, want %d (two sequential updates)", snap.Metadata.Step, base.Metadata.Step+2)
	}
}

// TestBulkUpdateStateEmptyInput verifies empty supersteps and empty inner
// supersteps are rejected.
func TestBulkUpdateStateEmptyInput(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := bulkUpdateGraph(t, saver)
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	cfg := checkpoint.Config{ThreadID: "t1"}

	if _, err := cg.BulkUpdateState(ctx, cfg, [][]BulkUpdate{}); err == nil ||
		!strings.Contains(err.Error(), "no supersteps") {
		t.Fatalf("BulkUpdateState() with no supersteps error = %v, want a \"no supersteps\" error", err)
	}
	if _, err := cg.BulkUpdateState(ctx, cfg, [][]BulkUpdate{{}}); err == nil ||
		!strings.Contains(err.Error(), "no updates") {
		t.Fatalf("BulkUpdateState() with an empty superstep error = %v, want a \"no updates\" error", err)
	}
}

// TestBulkUpdateStateRequiresCheckpointer verifies BulkUpdateState fails
// clearly without a checkpointer.
func TestBulkUpdateStateRequiresCheckpointer(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	if _, err := cg.BulkUpdateState(context.Background(), checkpoint.Config{ThreadID: "t1"},
		[][]BulkUpdate{{{Values: map[string]any{"x": 1}, AsNode: "a"}}}); err == nil ||
		!strings.Contains(err.Error(), "checkpointer") {
		t.Fatalf("BulkUpdateState() error = %v, want a checkpointer error", err)
	}
}
