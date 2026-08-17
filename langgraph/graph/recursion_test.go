package graph

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// newLoopGraph builds an infinite loop → loop graph compiled with the given
// recursion limit. With no exit edge, every invocation must trip the limit.
func newLoopGraph(t *testing.T, limit int) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("loop", func(runtime.Runtime, map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "loop")
	g.AddEdge("loop", "loop")
	cg, err := g.Compile(WithRecursionLimit(limit))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// asRecursionError asserts err carries a *types.GraphRecursionError and
// returns it.
func asRecursionError(t *testing.T, err error) *types.GraphRecursionError {
	t.Helper()
	if err == nil {
		t.Fatal("expected recursion limit error, got nil")
	}
	var rle *types.GraphRecursionError
	if !errors.As(err, &rle) {
		t.Fatalf("expected *types.GraphRecursionError, got %T: %v", err, err)
	}
	return rle
}

// TestGraphRecursionErrorTyped verifies the recursion failure is detectable
// via errors.As with the effective limit, and that the message text is
// unchanged from the pre-typed-error era.
func TestGraphRecursionErrorTyped(t *testing.T) {
	cg := newLoopGraph(t, 5)
	_, err := cg.Invoke(context.Background(), map[string]any{})
	rle := asRecursionError(t, err)
	if rle.Limit != 5 {
		t.Fatalf("rle.Limit = %d, want 5", rle.Limit)
	}
	if want := "graph: recursion limit (5) exceeded"; err.Error() != want {
		t.Fatalf("err.Error() = %q, want %q", err.Error(), want)
	}
	if rle.Node != "loop" {
		t.Fatalf("rle.Node = %q, want %q", rle.Node, "loop")
	}
}

// TestRecursionLimitRuntimeOverrideWins verifies a positive
// Options.RecursionLimit takes precedence over the compiled limit.
func TestRecursionLimitRuntimeOverrideWins(t *testing.T) {
	cg := newLoopGraph(t, 5)

	for _, tc := range []struct {
		name  string
		limit int
		want  int
	}{
		{name: "raise", limit: 10, want: 10},
		{name: "lower", limit: 2, want: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{RecursionLimit: tc.limit})
			rle := asRecursionError(t, err)
			if rle.Limit != tc.want {
				t.Fatalf("rle.Limit = %d, want %d", rle.Limit, tc.want)
			}
		})
	}
}

// TestRecursionLimitZeroKeepsCompiledDefault verifies zero and negative
// Options.RecursionLimit fall back to the compiled WithRecursionLimit value.
func TestRecursionLimitZeroKeepsCompiledDefault(t *testing.T) {
	cg := newLoopGraph(t, 5)

	for _, tc := range []struct {
		name string
		opts Options
	}{
		{name: "zero options", opts: Options{}},
		{name: "negative", opts: Options{RecursionLimit: -1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, tc.opts)
			rle := asRecursionError(t, err)
			if rle.Limit != 5 {
				t.Fatalf("rle.Limit = %d, want 5 (compiled default)", rle.Limit)
			}
		})
	}
}

// TestGraphRecursionErrorNodeNamesPendingNode verifies Node reports the node
// about to dispatch when the limit trips mid-graph (not just the entry node).
func TestGraphRecursionErrorNodeNamesPendingNode(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", func(runtime.Runtime, map[string]any) (any, error) { return nil, nil })
	g.AddNode("b", func(runtime.Runtime, map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", "b")
	cg, err := g.Compile(WithRecursionLimit(5))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(context.Background(), map[string]any{})
	rle := asRecursionError(t, err)
	if rle.Node != "b" {
		t.Fatalf("rle.Node = %q, want %q", rle.Node, "b")
	}
}

// TestGraphRecursionErrorNodeEmptyWhenAmbiguous verifies Node stays empty when
// the limit trips with more than one pending task (fan-out), as documented.
func TestGraphRecursionErrorNodeEmptyWhenAmbiguous(t *testing.T) {
	g := NewStateGraph()
	noop := func(runtime.Runtime, map[string]any) (any, error) { return nil, nil }
	g.AddNode("fan", noop)
	g.AddNode("a", noop)
	g.AddNode("b", noop)
	g.AddEdge(types.START, "fan")
	g.AddConditionalEdges("fan", func(runtime.Runtime, map[string]any) ([]any, error) {
		return []any{"a", "b"}, nil
	})
	g.AddEdge("a", "a")
	g.AddEdge("b", "b")
	cg, err := g.Compile(WithRecursionLimit(3))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(context.Background(), map[string]any{})
	rle := asRecursionError(t, err)
	if rle.Node != "" {
		t.Fatalf("rle.Node = %q, want empty (two pending tasks)", rle.Node)
	}
	if rle.Limit != 3 {
		t.Fatalf("rle.Limit = %d, want 3", rle.Limit)
	}
}

// TestRecursionLimitDefaultStays100 verifies the compiled default limit is
// still 100 when neither WithRecursionLimit nor an override is given.
func TestRecursionLimitDefaultStays100(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("loop", func(runtime.Runtime, map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "loop")
	g.AddEdge("loop", "loop")
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = cg.Invoke(context.Background(), map[string]any{})
	rle := asRecursionError(t, err)
	if rle.Limit != 100 {
		t.Fatalf("rle.Limit = %d, want 100 (default)", rle.Limit)
	}
}

// TestRecursionLimitInvokeStreamOverride verifies InvokeStream honors the
// per-invoke override (nil sink: the non-emitting event path).
func TestRecursionLimitInvokeStreamOverride(t *testing.T) {
	cg := newLoopGraph(t, 5)
	_, err := cg.InvokeStream(context.Background(), map[string]any{}, Options{RecursionLimit: 3}, nil)
	rle := asRecursionError(t, err)
	if rle.Limit != 3 {
		t.Fatalf("rle.Limit = %d, want 3", rle.Limit)
	}
}

// TestRecursionLimitStreamOverride verifies Stream (whose StreamOptions embed
// Options) surfaces the override failure as the final error pair.
func TestRecursionLimitStreamOverride(t *testing.T) {
	cg := newLoopGraph(t, 5)
	var lastErr error
	for _, err := range cg.Stream(context.Background(), map[string]any{}, StreamOptions{
		Options: Options{RecursionLimit: 3},
		Modes:   []StreamMode{StreamValues},
	}) {
		if err != nil {
			lastErr = err
			break
		}
	}
	rle := asRecursionError(t, lastErr)
	if rle.Limit != 3 {
		t.Fatalf("rle.Limit = %d, want 3", rle.Limit)
	}
}

// TestRecursionLimitSubgraphPropagation verifies the parent's per-invoke
// override propagates into an AddSubgraph child: the child (compiled with the
// default limit) trips the parent's small override, and the typed error
// survives the subgraph wrapping.
func TestRecursionLimitSubgraphPropagation(t *testing.T) {
	child := NewStateGraph()
	child.AddNode("loop", func(runtime.Runtime, map[string]any) (any, error) { return nil, nil })
	child.AddEdge(types.START, "loop")
	child.AddEdge("loop", "loop")
	childCg, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	parent := NewStateGraph()
	parent.AddSubgraph("sub", childCg)
	parent.AddEdge(types.START, "sub")
	parent.AddEdge("sub", types.END)
	parentCg, err := parent.Compile()
	if err != nil {
		t.Fatalf("parent Compile() error = %v", err)
	}

	_, err = parentCg.InvokeWithOptions(context.Background(), map[string]any{}, Options{RecursionLimit: 3})
	rle := asRecursionError(t, err)
	if rle.Limit != 3 {
		t.Fatalf("rle.Limit = %d, want 3 (parent override propagated to child)", rle.Limit)
	}
}

// TestRecursionLimitSubgraphPropagationCheckpointed verifies the override also
// reaches the child when the parent run is checkpointing — the path where
// invokeSubgraph rebuilds the child Options from scratch (ThreadID +
// checkpointNS) rather than leaving them zero.
func TestRecursionLimitSubgraphPropagationCheckpointed(t *testing.T) {
	child := NewStateGraph()
	child.AddNode("loop", func(runtime.Runtime, map[string]any) (any, error) { return nil, nil })
	child.AddEdge(types.START, "loop")
	child.AddEdge("loop", "loop")
	childCg, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	parent := NewStateGraph()
	parent.AddSubgraph("sub", childCg)
	parent.AddEdge(types.START, "sub")
	parent.AddEdge("sub", types.END)
	parentCg, err := parent.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("parent Compile() error = %v", err)
	}

	_, err = parentCg.InvokeWithOptions(context.Background(), map[string]any{}, Options{
		ThreadID:       "recursion-subgraph-checkpointed",
		RecursionLimit: 3,
	})
	rle := asRecursionError(t, err)
	if rle.Limit != 3 {
		t.Fatalf("rle.Limit = %d, want 3 (override survived the rebuilt child Options)", rle.Limit)
	}
}

// TestRecursionLimitNestedSubgraphPropagation verifies the override propagates
// transitively: the grandchild's invokeSubgraph reads the child run's
// execution meta, which already carries the parent's override.
func TestRecursionLimitNestedSubgraphPropagation(t *testing.T) {
	grandchild := NewStateGraph()
	grandchild.AddNode("loop", func(runtime.Runtime, map[string]any) (any, error) { return nil, nil })
	grandchild.AddEdge(types.START, "loop")
	grandchild.AddEdge("loop", "loop")
	grandchildCg, err := grandchild.Compile()
	if err != nil {
		t.Fatalf("grandchild Compile() error = %v", err)
	}

	child := NewStateGraph()
	child.AddSubgraph("inner", grandchildCg)
	child.AddEdge(types.START, "inner")
	child.AddEdge("inner", types.END)
	childCg, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	parent := NewStateGraph()
	parent.AddSubgraph("sub", childCg)
	parent.AddEdge(types.START, "sub")
	parent.AddEdge("sub", types.END)
	parentCg, err := parent.Compile()
	if err != nil {
		t.Fatalf("parent Compile() error = %v", err)
	}

	_, err = parentCg.InvokeWithOptions(context.Background(), map[string]any{}, Options{RecursionLimit: 2})
	rle := asRecursionError(t, err)
	if rle.Limit != 2 {
		t.Fatalf("rle.Limit = %d, want 2 (override propagated through nested subgraphs)", rle.Limit)
	}
}
