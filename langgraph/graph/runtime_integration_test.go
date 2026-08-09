package graph

// M1.1 acceptance tests for the Runtime convergence: verify the executor's
// buildRuntime populates Runtime.Context from the context_schema values the
// caller attached via runtime.ContextWithValues, and that a node receives a
// runtime.Runtime that satisfies context.Context (so the existing ctx
// idioms — Interrupt(ctx, ...), StreamWriterFromContext(ctx), etc. — still
// work).

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestBuildRuntimeContextSchemaValues is the M1.1 acceptance test: a caller
// attaches context_schema values to the Invoke context via
// runtime.ContextWithValues; the executor's buildRuntime surfaces them on
// Runtime.Context; the node reads them via rt.Context and via
// runtime.ValueFromRuntime. It also verifies rt.Value(runtime's key) still
// reaches the same map via context.Context delegation (the deprecated
// agents.ContextValue path survives).
func TestBuildRuntimeContextSchemaValues(t *testing.T) {
	g := NewStateGraph()

	var observedContext any
	var observedValue any
	var observedValueOK bool
	var observedExecInfo *runtime.ExecutionInfo

	g.AddNode("n", func(rt runtime.Runtime, state map[string]any) (any, error) {
		observedContext = rt.Context
		observedValue, observedValueOK = runtime.ValueFromRuntime(rt, "user_id")
		observedExecInfo = rt.ExecutionInfo
		return nil, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile error: %v", err)
	}

	ctx := runtime.ContextWithValues(context.Background(), map[string]any{
		"user_id": "u123",
		"tenant":  "acme",
	})
	if _, err := cg.Invoke(ctx, map[string]any{}); err != nil {
		t.Fatalf("Invoke error: %v", err)
	}

	m, ok := observedContext.(map[string]any)
	if !ok {
		t.Fatalf("rt.Context is %T, want map[string]any", observedContext)
	}
	if got, want := m["user_id"], "u123"; got != want {
		t.Errorf(`rt.Context["user_id"] = %v, want %v`, got, want)
	}
	if got, want := m["tenant"], "acme"; got != want {
		t.Errorf(`rt.Context["tenant"] = %v, want %v`, got, want)
	}
	if !observedValueOK {
		t.Errorf("ValueFromRuntime returned ok=false, want true")
	}
	if observedValue != "u123" {
		t.Errorf("ValueFromRuntime = %v, want u123", observedValue)
	}
	// ExecutionInfo is always populated by buildRuntime (even without a
	// checkpointer); with no checkpointer, ThreadID/CheckpointID are empty
	// and NodeAttempt defaults to 1.
	if observedExecInfo == nil {
		t.Fatalf("rt.ExecutionInfo = nil, want a non-nil value")
	}
	if got, want := observedExecInfo.NodeAttempt, 1; got != want {
		t.Errorf("ExecutionInfo.NodeAttempt = %d, want %d", got, want)
	}
}

// TestBuildRuntimeWithoutContextValues verifies a node still runs cleanly
// when no context_schema values are attached: rt.Context is nil and
// ValueFromRuntime returns (nil, false).
func TestBuildRuntimeWithoutContextValues(t *testing.T) {
	g := NewStateGraph()

	var observedContext any
	var observedOK bool

	g.AddNode("n", func(rt runtime.Runtime, state map[string]any) (any, error) {
		observedContext = rt.Context
		_, observedOK = runtime.ValueFromRuntime(rt, "anything")
		return nil, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile error: %v", err)
	}

	if _, err := cg.Invoke(context.Background(), map[string]any{}); err != nil {
		t.Fatalf("Invoke error: %v", err)
	}
	if observedContext != nil {
		t.Errorf("rt.Context = %v, want nil when no values attached", observedContext)
	}
	if observedOK {
		t.Errorf("ValueFromRuntime returned ok=true, want false when no values attached")
	}
}

// TestRuntimeSatisfiesContextInNode verifies a runtime.Runtime received by a
// node satisfies context.Context: Deadline/Done/Err/Value all delegate to
// the backing ctx, so existing ctx idioms survive the M1.1 signature change.
func TestRuntimeSatisfiesContextInNode(t *testing.T) {
	g := NewStateGraph()

	type testKey struct{}
	var doneCh <-chan struct{}
	var errVal error
	var val any

	g.AddNode("n", func(rt runtime.Runtime, state map[string]any) (any, error) {
		// rt satisfies context.Context: these are context.Context methods.
		doneCh = rt.Done()
		errVal = rt.Err()
		val = rt.Value(testKey{})
		return nil, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile error: %v", err)
	}

	ctx := context.WithValue(context.Background(), testKey{}, "hello")
	if _, err := cg.Invoke(ctx, map[string]any{}); err != nil {
		t.Fatalf("Invoke error: %v", err)
	}
	if doneCh != nil {
		t.Errorf("rt.Done() = non-nil, want nil for Background-derived ctx")
	}
	if errVal != nil {
		t.Errorf("rt.Err() = %v, want nil for active ctx", errVal)
	}
	if val != "hello" {
		t.Errorf("rt.Value(testKey{}) = %v, want %q (delegation)", val, "hello")
	}
}
