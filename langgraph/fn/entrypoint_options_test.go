package fn

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
)

// TestEntrypointCachePolicy mirrors Python's @entrypoint(cache_policy=...)
// (func/__init__.py:443, installed on the internal Pregel at :606): the
// workflow's own result is cached; a second identical invocation does not
// re-execute the function (hit/miss counting like test_pregel.py:5745).
func TestEntrypointCachePolicy(t *testing.T) {
	cache := checkpoint.NewInMemoryCache()
	var calls atomic.Int32
	e, err := NewEntrypoint[int, int, int](
		EntrypointOpts{Cache: cache, CachePolicy: &graph.CachePolicy{}},
		func(_ runtime.Runtime, in int, _ int, _ bool) (int, error) {
			calls.Add(1)
			return in * 2, nil
		},
	)
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}
	ctx := context.Background()

	v, err := e.Invoke(ctx, 5, graph.Options{})
	if err != nil || v != 10 {
		t.Fatalf("first Invoke = (%v, %v), want (10, nil)", v, err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1", got)
	}
	// Same input: served from the cache, the function does not re-run.
	v, err = e.Invoke(ctx, 5, graph.Options{})
	if err != nil || v != 10 {
		t.Fatalf("cached Invoke = (%v, %v), want (10, nil)", v, err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1 (second invoke served from cache)", got)
	}
	// Different input: cache key differs (the __start__ input channel is part
	// of the node-cache key input, graph.go:1054-1062), so it executes.
	v, err = e.Invoke(ctx, 7, graph.Options{})
	if err != nil || v != 14 {
		t.Fatalf("new-input Invoke = (%v, %v), want (14, nil)", v, err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls = %d, want 2 (new input missed the cache)", got)
	}
}

// TestEntrypointCachePolicyWithoutBackend: a policy with no Cache backend is
// inert (the graph's "policy without backend is inert" rule,
// policy.go:222-223).
func TestEntrypointCachePolicyWithoutBackend(t *testing.T) {
	var calls atomic.Int32
	e, err := NewEntrypoint[int, int, int](
		EntrypointOpts{CachePolicy: &graph.CachePolicy{}},
		func(_ runtime.Runtime, in int, _ int, _ bool) (int, error) {
			calls.Add(1)
			return in, nil
		},
	)
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}
	for i := 0; i < 2; i++ {
		if _, err := e.Invoke(context.Background(), 5, graph.Options{}); err != nil {
			t.Fatalf("Invoke %d: %v", i, err)
		}
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls = %d, want 2 (no backend, no caching)", got)
	}
}

// TestEntrypointTimeout mirrors Python's @entrypoint(timeout=...)
// (func/__init__.py:445): the whole workflow attempt is capped; Go surfaces
// the expiry as context.DeadlineExceeded (graph.TimeoutPolicy semantics,
// timeout_test.go:30-49 — Python's NodeTimeoutError analogue in this port).
func TestEntrypointTimeout(t *testing.T) {
	e, err := NewEntrypoint[int, int, int](
		EntrypointOpts{Timeout: &graph.TimeoutPolicy{RunTimeout: 50 * time.Millisecond}},
		func(rt runtime.Runtime, in int, _ int, _ bool) (int, error) {
			select {
			case <-time.After(time.Second):
				return in, nil
			case <-rt.Done():
				return 0, rt.Err()
			}
		},
	)
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}
	start := time.Now()
	_, err = e.Invoke(context.Background(), 1, graph.Options{})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Invoke error = %v, want context.DeadlineExceeded", err)
	}
	if d := time.Since(start); d > 500*time.Millisecond {
		t.Fatalf("timeout did not fire promptly: %v", d)
	}
}

// TestEntrypointContextSchema mirrors Python's @entrypoint(context_schema=...)
// (test_pregel.py:1284 uses a TypedDict with a required "model" key): the
// run-scoped context is validated before the function runs; a validation
// failure fails the invoke, a passing context is visible on rt.Context.
func TestEntrypointContextSchema(t *testing.T) {
	// Nil-tolerant schema: with no context attached, ValuesFromContext
	// returns nil (runtime.go:482-488) and buildRuntime leaves rt.Context
	// unset (graph.go:1777-1779), so the schema receives a nil any rather
	// than a map. The comma-ok assertion yields a nil map — a nil-map lookup
	// is safe in Go — so the missing-key branch fires instead of a
	// type-mismatch error.
	schema := func(ctx any) error {
		m, _ := ctx.(map[string]any)
		if _, ok := m["model"]; !ok {
			return errors.New(`missing required key "model"`)
		}
		return nil
	}
	var sawModel any
	e, err := NewEntrypoint[int, int, int](
		EntrypointOpts{ContextSchema: schema},
		func(rt runtime.Runtime, in int, _ int, _ bool) (int, error) {
			if m, ok := rt.Context.(map[string]any); ok {
				sawModel = m["model"]
			}
			return in, nil
		},
	)
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	// No context attached → rt.Context is nil → the nil-tolerant schema sees
	// an empty map and fails with the missing-key error.
	_, err = e.Invoke(context.Background(), 1, graph.Options{})
	if err == nil || !strings.Contains(err.Error(), `missing required key "model"`) {
		t.Fatalf("Invoke without context: error = %v, want context_schema validation failure", err)
	}

	// Valid context attached via runtime.ContextWithValues → runs, rt.Context
	// carries the values (buildRuntime surfaces them, graph.go:1777-1779).
	ctx := runtime.ContextWithValues(context.Background(), map[string]any{"model": "m1"})
	v, err := e.Invoke(ctx, 3, graph.Options{})
	if err != nil || v != 3 {
		t.Fatalf("Invoke with valid context = (%v, %v), want (3, nil)", v, err)
	}
	if sawModel != "m1" {
		t.Fatalf("rt.Context[model] = %v, want m1", sawModel)
	}
}

// TestEntrypointContextSchemaNil: no schema means no validation (context is
// passed through untouched).
func TestEntrypointContextSchemaNil(t *testing.T) {
	e, err := NewEntrypoint[int, int, int](
		EntrypointOpts{},
		func(_ runtime.Runtime, in int, _ int, _ bool) (int, error) { return in, nil },
	)
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}
	if v, err := e.Invoke(context.Background(), 9, graph.Options{}); err != nil || v != 9 {
		t.Fatalf("Invoke = (%v, %v), want (9, nil)", v, err)
	}
}

// TestEntrypointFinalOptions: the NewEntrypointFinal form honors the same
// policies (compileEntrypoint is shared) — CachePolicy + validation on the
// final-form function.
func TestEntrypointFinalOptions(t *testing.T) {
	cache := checkpoint.NewInMemoryCache()
	var calls atomic.Int32
	e, err := NewEntrypointFinal[string, int, string](
		EntrypointOpts{
			Cache:       cache,
			CachePolicy: &graph.CachePolicy{},
			ContextSchema: func(ctx any) error {
				if ctx == nil {
					return errors.New("context required")
				}
				return nil
			},
		},
		func(_ runtime.Runtime, in string, _ string, _ bool) (Final[int, string], error) {
			calls.Add(1)
			return Final[int, string]{Value: len(in), Save: in}, nil
		},
	)
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}
	if _, err := e.Invoke(context.Background(), "abc", graph.Options{}); err == nil {
		t.Fatal("Invoke without context: error = nil, want context_schema failure")
	}
	ctx := runtime.ContextWithValues(context.Background(), map[string]any{"k": "v"})
	v, err := e.Invoke(ctx, "abc", graph.Options{})
	if err != nil || v != 3 {
		t.Fatalf("Invoke = (%v, %v), want (3, nil)", v, err)
	}
	if _, err := e.Invoke(ctx, "abc", graph.Options{}); err != nil {
		t.Fatalf("cached Invoke: %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1 (second invoke served from cache)", got)
	}
}
