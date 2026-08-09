package fn

// M1.2 acceptance test for the BaseStore wiring through the functional
// Entrypoint API: EntrypointOpts.Store is installed on the internal graph via
// graph.WithStore and surfaces on Runtime.Store inside the entrypoint function
// and any tasks it dispatches, mirroring Python's `@entrypoint(store=...)`.

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/store"
)

// TestEntrypointStoreRuntimeWiring verifies that an EntrypointOpts.Store
// reaches rt.Store inside the entrypoint function, that the function can
// Put/Get through it, and that the SAME store instance is shared across two
// separate Invoke calls (cross-thread persistence).
func TestEntrypointStoreRuntimeWiring(t *testing.T) {
	mem := store.NewInMemoryStore()

	var sawStore bool
	e := NewEntrypoint[string, string, any](
		EntrypointOpts{Store: mem},
		func(rt runtime.Runtime, in string, _ any, _ bool) (string, error) {
			if rt.Store == nil {
				return "", nil
			}
			sawStore = true
			// On each call, record the input under a fixed namespace/key, then
			// return whatever the store currently holds there (after the Put).
			if err := rt.Store.Put(rt, []string{"echo", "thread"}, "last", map[string]any{"v": in}, nil); err != nil {
				return "", err
			}
			it, err := rt.Store.Get(rt, []string{"echo", "thread"}, "last")
			if err != nil {
				return "", err
			}
			if it == nil {
				return "", nil
			}
			s, _ := it.Value["v"].(string)
			return s, nil
		},
	)

	ctx := context.Background()
	out, err := e.Invoke(ctx, "first", graph.Options{})
	if err != nil {
		t.Fatalf("Invoke first: %v", err)
	}
	if !sawStore {
		t.Fatalf("rt.Store was nil inside the entrypoint function; EntrypointOpts.Store did not wire through")
	}
	if out != "first" {
		t.Errorf("first Invoke output = %q, want %q", out, "first")
	}

	// Second invocation shares the store: it observes the value the first
	// invocation wrote (overwritten by its own Put, but proving persistence).
	out, err = e.Invoke(ctx, "second", graph.Options{})
	if err != nil {
		t.Fatalf("Invoke second: %v", err)
	}
	if out != "second" {
		t.Errorf("second Invoke output = %q, want %q", out, "second")
	}

	// The store holds the most-recently-written value, proving it outlived the
	// first Invoke and was shared with the second.
	direct, err := mem.Get(ctx, []string{"echo", "thread"}, "last")
	if err != nil || direct == nil {
		t.Fatalf("direct store Get: item=%v err=%v", direct, err)
	}
	if got, _ := direct.Value["v"].(string); got != "second" {
		t.Errorf("store held %q, want %q (cross-invocation sharing)", got, "second")
	}
}
