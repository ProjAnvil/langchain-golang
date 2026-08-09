package graph

// M1.2 acceptance tests for the BaseStore wiring: verify the executor's
// buildRuntime surfaces a compile-option store.Store (installed via WithStore)
// on Runtime.Store, that a node can read it via rt.Store, and that the SAME
// store instance is shared across nodes within a run AND across separate
// Invoke calls (the cross-thread persistence that is the whole point of the
// BaseStore, mirroring Python's Runtime.store / create_agent(store=...)).

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestStoreRuntimeWiringWithinRun verifies that a store installed via
// WithStore reaches Runtime.Store, and that a "writer" node's Put is visible
// to a later "reader" node's Get in the same run (cross-node access via the
// shared store instance).
func TestStoreRuntimeWiringWithinRun(t *testing.T) {
	mem := store.NewInMemoryStore()

	var readerSaw *store.Item
	var storeWasNil bool

	g := NewStateGraph()
	g.AddNode("writer", func(rt runtime.Runtime, state map[string]any) (any, error) {
		if rt.Store == nil {
			storeWasNil = true
			return nil, nil
		}
		return nil, rt.Store.Put(rt, []string{"memories", "thread-1"}, "prefs", map[string]any{"theme": "dark"}, nil)
	})
	g.AddNode("reader", func(rt runtime.Runtime, state map[string]any) (any, error) {
		if rt.Store == nil {
			storeWasNil = true
			return nil, nil
		}
		it, err := rt.Store.Get(rt, []string{"memories", "thread-1"}, "prefs")
		if err != nil {
			return nil, err
		}
		readerSaw = it
		return nil, nil
	})
	g.AddEdge(types.START, "writer")
	g.AddEdge("writer", "reader")
	g.AddEdge("reader", types.END)

	cg, err := g.Compile(WithStore(mem))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	if _, err := cg.Invoke(context.Background(), map[string]any{}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	if storeWasNil {
		t.Fatalf("rt.Store was nil inside a node; WithStore did not surface the store")
	}
	if readerSaw == nil {
		t.Fatalf("reader node did not observe the item the writer node Put")
	}
	if got := readerSaw.Value["theme"]; got != "dark" {
		t.Errorf(`reader observed theme = %v, want "dark"`, got)
	}
}

// TestStoreSharedAcrossInvocations verifies the cross-thread contract: the
// SAME store instance, passed to WithStore, persists data across two separate
// Invoke calls. The first run's writer node writes an item; the second run's
// reader node (a DIFFERENT graph instance, reusing the store) reads it back —
// proving the store outlives any single graph run, mirroring Python's
// BaseStore persistence across threads.
func TestStoreSharedAcrossInvocations(t *testing.T) {
	mem := store.NewInMemoryStore()

	// First graph: a single writer node that stores a value.
	writeG := NewStateGraph()
	writeG.AddNode("writer", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return nil, rt.Store.Put(rt, []string{"memories", "user42"}, "note", map[string]any{"text": "hello"}, nil)
	})
	writeG.AddEdge(types.START, "writer")
	writeG.AddEdge("writer", types.END)
	writeCG, err := writeG.Compile(WithStore(mem))
	if err != nil {
		t.Fatalf("Compile writer: %v", err)
	}
	if _, err := writeCG.Invoke(context.Background(), map[string]any{}); err != nil {
		t.Fatalf("Invoke writer: %v", err)
	}

	// Second graph: a single reader node that reads the value the first run
	// wrote. It shares ONLY the store instance with the first graph.
	var read *store.Item
	readG := NewStateGraph()
	readG.AddNode("reader", func(rt runtime.Runtime, state map[string]any) (any, error) {
		var err error
		read, err = rt.Store.Get(rt, []string{"memories", "user42"}, "note")
		return nil, err
	})
	readG.AddEdge(types.START, "reader")
	readG.AddEdge("reader", types.END)
	readCG, err := readG.Compile(WithStore(mem))
	if err != nil {
		t.Fatalf("Compile reader: %v", err)
	}
	if _, err := readCG.Invoke(context.Background(), map[string]any{}); err != nil {
		t.Fatalf("Invoke reader: %v", err)
	}

	if read == nil {
		t.Fatalf("second invocation did not observe the item the first invocation wrote (cross-thread store not shared)")
	}
	if got := read.Value["text"]; got != "hello" {
		t.Errorf(`cross-invocation value = %v, want "hello"`, got)
	}

	// Also confirm the store is reachable directly (the source of truth),
	// independent of any graph runtime.
	direct, err := mem.Get(context.Background(), []string{"memories", "user42"}, "note")
	if err != nil || direct == nil {
		t.Fatalf("direct store Get after run: item=%v err=%v", direct, err)
	}
}

// TestStoreNilWhenNotConfigured verifies that without WithStore, rt.Store is
// nil and nodes that nil-check before use are unaffected (no panic).
func TestStoreNilWhenNotConfigured(t *testing.T) {
	var observed runtime.Store
	g := NewStateGraph()
	g.AddNode("n", func(rt runtime.Runtime, state map[string]any) (any, error) {
		observed = rt.Store
		return nil, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)

	cg, err := g.Compile() // no WithStore
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if _, err := cg.Invoke(context.Background(), map[string]any{}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if observed != nil {
		t.Errorf("rt.Store = %v, want nil when no store configured", observed)
	}
}
