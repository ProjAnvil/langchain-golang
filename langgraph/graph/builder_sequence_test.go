package graph

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestAddSequence mirrors Python's test_add_sequence
// (tests/test_pregel.py:4594): an empty sequence and duplicate names are
// errors (surfaced at Compile via the builder's sticky error, the Go
// analogue of Python's ValueError); a flat sequence runs its nodes in
// order, threading updates through the reducer channel.
func TestAddSequence(t *testing.T) {
	step1 := func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"foo": []string{"step1"}, "bar": "baz"}, nil
	}
	step2 := func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"foo": []string{"step2"}}, nil
	}

	// Python: add_sequence([]) raises ValueError("Sequence requires at least
	// one node.").
	if _, err := NewStateGraph().AddSequence().Compile(); err == nil ||
		!strings.Contains(err.Error(), "at least one node") {
		t.Fatalf("AddSequence() Compile error = %v, want 'at least one node'", err)
	}
	// Python: duplicate names raise ValueError("Node names must be unique...").
	if _, err := NewStateGraph().
		AddSequence(NamedNode{Name: "step1", Fn: step1}, NamedNode{Name: "step1", Fn: step1}).
		Compile(); err == nil || !strings.Contains(err.Error(), `duplicate node "step1"`) {
		t.Fatalf("duplicate AddSequence Compile error = %v, want duplicate node error", err)
	}

	// Flat sequence: step1 -> step2 (Python asserts result ==
	// {"foo": ["step1", "step2"], "bar": "baz"} over an operator.add channel;
	// AppendSliceReducer is the Go analogue).
	g := NewStateGraph().AddReducer("foo", channels.AppendSliceReducer)
	g.AddSequence(NamedNode{Name: "step1", Fn: step1}, NamedNode{Name: "step2", Fn: step2})
	g.AddEdge(types.START, "step1")
	g.SetFinishPoint("step2")
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"foo": []string{}})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if got := fmt.Sprintf("%v", res.Values["foo"]); got != "[step1 step2]" {
		t.Fatalf("foo = %v, want [step1 step2] (sequence order)", res.Values["foo"])
	}
	if res.Values["bar"] != "baz" {
		t.Fatalf("bar = %v, want baz", res.Values["bar"])
	}
}

// TestSetConditionalEntryPointFanOutInvalidUpdate mirrors
// tests/test_pregel.py:784-797: a conditional entry point routing to two
// nodes that both write a LastValue channel fails with InvalidUpdateError.
func TestSetConditionalEntryPointFanOutInvalidUpdate(t *testing.T) {
	myNode := func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"hello": "world"}, nil
	}
	cg, err := NewStateGraph().
		AddChannel("hello", channels.NewLastValue()).
		AddNode("one", myNode).
		AddNode("two", myNode).
		SetConditionalEntryPoint(func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
			return To("one", "two"), nil
		}).
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	_, err = cg.Invoke(context.Background(), map[string]any{"hello": "there"})
	var iu *channels.InvalidUpdateError
	if !errors.As(err, &iu) {
		t.Fatalf("Invoke error = %v, want *channels.InvalidUpdateError", err)
	}
}

// TestSetConditionalEntryPointRoutes: the router sees the post-input state
// and picks the first node (Python's set_conditional_entry_point is
// add_conditional_edges(START, path); tests/test_pregel.py:1747 exercises
// the same routing with a path_map, which Go does not need — routers return
// node names directly).
func TestSetConditionalEntryPointRoutes(t *testing.T) {
	ran := map[string]int{}
	node := func(name string) NodeFunc {
		return func(_ runtime.Runtime, _ map[string]any) (any, error) {
			ran[name]++
			return map[string]any{"k": name}, nil
		}
	}
	cg, err := NewStateGraph().
		AddNode("one", node("one")).
		AddNode("two", node("two")).
		SetConditionalEntryPoint(func(_ runtime.Runtime, state map[string]any) ([]any, error) {
			if state["pick"] == "two" {
				return To("two"), nil
			}
			return To("one"), nil
		}).
		SetFinishPoint("one").
		SetFinishPoint("two").
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"pick": "two"})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if ran["two"] != 1 || ran["one"] != 0 {
		t.Fatalf("ran = %v, want only \"two\" executed", ran)
	}
	if res.Values["k"] != "two" {
		t.Fatalf("k = %v, want two", res.Values["k"])
	}
}

// TestSetConditionalEntryPointToEND: routing to END only stops execution
// immediately (Python: "If it returns END, the graph will stop execution",
// state.py:1093).
func TestSetConditionalEntryPointToEND(t *testing.T) {
	called := false
	cg, err := NewStateGraph().
		AddNode("one", func(_ runtime.Runtime, _ map[string]any) (any, error) {
			called = true
			return nil, nil
		}).
		SetConditionalEntryPoint(func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
			return To(types.END), nil
		}).
		SetFinishPoint("one").
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if called {
		t.Fatal("node ran despite END routing")
	}
	if res.Values["x"] != 1 {
		t.Fatalf("Values = %v, want the input state", res.Values)
	}
}

// TestEntryPointConflictAndMissing: exactly one entry mechanism must be set
// (Python lets set_entry_point/set_conditional_entry_point both register;
// Go's single-entry model rejects the ambiguity, consistent with the
// existing duplicate-SetEntryPoint error at graph.go:304-306).
func TestEntryPointConflictAndMissing(t *testing.T) {
	fn := func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil }
	router := func(_ runtime.Runtime, _ map[string]any) ([]any, error) { return To("a"), nil }

	if _, err := NewStateGraph().AddNode("a", fn).SetEntryPoint("a").
		SetConditionalEntryPoint(router).Compile(); err == nil ||
		!strings.Contains(err.Error(), "entry point already set") {
		t.Fatalf("SetEntryPoint then SetConditionalEntryPoint: error = %v, want conflict", err)
	}
	if _, err := NewStateGraph().AddNode("a", fn).SetConditionalEntryPoint(router).
		SetEntryPoint("a").Compile(); err == nil ||
		!strings.Contains(err.Error(), "conditional entry point already set") {
		t.Fatalf("SetConditionalEntryPoint then SetEntryPoint: error = %v, want conflict", err)
	}
	if _, err := NewStateGraph().AddNode("a", fn).SetConditionalEntryPoint(router).
		SetConditionalEntryPoint(router).Compile(); err == nil ||
		!strings.Contains(err.Error(), "conditional entry point already set") {
		t.Fatalf("duplicate SetConditionalEntryPoint: error = %v, want conflict", err)
	}
	if _, err := NewStateGraph().AddNode("a", fn).
		SetConditionalEntryPoint(nil).Compile(); err == nil ||
		!strings.Contains(err.Error(), "must not be nil") {
		t.Fatalf("nil router: error = %v, want nil-router error", err)
	}
	// Missing entry keeps failing (message extended, not removed).
	if _, err := NewStateGraph().AddNode("a", fn).Compile(); err == nil ||
		!strings.Contains(err.Error(), "entry point not set") {
		t.Fatalf("missing entry: error = %v, want 'entry point not set'", err)
	}
}

// TestSetConditionalEntryPointRouterError: a router failure surfaces as an
// Invoke error, wrapped with the conditional-entry-point context (the
// entryTasks error path).
func TestSetConditionalEntryPointRouterError(t *testing.T) {
	cg, err := NewStateGraph().
		AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil }).
		SetConditionalEntryPoint(func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
			return nil, errors.New("router boom")
		}).
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	_, err = cg.Invoke(context.Background(), map[string]any{"x": 1})
	if err == nil || !strings.Contains(err.Error(), "router boom") ||
		!strings.Contains(err.Error(), "conditional entry point") {
		t.Fatalf("Invoke error = %v, want wrapped router error", err)
	}
}

// TestSetFinishPoint mirrors the set_finish_point usage at
// tests/test_pregel.py:2871 and tests/test_state.py:123: sugar for
// AddEdge(node, types.END).
func TestSetFinishPoint(t *testing.T) {
	order := []string{}
	node := func(name string) NodeFunc {
		return func(_ runtime.Runtime, _ map[string]any) (any, error) {
			order = append(order, name)
			return map[string]any{"k": name}, nil
		}
	}
	cg, err := NewStateGraph().
		AddNode("n1", node("n1")).
		AddNode("n2", node("n2")).
		SetEntryPoint("n1").
		AddEdge("n1", "n2").
		SetFinishPoint("n2").
		Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	res, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(order) != 2 || order[0] != "n1" || order[1] != "n2" {
		t.Fatalf("order = %v, want [n1 n2]", order)
	}
	if res.Values["k"] != "n2" {
		t.Fatalf("k = %v, want n2", res.Values["k"])
	}
}
