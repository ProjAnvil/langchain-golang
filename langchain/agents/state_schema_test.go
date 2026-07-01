package agents

// Tests for the Step 3c state_schema + context_schema support
// (migration_plan/state-schema-design.md, "## Tests" section).
//
// state_schema:
//  1. Register a custom field; a node writes then reads it across two
//     supersteps; default LastValue replace semantics verified.
//  2. Register a custom field with a user-supplied reducer
//     (channels.AppendSliceReducer over []string); accumulates across steps.
//  3. A StateField whose Name collides with a default key ("messages")
//     overrides its reducer.
//
// context_schema:
//  4. WithContextValues -> ContextValue round-trip; absent key returns
//     (nil, false).
//  5. A value attached before InvokeWithState is visible inside a node.
//  6. Without WithAgentContextSchema declared, WithContextValues /
//     ContextValue still work (schema is declarative-only).
//
// All existing tests stay green (covered by the rest of the package's
// _test.go files; these tests are additive).

import (
	"context"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/channels"
	graphpkg "github.com/projanvil/langchain-golang/langchain/internal/agentruntime/graph"
)

// ===========================================================================
// state_schema
// ===========================================================================

// TestStateSchema_LastValueAcrossSupersteps verifies that a custom state
// field declared via the same wiring CreateAgent uses for a nil-reducer
// StateField (g.AddReducer(name, channels.LastValueReducer)) is writable from
// one node and persists to the next superstep, and that LastValue replace
// semantics hold: a single write fully establishes the value (no
// accumulation). This mirrors what WithAgentStateFields(StateField{Name:
// "counter"}) compiles to inside CreateAgent.
//
// Spec mapping: state_schema test #1.
func TestStateSchema_LastValueAcrossSupersteps(t *testing.T) {
	// A tiny standalone graph (the full model<->tools loop isn't needed to
	// exercise reducer semantics): "writer" writes "counter", then "reader"
	// observes the persisted value and routes to END.
	g := graphpkg.NewStateGraph()

	var readerSaw any
	var readerReadOK bool

	g.AddNode("writer", func(_ context.Context, state map[string]any) (any, error) {
		// First-read-tolerant: "counter" is absent on the writer's first read
		// (nodes must tolerate an absent key, per the spec's open notes).
		if _, ok := state["counter"]; ok {
			t.Errorf("writer unexpectedly saw a pre-existing counter field")
		}
		return map[string]any{"counter": 41}, nil
	})
	g.AddNode("reader", func(_ context.Context, state map[string]any) (any, error) {
		readerSaw, readerReadOK = state["counter"], true
		return nil, nil
	})
	g.SetEntryPoint("writer")
	g.AddEdge("writer", "reader")
	g.AddEdge("reader", agentruntime.END)

	// Register "counter" with LastValue (the default CreateAgent's loop picks
	// when a StateField carries a nil Reducer). Calling g.AddReducer directly
	// mirrors exactly what CreateAgent's wiring does.
	g.AddReducer("counter", channels.LastValueReducer)

	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	res, err := compiled.Invoke(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !readerReadOK {
		t.Fatalf("reader node never ran / never observed the counter field")
	}
	if readerSaw != 41 {
		t.Fatalf("expected counter=41 in reader node, got %#v", readerSaw)
	}
	if res.Values["counter"] != 41 {
		t.Fatalf("expected counter=41 in final state, got %#v", res.Values["counter"])
	}
}

// TestStateSchema_CustomReducerAccumulates verifies that a StateField with a
// user-supplied reducer (channels.AppendSliceReducer over []string) accumulates
// values across two successive writes from two nodes, rather than replacing.
//
// Spec mapping: state_schema test #2.
func TestStateSchema_CustomReducerAccumulates(t *testing.T) {
	g := graphpkg.NewStateGraph()

	g.AddNode("first", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"a"}}, nil
	})
	g.AddNode("second", func(_ context.Context, _ map[string]any) (any, error) {
		// Append onto whatever "first" already produced; AppendSliceReducer
		// concatenates rather than replaces.
		return map[string]any{"docs": []string{"b", "c"}}, nil
	})
	g.SetEntryPoint("first")
	g.AddEdge("first", "second")
	g.AddEdge("second", agentruntime.END)

	// A user-supplied reducer, exactly as a caller would pass via
	// WithAgentStateFields(StateField{Name: "docs", Reducer: channels.AppendSliceReducer}).
	g.AddReducer("docs", channels.AppendSliceReducer)

	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	res, err := compiled.Invoke(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	got, ok := res.Values["docs"].([]string)
	if !ok {
		t.Fatalf("expected []string under docs, got %T: %#v", res.Values["docs"], res.Values["docs"])
	}
	want := []string{"a", "b", "c"}
	if len(got) != len(want) || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Fatalf("AppendSliceReducer did not accumulate as expected: got %v, want %v", got, want)
	}
}

// TestStateSchema_NameCollisionOverridesDefault verifies that a StateField
// whose Name collides with a default key overrides that key's reducer. We
// drive this through the real CreateAgent wiring and assert the override
// actually changes merge behavior on "messages": swap MessagesReducer for a
// custom reducer that records whether it was invoked.
//
// Spec mapping: state_schema test #3.
func TestStateSchema_NameCollisionOverridesDefault(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("ok")}}

	var mu sync.Mutex
	invocations := 0
	called := func() bool {
		mu.Lock()
		defer mu.Unlock()
		return invocations > 0
	}
	// A custom reducer that simply replaces (mimicking LastValue so the agent
	// loop still completes) but flips a flag to prove the graph dispatched to
	// it rather than the default MessagesReducer.
	customMessagesReducer := func(existing any, update any) (any, error) {
		mu.Lock()
		invocations++
		mu.Unlock()
		_ = existing
		return update, nil
	}

	agent, err := CreateAgent(
		model,
		nil,
		WithAgentStateFields(StateField{Name: "messages", Reducer: customMessagesReducer}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !called() {
		t.Fatalf("expected the override reducer for \"messages\" to be invoked; the default MessagesReducer was used instead (Name collision did not override)")
	}
}

// ===========================================================================
// context_schema
// ===========================================================================

// TestContextSchema_RoundTrip verifies WithContextValues -> ContextValue
// round-trips a value and that an absent key returns (nil, false).
//
// Spec mapping: context_schema test #4.
func TestContextSchema_RoundTrip(t *testing.T) {
	ctx := WithContextValues(context.Background(), map[string]any{
		"user_id": "u123",
		"count":   7,
	})

	if v, ok := ContextValue(ctx, "user_id"); !ok || v != "u123" {
		t.Fatalf("user_id round-trip: got (%v, %v), want (u123, true)", v, ok)
	}
	if v, ok := ContextValue(ctx, "count"); !ok || v != 7 {
		t.Fatalf("count round-trip: got (%v, %v), want (7, true)", v, ok)
	}
	// Absent key on a context that DOES carry a values map.
	if v, ok := ContextValue(ctx, "absent"); ok || v != nil {
		t.Fatalf("absent key: got (%v, %v), want (nil, false)", v, ok)
	}
}

// TestContextSchema_AbsentMapReturnsFalse verifies that when no values map was
// attached to the context at all, ContextValue returns (nil, false) for every
// key (rather than panicking on a missing/typed-wrong value).
//
// Spec mapping: context_schema test #4 (absent-key half, stressed).
func TestContextSchema_AbsentMapReturnsFalse(t *testing.T) {
	ctx := context.Background() // no WithContextValues call
	if v, ok := ContextValue(ctx, "anything"); ok || v != nil {
		t.Fatalf("context with no values map: got (%v, %v), want (nil, false)", v, ok)
	}
}

// TestContextSchema_ValueVisibleInsideNodeDuringInvoke verifies that a value
// attached via WithContextValues before Agent.InvokeWithState is visible
// inside a node via ContextValue. The node is plumbed in through a
// BeforeAgent hook (BeforeAgentHook runs as its own dedicated graph node,
// see the create_agent.go package doc comment), which is the lowest-friction
// injection point that doesn't require constructing a custom graph.
//
// Spec mapping: context_schema test #5.
func TestContextSchema_ValueVisibleInsideNodeDuringInvoke(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	var observedValue any
	var observedOK bool
	hook := &ctxReadingBeforeAgent{
		onBeforeAgent: func(ctx context.Context) {
			observedValue, observedOK = ContextValue(ctx, "tenant_id")
		},
	}

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	ctx := WithContextValues(context.Background(), map[string]any{"tenant_id": "acme"})
	if _, err := agent.InvokeWithState(ctx, []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !observedOK || observedValue != "acme" {
		t.Fatalf("expected ContextValue inside before_agent node to return (acme, true), got (%v, %v)", observedValue, observedOK)
	}
}

// TestContextSchema_WorksWithoutSchemaDeclared verifies that
// WithContextValues / ContextValue work even when no WithAgentContextSchema
// was declared — the schema layer is purely declarative for now.
//
// Spec mapping: context_schema test #6.
func TestContextSchema_WorksWithoutSchemaDeclared(t *testing.T) {
	// Deliberately do NOT pass WithAgentContextSchema. CreateAgent succeeds
	// (no schema gating), and ContextValue still reads the attached value.
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	hook := &ctxReadingBeforeAgent{
		onBeforeAgent: func(ctx context.Context) {
			if v, ok := ContextValue(ctx, "k"); !ok || v != "v" {
				t.Errorf("ContextValue(k) inside node returned (%v, %v); expected (v, true) even without schema declared", v, ok)
			}
		},
	}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	ctx := WithContextValues(context.Background(), map[string]any{"k": "v"})
	if _, err := agent.InvokeWithState(ctx, []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
}

// ctxReadingBeforeAgent is a minimal BeforeAgentHook that hands the ctx it
// receives to a callback, so a test can read runtime-context values from
// inside the node's execution.
type ctxReadingBeforeAgent struct {
	onBeforeAgent func(ctx context.Context)
}

func (h *ctxReadingBeforeAgent) BeforeAgent(ctx context.Context, _ map[string]any) (map[string]any, error) {
	if h.onBeforeAgent != nil {
		h.onBeforeAgent(ctx)
	}
	return nil, nil
}

// Compile-time assertion that the helper satisfies the hook interface
// CreateAgent probes for, so a future rename/reshape of that interface turns
// this file red at compile time rather than silently skipping the hook.
var _ BeforeAgentHook = (*ctxReadingBeforeAgent)(nil)
