package graph

import (
	"context"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/checkpoint"
	lcgraph "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
)

// TestAliasesAndConstants verifies the shim's declared aliases and event-kind
// constants remain identical to the langgraph/graph ones. The assignments
// fail to compile if an alias ever drifts into a defined type.
func TestAliasesAndConstants(t *testing.T) {
	var fn NodeFunc = func(runtime.Runtime, map[string]any) (any, error) { return nil, nil }
	var _ lcgraph.NodeFunc = fn

	var router ConditionalEdge = func(runtime.Runtime, map[string]any) ([]any, error) { return nil, nil }
	var _ lcgraph.ConditionalEdge = router

	var g *StateGraph
	var _ *lcgraph.StateGraph = g

	var cg *CompiledGraph
	var _ *lcgraph.CompiledGraph = cg

	var opt CompileOption
	var _ lcgraph.CompileOption = opt

	var opts Options
	var _ lcgraph.Options = opts

	var res Result
	var _ lcgraph.Result = res

	var kind RawEventKind
	var _ lcgraph.RawEventKind = kind

	var ev RawEvent
	var _ lcgraph.RawEvent = ev

	var sink NodeEventSink
	var _ lcgraph.NodeEventSink = sink

	if RawNodeStart != lcgraph.RawNodeStart {
		t.Errorf("RawNodeStart = %q, want langgraph/graph.RawNodeStart %q", RawNodeStart, lcgraph.RawNodeStart)
	}
	if RawNodeStart != "node_start" {
		t.Errorf("RawNodeStart = %q, want %q", RawNodeStart, "node_start")
	}
	if RawNodeEnd != lcgraph.RawNodeEnd {
		t.Errorf("RawNodeEnd = %q, want langgraph/graph.RawNodeEnd %q", RawNodeEnd, lcgraph.RawNodeEnd)
	}
	if RawNodeEnd != "node_end" {
		t.Errorf("RawNodeEnd = %q, want %q", RawNodeEnd, "node_end")
	}
}

// TestTo verifies the To helper packs node names into the []any routing
// slice the executor expects.
func TestTo(t *testing.T) {
	got := To("a", "b")
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("To(a, b) = %v, want [a b]", got)
	}
	if empty := To(); len(empty) != 0 {
		t.Errorf("To() = %v, want empty", empty)
	}
}

// TestLinearGraphThroughShim builds and runs a graph entirely through the
// shim's API surface — including the agentruntime START/END sentinels a
// shim consumer would pair it with — and checks the state reducers ran.
func TestLinearGraphThroughShim(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("step1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"count": state["count"].(int) + 1}, nil
	})
	g.AddNode("step2", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"count": state["count"].(int) * 10}, nil
	})
	g.AddEdge(agentruntime.START, "step1")
	g.AddEdge("step1", "step2")
	g.AddEdge("step2", agentruntime.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"count": 2})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["count"] != 30 {
		t.Errorf("count = %v, want 30", res.Values["count"])
	}
	if len(res.Interrupts) != 0 {
		t.Errorf("expected no interrupts, got %+v", res.Interrupts)
	}
}

// TestConditionalEdgeThroughShim verifies a ConditionalEdge router built
// with To drives execution to the chosen branch.
func TestConditionalEdgeThroughShim(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("router", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("left", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"branch": "left"}, nil
	})
	g.AddNode("right", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"branch": "right"}, nil
	})
	g.AddEdge(agentruntime.START, "router")
	g.AddConditionalEdges("router", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To("right"), nil
	})
	g.AddEdge("left", agentruntime.END)
	g.AddEdge("right", agentruntime.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["branch"] != "right" {
		t.Errorf("branch = %v, want right", res.Values["branch"])
	}
}

// TestInterruptAndResumeThroughShim exercises the full pause/resume cycle a
// shim consumer relies on: WithCheckpointer wired to a checkpoint-shim saver,
// an in-node Interrupt call, and a resume via Options.
func TestInterruptAndResumeThroughShim(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		answer := Interrupt(rt, "what is your name?")
		return map[string]any{"name": answer}, nil
	})
	g.AddEdge(agentruntime.START, "ask")
	g.AddEdge("ask", agentruntime.END)

	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "what is your name?" {
		t.Fatalf("first.Interrupts = %+v, want one interrupt asking the question", first.Interrupts)
	}
	// The surfaced interrupt must be usable through the agentruntime shim's
	// Interrupt alias too.
	var _ agentruntime.Interrupt = first.Interrupts[0]

	second, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t1", Resume: "Ada"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if second.Values["name"] != "Ada" {
		t.Errorf("name = %v, want Ada", second.Values["name"])
	}
}

// TestInterruptBeforeThroughShim verifies WithInterruptBefore pauses the run
// before the named node and a nil-resume invocation completes it.
func TestInterruptBeforeThroughShim(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	ran := map[string]bool{}
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		ran["a"] = true
		return nil, nil
	})
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		ran["b"] = true
		return nil, nil
	})
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", agentruntime.END)

	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("b"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) == 0 {
		t.Fatal("expected an interrupt before node b")
	}
	if !ran["a"] || ran["b"] {
		t.Errorf("ran = %v, want a executed and b not yet run", ran)
	}

	second, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t1", Resume: nil})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 || !ran["b"] {
		t.Errorf("after resume: interrupts = %+v, ran = %v; want b completed", second.Interrupts, ran)
	}
}

// TestInterruptAfterThroughShim verifies WithInterruptAfter pauses the run
// after the named node has executed.
func TestInterruptAfterThroughShim(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	ran := map[string]bool{}
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		ran["a"] = true
		return map[string]any{"step": "a"}, nil
	})
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		ran["b"] = true
		return nil, nil
	})
	g.AddEdge(agentruntime.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", agentruntime.END)

	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptAfter("a"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) == 0 {
		t.Fatal("expected an interrupt after node a")
	}
	if !ran["a"] || ran["b"] {
		t.Errorf("ran = %v, want a executed and b not yet run", ran)
	}

	if _, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t1", Resume: nil}); err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if !ran["b"] {
		t.Error("expected b to run after resume")
	}
}

// TestRecursionLimitThroughShim verifies WithRecursionLimit stops a looping
// graph with an error instead of running forever.
func TestRecursionLimitThroughShim(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("loop", func(_ runtime.Runtime, state map[string]any) (any, error) {
		n, _ := state["n"].(int)
		return map[string]any{"n": n + 1}, nil
	})
	g.AddEdge(agentruntime.START, "loop")
	g.AddConditionalEdges("loop", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To("loop"), nil // loop forever; only the recursion limit stops it
	})

	cg, err := g.Compile(WithRecursionLimit(5))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.Invoke(context.Background(), map[string]any{"n": 0}); err == nil {
		t.Error("expected a recursion-limit error from an unbounded loop")
	}
}

type recordingSink struct {
	mu     sync.Mutex
	events []RawEvent
}

func (s *recordingSink) EmitRawEvent(ev RawEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, ev)
}

// TestEventSinkThroughShim verifies the context helpers round-trip a sink
// and that InvokeStream emits balanced node_start/node_end events through
// the shim's sink and event types.
func TestEventSinkThroughShim(t *testing.T) {
	if EventSinkFromContext(context.Background()) != nil {
		t.Error("EventSinkFromContext(background) != nil, want nil")
	}

	sink := &recordingSink{}
	ctx := ContextWithEventSink(context.Background(), sink)
	if EventSinkFromContext(ctx) != NodeEventSink(sink) {
		t.Error("EventSinkFromContext did not return the installed sink")
	}

	g := NewStateGraph()
	g.AddNode("n1", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		// A node observes the streaming sink through the shim's helper.
		if EventSinkFromContext(rt) == nil {
			t.Error("node context has no event sink during InvokeStream")
		}
		return nil, nil
	})
	g.AddEdge(agentruntime.START, "n1")
	g.AddEdge("n1", agentruntime.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeStream(context.Background(), map[string]any{}, Options{}, sink); err != nil {
		t.Fatalf("InvokeStream() error = %v", err)
	}

	sink.mu.Lock()
	defer sink.mu.Unlock()
	if len(sink.events) != 2 {
		t.Fatalf("events = %+v, want one start/end pair", sink.events)
	}
	if sink.events[0].Kind != RawNodeStart || sink.events[0].Node != "n1" {
		t.Errorf("events[0] = %+v, want node_start for n1", sink.events[0])
	}
	if sink.events[1].Kind != RawNodeEnd || sink.events[1].Node != "n1" {
		t.Errorf("events[1] = %+v, want node_end for n1", sink.events[1])
	}
}
