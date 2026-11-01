// Tests for per-node TracePolicy (langgraph 1.2.11 add_node(trace_policy=),
// PR #8523, with #9486's tracer-only scoping): transforms apply on the emit
// side, so every tracer observing events emitted within the node — including
// chain events from runnables the node invokes — sees transformed payloads,
// while the messages stream bridge keeps seeing the raw ones.
package graph

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// tracePolicyGraph compiles a single-node graph whose node invokes a Func
// runnable with the node context's callback manager fanned into its config —
// the documented pattern for emitting chain events from inside a node (the
// node discovers the manager via callbacks.ManagerFromContext and threads it
// through runnables.WithCallbacks).
func tracePolicyGraph(t *testing.T, policies NodePolicies) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNodeWithPolicies("pii", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		manager, ok := callbacks.ManagerFromContext(rt)
		if !ok {
			return nil, errors.New("trace policy test: no callback manager in the node context")
		}
		fn := runnables.NewFunc(
			func(_ context.Context, input string, _ ...runnables.Option) (string, error) {
				return strings.ToUpper(input), nil
			},
			schema.String(""), schema.String(""),
		)
		out, err := fn.Invoke(rt, "secret", runnables.WithCallbacks(manager))
		if err != nil {
			return nil, err
		}
		return map[string]any{"out": out}, nil
	}, policies)
	g.AddEdge(types.START, "pii")
	g.AddEdge("pii", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// runTracePolicyGraph runs the graph with a caller-installed recorder-backed
// manager (a stand-in tracer) and returns what the tracer observed.
func runTracePolicyGraph(t *testing.T, cg *CompiledGraph) []callbacks.Event {
	t.Helper()
	recorder := callbacks.NewRecorder()
	ctx := callbacks.ContextWithManager(t.Context(), callbacks.NewManager(recorder))
	if _, err := cg.Invoke(ctx, nil); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	return recorder.Events()
}

func TestNodeTracePolicyAppliesToRunnablesInsideNode(t *testing.T) {
	// Control run (no Trace policy): the tracer sees the runnable's raw
	// payloads, establishing the baseline the policy must change.
	controlEvents := runTracePolicyGraph(t, tracePolicyGraph(t, NodePolicies{}))
	if len(controlEvents) != 2 {
		t.Fatalf("control run recorded %d events, want 2 (chain_start + chain_end)", len(controlEvents))
	}
	if got, want := controlEvents[0].Input, "secret"; got != want {
		t.Fatalf("control chain_start Input = %v, want %q", got, want)
	}
	if got, want := controlEvents[1].Output, "SECRET"; got != want {
		t.Fatalf("control chain_end Output = %v, want %q", got, want)
	}

	// Policy run: OmitPayload on both sides. The tracer still sees the events
	// with their run metadata intact, but the payloads are gone.
	policyEvents := runTracePolicyGraph(t, tracePolicyGraph(t, NodePolicies{
		Trace: &TracePolicy{ProcessInputs: OmitPayload, ProcessOutputs: OmitPayload},
	}))
	if len(policyEvents) != 2 {
		t.Fatalf("policy run recorded %d events, want 2 (events are delivered, only payloads omitted)", len(policyEvents))
	}
	for i, event := range policyEvents {
		if event.Input != nil || event.Output != nil {
			t.Fatalf("event %d (%s:%s): Input = %v, Output = %v, want both omitted", i, event.Kind, event.Name, event.Input, event.Output)
		}
		if event.Kind != controlEvents[i].Kind || event.Name != controlEvents[i].Name {
			t.Fatalf("event %d metadata changed by the policy: got %s:%s, want %s:%s",
				i, event.Kind, event.Name, controlEvents[i].Kind, controlEvents[i].Name)
		}
		if event.RunID == "" {
			t.Fatalf("event %d (%s:%s) lost its RunID", i, event.Kind, event.Name)
		}
	}
}

// scrubMessageContent is a stand-in PII scrubber for message payloads: it
// replaces a message's content wholesale, leaving other values unchanged.
func scrubMessageContent(value any) any {
	if m, ok := value.(messages.Message); ok {
		m.Content = "[REDACTED]"
		return m
	}
	return value
}

// TestNodeTracePolicyScrubsTracersNotMessagesStream pins the scoping upstream
// fixed in langgraph #9486: a node's trace_policy rewrites what TRACERS
// record — never what the messages stream mode delivers to the caller. The
// same policy-covered node is observed from both sides: (a) the
// StreamMessages bridge receives the event's raw payload and streams it, and
// (b) a handler registered as a tracer under the policy receives the
// scrubbed payload.
//
// Go's messages-mode manager owns the node's handler set (see
// streamEmitter.nodeContext), so the tracer side is observed on a plain
// Invoke with the tracer installed as the node's callback manager — the very
// manager the executor wraps with the policy.
func TestNodeTracePolicyScrubsTracersNotMessagesStream(t *testing.T) {
	newGraph := func() *CompiledGraph {
		t.Helper()
		g := NewStateGraph()
		g.AddNodeWithPolicies("pii", func(rt runtime.Runtime, _ map[string]any) (any, error) {
			manager, ok := callbacks.ManagerFromContext(rt)
			if !ok {
				return nil, errors.New("trace policy test: no callback manager in the node context")
			}
			final := messages.AI("reach me at secret@example.com")
			final.ID = "run-1"
			if err := manager.Emit(rt, callbacks.Event{Kind: callbacks.EventChatModelEnd, Output: final}); err != nil {
				return nil, err
			}
			return map[string]any{"done": true}, nil
		}, NodePolicies{Trace: &TracePolicy{ProcessOutputs: scrubMessageContent}})
		g.AddEdge(types.START, "pii")
		g.AddEdge("pii", types.END)
		cg, err := g.Compile()
		if err != nil {
			t.Fatalf("Compile() error = %v", err)
		}
		return cg
	}

	// (a) messages mode: the stream sees the RAW payload. Before the fix the
	// bridge sat inside the policy-wrapped manager, so messages mode received
	// the scrubbed payload (or none at all under omit_payload).
	chunks, err := collectStream(t, newGraph().Stream(t.Context(), map[string]any{}, StreamOptions{
		Modes: []StreamMode{StreamMessages},
	}))
	if err != nil {
		t.Fatalf("Stream(messages) error = %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("messages mode got %d chunks, want 1 (the node's end message, not dropped by the policy): %#v", len(chunks), chunks)
	}
	mc := messageChunkPayload(t, chunks[0])
	if got, want := mc.Message.Content, "reach me at secret@example.com"; got != want {
		t.Errorf("messages chunk content = %q, want the RAW %q (policy must not rewrite streamed payloads)", got, want)
	}
	if got := mc.Metadata["langgraph_node"]; got != "pii" {
		t.Errorf("messages chunk langgraph_node metadata = %v, want %q", got, "pii")
	}

	// (b) tracer side: a handler under the policy-wrapped manager sees the
	// SCRUBBED payload — the policy still does its job for tracers.
	recorder := callbacks.NewRecorder()
	ctx := callbacks.ContextWithManager(t.Context(), callbacks.NewManager(recorder))
	if _, err := newGraph().Invoke(ctx, map[string]any{}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("tracer recorded %d events, want 1 (chat_model_end)", len(events))
	}
	out, ok := events[0].Output.(messages.Message)
	if !ok {
		t.Fatalf("tracer event Output type = %T, want messages.Message", events[0].Output)
	}
	if got, want := out.Content, "[REDACTED]"; got != want {
		t.Errorf("tracer saw Output content = %q, want the SCRUBBED %q", got, want)
	}
}
