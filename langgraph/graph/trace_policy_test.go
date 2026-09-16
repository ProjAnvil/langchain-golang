// Tests for per-node TracePolicy (langgraph 1.2.11 add_node(trace_policy=),
// PR #8523): transforms apply on the emit side, so every tracer observing
// events emitted within the node — including chain events from runnables the
// node invokes — sees transformed payloads.
package graph

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
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
