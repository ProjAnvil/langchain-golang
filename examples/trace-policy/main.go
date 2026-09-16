// Command trace-policy demonstrates per-node payload scrubbing for traced
// events (langgraph's add_node(trace_policy=)): a node installed with
// NodePolicies.Trace has its event payloads transformed on the emit side —
// BEFORE any tracer observes them — while a sibling node without the policy
// emits raw payloads. A recording callback handler stands in for LangSmith /
// an OTel bridge so the redaction is visible in the terminal.
//
// Two transforms are shown: a custom redactor (emails -> [REDACTED_EMAIL])
// on inputs, and graph.OmitPayload on outputs (payload dropped entirely,
// event metadata preserved).
//
// Usage:
//
//	go run ./examples/trace-policy
package main

import (
	"context"
	"fmt"
	"regexp"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

var emailPattern = regexp.MustCompile(`\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`)

// redactEmails is a PII-scrubbing TracePolicy transform: it rewrites any
// email address in a string payload before the payload reaches a tracer.
func redactEmails(value any) any {
	text, ok := value.(string)
	if !ok {
		return value
	}
	return emailPattern.ReplaceAllString(text, "[REDACTED_EMAIL]")
}

// piiNode builds a node that invokes a Func runnable with the node context's
// callback manager fanned in — the documented pattern for emitting chain
// events from inside a node. Everything the runnable emits (chain start with
// its input, chain end with its output) is traceable payload.
func piiNode(name string) graphpkg.NodeFunc {
	return func(rt runtime.Runtime, _ map[string]any) (any, error) {
		manager, ok := callbacks.ManagerFromContext(rt)
		if !ok {
			return nil, fmt.Errorf("node %s: no callback manager in context", name)
		}
		fn := runnables.NewFunc(
			func(_ context.Context, input string, _ ...runnables.Option) (string, error) {
				return "processed: " + input, nil
			},
			schema.String(""), schema.String(""),
		)
		out, err := fn.Invoke(rt, "contact alice@example.com about order 42", runnables.WithCallbacks(manager))
		if err != nil {
			return nil, err
		}
		return map[string]any{"out": out}, nil
	}
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()

	// open_node: no Trace policy — its raw payloads reach the tracer.
	g.AddNode("open_node", piiNode("open_node"))

	// guarded_node: NodePolicies.Trace scrubs start-kind payloads with the
	// custom redactor and omits end-kind payloads entirely.
	g.AddNodeWithPolicies("guarded_node", piiNode("guarded_node"), graphpkg.NodePolicies{
		Trace: &graphpkg.TracePolicy{
			ProcessInputs:  redactEmails,
			ProcessOutputs: graphpkg.OmitPayload,
		},
	})

	g.AddEdge(types.START, "open_node")
	g.AddEdge("open_node", "guarded_node")
	g.AddEdge("guarded_node", types.END)

	cg, err := g.Compile()
	if err != nil {
		fmt.Println("compile:", err)
		return
	}

	// The recorder is the "tracer": every event the graph emits lands here
	// with whatever payload the trace policy let through.
	recorder := callbacks.NewRecorder()
	traceCtx := callbacks.ContextWithManager(ctx, callbacks.NewManager(recorder))
	if _, err := cg.Invoke(traceCtx, nil); err != nil {
		fmt.Println("invoke:", err)
		return
	}

	fmt.Println("events observed by the tracer:")
	for _, ev := range recorder.Events() {
		fmt.Printf("  kind=%s name=%s input=%v output=%v\n",
			ev.Kind, ev.Name, ev.Input, ev.Output)
	}
	fmt.Println()
	fmt.Println("open_node payloads carry the raw email; guarded_node payloads are")
	fmt.Println("redacted on input and omitted on output (graph.OmitPayload), while the")
	fmt.Println("events themselves — kind, name, run ids — are preserved.")

	// A related surface: middleware implementing middleware.TracePolicyProvider
	// applies the same emit-side transforms to everything inside its
	// wrap_model_call layer on an agents.CreateAgent agent; see
	// langchain/agents/middleware/trace_policy_test.go for that composition.
}
