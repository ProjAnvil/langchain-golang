// Command fault-tolerance demonstrates langgraph's per-node resilience
// policies (AddNodeWithPolicies) in one graph:
//
//   - a RetryPolicy retries a flaky node until it succeeds (with backoff;
//     jitter disabled so the demo stays fast),
//   - an ErrorHandlerPolicy recovers a permanently failing node, once with a
//     plain state update and once by routing around it with a Command.
//
// Usage:
//
//	go run ./examples/fault-tolerance
package main

import (
	"context"
	"errors"
	"fmt"
	"time"

	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// errPaymentProvider is the simulated transient upstream failure.
var errPaymentProvider = errors.New("payment provider: connection reset")

func main() {
	ctx := context.Background()

	// flakyCharge fails the first two attempts and succeeds on the third;
	// the RetryPolicy carries it through transparently.
	var chargeAttempts int
	// captureRouter is always broken; its error handler salvages the run by
	// routing to the "manual_review" node with a state update.
	var handlerCalls int

	g := graphpkg.NewStateGraph()
	g.AddNodeWithPolicies("flaky_charge", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		chargeAttempts++
		if chargeAttempts < 3 {
			return nil, fmt.Errorf("attempt %d: %w", chargeAttempts, errPaymentProvider)
		}
		return map[string]any{"charged": true}, nil
	}, graphpkg.NodePolicies{
		Retry: &graphpkg.RetryPolicy{
			MaxAttempts:     3,
			InitialInterval: 10 * time.Millisecond,
			BackoffFactor:   2,
			NoJitter:        true, // keep the demo sub-second; Python-parity default is jittered
			RetryOn: func(err error) bool {
				// Retry transient provider errors; a NonRetryable-wrapped
				// error (or context cancellation) would abort immediately.
				return graphpkg.DefaultRetryOn(err)
			},
		},
	})

	g.AddNodeWithPolicies("capture_receipt", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return nil, errors.New("receipt API is down for maintenance")
	}, graphpkg.NodePolicies{
		// No RetryPolicy: the first failure goes straight to the handler.
		ErrorHandler: &graphpkg.ErrorHandlerPolicy{
			Handler: func(state map[string]any, nerr *graphpkg.NodeError) (any, error) {
				handlerCalls++
				fmt.Printf("  handler saw node=%q attempt=%d err=%v (charged so far: %v)\n",
					nerr.Node, nerr.Attempt, nerr.Err, state["charged"])
				// Recover by routing around the broken node: the handler's
				// *types.Command is normalized exactly like a node result,
				// so Goto bypasses the static "capture_receipt" edge.
				return &types.Command{
					Update: map[string]any{"receipt": "queued for manual review"},
					Goto:   graphpkg.To("manual_review"),
				}, nil
			},
		},
	})

	manualRuns := 0
	g.AddNode("manual_review", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		manualRuns++
		return map[string]any{"review": "opened ticket PAY-1042"}, nil
	})

	g.AddEdge(types.START, "flaky_charge")
	g.AddEdge("flaky_charge", "capture_receipt")
	g.AddEdge("capture_receipt", types.END) // static edge; bypassed by the handler's Goto
	g.AddEdge("manual_review", types.END)

	cg, err := g.Compile()
	if err != nil {
		fmt.Println("compile:", err)
		return
	}

	fmt.Println("invoking graph (flaky node retried, broken node recovered via handler)...")
	result, err := cg.Invoke(ctx, nil)
	if err != nil {
		fmt.Println("invoke:", err)
		return
	}
	fmt.Printf("charge attempts: %d (retry policy exhausted only on success)\n", chargeAttempts)
	fmt.Printf("error handler calls: %d, manual_review executions: %d\n", handlerCalls, manualRuns)
	fmt.Printf("final state: %v\n", result.Values)
}
