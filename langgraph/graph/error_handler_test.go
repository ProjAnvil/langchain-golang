package graph

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Node-level error handler runtime coverage (spec §6.1, Python langgraph 1.2.0
// add_node(error_handler=)): handlers run after retries are exhausted (or
// immediately with no RetryPolicy), receive the pre-superstep state snapshot
// plus a typed NodeError, and may recover with a plain update or a routing
// Command; a failing handler's error propagates with the original failure
// still matchable. Resume/crash-recovery cases live at the bottom of this
// file (Task 2).

// TestErrorHandlerRunsAfterRetriesExhausted pins the exhaustion contract: the
// node runs exactly MaxAttempts times, then the handler fires ONCE with the
// 1-based attempt count that exhausted the budget and a readable state
// snapshot, and its update lands in the same superstep's commit.
func TestErrorHandlerRunsAfterRetriesExhausted(t *testing.T) {
	var attempts, handlerCalls atomic.Int32
	var got NodeError
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		attempts.Add(1)
		return nil, errFlaky
	}, NodePolicies{
		Retry: fastRetryPolicy(2),
		ErrorHandler: &ErrorHandlerPolicy{
			Handler: func(state map[string]any, nerr *NodeError) (any, error) {
				handlerCalls.Add(1)
				got = *nerr
				if state["seed"] != 7 {
					t.Errorf("handler state[\"seed\"] = %v, want 7 (pre-superstep snapshot must be readable)", state["seed"])
				}
				return map[string]any{"recovered": true}, nil
			},
		},
	})

	res, err := cg.Invoke(t.Context(), map[string]any{"seed": 7})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want handler recovery to complete the run", err)
	}
	if n := attempts.Load(); n != 2 {
		t.Fatalf("node attempts = %d, want 2 (MaxAttempts)", n)
	}
	if n := handlerCalls.Load(); n != 1 {
		t.Fatalf("handler calls = %d, want exactly 1 (after retries exhausted)", n)
	}
	if got.Node != "node" {
		t.Fatalf("NodeError.Node = %q, want %q", got.Node, "node")
	}
	if got.Attempt != 2 {
		t.Fatalf("NodeError.Attempt = %d, want 2 (1-based exhausted attempt)", got.Attempt)
	}
	if !errors.Is(got.Err, errFlaky) {
		t.Fatalf("NodeError.Err = %v, want the original failure (errors.Is errFlaky)", got.Err)
	}
	if res.Values["recovered"] != true {
		t.Fatalf("Values[\"recovered\"] = %v, want true", res.Values["recovered"])
	}
}

// TestErrorHandlerNoRetryRunsImmediately: with no RetryPolicy the first
// failure goes straight to the handler with Attempt == 1.
func TestErrorHandlerNoRetryRunsImmediately(t *testing.T) {
	var attempts, handlerCalls atomic.Int32
	var gotAttempt int
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		attempts.Add(1)
		return nil, errFlaky
	}, NodePolicies{ErrorHandler: &ErrorHandlerPolicy{
		Handler: func(_ map[string]any, nerr *NodeError) (any, error) {
			handlerCalls.Add(1)
			gotAttempt = nerr.Attempt
			return map[string]any{"recovered": true}, nil
		},
	}})

	res, err := cg.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if n := attempts.Load(); n != 1 {
		t.Fatalf("node attempts = %d, want 1 (no retry policy)", n)
	}
	if n := handlerCalls.Load(); n != 1 {
		t.Fatalf("handler calls = %d, want 1", n)
	}
	if gotAttempt != 1 {
		t.Fatalf("NodeError.Attempt = %d, want 1", gotAttempt)
	}
	if res.Values["recovered"] != true {
		t.Fatalf("Values[\"recovered\"] = %v, want true", res.Values["recovered"])
	}
}

// TestErrorHandlerPlainUpdateBecomesOutcome: a handler returning a plain
// map[string]any is normalized exactly like a node result and the graph
// continues to END.
func TestErrorHandlerPlainUpdateBecomesOutcome(t *testing.T) {
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return nil, errFlaky
	}, NodePolicies{ErrorHandler: &ErrorHandlerPolicy{
		Handler: func(_ map[string]any, _ *NodeError) (any, error) {
			return map[string]any{"recovered": true}, nil
		},
	}})

	res, err := cg.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["recovered"] != true {
		t.Fatalf("Values[\"recovered\"] = %v, want true", res.Values["recovered"])
	}
}

// compileTwoNodeGraph builds a two-node graph where node a runs under
// policies and carries a static edge straight to END, while b is reachable
// only via a Command.Goto from a (or a's handler): reaching b proves dynamic
// routing bypassed the static edge.
func compileTwoNodeGraph(t *testing.T, aFn NodeFunc, aPolicies NodePolicies, bFn NodeFunc) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNodeWithPolicies("a", aFn, aPolicies)
	g.AddNode("b", bFn)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	g.AddEdge("b", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestErrorHandlerCommandRoutes: a handler returning a *types.Command routes
// exactly like a node's Command — the static edge is bypassed and the Goto
// destination runs.
func TestErrorHandlerCommandRoutes(t *testing.T) {
	var aRuns, bRuns atomic.Int32
	cg := compileTwoNodeGraph(t,
		func(_ runtime.Runtime, _ map[string]any) (any, error) {
			aRuns.Add(1)
			return nil, errFlaky
		},
		NodePolicies{ErrorHandler: &ErrorHandlerPolicy{
			Handler: func(_ map[string]any, _ *NodeError) (any, error) {
				return &types.Command{Update: map[string]any{"a_ok": true}, Goto: []any{"b"}}, nil
			},
		}},
		func(_ runtime.Runtime, _ map[string]any) (any, error) {
			bRuns.Add(1)
			return map[string]any{"b_ran": true}, nil
		})

	res, err := cg.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if n := aRuns.Load(); n != 1 {
		t.Fatalf("node a runs = %d, want 1", n)
	}
	if n := bRuns.Load(); n != 1 {
		t.Fatalf("node b runs = %d, want 1 (handler Command.Goto must dispatch b)", n)
	}
	if res.Values["a_ok"] != true || res.Values["b_ran"] != true {
		t.Fatalf("Values = %v, want a_ok and b_ran", res.Values)
	}
}

// TestErrorHandlerFailurePropagates: a failing handler's error surfaces from
// Invoke with BOTH the handler error and the original node error matchable
// via errors.Is.
func TestErrorHandlerFailurePropagates(t *testing.T) {
	errHandlerFailed := errors.New("handler also failed")
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return nil, errFlaky
	}, NodePolicies{ErrorHandler: &ErrorHandlerPolicy{
		Handler: func(_ map[string]any, _ *NodeError) (any, error) {
			return nil, errHandlerFailed
		},
	}})

	_, err := cg.Invoke(t.Context(), nil)
	if err == nil {
		t.Fatal("Invoke() error = nil, want the handler failure to propagate")
	}
	if !errors.Is(err, errHandlerFailed) {
		t.Fatalf("Invoke() error = %v, want errors.Is(err, handler error) to match", err)
	}
	if !errors.Is(err, errFlaky) {
		t.Fatalf("Invoke() error = %v, want errors.Is(err, original node error) to match", err)
	}
	if !strings.Contains(err.Error(), `error handler for node "node"`) {
		t.Fatalf("Invoke() error = %v, want it to name the error handler for node %q", err, "node")
	}
}

// TestErrorHandlerTimeoutTriggersHandler: a TimeoutPolicy firing mid-node is
// an ordinary task failure as far as the handler is concerned — the handler's
// NodeError.Err satisfies errors.Is(err, context.DeadlineExceeded).
func TestErrorHandlerTimeoutTriggersHandler(t *testing.T) {
	var gotErr error
	cg := compileRetryGraph(t, func(rt runtime.Runtime, _ map[string]any) (any, error) {
		select {
		case <-time.After(time.Second):
			return "done", nil
		case <-rt.Done():
			return nil, rt.Err()
		}
	}, NodePolicies{
		Timeout: &TimeoutPolicy{RunTimeout: 50 * time.Millisecond},
		ErrorHandler: &ErrorHandlerPolicy{
			Handler: func(_ map[string]any, nerr *NodeError) (any, error) {
				gotErr = nerr.Err
				return map[string]any{"recovered": true}, nil
			},
		},
	})

	res, err := cg.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if !errors.Is(gotErr, context.DeadlineExceeded) {
		t.Fatalf("NodeError.Err = %v, want context.DeadlineExceeded", gotErr)
	}
	if res.Values["recovered"] != true {
		t.Fatalf("Values[\"recovered\"] = %v, want true", res.Values["recovered"])
	}
}
