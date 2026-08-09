package graph

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// compileTimeoutGraph builds a single-node graph whose node runs under the
// given per-node policies, with START -> node -> END.
func compileTimeoutGraph(t *testing.T, fn NodeFunc, policies NodePolicies) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNodeWithPolicies("node", fn, policies)
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	return cg
}

// TestTimeoutPolicyRunTimeout verifies a RunTimeout deadline cancels a node
// that would otherwise overrun it.
func TestTimeoutPolicyRunTimeout(t *testing.T) {
	cg := compileTimeoutGraph(t, func(rt runtime.Runtime, _ map[string]any) (any, error) {
		select {
		case <-time.After(time.Second):
			return "done", nil
		case <-rt.Done():
			return nil, rt.Err()
		}
	}, NodePolicies{Timeout: &TimeoutPolicy{RunTimeout: 50 * time.Millisecond}})

	start := time.Now()
	_, err := cg.Invoke(context.Background(), nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded, got %v", err)
	}
	if d := time.Since(start); d > 500*time.Millisecond {
		t.Fatalf("run timeout did not fire promptly: %v", d)
	}
}

// TestTimeoutPolicyIdleTimeout verifies an IdleTimeout cancels a node that
// blocks without emitting any heartbeat progress.
func TestTimeoutPolicyIdleTimeout(t *testing.T) {
	cg := compileTimeoutGraph(t, func(rt runtime.Runtime, _ map[string]any) (any, error) {
		select {
		case <-time.After(time.Second):
			return "done", nil
		case <-rt.Done():
			return nil, rt.Err()
		}
	}, NodePolicies{Timeout: &TimeoutPolicy{IdleTimeout: 50 * time.Millisecond, RefreshOn: "heartbeat"}})

	start := time.Now()
	_, err := cg.Invoke(context.Background(), nil)
	if err == nil {
		t.Fatal("expected cancellation from idle timeout, got nil")
	}
	if d := time.Since(start); d > 500*time.Millisecond {
		t.Fatalf("idle timeout did not fire promptly: %v", d)
	}
}

// TestTimeoutPolicyHeartbeatRefreshesIdle verifies that periodic
// rt.Heartbeat() calls keep an idle-timed node alive past its IdleTimeout.
func TestTimeoutPolicyHeartbeatRefreshesIdle(t *testing.T) {
	cg := compileTimeoutGraph(t, func(rt runtime.Runtime, _ map[string]any) (any, error) {
		deadline := time.After(300 * time.Millisecond)
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-deadline:
				return map[string]any{"done": true}, nil
			case <-ticker.C:
				rt.Heartbeat()
			case <-rt.Done():
				return nil, rt.Err()
			}
		}
	}, NodePolicies{Timeout: &TimeoutPolicy{IdleTimeout: 50 * time.Millisecond, RefreshOn: "heartbeat"}})

	result, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("expected node to complete via heartbeats, got err=%v", err)
	}
	if result.Values["done"] != true {
		t.Fatalf("unexpected result: %+v", result)
	}
}

// TestTimeoutPolicyValidation verifies AddNodeWithPolicies rejects invalid
// timeout policies at build time.
func TestTimeoutPolicyValidation(t *testing.T) {
	cases := []struct {
		name string
		p    *TimeoutPolicy
	}{
		{"both zero", &TimeoutPolicy{}},
		{"negative run", &TimeoutPolicy{RunTimeout: -1}},
		{"bad refresh", &TimeoutPolicy{IdleTimeout: time.Second, RefreshOn: "nope"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewStateGraph()
			g.AddNodeWithPolicies("node", func(runtime.Runtime, map[string]any) (any, error) {
				return nil, nil
			}, NodePolicies{Timeout: tc.p})
			g.AddEdge(types.START, "node")
			g.AddEdge("node", types.END)
			if _, err := g.Compile(); err == nil {
				t.Fatal("expected Compile to reject invalid timeout policy")
			}
		})
	}
}
