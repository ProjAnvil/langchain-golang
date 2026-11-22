package middleware

import (
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/graph"
)

// Tests for the process-global default TracePolicy (langchain_v1's
// configure_trace_policy / _node_trace_policy). These use distinctive helper
// names to avoid colliding with the TracePolicyProvider tests in the external
// middleware_test package.

func defaultTraceReset(t *testing.T) {
	t.Helper()
	ConfigureTracePolicy(nil)
	t.Cleanup(func() { ConfigureTracePolicy(nil) })
}

func defaultTraceIn(any) any  { return "default-in" }
func defaultTraceOut(any) any { return "default-out" }
func mwTraceIn(any) any       { return "mw-in" }

func TestNodeTracePolicyDefaultPassthrough(t *testing.T) {
	defaultTraceReset(t)
	policy := NodeTracePolicy(nil)
	if got := policy.ProcessInputs("raw"); got != "raw" {
		t.Fatalf("ProcessInputs = %v, want passthrough raw", got)
	}
	if got := policy.ProcessOutputs("raw"); got != "raw" {
		t.Fatalf("ProcessOutputs = %v, want passthrough raw", got)
	}
}

func TestConfigureTracePolicyAppliesDefault(t *testing.T) {
	defaultTraceReset(t)
	ConfigureTracePolicy(&graph.TracePolicy{
		ProcessInputs:  defaultTraceIn,
		ProcessOutputs: defaultTraceOut,
	})
	policy := NodeTracePolicy(nil)
	if got := policy.ProcessInputs("raw"); got != "default-in" {
		t.Fatalf("ProcessInputs = %v, want default-in", got)
	}
	if got := policy.ProcessOutputs("raw"); got != "default-out" {
		t.Fatalf("ProcessOutputs = %v, want default-out", got)
	}
}

func TestNodeTracePolicyCallTimeOverride(t *testing.T) {
	defaultTraceReset(t)
	// Build the node policy BEFORE the default is configured. Because the
	// transforms resolve the global at call time, the late configuration still
	// applies (the same policy object changes behavior).
	policy := NodeTracePolicy(nil)
	if got := policy.ProcessInputs("raw"); got != "raw" {
		t.Fatalf("before configure: ProcessInputs = %v, want raw", got)
	}

	ConfigureTracePolicy(&graph.TracePolicy{
		ProcessInputs:  defaultTraceIn,
		ProcessOutputs: defaultTraceOut,
	})
	if got := policy.ProcessInputs("raw"); got != "default-in" {
		t.Fatalf("after configure: ProcessInputs = %v, want default-in", got)
	}
	if got := policy.ProcessOutputs("raw"); got != "default-out" {
		t.Fatalf("after configure: ProcessOutputs = %v, want default-out", got)
	}

	// Clearing the default restores passthrough on the same policy object.
	ConfigureTracePolicy(nil)
	if got := policy.ProcessInputs("raw"); got != "raw" {
		t.Fatalf("after clear: ProcessInputs = %v, want raw", got)
	}
}

func TestNodeTracePolicyMiddlewareOverrideWholesale(t *testing.T) {
	defaultTraceReset(t)
	ConfigureTracePolicy(&graph.TracePolicy{
		ProcessInputs:  defaultTraceIn,
		ProcessOutputs: defaultTraceOut,
	})
	// A middleware policy that defines ONLY ProcessInputs wins wholesale: its
	// unset ProcessOutputs is NOT merged from the global default.
	mw := &graph.TracePolicy{ProcessInputs: mwTraceIn}
	policy := NodeTracePolicy(mw)
	if got := policy.ProcessInputs("raw"); got != "mw-in" {
		t.Fatalf("ProcessInputs = %v, want mw-in", got)
	}
	if got := policy.ProcessOutputs("raw"); got != "raw" {
		t.Fatalf("ProcessOutputs = %v, want passthrough raw (no field-level merge with global)", got)
	}
}

func TestConfigureTracePolicyClear(t *testing.T) {
	defaultTraceReset(t)
	ConfigureTracePolicy(&graph.TracePolicy{ProcessInputs: defaultTraceIn})
	if got := NodeTracePolicy(nil).ProcessInputs("raw"); got != "default-in" {
		t.Fatalf("ProcessInputs = %v, want default-in", got)
	}
	ConfigureTracePolicy(nil)
	if got := NodeTracePolicy(nil).ProcessInputs("raw"); got != "raw" {
		t.Fatalf("ProcessInputs after clear = %v, want raw", got)
	}
}

func TestResolveTracePolicy(t *testing.T) {
	defaultTraceReset(t)
	if got := resolveTracePolicy(nil); got != nil {
		t.Fatalf("resolveTracePolicy(nil) = %v, want nil when no default is set", got)
	}
	ConfigureTracePolicy(&graph.TracePolicy{ProcessInputs: defaultTraceIn})
	if got := resolveTracePolicy(nil); got == nil || got.ProcessInputs == nil {
		t.Fatalf("resolveTracePolicy(nil) = %v, want the configured default", got)
	}
	mw := &graph.TracePolicy{ProcessInputs: mwTraceIn}
	if got := resolveTracePolicy(mw); got != mw {
		t.Fatalf("resolveTracePolicy(mw) = %v, want the middleware policy itself", got)
	}
}

func TestNodeTracePolicyUsesOmitPayload(t *testing.T) {
	defaultTraceReset(t)
	ConfigureTracePolicy(&graph.TracePolicy{ProcessInputs: graph.OmitPayload})
	policy := NodeTracePolicy(nil)
	if got := policy.ProcessInputs("secret"); got != nil {
		t.Fatalf("ProcessInputs = %v, want nil (OmitPayload)", got)
	}
	if got := policy.ProcessOutputs("kept"); got != "kept" {
		t.Fatalf("ProcessOutputs = %v, want passthrough kept", got)
	}
}
