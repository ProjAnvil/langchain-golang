package middleware

import (
	"sync/atomic"

	"github.com/projanvil/langchain-golang/langgraph/graph"
)

// This file implements the process-global default TracePolicy for agent
// middleware hook spans, mirroring langchain_v1's
// langchain.agents.middleware._trace_policy. It is deliberately separate from
// the TracePolicyConfig / TracePolicyProvider hook in types.go (a distinct
// langchain feature): here a process-wide default is configured once and read
// at call time, so it applies to node policies built before or after the call.

// defaultTracePolicy is the process-wide default. It is stored atomically and
// read at call time by the transforms returned from NodeTracePolicy, so
// ConfigureTracePolicy can run after a node policy was built and still apply.
var defaultTracePolicy atomic.Pointer[graph.TracePolicy]

// ConfigureTracePolicy sets the process-wide default TracePolicy for agent
// middleware hook spans. Call once at startup. A middleware's own policy
// overrides this wholesale (no field-level merge). Pass nil to clear the
// default. The policy is copied, so later mutations by the caller do not
// race with readers.
func ConfigureTracePolicy(policy *graph.TracePolicy) {
	if policy == nil {
		defaultTracePolicy.Store(nil)
		return
	}
	cloned := *policy
	defaultTracePolicy.Store(&cloned)
}

// traceField identifies which transform of a TracePolicy to apply, mirroring
// the Literal["process_inputs", "process_outputs"] field of Python's
// _resolved_transform.
type traceField string

const (
	fieldProcessInputs  traceField = "process_inputs"
	fieldProcessOutputs traceField = "process_outputs"
)

// fieldTransform returns the transform for field on policy, or nil when the
// policy does not define it. It mirrors getattr(effective, field) in Python.
func fieldTransform(policy *graph.TracePolicy, field traceField) func(any) any {
	if policy == nil {
		return nil
	}
	switch field {
	case fieldProcessInputs:
		return policy.ProcessInputs
	case fieldProcessOutputs:
		return policy.ProcessOutputs
	default:
		return nil
	}
}

// resolveTracePolicy returns the effective policy: the middleware's own if
// set, otherwise the process-wide default. A non-nil middleware policy wins
// wholesale, so its unset fields are NOT merged with the default. It mirrors
// _resolve.
func resolveTracePolicy(mwPolicy *graph.TracePolicy) *graph.TracePolicy {
	if mwPolicy != nil {
		return mwPolicy
	}
	return defaultTracePolicy.Load()
}

// resolvedTransform returns a trace processor that resolves the effective
// policy at call time. Reading the global at call time (not build time) means
// ConfigureTracePolicy can run after NodeTracePolicy and still apply. The
// value passes through unchanged when no effective policy defines the field.
// It mirrors _resolved_transform.
func resolvedTransform(mwPolicy *graph.TracePolicy, field traceField) func(any) any {
	return func(value any) any {
		effective := resolveTracePolicy(mwPolicy)
		transform := fieldTransform(effective, field)
		if transform == nil {
			return value
		}
		return transform(value)
	}
}

// NodeTracePolicy builds the resolver-backed TracePolicy for a middleware's
// node hook, mirroring _node_trace_policy. mwPolicy is the middleware's own
// policy (nil when it has none). The returned ProcessInputs and
// ProcessOutputs resolve the effective policy at call time, so the global
// default set by ConfigureTracePolicy is honored even when it is configured
// after this function returns.
func NodeTracePolicy(mwPolicy *graph.TracePolicy) graph.TracePolicy {
	return graph.TracePolicy{
		ProcessInputs:  resolvedTransform(mwPolicy, fieldProcessInputs),
		ProcessOutputs: resolvedTransform(mwPolicy, fieldProcessOutputs),
	}
}
