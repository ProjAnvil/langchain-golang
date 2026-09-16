package graph

// TracePolicy controls what traced payloads a node contributes, mirroring
// langgraph 1.2.11's add_node(trace_policy=) (PR #8523): transforms are
// applied where events are EMITTED, before any tracer (LangSmith, console, a
// future OTel bridge) observes them, so every handler sees the same scrubbed
// payloads. Installed per node via NodePolicies.Trace
// (StateGraph.AddNodeWithPolicies); the executor wraps the node context's
// callback manager in a callbacks.NewPayloadPolicyManager for the duration of
// the node's task, which covers both events the node emits itself and chain
// events from runnables it invokes (they discover the wrapped manager via
// callbacks.ManagerFromContext).
//
// Fail-closed: a transform that panics drops the corresponding payload rather
// than leaking it — a deliberate divergence from upstream's fail-open
// behavior (which logs and records the UNTRANSFORMED payload); this port's
// rationale is the feature's own motivation, PII/compliance scrubbing, where
// failing open would silently leak the raw value. See DIVERGENCES.md.
type TracePolicy struct {
	// ProcessInputs transforms start-kind event payloads (the Input field of
	// chain/model/tool/retriever start events). Nil leaves inputs untouched.
	ProcessInputs func(value any) any
	// ProcessOutputs transforms end-kind event payloads (the Output field of
	// end events). Nil leaves outputs untouched.
	ProcessOutputs func(value any) any
}

// OmitPayload drops a payload entirely; assign it to TracePolicy.ProcessInputs
// or TracePolicy.ProcessOutputs (the Go analogue of Python's omit_payload
// helper, which is a module-level helper rather than a policy method).
func OmitPayload(any) any { return nil }
