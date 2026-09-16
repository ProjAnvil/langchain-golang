package callbacks

import "context"

// NewPayloadPolicyManager wraps parent so every event's Input and Output pass
// through the supplied transforms BEFORE handlers observe them — the
// emit-side application point of langgraph 1.2.11's trace_policy (PR #8523)
// and langchain 1.3.15's middleware trace_policy (#38910). Any tracer
// attached to parent (LangSmith, console, a future OTel bridge) therefore
// sees the transformed payloads; the raw values never reach a handler.
//
// Manager is a value struct with no virtual dispatch, so the wrapper is built
// by composition rather than embedding: the returned manager's single handler
// transforms a copy of the event and re-emits it through parent (Manager's
// HandleEvent exists exactly for such nesting). Derivation methods on the
// returned manager (Child, WithTags, WithMetadata) retain the policy, because
// the transforming handler is part of the manager's handler set and survives
// the value copy — a tracer observing child runs sees transformed payloads
// too.
//
// Fail-closed: a transform that panics drops the corresponding payload (the
// field becomes nil) while the event itself is still delivered — a deliberate
// divergence from upstream's fail-open behavior (which logs and records the
// UNTRANSFORMED payload). The feature exists for PII/compliance scrubbing, so
// failing open would silently leak the raw value. See DIVERGENCES.md.
func NewPayloadPolicyManager(parent Manager, processInputs, processOutputs func(any) any) Manager {
	return NewManager(payloadPolicyHandler{parent: parent, processInputs: processInputs, processOutputs: processOutputs})
}

// payloadPolicyHandler is the transforming shim installed by
// NewPayloadPolicyManager: it applies the payload transforms to its copy of
// the event and delegates delivery to the wrapped parent manager.
type payloadPolicyHandler struct {
	parent         Manager
	processInputs  func(any) any
	processOutputs func(any) any
}

// HandleEvent transforms the event's payloads (skipping nil fields and nil
// transforms) and re-emits it through the parent manager, so the parent's
// own prepareEvent pass (inherited parent run ID, tags, metadata) still
// applies after the transforms.
func (h payloadPolicyHandler) HandleEvent(ctx context.Context, event Event) error {
	if event.Input != nil && h.processInputs != nil {
		event.Input = applyPayloadTransform(event.Input, h.processInputs)
	}
	if event.Output != nil && h.processOutputs != nil {
		event.Output = applyPayloadTransform(event.Output, h.processOutputs)
	}
	return h.parent.Emit(ctx, event)
}

// applyPayloadTransform runs transform under fail-closed recovery: a panic
// drops the payload (returns nil) instead of crashing the run or leaking the
// raw value.
func applyPayloadTransform(value any, transform func(any) any) (payload any) {
	defer func() {
		if recover() != nil {
			payload = nil // fail closed: drop the payload, deliver the event
		}
	}()
	return transform(value)
}
