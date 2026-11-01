package callbacks

import "context"

// NewPayloadPolicyManager wraps parent so every event's Input and Output pass
// through the supplied transforms BEFORE tracer handlers observe them — the
// emit-side application point of langgraph 1.2.11's trace_policy (PR #8523)
// and langchain 1.3.15's middleware trace_policy (#38910). Any tracer
// attached to parent (LangSmith, console, a future OTel bridge) therefore
// sees the transformed payloads. Handlers that opted into raw payloads
// (RawPayloadHandler — stream bridges) still see the real ones: since
// langgraph #9486 the policy scopes to tracers, because it exists to keep
// tracer ingest small, not to rewrite what streams deliver to the caller.
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

// RawPayloadHandler is a Handler that must observe a run's REAL payloads even
// when a payload policy (trace_policy) is active: the policy rewrites what
// tracers record, never what the handler itself delivers. Implement it —
// returning true — for stream bridges whose output belongs to the caller
// (langgraph's messages mode does; upstream #9486).
type RawPayloadHandler interface {
	Handler
	// RawPayloads reports whether the handler opts out of payload-policy
	// transforms.
	RawPayloads() bool
}

// payloadPolicyHandler is the transforming shim installed by
// NewPayloadPolicyManager: it applies the payload transforms to its copy of
// the event and delivers per-handler payloads through the wrapped parent
// manager (raw for RawPayloadHandler handlers, transformed for the rest).
type payloadPolicyHandler struct {
	parent         Manager
	processInputs  func(any) any
	processOutputs func(any) any
}

// HandleEvent fans the event out to the parent manager's handlers with
// per-handler payloads, mirroring parent.Emit (prepareEvent first, handlers
// in order, first error wins): raw-payload handlers get the event untouched,
// every other handler — tracers — gets the transformed copy (skipping nil
// fields and nil transforms). This is the Go shape of upstream #9486's
// handler split: tracers get the processed payload, stream handlers the real
// one.
func (h payloadPolicyHandler) HandleEvent(ctx context.Context, event Event) error {
	event = h.parent.prepareEvent(event)
	transformed := h.transform(event)
	for _, handler := range h.parent.handlers {
		delivery := transformed
		if raw, ok := handler.(RawPayloadHandler); ok && raw.RawPayloads() {
			delivery = event
		}
		if err := handler.HandleEvent(ctx, delivery); err != nil {
			return err
		}
	}
	return nil
}

// transform returns a copy of event with its payloads transformed (skipping
// nil fields and nil transforms).
func (h payloadPolicyHandler) transform(event Event) Event {
	if event.Input != nil && h.processInputs != nil {
		event.Input = applyPayloadTransform(event.Input, h.processInputs)
	}
	if event.Output != nil && h.processOutputs != nil {
		event.Output = applyPayloadTransform(event.Output, h.processOutputs)
	}
	return event
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
