// Tests for NewPayloadPolicyManager: emit-side payload transforms applied
// before any handler observes an event (langgraph 1.2.11 trace_policy parity,
// PR #8523). Every tracer attached to the wrapped manager's parent must see
// the transformed payloads; a panicking transform fails closed (payload
// dropped, event delivered).
package callbacks

import (
	"fmt"
	"strings"
	"testing"
)

// redactPayload is a stand-in payload transform: it replaces every occurrence
// of "secret" with "[REDACTED]", leaving other values unchanged.
func redactPayload(value any) any {
	if s, ok := value.(string); ok {
		return strings.ReplaceAll(s, "secret", "[REDACTED]")
	}
	return value
}

// assertNoRawPayload fails when any recorded event still carries the raw
// "secret" substring in its payloads.
func assertNoRawPayload(t *testing.T, name string, events []Event) {
	t.Helper()
	for _, event := range events {
		raw := fmt.Sprint(event.Input) + " " + fmt.Sprint(event.Output)
		if strings.Contains(raw, "secret") {
			t.Fatalf("%s: raw payload leaked to a handler: %+v", name, event)
		}
	}
}

func TestPayloadPolicyManagerTransformsInputOutput(t *testing.T) {
	// Two handlers stand in for two concurrent tracers (e.g. a LangSmith
	// tracer and a console tracer): BOTH must observe the transformed
	// payloads — the transform lives on the emit side, before fan-out.
	langsmithTracer := NewRecorder()
	consoleTracer := NewRecorder()
	parent := NewManager(langsmithTracer, consoleTracer)

	wrapped := NewPayloadPolicyManager(parent, redactPayload, redactPayload)

	if err := wrapped.Emit(t.Context(), Event{Kind: EventChainStart, Name: "node", Input: "feed secret words"}); err != nil {
		t.Fatalf("emit chain_start: %v", err)
	}
	if err := wrapped.Emit(t.Context(), Event{Kind: EventChainEnd, Name: "node", Output: "out secret"}); err != nil {
		t.Fatalf("emit chain_end: %v", err)
	}

	for name, tracer := range map[string]*Recorder{"langsmith tracer": langsmithTracer, "console tracer": consoleTracer} {
		events := tracer.Events()
		if len(events) != 2 {
			t.Fatalf("%s: recorded %d events, want 2", name, len(events))
		}
		if got, want := events[0].Input, "feed [REDACTED] words"; got != want {
			t.Fatalf("%s: chain_start Input = %v, want %q", name, got, want)
		}
		if got, want := events[1].Output, "out [REDACTED]"; got != want {
			t.Fatalf("%s: chain_end Output = %v, want %q", name, got, want)
		}
		assertNoRawPayload(t, name, events)
	}
}

func TestPayloadPolicyManagerPanicFailsClosed(t *testing.T) {
	// A panicking transform must fail CLOSED: the payload is dropped (nil)
	// but the event itself is still delivered, and the OTHER field's
	// transform still applies. Deliberate divergence from upstream's
	// fail-open behavior (see DIVERGENCES.md): the feature's motivation is
	// PII/compliance, so failing open would silently leak the raw payload.
	recorder := NewRecorder()
	parent := NewManager(recorder)
	panicky := func(any) any { panic("transform exploded") }

	wrapped := NewPayloadPolicyManager(parent, panicky, redactPayload)

	if err := wrapped.Emit(t.Context(), Event{Kind: EventChainStart, Input: "secret", Output: "clean secret"}); err != nil {
		t.Fatalf("emit: %v", err)
	}

	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("recorded %d events, want 1 (fail-closed drops the payload, not the event)", len(events))
	}
	if events[0].Input != nil {
		t.Fatalf("Input = %v, want nil after the transform panicked", events[0].Input)
	}
	if got, want := events[0].Output, "clean [REDACTED]"; got != want {
		t.Fatalf("Output = %v, want %q (an independent transform must still apply)", got, want)
	}
	assertNoRawPayload(t, "recorder", events)
}

func TestPayloadPolicyManagerChildRetainsPolicy(t *testing.T) {
	// Managers derived from the wrapped manager (Child for nested runs,
	// WithMetadata for inherited metadata) must keep transforming payloads —
	// tracers observe child runs through derived managers.
	recorder := NewRecorder()
	parent := NewManager(recorder).WithTags("root")
	wrapped := NewPayloadPolicyManager(parent, redactPayload, redactPayload)

	child := wrapped.Child("run-42").WithMetadata(map[string]any{"step": 2})

	if err := child.Emit(t.Context(), Event{Kind: EventChainEnd, Output: "secret output"}); err != nil {
		t.Fatalf("emit: %v", err)
	}

	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("recorded %d events, want 1", len(events))
	}
	if got, want := events[0].Output, "[REDACTED] output"; got != want {
		t.Fatalf("Output = %v, want %q (policy must survive Child/WithMetadata derivation)", got, want)
	}
	if events[0].ParentID != "run-42" {
		t.Fatalf("ParentID = %q, want %q (derivation semantics preserved)", events[0].ParentID, "run-42")
	}
	if events[0].Metadata["step"] != 2 {
		t.Fatalf("Metadata = %v, want the WithMetadata value preserved", events[0].Metadata)
	}
	if len(events[0].Tags) != 1 || events[0].Tags[0] != "root" {
		t.Fatalf("Tags = %v, want the parent's [root] preserved", events[0].Tags)
	}
	assertNoRawPayload(t, "recorder", events)
}

func TestPayloadPolicyManagerNilTransformsPassthrough(t *testing.T) {
	// Both transforms nil: events pass through unchanged. A nil payload must
	// also skip a NON-nil transform (transforms are never invoked with nil).
	recorder := NewRecorder()

	nilPolicy := NewPayloadPolicyManager(NewManager(recorder), nil, nil)
	if err := nilPolicy.Emit(t.Context(), Event{Kind: EventChainStart, Input: "raw secret"}); err != nil {
		t.Fatalf("emit through nil-transform policy: %v", err)
	}

	inputCalls := 0
	countingInputs := func(value any) any {
		inputCalls++
		return value
	}
	partialPolicy := NewPayloadPolicyManager(NewManager(recorder), countingInputs, nil)
	// Input is nil, so the non-nil ProcessInputs transform must be skipped.
	if err := partialPolicy.Emit(t.Context(), Event{Kind: EventChainEnd, Output: "raw out"}); err != nil {
		t.Fatalf("emit with nil Input: %v", err)
	}

	events := recorder.Events()
	if len(events) != 2 {
		t.Fatalf("recorded %d events, want 2", len(events))
	}
	if got, want := events[0].Input, "raw secret"; got != want {
		t.Fatalf("nil-transform Input = %v, want %q (unchanged)", got, want)
	}
	if got, want := events[1].Output, "raw out"; got != want {
		t.Fatalf("skipped-payload Output = %v, want %q (unchanged)", got, want)
	}
	if inputCalls != 0 {
		t.Fatalf("ProcessInputs called %d times, want 0 (nil payloads skip the transform)", inputCalls)
	}
}
