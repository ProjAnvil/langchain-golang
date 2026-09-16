package graph

import (
	"testing"
)

func TestEventSinkContextRoundTrip(t *testing.T) {
	if got := EventSinkFromContext(t.Context()); got != nil {
		t.Fatalf("EventSinkFromContext() = %v, want nil when no sink is installed", got)
	}

	sink := &retryRecordingSink{}
	ctx := ContextWithEventSink(t.Context(), sink)
	if got := EventSinkFromContext(ctx); got != sink {
		t.Fatalf("EventSinkFromContext() = %v, want the installed sink", got)
	}
}

func TestContextWithEventSinkNil(t *testing.T) {
	// A nil sink leaves the context unchanged: EventSinkFromContext must still
	// report nil (the non-streaming zero-overhead path).
	ctx := ContextWithEventSink(t.Context(), nil)
	if got := EventSinkFromContext(ctx); got != nil {
		t.Fatalf("EventSinkFromContext() = %v, want nil for a nil sink", got)
	}
}
