package graph

import (
	"context"
	"testing"
)

func TestEventSinkContextRoundTrip(t *testing.T) {
	if got := EventSinkFromContext(context.Background()); got != nil {
		t.Fatalf("EventSinkFromContext() = %v, want nil when no sink is installed", got)
	}

	sink := &retryRecordingSink{}
	ctx := ContextWithEventSink(context.Background(), sink)
	if got := EventSinkFromContext(ctx); got != sink {
		t.Fatalf("EventSinkFromContext() = %v, want the installed sink", got)
	}
}

func TestContextWithEventSinkNil(t *testing.T) {
	// A nil sink leaves the context unchanged: EventSinkFromContext must still
	// report nil (the non-streaming zero-overhead path).
	ctx := ContextWithEventSink(context.Background(), nil)
	if got := EventSinkFromContext(ctx); got != nil {
		t.Fatalf("EventSinkFromContext() = %v, want nil for a nil sink", got)
	}
}
