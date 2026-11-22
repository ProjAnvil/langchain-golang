package callbacks

import (
	"context"
	"testing"
)

func TestInternalCallMarker(t *testing.T) {
	bg := t.Context()
	if IsInternalCall(bg) {
		t.Fatal("plain context must not be internal")
	}
	marked := WithInternalCall(bg)
	if !IsInternalCall(marked) {
		t.Fatal("WithInternalCall context must be internal")
	}
	// Child contexts inherit the marker.
	child, cancel := context.WithCancel(marked)
	defer cancel()
	if !IsInternalCall(child) {
		t.Fatal("child context must inherit the marker")
	}
}

// TestInternalCallSourceRoundTrip pins the lc_source context channel
// (upstream tags internal runs' metadata with lc_source, #41579).
func TestInternalCallSourceRoundTrip(t *testing.T) {
	if source := InternalCallSource(t.Context()); source != "" {
		t.Fatalf("plain context source: %q", source)
	}
	ctx := WithInternalCallSource(t.Context(), "summarization")
	if !IsInternalCall(ctx) {
		t.Fatal("WithInternalCallSource must also mark the call internal")
	}
	if source := InternalCallSource(ctx); source != "summarization" {
		t.Fatalf("source: %q", source)
	}
}
