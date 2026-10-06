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
