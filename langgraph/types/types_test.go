package types

import (
	"errors"
	"fmt"
	"testing"
	"time"
)

func TestSentinelConstants(t *testing.T) {
	if START == END {
		t.Errorf("START and END must be distinct, both are %q", START)
	}
	if ParentGraph == "" {
		t.Error("ParentGraph must be non-empty")
	}
}

func TestGraphInterruptErrorMessage(t *testing.T) {
	err := &GraphInterrupt{Interrupt: Interrupt{Value: "need approval", ID: "int-1"}}
	got := err.Error()
	want := "types: interrupted with value need approval (id=int-1)"
	if got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestGraphInterruptErrorsAs(t *testing.T) {
	// CompiledGraph.Invoke recovers GraphInterrupt internally; callers must be
	// able to recognize it through wrapped errors via errors.As.
	inner := &GraphInterrupt{Interrupt: Interrupt{Value: 42, ID: "int-2"}}
	wrapped := fmt.Errorf("node failed: %w", inner)

	var target *GraphInterrupt
	if !errors.As(wrapped, &target) {
		t.Fatal("errors.As should unwrap a GraphInterrupt")
	}
	if target.Interrupt.ID != "int-2" || target.Interrupt.Value != 42 {
		t.Errorf("unwrapped interrupt = %+v, want value 42 id int-2", target.Interrupt)
	}
}

func TestNewCacheKey(t *testing.T) {
	ns := []string{"a", "b"}
	ttl := 5 * time.Minute
	key := NewCacheKey(ns, "k", ttl)

	if key.Key != "k" {
		t.Errorf("Key = %q, want %q", key.Key, "k")
	}
	if key.TTL != ttl {
		t.Errorf("TTL = %v, want %v", key.TTL, ttl)
	}
	if len(key.Namespace) != 2 || key.Namespace[0] != "a" || key.Namespace[1] != "b" {
		t.Errorf("Namespace = %v, want [a b]", key.Namespace)
	}
}

func TestNewCacheKeyZeroTTLMeansNoOverride(t *testing.T) {
	key := NewCacheKey(nil, "k", 0)
	if key.TTL != 0 {
		t.Errorf("TTL = %v, want 0 (no override)", key.TTL)
	}
	if key.Namespace != nil {
		t.Errorf("Namespace = %v, want nil", key.Namespace)
	}
}

func TestCommandZeroValue(t *testing.T) {
	var cmd Command
	if cmd.Graph != "" {
		t.Errorf("zero Command.Graph = %q, want empty (current graph)", cmd.Graph)
	}
	if cmd.Goto != nil || cmd.Update != nil || cmd.Resume != nil {
		t.Errorf("zero Command should have nil Goto/Update/Resume, got %+v", cmd)
	}
}
