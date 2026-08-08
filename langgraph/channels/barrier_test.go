package channels

import (
	"errors"
	"reflect"
	"testing"
)

// requireUnavailable asserts that ch has no readable value (Get fails with
// ErrEmptyChannel, IsAvailable is false) without requiring checkpoint
// omission — unlike the other channels, a partially-arrived Barrier still
// persists its seen set, so requireEmpty does not apply to it.
func requireUnavailable(t *testing.T, ch Channel) {
	t.Helper()
	if _, err := ch.Get(); !errors.Is(err, ErrEmptyChannel) {
		t.Fatalf("Get() error = %v, want ErrEmptyChannel", err)
	}
	if ch.IsAvailable() {
		t.Fatal("IsAvailable() = true, want false")
	}
}

// TestBarrierAccumulatesIdempotently mirrors NamedBarrierValue.update
// (named_barrier_value.py:56-67): arrivals accumulate, repeat arrivals are
// no-ops, and the channel reports change only on NEW arrivals.
func TestBarrierAccumulatesIdempotently(t *testing.T) {
	b := NewBarrier("a", "b")
	requireEmpty(t, b)

	if !update(t, b, "a") {
		t.Fatal("Update(a) changed = false, want true (new arrival)")
	}
	if update(t, b, "a") {
		t.Fatal("Update(a) changed = true, want false (idempotent)")
	}
	requireUnavailable(t, b) // 1 of 2 arrivals: still unavailable

	if !update(t, b, "a", "b") { // mixed repeat + new in one batch
		t.Fatal("Update(a, b) changed = false, want true")
	}
	if !b.IsAvailable() {
		t.Fatal("IsAvailable() = false, want true (all names arrived)")
	}
	if got := get(t, b); got != nil {
		t.Fatalf("Get() = %v, want nil (Python get() returns None)", got)
	}
}

// TestBarrierRejectsUnknownName mirrors the InvalidUpdateError branch
// (named_barrier_value.py:63-66).
func TestBarrierRejectsUnknownName(t *testing.T) {
	b := NewBarrier("a", "b")
	_, err := b.Update([]any{"c"})
	var iu *InvalidUpdateError
	if !errors.As(err, &iu) {
		t.Fatalf("Update(c) error = %v, want *InvalidUpdateError", err)
	}
	if _, err := b.Update([]any{42}); !errors.As(err, &iu) {
		t.Fatalf("Update(42) error = %v, want *InvalidUpdateError (non-string)", err)
	}
	if b.IsAvailable() {
		t.Fatal("IsAvailable() = true after rejected update")
	}
}

// TestBarrierSurvivesStepBoundary: an empty Update (the applyWrites
// step-boundary notification) must NOT expire the barrier — unlike Ephemeral,
// arrivals persist across supersteps until Consume.
func TestBarrierSurvivesStepBoundary(t *testing.T) {
	b := NewBarrier("a", "b")
	update(t, b, "a", "b")
	if update(t, b /* no values: step boundary */) {
		t.Fatal("Update() changed = true, want false")
	}
	if !b.IsAvailable() {
		t.Fatal("IsAvailable() = false after step boundary, want true")
	}
}

// TestBarrierConsumeResets mirrors consume (named_barrier_value.py:77-81):
// no-op unless full; when full, resets so a loop round can re-accumulate.
func TestBarrierConsumeResets(t *testing.T) {
	b := NewBarrier("a", "b")
	if b.Consume() {
		t.Fatal("Consume() = true on partial barrier, want false")
	}
	update(t, b, "a", "b")
	if !b.Consume() {
		t.Fatal("Consume() = false on full barrier, want true")
	}
	requireEmpty(t, b)
	// Re-accumulation after reset (loop re-trigger).
	update(t, b, "b")
	requireUnavailable(t, b)
	update(t, b, "a")
	if !b.IsAvailable() {
		t.Fatal("IsAvailable() = false after re-accumulation, want true")
	}
}

// TestBarrierCheckpointRoundTrip: Checkpoint omits an empty barrier, persists
// partial arrivals as a sorted []string (serde registry nameStrings,
// serde/json.go:40), and FromCheckpoint restores them — the "parent A
// arrived, parent B interrupted" pause/resume closure.
func TestBarrierCheckpointRoundTrip(t *testing.T) {
	proto := NewBarrier("a", "b")
	if _, ok := proto.Checkpoint(); ok {
		t.Fatal("Checkpoint() ok = true on empty barrier, want false (omit)")
	}

	b := NewBarrier("a", "b")
	update(t, b, "b")
	v, ok := b.Checkpoint()
	if !ok {
		t.Fatal("Checkpoint() ok = false on partial barrier, want true")
	}
	if !reflect.DeepEqual(v, []string{"b"}) {
		t.Fatalf("Checkpoint() = %v, want []string{\"b\"}", v)
	}

	restored := proto.FromCheckpoint(v)
	if restored.IsAvailable() {
		t.Fatal("restored IsAvailable() = true, want false (partial)")
	}
	if !update(t, restored.(*Barrier), "a") {
		t.Fatal("Update(a) on restored changed = false, want true")
	}
	if !restored.IsAvailable() {
		t.Fatal("restored IsAvailable() = false after completing arrival")
	}
	// The prototype is never mutated by FromCheckpoint.
	if proto.IsAvailable() {
		t.Fatal("prototype mutated by FromCheckpoint")
	}

	// Defensive branch: a JSON-decoded []any restore (serde without the
	// registry entry) still lands.
	def := proto.FromCheckpoint([]any{"a"})
	if !update(t, def.(*Barrier), "b") || !def.IsAvailable() {
		t.Fatal("[]any restore did not preserve arrival")
	}
	// FromCheckpoint(nil) yields an empty barrier that keeps the name set.
	empty := proto.FromCheckpoint(nil)
	requireEmpty(t, empty)
	if _, err := empty.Update([]any{"zzz"}); err == nil {
		t.Fatal("FromCheckpoint(nil) lost the name set: unknown name accepted")
	}
}
