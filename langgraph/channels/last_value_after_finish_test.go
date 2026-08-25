package channels

import (
	"errors"
	"testing"
)

// Tests for LastValueAfterFinish, authored directly from the Python class
// semantics (channels/last_value.py:81-151) — Python has no dedicated tests
// for this channel.

// Before Finish, the channel buffers updates but stays unavailable
// ("only made available after finish()").
func TestLastValueAfterFinishBuffersUntilFinish(t *testing.T) {
	c := NewLastValueAfterFinish()
	if c.IsAvailable() {
		t.Fatal("fresh channel must not be available")
	}
	if _, err := c.Get(); !errors.Is(err, ErrEmptyChannel) {
		t.Fatalf("Get on fresh channel = %v, want ErrEmptyChannel", err)
	}
	changed, err := c.Update([]any{"a"})
	if err != nil || !changed {
		t.Fatalf("Update = (%v, %v), want (true, nil)", changed, err)
	}
	if c.IsAvailable() {
		t.Fatal("updated but unfinished channel must not be available")
	}
	if _, err := c.Get(); !errors.Is(err, ErrEmptyChannel) {
		t.Fatalf("Get before Finish = %v, want ErrEmptyChannel", err)
	}
}

// Finish publishes the buffered value; any number of writes per step is
// allowed and the LAST one wins (unlike LastValue).
func TestLastValueAfterFinishFinish(t *testing.T) {
	c := NewLastValueAfterFinish()
	if _, err := c.Update([]any{"a", "b", "c"}); err != nil {
		t.Fatalf("Update: %v", err)
	}
	fc, ok := c.(interface{ Finish() bool })
	if !ok {
		t.Fatal("LastValueAfterFinish must expose Finish() bool")
	}
	if !fc.Finish() {
		t.Fatal("Finish after update = false, want true")
	}
	if fc.Finish() {
		t.Fatal("second Finish = true, want false (already finished)")
	}
	v, err := c.Get()
	if err != nil || v != "c" {
		t.Fatalf("Get = (%v, %v), want (c, nil) — last write wins", v, err)
	}
	if !c.IsAvailable() {
		t.Fatal("IsAvailable = false after Finish, want true")
	}
}

// Finish on an empty channel is a no-op (Python: `if not self.finished and
// self.value is not MISSING`).
func TestLastValueAfterFinishFinishEmpty(t *testing.T) {
	c := NewLastValueAfterFinish()
	fc := c.(interface{ Finish() bool })
	if fc.Finish() {
		t.Fatal("Finish on empty channel = true, want false")
	}
	if _, err := c.Get(); !errors.Is(err, ErrEmptyChannel) {
		t.Fatalf("Get = %v, want ErrEmptyChannel", err)
	}
}

// A new update clears the finished flag: the value must be re-Finished
// before it is readable again (Python update sets `self.finished = False`).
func TestLastValueAfterFinishUpdateResetsFinished(t *testing.T) {
	c := NewLastValueAfterFinish()
	fc := c.(interface{ Finish() bool })
	_, _ = c.Update([]any{"a"})
	fc.Finish()
	if _, err := c.Update([]any{"b"}); err != nil {
		t.Fatalf("Update: %v", err)
	}
	if c.IsAvailable() {
		t.Fatal("IsAvailable = true after re-update, want false (finished reset)")
	}
	if _, err := c.Get(); !errors.Is(err, ErrEmptyChannel) {
		t.Fatalf("Get = %v, want ErrEmptyChannel", err)
	}
	fc.Finish()
	if v, err := c.Get(); err != nil || v != "b" {
		t.Fatalf("Get = (%v, %v), want (b, nil)", v, err)
	}
}

// Consume on a finished channel clears the value and reports true; on an
// unfinished channel it is a no-op false (Python: consume()).
func TestLastValueAfterFinishConsume(t *testing.T) {
	c := NewLastValueAfterFinish()
	cc := c.(interface{ Consume() bool })
	fc := c.(interface{ Finish() bool })
	if cc.Consume() {
		t.Fatal("Consume on fresh channel = true, want false")
	}
	_, _ = c.Update([]any{"a"})
	if cc.Consume() {
		t.Fatal("Consume before Finish = true, want false")
	}
	fc.Finish()
	if !cc.Consume() {
		t.Fatal("Consume after Finish = false, want true")
	}
	if c.IsAvailable() {
		t.Fatal("IsAvailable = true after Consume, want false")
	}
	if _, err := c.Get(); !errors.Is(err, ErrEmptyChannel) {
		t.Fatalf("Get after Consume = %v, want ErrEmptyChannel", err)
	}
	if cc.Consume() {
		t.Fatal("second Consume = true, want false")
	}
}

// An empty step-boundary update changes nothing (Python: len(values)==0 →
// False) — the buffered value survives.
func TestLastValueAfterFinishEmptyUpdate(t *testing.T) {
	c := NewLastValueAfterFinish()
	_, _ = c.Update([]any{"a"})
	changed, err := c.Update(nil)
	if err != nil || changed {
		t.Fatalf("Update(nil) = (%v, %v), want (false, nil)", changed, err)
	}
	c.(interface{ Finish() bool }).Finish()
	if v, err := c.Get(); err != nil || v != "a" {
		t.Fatalf("Get = (%v, %v), want (a, nil)", v, err)
	}
}

// Checkpoint omits an empty channel and round-trips (value, finished)
// through FromCheckpoint (Python: checkpoint() returns MISSING or
// (value, finished); from_checkpoint unpacks the pair).
func TestLastValueAfterFinishCheckpointRoundTrip(t *testing.T) {
	c := NewLastValueAfterFinish()
	if _, ok := c.Checkpoint(); ok {
		t.Fatal("Checkpoint on empty channel: ok = true, want false")
	}
	_, _ = c.Update([]any{"a"})
	c.(interface{ Finish() bool }).Finish()
	value, ok := c.Checkpoint()
	if !ok {
		t.Fatal("Checkpoint after Finish: ok = false, want true")
	}
	restored := NewLastValueAfterFinish().FromCheckpoint(value)
	if !restored.IsAvailable() {
		t.Fatal("restored channel not available (finished flag lost)")
	}
	if v, err := restored.Get(); err != nil || v != "a" {
		t.Fatalf("restored Get = (%v, %v), want (a, nil)", v, err)
	}
	// Unfinished state round-trips too: buffered but not yet available.
	c2 := NewLastValueAfterFinish()
	_, _ = c2.Update([]any{"b"})
	v2, ok := c2.Checkpoint()
	if !ok {
		t.Fatal("Checkpoint of unfinished channel: ok = false, want true")
	}
	r2 := NewLastValueAfterFinish().FromCheckpoint(v2)
	if r2.IsAvailable() {
		t.Fatal("restored unfinished channel available, want buffered-only")
	}
	r2.(interface{ Finish() bool }).Finish()
	if v, err := r2.Get(); err != nil || v != "b" {
		t.Fatalf("restored-finished Get = (%v, %v), want (b, nil)", v, err)
	}
	// FromCheckpoint(nil) is the omitted-channel restore: a fresh channel.
	if fresh := NewLastValueAfterFinish().FromCheckpoint(nil); fresh.IsAvailable() {
		t.Fatal("FromCheckpoint(nil) must produce an empty channel")
	}
}
