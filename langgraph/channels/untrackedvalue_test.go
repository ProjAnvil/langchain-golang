package channels

import (
	"errors"
	"testing"
)

func TestUntrackedValue(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		requireEmpty(t, NewUntrackedValue(false))
		requireEmpty(t, NewUntrackedValue(true))
	})

	t.Run("single write", func(t *testing.T) {
		ch := NewUntrackedValue(true)
		if !update(t, ch, "a") {
			t.Fatal("Update() changed = false, want true")
		}
		if got := get(t, ch); got != "a" {
			t.Fatalf("Get() = %v, want %q", got, "a")
		}
		if !ch.IsAvailable() {
			t.Fatal("IsAvailable() = false, want true")
		}
	})

	t.Run("guard=true errors on multiple writes", func(t *testing.T) {
		ch := NewUntrackedValue(true)
		_, err := ch.Update([]any{"a", "b"})
		var iuErr *InvalidUpdateError
		if !errors.As(err, &iuErr) {
			t.Fatalf("Update() error = %v, want *InvalidUpdateError", err)
		}
		if iuErr.Channel != "UntrackedValue" {
			t.Fatalf("InvalidUpdateError.Channel = %q, want %q", iuErr.Channel, "UntrackedValue")
		}
	})

	t.Run("guard=false takes the last write", func(t *testing.T) {
		ch := NewUntrackedValue(false)
		update(t, ch, "a", "b")
		if got := get(t, ch); got != "b" {
			t.Fatalf("Get() = %v, want %q", got, "b")
		}
	})

	t.Run("empty update is a no-op", func(t *testing.T) {
		ch := NewUntrackedValue(false)
		update(t, ch, "a")
		if update(t, ch) {
			t.Fatal("Update([]) changed = true, want false")
		}
		if got := get(t, ch); got != "a" {
			t.Fatalf("Get() = %v, want %q", got, "a")
		}
	})

	t.Run("checkpoint always omitted", func(t *testing.T) {
		ch := NewUntrackedValue(false)
		if _, ok := ch.Checkpoint(); ok {
			t.Fatal("Checkpoint() ok = true on empty, want false")
		}
		update(t, ch, "a")
		if _, ok := ch.Checkpoint(); ok {
			t.Fatal("Checkpoint() ok = true after update, want false (untracked)")
		}
	})

	t.Run("from checkpoint never restores", func(t *testing.T) {
		ch := NewUntrackedValue(true)
		update(t, ch, "a")
		restored := ch.FromCheckpoint("a")
		requireEmpty(t, restored)

		// The guard setting must survive a FromCheckpoint round-trip.
		_, err := restored.Update([]any{"x", "y"})
		var iuErr *InvalidUpdateError
		if !errors.As(err, &iuErr) {
			t.Fatalf("restored guard lost: Update() error = %v, want *InvalidUpdateError", err)
		}

		requireEmpty(t, ch.FromCheckpoint(nil))
	})
}
