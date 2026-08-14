package channels

import (
	"testing"
)

func TestAnyValue(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		requireEmpty(t, NewAnyValue())
	})

	t.Run("single write", func(t *testing.T) {
		ch := NewAnyValue()
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

	t.Run("multiple writes take the last", func(t *testing.T) {
		ch := NewAnyValue()
		if !update(t, ch, "a", "b", "c") {
			t.Fatal("Update() changed = false, want true")
		}
		if got := get(t, ch); got != "c" {
			t.Fatalf("Get() = %v, want %q", got, "c")
		}
	})

	t.Run("empty update is a no-op when unset", func(t *testing.T) {
		ch := NewAnyValue()
		if update(t, ch) {
			t.Fatal("Update([]) changed = true, want false")
		}
		requireEmpty(t, ch)
	})

	t.Run("empty update clears a set value", func(t *testing.T) {
		ch := NewAnyValue()
		update(t, ch, "a")
		if !update(t, ch) {
			t.Fatal("Update([]) changed = false, want true (cleared stored value)")
		}
		requireEmpty(t, ch)
	})

	t.Run("checkpoint round-trip", func(t *testing.T) {
		ch := NewAnyValue()
		update(t, ch, "a", "b")
		requireRoundTrip(t, ch)
	})
}
