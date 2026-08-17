package channels

import (
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// update calls ch.Update and fails the test on an unexpected error.
func update(t *testing.T, ch Channel, values ...any) bool {
	t.Helper()
	changed, err := ch.Update(values)
	if err != nil {
		t.Fatalf("Update(%v) error = %v", values, err)
	}
	return changed
}

// get calls ch.Get and fails the test on an unexpected error.
func get(t *testing.T, ch Channel) any {
	t.Helper()
	got, err := ch.Get()
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return got
}

// requireEmpty asserts that ch has no value: Get fails with ErrEmptyChannel,
// IsAvailable is false, and Checkpoint omits the channel.
func requireEmpty(t *testing.T, ch Channel) {
	t.Helper()
	if _, err := ch.Get(); !errors.Is(err, ErrEmptyChannel) {
		t.Fatalf("Get() error = %v, want ErrEmptyChannel", err)
	}
	if ch.IsAvailable() {
		t.Fatal("IsAvailable() = true, want false")
	}
	if _, ok := ch.Checkpoint(); ok {
		t.Fatal("Checkpoint() ok = true, want false (omit from checkpoint)")
	}
}

// requireRoundTrip asserts that a non-empty channel snapshots via Checkpoint
// and restores an equal value through FromCheckpoint, and that
// FromCheckpoint(nil) yields an empty channel.
func requireRoundTrip(t *testing.T, ch Channel) {
	t.Helper()
	cp, ok := ch.Checkpoint()
	if !ok {
		t.Fatal("Checkpoint() ok = false, want true")
	}
	restored := ch.FromCheckpoint(cp)
	if got := get(t, restored); !reflect.DeepEqual(got, get(t, ch)) {
		t.Fatalf("FromCheckpoint(Checkpoint()) Get() = %v, want %v", got, cp)
	}
	// FromCheckpoint must not mutate the receiver.
	if got := get(t, ch); !reflect.DeepEqual(got, cp) {
		t.Fatalf("FromCheckpoint mutated receiver: Get() = %v, want %v", got, cp)
	}
	requireEmpty(t, ch.FromCheckpoint(nil))
}

func TestInvalidUpdateErrorMessage(t *testing.T) {
	err := &InvalidUpdateError{Channel: "LastValue", Reason: "can receive only one value per super-step"}
	got := err.Error()
	want := "channels: invalid update for LastValue channel: can receive only one value per super-step"
	if got != want {
		t.Fatalf("Error() = %q, want %q", got, want)
	}
}

func TestLastValue(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		requireEmpty(t, NewLastValue())
	})

	t.Run("single write", func(t *testing.T) {
		ch := NewLastValue()
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

	t.Run("multiple writes in one step error", func(t *testing.T) {
		ch := NewLastValue()
		_, err := ch.Update([]any{"a", "b"})
		var iuErr *InvalidUpdateError
		if !errors.As(err, &iuErr) {
			t.Fatalf("Update() error = %v, want *InvalidUpdateError", err)
		}
	})

	t.Run("empty update is a no-op", func(t *testing.T) {
		ch := NewLastValue()
		update(t, ch, "a")
		if update(t, ch) {
			t.Fatal("Update([]) changed = true, want false")
		}
		if got := get(t, ch); got != "a" {
			t.Fatalf("Get() = %v, want %q", got, "a")
		}
	})

	t.Run("checkpoint round-trip", func(t *testing.T) {
		ch := NewLastValue()
		update(t, ch, "a")
		requireRoundTrip(t, ch)
	})
}

func TestTopic(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		requireEmpty(t, NewTopic(false))
		requireEmpty(t, NewTopic(true))
	})

	t.Run("single write", func(t *testing.T) {
		ch := NewTopic(false)
		if !update(t, ch, "a") {
			t.Fatal("Update() changed = false, want true")
		}
		if got := get(t, ch); !reflect.DeepEqual(got, []any{"a"}) {
			t.Fatalf("Get() = %v, want %v", got, []any{"a"})
		}
	})

	t.Run("flattens list values one level", func(t *testing.T) {
		ch := NewTopic(false)
		update(t, ch, []any{"a", "b"}, "c")
		if got := get(t, ch); !reflect.DeepEqual(got, []any{"a", "b", "c"}) {
			t.Fatalf("Get() = %v, want %v", got, []any{"a", "b", "c"})
		}
	})

	t.Run("accumulate=true keeps prior values", func(t *testing.T) {
		ch := NewTopic(true)
		update(t, ch, "a")
		update(t, ch, "b")
		if got := get(t, ch); !reflect.DeepEqual(got, []any{"a", "b"}) {
			t.Fatalf("Get() = %v, want %v", got, []any{"a", "b"})
		}
		if update(t, ch) {
			t.Fatal("Update([]) changed = true, want false")
		}
		if got := get(t, ch); !reflect.DeepEqual(got, []any{"a", "b"}) {
			t.Fatalf("Get() after empty Update = %v, want %v", got, []any{"a", "b"})
		}
	})

	t.Run("accumulate=false replaces prior values", func(t *testing.T) {
		ch := NewTopic(false)
		update(t, ch, "a")
		update(t, ch, "b")
		if got := get(t, ch); !reflect.DeepEqual(got, []any{"b"}) {
			t.Fatalf("Get() = %v, want %v", got, []any{"b"})
		}
	})

	t.Run("accumulate=false empty update clears", func(t *testing.T) {
		ch := NewTopic(false)
		update(t, ch, "a")
		if !update(t, ch) {
			t.Fatal("Update([]) changed = false, want true (cleared stored values)")
		}
		requireEmpty(t, ch)
	})

	t.Run("checkpoint round-trip", func(t *testing.T) {
		for _, accumulate := range []bool{false, true} {
			ch := NewTopic(accumulate)
			update(t, ch, "a", []any{"b"})
			requireRoundTrip(t, ch)
		}
	})
}

func TestBinaryOperator(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		requireEmpty(t, NewBinaryOperator(AppendSliceReducer))
	})

	t.Run("first value seeds without applying op", func(t *testing.T) {
		ch := NewBinaryOperator(AppendSliceReducer)
		if !update(t, ch, []int{1}) {
			t.Fatal("Update() changed = false, want true")
		}
		if got := get(t, ch); !reflect.DeepEqual(got, []int{1}) {
			t.Fatalf("Get() = %v, want %v", got, []int{1})
		}
	})

	t.Run("folds values left-to-right", func(t *testing.T) {
		ch := NewBinaryOperator(AppendSliceReducer)
		update(t, ch, []int{1})
		update(t, ch, []int{2}, []int{3})
		if got := get(t, ch); !reflect.DeepEqual(got, []int{1, 2, 3}) {
			t.Fatalf("Get() = %v, want %v", got, []int{1, 2, 3})
		}
	})

	t.Run("folds messages by ID", func(t *testing.T) {
		ch := NewBinaryOperator(MessagesReducer)
		update(t, ch, []messages.Message{withID(messages.AI("old"), "a")})
		update(t, ch, []messages.Message{
			withID(messages.AI("new"), "a"),
			withID(messages.Human("hi"), "b"),
		})
		want := []messages.Message{
			withID(messages.AI("new"), "a"),
			withID(messages.Human("hi"), "b"),
		}
		if got := get(t, ch); !reflect.DeepEqual(got, want) {
			t.Fatalf("Get() = %v, want %v", got, want)
		}
	})

	t.Run("empty update is a no-op", func(t *testing.T) {
		ch := NewBinaryOperator(AppendSliceReducer)
		update(t, ch, []int{1})
		if update(t, ch) {
			t.Fatal("Update([]) changed = true, want false")
		}
		if got := get(t, ch); !reflect.DeepEqual(got, []int{1}) {
			t.Fatalf("Get() = %v, want %v", got, []int{1})
		}
	})

	t.Run("reducer error propagates", func(t *testing.T) {
		ch := NewBinaryOperator(AppendSliceReducer)
		update(t, ch, []int{1})
		if _, err := ch.Update([]any{"not-a-slice-of-int"}); err == nil {
			t.Fatal("Update() error = nil, want reducer error")
		}
	})

	t.Run("checkpoint round-trip", func(t *testing.T) {
		ch := NewBinaryOperator(AppendSliceReducer)
		update(t, ch, []int{1}, []int{2})
		requireRoundTrip(t, ch)
	})
}

func TestEphemeral(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		requireEmpty(t, NewEphemeral(false))
		requireEmpty(t, NewEphemeral(true))
	})

	t.Run("single write", func(t *testing.T) {
		ch := NewEphemeral(true)
		if !update(t, ch, "a") {
			t.Fatal("Update() changed = false, want true")
		}
		if got := get(t, ch); got != "a" {
			t.Fatalf("Get() = %v, want %q", got, "a")
		}
	})

	t.Run("guard=true errors on multiple writes", func(t *testing.T) {
		ch := NewEphemeral(true)
		_, err := ch.Update([]any{"a", "b"})
		var iuErr *InvalidUpdateError
		if !errors.As(err, &iuErr) {
			t.Fatalf("Update() error = %v, want *InvalidUpdateError", err)
		}
	})

	t.Run("guard=false takes the last write", func(t *testing.T) {
		ch := NewEphemeral(false)
		update(t, ch, "a", "b")
		if got := get(t, ch); got != "b" {
			t.Fatalf("Get() = %v, want %q", got, "b")
		}
	})

	t.Run("empty update clears the stored value", func(t *testing.T) {
		ch := NewEphemeral(false)
		update(t, ch, "a")
		if !update(t, ch) {
			t.Fatal("Update([]) changed = false, want true (expired stored value)")
		}
		requireEmpty(t, ch)
		if update(t, ch) {
			t.Fatal("Update([]) on empty channel changed = true, want false")
		}
	})

	t.Run("checkpoint round-trip", func(t *testing.T) {
		ch := NewEphemeral(false)
		update(t, ch, "a")
		requireRoundTrip(t, ch)
	})
}
