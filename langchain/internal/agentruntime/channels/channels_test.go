package channels

import (
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	lcchannels "github.com/projanvil/langchain-golang/langgraph/channels"
)

// TestReducerAliasesAreIdentical verifies the shim's reducer variables still
// point at the langgraph/channels implementations, so a graph wired through
// this package gets exactly the upstream merge semantics.
func TestReducerAliasesAreIdentical(t *testing.T) {
	if reflect.ValueOf(LastValueReducer).Pointer() != reflect.ValueOf(lcchannels.LastValueReducer).Pointer() {
		t.Error("LastValueReducer does not delegate to langgraph/channels.LastValueReducer")
	}
	if reflect.ValueOf(AppendSliceReducer).Pointer() != reflect.ValueOf(lcchannels.AppendSliceReducer).Pointer() {
		t.Error("AppendSliceReducer does not delegate to langgraph/channels.AppendSliceReducer")
	}
	if reflect.ValueOf(MessagesReducer).Pointer() != reflect.ValueOf(lcchannels.MessagesReducer).Pointer() {
		t.Error("MessagesReducer does not delegate to langgraph/channels.MessagesReducer")
	}

	// The Reducer alias must remain a true alias, not a distinct func type.
	var r Reducer = lcchannels.Reducer(LastValueReducer)
	var _ lcchannels.Reducer = r
}

// TestLastValueReducer exercises the "last write wins" channel behavior
// through the shim.
func TestLastValueReducer(t *testing.T) {
	got, err := LastValueReducer("old", "new")
	if err != nil {
		t.Fatalf("LastValueReducer() error = %v", err)
	}
	if got != "new" {
		t.Errorf("LastValueReducer(old, new) = %v, want new", got)
	}

	got, err = LastValueReducer("old", nil)
	if err != nil {
		t.Fatalf("LastValueReducer() error = %v", err)
	}
	if got != nil {
		t.Errorf("LastValueReducer(old, nil) = %v, want nil", got)
	}
}

// TestAppendSliceReducer exercises the accumulate channel behavior through
// the shim, including its type-checking error paths.
func TestAppendSliceReducer(t *testing.T) {
	t.Run("nil existing yields update", func(t *testing.T) {
		update := []string{"a", "b"}
		got, err := AppendSliceReducer(nil, update)
		if err != nil {
			t.Fatalf("AppendSliceReducer() error = %v", err)
		}
		if !reflect.DeepEqual(got, update) {
			t.Errorf("AppendSliceReducer(nil, %v) = %v", update, got)
		}
	})

	t.Run("nil update keeps existing", func(t *testing.T) {
		existing := []int{1}
		got, err := AppendSliceReducer(existing, nil)
		if err != nil {
			t.Fatalf("AppendSliceReducer() error = %v", err)
		}
		if !reflect.DeepEqual(got, existing) {
			t.Errorf("AppendSliceReducer(%v, nil) = %v", existing, got)
		}
	})

	t.Run("concatenates matching slice types", func(t *testing.T) {
		got, err := AppendSliceReducer([]string{"a"}, []string{"b", "c"})
		if err != nil {
			t.Fatalf("AppendSliceReducer() error = %v", err)
		}
		if !reflect.DeepEqual(got, []string{"a", "b", "c"}) {
			t.Errorf("AppendSliceReducer() = %v, want [a b c]", got)
		}
	})

	t.Run("rejects non-slice values", func(t *testing.T) {
		if _, err := AppendSliceReducer("a", []string{"b"}); err == nil {
			t.Error("expected error for non-slice existing value")
		}
		if _, err := AppendSliceReducer([]string{"a"}, 42); err == nil {
			t.Error("expected error for non-slice update value")
		}
	})

	t.Run("rejects mismatched slice types", func(t *testing.T) {
		_, err := AppendSliceReducer([]string{"a"}, []int{1})
		if err == nil {
			t.Fatal("expected error for mismatched slice types")
		}
		if !strings.Contains(err.Error(), "matching slice types") {
			t.Errorf("error = %v, want it to mention matching slice types", err)
		}
	})
}

// TestMessagesReducer exercises the add_messages channel behavior through the
// shim: merge by ID, removal sentinels, and the type-checking error paths.
func TestMessagesReducer(t *testing.T) {
	t.Run("nil handling", func(t *testing.T) {
		got, err := MessagesReducer(nil, nil)
		if err != nil {
			t.Fatalf("MessagesReducer(nil, nil) error = %v", err)
		}
		if msgs := got.([]messages.Message); len(msgs) != 0 {
			t.Errorf("MessagesReducer(nil, nil) = %v, want empty", msgs)
		}

		update := []messages.Message{messages.Human("hi")}
		got, err = MessagesReducer(nil, update)
		if err != nil {
			t.Fatalf("MessagesReducer() error = %v", err)
		}
		if !reflect.DeepEqual(got, update) {
			t.Errorf("MessagesReducer(nil, update) = %v", got)
		}
	})

	t.Run("appends new IDs and replaces matching IDs in place", func(t *testing.T) {
		m1 := messages.Human("first")
		m1.ID = "m1"
		m2 := messages.AI("second")
		m2.ID = "m2"
		m2new := messages.AI("second revised")
		m2new.ID = "m2"
		m3 := messages.Human("third")
		m3.ID = "m3"

		got, err := MessagesReducer([]messages.Message{m1, m2}, []messages.Message{m2new, m3})
		if err != nil {
			t.Fatalf("MessagesReducer() error = %v", err)
		}
		want := []messages.Message{m1, m2new, m3}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("MessagesReducer() = %+v, want %+v", got, want)
		}
	})

	t.Run("RemoveMessage deletes by ID", func(t *testing.T) {
		m1 := messages.Human("keep")
		m1.ID = "m1"
		m2 := messages.AI("drop")
		m2.ID = "m2"

		got, err := MessagesReducer([]messages.Message{m1, m2}, []messages.RemoveMessage{{ID: "m2"}})
		if err != nil {
			t.Fatalf("MessagesReducer() error = %v", err)
		}
		if !reflect.DeepEqual(got, []messages.Message{m1}) {
			t.Errorf("MessagesReducer() = %+v, want only m1 kept", got)
		}
	})

	t.Run("RemoveMessage with empty ID removes all and keeps following updates", func(t *testing.T) {
		m1 := messages.Human("old")
		m1.ID = "m1"
		fresh := messages.Human("fresh start")

		got, err := MessagesReducer([]messages.Message{m1}, []messages.MessageUpdate{
			messages.RemoveMessage{ID: ""},
			fresh,
		})
		if err != nil {
			t.Fatalf("MessagesReducer() error = %v", err)
		}
		if !reflect.DeepEqual(got, []messages.Message{fresh}) {
			t.Errorf("MessagesReducer() = %+v, want only the post-remove-all message", got)
		}
	})

	t.Run("removing a nonexistent ID errors", func(t *testing.T) {
		_, err := MessagesReducer(nil, []messages.RemoveMessage{{ID: "missing"}})
		if err == nil {
			t.Fatal("expected error removing a message ID that does not exist")
		}
		if !strings.Contains(err.Error(), "missing") {
			t.Errorf("error = %v, want it to name the missing ID", err)
		}
	})

	t.Run("rejects wrong value types", func(t *testing.T) {
		if _, err := MessagesReducer("not-messages", nil); err == nil {
			t.Error("expected error for non-[]messages.Message existing value")
		}
		if _, err := MessagesReducer(nil, 42); err == nil {
			t.Error("expected error for unsupported update type")
		}
	})
}
