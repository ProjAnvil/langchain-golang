package channels

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func withID(msg messages.Message, id string) messages.Message {
	msg.ID = id
	return msg
}

func TestLastValueReducer(t *testing.T) {
	got, err := LastValueReducer("old", "new")
	if err != nil {
		t.Fatalf("LastValueReducer() error = %v", err)
	}
	if got != "new" {
		t.Fatalf("LastValueReducer() = %v, want %q", got, "new")
	}
}

func TestAppendSliceReducer(t *testing.T) {
	existing := []string{"a", "b"}
	update := []string{"c"}
	got, err := AppendSliceReducer(existing, update)
	if err != nil {
		t.Fatalf("AppendSliceReducer() error = %v", err)
	}
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("AppendSliceReducer() = %v, want %v", got, want)
	}

	// existing untouched (no aliasing).
	existing[0] = "mutated"
	if got.([]string)[0] != "a" {
		t.Fatalf("AppendSliceReducer() result aliases existing slice: %v", got)
	}
}

func TestAppendSliceReducerNilHandling(t *testing.T) {
	got, err := AppendSliceReducer(nil, []int{1, 2})
	if err != nil {
		t.Fatalf("AppendSliceReducer() error = %v", err)
	}
	if !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("AppendSliceReducer() = %v", got)
	}

	got, err = AppendSliceReducer([]int{1}, nil)
	if err != nil {
		t.Fatalf("AppendSliceReducer() error = %v", err)
	}
	if !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("AppendSliceReducer() = %v", got)
	}
}

func TestAppendSliceReducerTypeMismatch(t *testing.T) {
	if _, err := AppendSliceReducer([]int{1}, []string{"a"}); err == nil {
		t.Fatal("expected error for mismatched slice element types")
	}
	if _, err := AppendSliceReducer(5, []int{1}); err == nil {
		t.Fatal("expected error for non-slice existing value")
	}
}

func TestMessagesReducerAppendsAndReplacesByID(t *testing.T) {
	existing := []messages.Message{
		withID(messages.Human("hi"), "1"),
		withID(messages.AI("hello"), "2"),
	}
	update := []messages.Message{
		withID(messages.AI("hello again"), "2"), // replaces id=2 in place
		messages.Human("new message"),           // no id, appended
	}

	got, err := MessagesReducer(existing, update)
	if err != nil {
		t.Fatalf("MessagesReducer() error = %v", err)
	}
	merged, ok := got.([]messages.Message)
	if !ok {
		t.Fatalf("MessagesReducer() returned %T, want []messages.Message", got)
	}
	if len(merged) != 3 {
		t.Fatalf("expected 3 messages, got %d: %+v", len(merged), merged)
	}
	if merged[0].ID != "1" || merged[0].Content != "hi" {
		t.Fatalf("expected first message unchanged, got %+v", merged[0])
	}
	if merged[1].ID != "2" || merged[1].Content != "hello again" {
		t.Fatalf("expected second message replaced in place, got %+v", merged[1])
	}
	if merged[2].Content != "new message" {
		t.Fatalf("expected third message appended, got %+v", merged[2])
	}

	// existing untouched (no aliasing).
	if existing[1].Content != "hello" {
		t.Fatalf("MessagesReducer mutated existing slice: %+v", existing)
	}
}

func TestMessagesReducerNilHandling(t *testing.T) {
	got, err := MessagesReducer(nil, []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("MessagesReducer() error = %v", err)
	}
	merged := got.([]messages.Message)
	if len(merged) != 1 {
		t.Fatalf("expected 1 message, got %d", len(merged))
	}

	got, err = MessagesReducer([]messages.Message{messages.Human("hi")}, nil)
	if err != nil {
		t.Fatalf("MessagesReducer() error = %v", err)
	}
	merged = got.([]messages.Message)
	if len(merged) != 1 {
		t.Fatalf("expected 1 message, got %d", len(merged))
	}
}

func TestMessagesReducerTypeMismatch(t *testing.T) {
	if _, err := MessagesReducer("not messages", nil); err == nil {
		t.Fatal("expected error for non-message existing value")
	}
	if _, err := MessagesReducer(nil, "not messages"); err == nil {
		t.Fatal("expected error for non-message update value")
	}
}
