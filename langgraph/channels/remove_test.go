package channels

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestMessagesReducerRemoveByID(t *testing.T) {
	existing := []messages.Message{
		withID(messages.Human("hi"), "1"),
		withID(messages.AI("hello"), "2"),
		withID(messages.Human("bye"), "3"),
	}
	update := []messages.MessageUpdate{
		messages.RemoveMessage{ID: "2"},
	}

	got, err := MessagesReducer(existing, update)
	if err != nil {
		t.Fatalf("MessagesReducer() error = %v", err)
	}
	merged, ok := got.([]messages.Message)
	if !ok {
		t.Fatalf("MessagesReducer() returned %T, want []messages.Message", got)
	}
	if len(merged) != 2 {
		t.Fatalf("expected 2 messages after removal, got %d: %+v", len(merged), merged)
	}
	if merged[0].ID != "1" || merged[1].ID != "3" {
		t.Fatalf("expected IDs [1 3], got [%s %s]", merged[0].ID, merged[1].ID)
	}
}

func TestMessagesReducerRemoveAllEmptyID(t *testing.T) {
	existing := []messages.Message{
		withID(messages.Human("hi"), "1"),
		withID(messages.AI("hello"), "2"),
	}

	got, err := MessagesReducer(existing, []messages.MessageUpdate{messages.RemoveMessage{ID: ""}})
	if err != nil {
		t.Fatalf("MessagesReducer() error = %v", err)
	}
	merged := got.([]messages.Message)
	if len(merged) != 0 {
		t.Fatalf("expected empty list after remove-all, got %d: %+v", len(merged), merged)
	}
}

func TestMessagesReducerRemoveAllKeepsSubsequent(t *testing.T) {
	existing := []messages.Message{withID(messages.Human("hi"), "1")}
	update := []messages.MessageUpdate{
		messages.Human("dropped"),
		messages.RemoveMessage{ID: ""},
		messages.Human("kept"),
	}

	got, err := MessagesReducer(existing, update)
	if err != nil {
		t.Fatalf("MessagesReducer() error = %v", err)
	}
	merged := got.([]messages.Message)
	if len(merged) != 1 || merged[0].Content != "kept" {
		t.Fatalf("expected only the message after the remove-all sentinel, got %+v", merged)
	}
}

func TestMessagesReducerRemoveUnknownID(t *testing.T) {
	existing := []messages.Message{withID(messages.Human("hi"), "1")}
	if _, err := MessagesReducer(existing, []messages.MessageUpdate{messages.RemoveMessage{ID: "nope"}}); err == nil {
		t.Fatal("expected error when removing an ID that doesn't exist")
	}
}
