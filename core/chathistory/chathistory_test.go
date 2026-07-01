package chathistory

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestAddMessageImplementationOnlySupportsBulkAdd(t *testing.T) {
	ctx := context.Background()
	store := []messages.Message{}
	history := &BaseChatMessageHistory{
		AddMessageFunc: func(_ context.Context, message messages.Message) error {
			store = append(store, message)
			return nil
		},
		ClearFunc: func(context.Context) error { return nil },
	}

	if err := history.AddMessage(ctx, messages.Human("Hello")); err != nil {
		t.Fatalf("add message: %v", err)
	}
	if err := history.AddMessage(ctx, messages.Human("World")); err != nil {
		t.Fatalf("add message: %v", err)
	}
	if err := history.AddMessages(ctx, []messages.Message{
		messages.Human("Hello"),
		messages.Human("World"),
	}); err != nil {
		t.Fatalf("add messages: %v", err)
	}

	want := []messages.Message{
		messages.Human("Hello"),
		messages.Human("World"),
		messages.Human("Hello"),
		messages.Human("World"),
	}
	if !reflect.DeepEqual(store, want) {
		t.Fatalf("store mismatch:\n got %#v\nwant %#v", store, want)
	}
}

func TestBulkMessageImplementationOnlySupportsSingleAdd(t *testing.T) {
	ctx := context.Background()
	store := []messages.Message{}
	history := &BaseChatMessageHistory{
		AddMessagesFunc: func(_ context.Context, batch []messages.Message) error {
			store = append(store, batch...)
			return nil
		},
		ClearFunc: func(context.Context) error { return nil },
	}

	if err := history.AddMessage(ctx, messages.Human("Hello")); err != nil {
		t.Fatalf("add message: %v", err)
	}
	if err := history.AddMessage(ctx, messages.Human("World")); err != nil {
		t.Fatalf("add message: %v", err)
	}
	if err := history.AddMessages(ctx, []messages.Message{
		messages.Human("Hello"),
		messages.Human("World"),
	}); err != nil {
		t.Fatalf("add messages: %v", err)
	}

	want := []messages.Message{
		messages.Human("Hello"),
		messages.Human("World"),
		messages.Human("Hello"),
		messages.Human("World"),
	}
	if !reflect.DeepEqual(store, want) {
		t.Fatalf("store mismatch:\n got %#v\nwant %#v", store, want)
	}
}

func TestInMemoryChatMessageHistory(t *testing.T) {
	ctx := context.Background()
	history := NewInMemoryChatMessageHistory()

	if err := history.AddMessages(ctx, []messages.Message{
		messages.Human("Hello"),
		messages.Human("World"),
	}); err != nil {
		t.Fatalf("add messages: %v", err)
	}
	if err := history.AddMessage(ctx, messages.Human("!")); err != nil {
		t.Fatalf("add message: %v", err)
	}

	got, err := history.Messages(ctx)
	if err != nil {
		t.Fatalf("messages: %v", err)
	}
	want := []messages.Message{
		messages.Human("Hello"),
		messages.Human("World"),
		messages.Human("!"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("messages mismatch:\n got %#v\nwant %#v", got, want)
	}

	got[0] = messages.Human("mutated")
	again, err := history.Messages(ctx)
	if err != nil {
		t.Fatalf("messages again: %v", err)
	}
	if again[0].Content != "Hello" {
		t.Fatalf("history exposed internal message slice: got %q", again[0].Content)
	}

	if err := history.Clear(ctx); err != nil {
		t.Fatalf("clear: %v", err)
	}
	empty, err := history.Messages(ctx)
	if err != nil {
		t.Fatalf("messages after clear: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("expected empty history, got %#v", empty)
	}
}
