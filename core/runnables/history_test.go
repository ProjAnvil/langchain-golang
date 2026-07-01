package runnables

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/chathistory"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestRunnableWithMessageHistoryListInput(t *testing.T) {
	ctx := context.Background()
	store := map[string]*chathistory.InMemoryChatMessageHistory{
		"abc": chathistory.NewInMemoryChatMessageHistory(messages.AI("hello")),
	}
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		batch := input.([]messages.Message)
		if len(batch) != 2 || batch[0].Content != "hello" || batch[1].Content != "next" {
			t.Fatalf("input messages: %#v", batch)
		}
		return messages.AI("answer"), nil
	}, schema.Schema{"type": "array"}, schema.Schema{})

	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(store))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	output, err := wrapped.Invoke(ctx, "next", WithConfigurable("session_id", "abc"))
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output.(messages.Message).Content != "answer" {
		t.Fatalf("output: %#v", output)
	}
	history, err := store["abc"].Messages(ctx)
	if err != nil {
		t.Fatalf("history messages: %v", err)
	}
	assertMessageContents(t, history, []string{"hello", "next", "answer"})
}

func TestRunnableWithMessageHistorySeparateHistoryKey(t *testing.T) {
	ctx := context.Background()
	store := map[string]*chathistory.InMemoryChatMessageHistory{
		"thread": chathistory.NewInMemoryChatMessageHistory(messages.Human("old")),
	}
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		values := input.(map[string]any)
		history := values["history"].([]messages.Message)
		if len(history) != 1 || history[0].Content != "old" {
			t.Fatalf("history input: %#v", history)
		}
		if values["question"] != "new" {
			t.Fatalf("question: %#v", values["question"])
		}
		return map[string]any{"message": messages.AI("done")}, nil
	}, schema.Schema{"type": "object"}, schema.Schema{"type": "object"})

	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(store))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	wrapped.InputMessagesKey = "question"
	wrapped.HistoryMessagesKey = "history"
	wrapped.OutputMessagesKey = "message"

	_, err = wrapped.Invoke(ctx, map[string]any{"question": "new"}, WithConfigurable("session_id", "thread"))
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	history, err := store["thread"].Messages(ctx)
	if err != nil {
		t.Fatalf("history messages: %v", err)
	}
	assertMessageContents(t, history, []string{"old", "new", "done"})
}

func TestRunnableWithMessageHistoryStreamUpdatesOnEOF(t *testing.T) {
	ctx := context.Background()
	store := map[string]*chathistory.InMemoryChatMessageHistory{
		"s": chathistory.NewInMemoryChatMessageHistory(),
	}
	base := streamOnlyRunnable{
		stream: []any{"hel", "lo"},
	}
	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(store))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	stream, err := wrapped.Stream(ctx, "hi", WithConfigurable("session_id", "s"))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	for {
		_, ok, err := stream.Next(ctx)
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
	}
	history, err := store["s"].Messages(ctx)
	if err != nil {
		t.Fatalf("history messages: %v", err)
	}
	assertMessageContents(t, history, []string{"hi", "hello"})
}

func TestRunnableWithMessageHistoryMissingConfigurableKey(t *testing.T) {
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		return input, nil
	}, schema.Schema{}, schema.Schema{})
	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(map[string]*chathistory.InMemoryChatMessageHistory{}))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	_, err = wrapped.Invoke(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), "session_id") {
		t.Fatalf("err: %v", err)
	}
}

func historyFactory(store map[string]*chathistory.InMemoryChatMessageHistory) MessageHistoryFactory {
	return func(_ context.Context, values map[string]any) (chathistory.History, error) {
		sessionID := values["session_id"].(string)
		history := store[sessionID]
		if history == nil {
			history = chathistory.NewInMemoryChatMessageHistory()
			store[sessionID] = history
		}
		return history, nil
	}
}

func assertMessageContents(t *testing.T, got []messages.Message, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len got %d want %d: %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].Content != want[i] {
			t.Fatalf("message[%d] got %q want %q", i, got[i].Content, want[i])
		}
	}
}

type streamOnlyRunnable struct {
	stream []any
}

func (r streamOnlyRunnable) Invoke(context.Context, any, ...Option) (any, error) {
	return nil, nil
}

func (r streamOnlyRunnable) Batch(context.Context, []any, ...Option) ([]any, error) {
	return nil, nil
}

func (r streamOnlyRunnable) Stream(context.Context, any, ...Option) (Stream[any], error) {
	return NewSliceStream(r.stream), nil
}

func (r streamOnlyRunnable) InputSchema() schema.Schema {
	return schema.Schema{}
}

func (r streamOnlyRunnable) OutputSchema() schema.Schema {
	return schema.Schema{}
}
