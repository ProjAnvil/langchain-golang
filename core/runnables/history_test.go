package runnables

import (
	"context"
	"reflect"
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

func TestNewRunnableWithMessageHistoryErrors(t *testing.T) {
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		return input, nil
	}, schema.Schema{}, schema.Schema{})
	factory := historyFactory(map[string]*chathistory.InMemoryChatMessageHistory{})

	if _, err := NewRunnableWithMessageHistory(nil, factory); err == nil {
		t.Fatal("expected error for nil runnable")
	}
	if _, err := NewRunnableWithMessageHistory(base, nil); err == nil {
		t.Fatal("expected error for nil factory")
	}
}

func TestRunnableWithMessageHistoryBatchAndSchemas(t *testing.T) {
	ctx := context.Background()
	store := map[string]*chathistory.InMemoryChatMessageHistory{}
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		batch := input.([]messages.Message)
		return messages.AI(batch[len(batch)-1].Content + "?"), nil
	}, schema.Schema{"type": "array", "description": "in"}, schema.Schema{"description": "out"})

	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(store))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	got, err := wrapped.Batch(ctx, []any{"one", "two"}, WithConfigurable("session_id", "s"))
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if len(got) != 2 || got[0].(messages.Message).Content != "one?" || got[1].(messages.Message).Content != "two?" {
		t.Fatalf("batch got %#v", got)
	}

	if wrapped.InputSchema()["description"] != "in" {
		t.Fatalf("input schema: %#v", wrapped.InputSchema())
	}
	if wrapped.OutputSchema()["description"] != "out" {
		t.Fatalf("output schema: %#v", wrapped.OutputSchema())
	}
}

func TestRunnableWithMessageHistoryFactoryFailures(t *testing.T) {
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		return input, nil
	}, schema.Schema{}, schema.Schema{})

	failing, err := NewRunnableWithMessageHistory(base, func(context.Context, map[string]any) (chathistory.History, error) {
		return nil, errTestSentinel
	})
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	if _, err := failing.Invoke(context.Background(), "x", WithConfigurable("session_id", "s")); err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}

	nilFactory, err := NewRunnableWithMessageHistory(base, func(context.Context, map[string]any) (chathistory.History, error) {
		return nil, nil
	})
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	if _, err := nilFactory.Invoke(context.Background(), "x", WithConfigurable("session_id", "s")); err == nil {
		t.Fatal("expected error for nil history from factory")
	}
}

func TestRunnableWithMessageHistoryUsesProvidedHistory(t *testing.T) {
	ctx := context.Background()
	history := chathistory.NewInMemoryChatMessageHistory()
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		return messages.AI("done"), nil
	}, schema.Schema{}, schema.Schema{})

	wrapped, err := NewRunnableWithMessageHistory(base, func(context.Context, map[string]any) (chathistory.History, error) {
		t.Fatal("factory must not be called when message_history is provided")
		return nil, nil
	})
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("message_history", chathistory.History(history))); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	assertMessageContents(t, mustMessages(t, history), []string{"hi", "done"})
}

func TestRunnableWithMessageHistoryCustomFactoryKeys(t *testing.T) {
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		return messages.AI("ok"), nil
	}, schema.Schema{}, schema.Schema{})
	wrapped, err := NewRunnableWithMessageHistory(base, func(_ context.Context, values map[string]any) (chathistory.History, error) {
		if values["user_id"] != "u1" {
			t.Fatalf("factory values: %#v", values)
		}
		return chathistory.NewInMemoryChatMessageHistory(), nil
	}, "user_id")
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}

	if _, err := wrapped.Invoke(context.Background(), "hi"); err == nil || !strings.Contains(err.Error(), "user_id") {
		t.Fatalf("expected missing key error, got %v", err)
	}
	if _, err := wrapped.Invoke(context.Background(), "hi", WithConfigurable("user_id", "u1")); err != nil {
		t.Fatalf("invoke: %v", err)
	}
}

func TestRunnableWithMessageHistoryPrepareInputErrors(t *testing.T) {
	ctx := context.Background()
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		return messages.AI("ok"), nil
	}, schema.Schema{}, schema.Schema{})

	// HistoryMessagesKey requires map input.
	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(map[string]*chathistory.InMemoryChatMessageHistory{}))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	wrapped.HistoryMessagesKey = "history"
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("session_id", "s")); err == nil {
		t.Fatal("expected error for non-map input with history_messages_key")
	}

	// InputMessagesKey requires map input.
	wrapped.HistoryMessagesKey = ""
	wrapped.InputMessagesKey = "question"
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("session_id", "s")); err == nil {
		t.Fatal("expected error for non-map input with input_messages_key")
	}

	// A failing Messages() call surfaces the history error.
	brokenHistory := &chathistory.BaseChatMessageHistory{
		MessagesFunc: func(context.Context) ([]messages.Message, error) {
			return nil, errTestSentinel
		},
	}
	if _, err := wrapped.Invoke(ctx, map[string]any{"question": "hi"}, WithConfigurable("message_history", chathistory.History(brokenHistory))); err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}
}

func TestRunnableWithMessageHistoryMapInputKeySelection(t *testing.T) {
	ctx := context.Background()
	var seenInput any
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		seenInput = input
		return map[string]any{"answer": messages.AI("done")}, nil
	}, schema.Schema{}, schema.Schema{})
	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(map[string]*chathistory.InMemoryChatMessageHistory{}))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}

	// A single-key map uses that key as the message key; the wrapped runnable
	// receives historic + new messages as a plain list.
	if _, err := wrapped.Invoke(ctx, map[string]any{"question": "hi"}, WithConfigurable("session_id", "s")); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	batch, ok := seenInput.([]messages.Message)
	if !ok || len(batch) != 1 || batch[0].Content != "hi" {
		t.Fatalf("seen input: %#v", seenInput)
	}

	// A multi-key map falls back to the "input" key.
	_, err = wrapped.Invoke(ctx, map[string]any{"input": "next", "other": 1}, WithConfigurable("session_id", "s"))
	if err != nil {
		t.Fatalf("multi-key invoke: %v", err)
	}

	// Output map with a single key uses that key for the output message.
	history, err := wrapped.GetSessionHistory(ctx, map[string]any{"session_id": "s"})
	if err != nil {
		t.Fatalf("get history: %v", err)
	}
	got, err := history.Messages(ctx)
	if err != nil {
		t.Fatalf("messages: %v", err)
	}
	assertMessageContents(t, got, []string{"hi", "done", "next", "done"})
}

func TestInputMessageVariants(t *testing.T) {
	ctx := context.Background()
	var seenInput any
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		seenInput = input
		return "ok", nil
	}, schema.Schema{}, schema.Schema{})
	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(map[string]*chathistory.InMemoryChatMessageHistory{}))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	opts := WithConfigurable("session_id", "s")

	// A single message input is accepted directly.
	if _, err := wrapped.Invoke(ctx, messages.Human("direct"), opts); err != nil {
		t.Fatalf("message invoke: %v", err)
	}
	// A typed message slice input is accepted.
	if _, err := wrapped.Invoke(ctx, []messages.Message{messages.Human("a"), messages.AI("b")}, opts); err != nil {
		t.Fatalf("slice invoke: %v", err)
	}
	// An []any of messages is accepted.
	if _, err := wrapped.Invoke(ctx, []any{messages.Human("c")}, opts); err != nil {
		t.Fatalf("any slice invoke: %v", err)
	}
	if seenInput == nil {
		t.Fatal("base runnable was never invoked")
	}

	// An []any containing a non-message is rejected.
	if _, err := wrapped.Invoke(ctx, []any{"not-a-message"}, opts); err == nil {
		t.Fatal("expected error for non-message element")
	}
	// An unsupported input type is rejected.
	if _, err := wrapped.Invoke(ctx, 42, opts); err == nil {
		t.Fatal("expected error for unsupported input type")
	}
}

func TestOutputMessageVariants(t *testing.T) {
	ctx := context.Background()
	newWrapper := func(output any) RunnableWithMessageHistory {
		base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
			return output, nil
		}, schema.Schema{}, schema.Schema{})
		wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(map[string]*chathistory.InMemoryChatMessageHistory{}))
		if err != nil {
			t.Fatalf("new wrapper: %v", err)
		}
		return wrapped
	}

	// String outputs become AI messages.
	wrapped := newWrapper("plain")
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("session_id", "s")); err != nil {
		t.Fatalf("string output invoke: %v", err)
	}
	// Message slice outputs are appended as-is.
	wrapped = newWrapper([]messages.Message{messages.AI("x"), messages.AI("y")})
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("session_id", "s")); err != nil {
		t.Fatalf("slice output invoke: %v", err)
	}
	// An []any output with a non-message element fails after invocation.
	wrapped = newWrapper([]any{"bad"})
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("session_id", "s")); err == nil {
		t.Fatal("expected error for invalid output element")
	}
	// An unsupported output type fails after invocation.
	wrapped = newWrapper(42)
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("session_id", "s")); err == nil {
		t.Fatal("expected error for unsupported output type")
	}
}

// errNextStream fails on the first Next call.
type errNextStream struct{}

func (errNextStream) Next(context.Context) (any, bool, error) { return nil, false, errTestSentinel }
func (errNextStream) Close() error                            { return nil }

type errNextStreamRunnable struct{}

func (errNextStreamRunnable) Invoke(context.Context, any, ...Option) (any, error) { return nil, nil }
func (errNextStreamRunnable) Batch(context.Context, []any, ...Option) ([]any, error) {
	return nil, nil
}
func (errNextStreamRunnable) Stream(context.Context, any, ...Option) (Stream[any], error) {
	return errNextStream{}, nil
}
func (errNextStreamRunnable) InputSchema() schema.Schema  { return schema.Schema{} }
func (errNextStreamRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

func TestRunnableWithMessageHistoryStreamErrorPassthrough(t *testing.T) {
	ctx := context.Background()
	store := map[string]*chathistory.InMemoryChatMessageHistory{"s": chathistory.NewInMemoryChatMessageHistory()}
	wrapped, err := NewRunnableWithMessageHistory(errNextStreamRunnable{}, historyFactory(store))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	stream, err := wrapped.Stream(ctx, "hi", WithConfigurable("session_id", "s"))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if _, _, err := stream.Next(ctx); err != errTestSentinel {
		t.Fatalf("next err: got %v want %v", err, errTestSentinel)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	// The failed stream must not write history.
	assertMessageContents(t, mustMessages(t, store["s"]), nil)
}

func TestRunnableWithMessageHistoryStreamFinalizationErrors(t *testing.T) {
	ctx := context.Background()

	// Output messages that cannot be converted fail the final Next call.
	store := map[string]*chathistory.InMemoryChatMessageHistory{"s": chathistory.NewInMemoryChatMessageHistory()}
	wrapped, err := NewRunnableWithMessageHistory(streamOnlyRunnable{stream: []any{42, 43}}, historyFactory(store))
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
			break
		}
		if !ok {
			t.Fatal("expected finalization error, got clean EOF")
		}
	}
	assertMessageContents(t, mustMessages(t, store["s"]), nil)

	// A failing AddMessages call fails the final Next call.
	brokenHistory := &chathistory.BaseChatMessageHistory{
		MessagesFunc: func(context.Context) ([]messages.Message, error) { return nil, nil },
		AddMessagesFunc: func(context.Context, []messages.Message) error {
			return errTestSentinel
		},
	}
	wrapped, err = NewRunnableWithMessageHistory(streamOnlyRunnable{stream: []any{"a"}}, historyFactory(store))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	stream, err = wrapped.Stream(ctx, "hi", WithConfigurable("message_history", chathistory.History(brokenHistory)))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if _, _, err := stream.Next(ctx); err != nil {
		t.Fatalf("first next: %v", err)
	}
	if _, _, err := stream.Next(ctx); err != errTestSentinel {
		t.Fatalf("final next err: got %v want %v", err, errTestSentinel)
	}
	// A subsequent Next after finalization reports plain EOF.
	if _, ok, err := stream.Next(ctx); err != nil || ok {
		t.Fatalf("post-finalization next: ok=%v err=%v", ok, err)
	}
}

func TestStreamOutputValueCombinations(t *testing.T) {
	if got := streamOutputValue([]any{"solo"}); got != "solo" {
		t.Fatalf("single chunk: %#v", got)
	}
	if got := streamOutputValue([]any{"hel", "lo"}); got != "hello" {
		t.Fatalf("string chunks: %#v", got)
	}
	msgs := streamOutputValue([]any{messages.AI("a"), messages.AI("b")})
	if batch, ok := msgs.([]messages.Message); !ok || len(batch) != 2 {
		t.Fatalf("message chunks: %#v", msgs)
	}
	mixed := []any{"text", messages.AI("a")}
	if got := streamOutputValue(mixed); !reflect.DeepEqual(got, any(mixed)) {
		t.Fatalf("mixed chunks: %#v", got)
	}
	other := []any{1, 2}
	if got := streamOutputValue(other); !reflect.DeepEqual(got, any(other)) {
		t.Fatalf("non-message chunks: %#v", got)
	}
}

func mustMessages(t *testing.T, history chathistory.History) []messages.Message {
	t.Helper()
	got, err := history.Messages(context.Background())
	if err != nil {
		t.Fatalf("messages: %v", err)
	}
	return got
}

func TestRunnableWithMessageHistoryStreamSetupErrors(t *testing.T) {
	ctx := context.Background()
	store := map[string]*chathistory.InMemoryChatMessageHistory{}

	wrapped, err := NewRunnableWithMessageHistory(streamOnlyRunnable{stream: []any{"a"}}, historyFactory(store))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	// Missing factory key fails before streaming.
	if _, err := wrapped.Stream(ctx, "hi"); err == nil {
		t.Fatal("expected missing session_id error")
	}
	// An input that cannot become messages fails before streaming.
	if _, err := wrapped.Stream(ctx, 42, WithConfigurable("session_id", "s")); err == nil {
		t.Fatal("expected input conversion error")
	}

	// A wrapped runnable whose Stream constructor fails propagates the error.
	wrapped, err = NewRunnableWithMessageHistory(streamConstructErrRunnable{}, historyFactory(store))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	if _, err := wrapped.Stream(ctx, "hi", WithConfigurable("session_id", "s")); err != errTestSentinel {
		t.Fatalf("stream err: got %v want %v", err, errTestSentinel)
	}
}

// streamConstructErrRunnable fails Stream construction.
type streamConstructErrRunnable struct{}

func (streamConstructErrRunnable) Invoke(context.Context, any, ...Option) (any, error) {
	return nil, nil
}

func (streamConstructErrRunnable) Batch(context.Context, []any, ...Option) ([]any, error) {
	return nil, nil
}

func (streamConstructErrRunnable) Stream(context.Context, any, ...Option) (Stream[any], error) {
	return nil, errTestSentinel
}

func (streamConstructErrRunnable) InputSchema() schema.Schema  { return schema.Schema{} }
func (streamConstructErrRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

func TestRunnableWithMessageHistoryAddMessagesError(t *testing.T) {
	ctx := context.Background()
	brokenHistory := &chathistory.BaseChatMessageHistory{
		MessagesFunc: func(context.Context) ([]messages.Message, error) { return nil, nil },
		AddMessagesFunc: func(context.Context, []messages.Message) error {
			return errTestSentinel
		},
	}
	base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
		return messages.AI("done"), nil
	}, schema.Schema{}, schema.Schema{})
	wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(map[string]*chathistory.InMemoryChatMessageHistory{}))
	if err != nil {
		t.Fatalf("new wrapper: %v", err)
	}
	_, err = wrapped.Invoke(ctx, "hi", WithConfigurable("message_history", chathistory.History(brokenHistory)))
	if err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}
}

func TestOutputMessageSingleAndAnySlice(t *testing.T) {
	ctx := context.Background()
	newWrapper := func(output any) RunnableWithMessageHistory {
		base := NewFunc(func(_ context.Context, input any, _ ...Option) (any, error) {
			return output, nil
		}, schema.Schema{}, schema.Schema{})
		wrapped, err := NewRunnableWithMessageHistory(base, historyFactory(map[string]*chathistory.InMemoryChatMessageHistory{}))
		if err != nil {
			t.Fatalf("new wrapper: %v", err)
		}
		return wrapped
	}

	// A single message output is appended directly.
	wrapped := newWrapper(messages.AI("single"))
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("session_id", "s")); err != nil {
		t.Fatalf("message output invoke: %v", err)
	}
	// An []any of messages is appended element-wise.
	wrapped = newWrapper([]any{messages.AI("x"), messages.AI("y")})
	if _, err := wrapped.Invoke(ctx, "hi", WithConfigurable("session_id", "s")); err != nil {
		t.Fatalf("any slice output invoke: %v", err)
	}
}
