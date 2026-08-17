package chathistory

import (
	"context"
	"errors"
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

func TestBaseChatMessageHistoryMessages(t *testing.T) {
	ctx := context.Background()
	want := []messages.Message{messages.Human("hi")}

	history := &BaseChatMessageHistory{
		MessagesFunc: func(context.Context) ([]messages.Message, error) {
			return want, nil
		},
	}
	got, err := history.Messages(ctx)
	if err != nil {
		t.Fatalf("messages: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("messages mismatch:\n got %#v\nwant %#v", got, want)
	}

	bare := &BaseChatMessageHistory{}
	if _, err := bare.Messages(ctx); !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("expected ErrNotImplemented, got %v", err)
	}
}

func TestBaseChatMessageHistoryUnimplemented(t *testing.T) {
	ctx := context.Background()
	history := &BaseChatMessageHistory{}

	if err := history.AddMessage(ctx, messages.Human("hi")); !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("AddMessage: expected ErrNotImplemented, got %v", err)
	}
	if err := history.AddMessages(ctx, []messages.Message{messages.Human("hi")}); !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("AddMessages: expected ErrNotImplemented, got %v", err)
	}
	if err := history.Clear(ctx); !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Clear: expected ErrNotImplemented, got %v", err)
	}

	cleared := false
	history.ClearFunc = func(context.Context) error {
		cleared = true
		return nil
	}
	if err := history.Clear(ctx); err != nil {
		t.Fatalf("Clear: %v", err)
	}
	if !cleared {
		t.Fatal("ClearFunc was not invoked")
	}
}

func TestBaseChatMessageHistoryAddMessagesPropagatesError(t *testing.T) {
	ctx := context.Background()
	boom := errors.New("boom")
	history := &BaseChatMessageHistory{
		AddMessageFunc: func(_ context.Context, message messages.Message) error {
			if message.Content == "bad" {
				return boom
			}
			return nil
		},
	}

	err := history.AddMessages(ctx, []messages.Message{
		messages.Human("ok"),
		messages.Human("bad"),
		messages.Human("never reached"),
	})
	if !errors.Is(err, boom) {
		t.Fatalf("expected boom, got %v", err)
	}
}

func TestBaseChatMessageHistoryAddUserAndAIMessage(t *testing.T) {
	ctx := context.Background()
	store := []messages.Message{}
	history := &BaseChatMessageHistory{
		AddMessageFunc: func(_ context.Context, message messages.Message) error {
			store = append(store, message)
			return nil
		},
	}

	if err := history.AddUserMessage(ctx, "hello"); err != nil {
		t.Fatalf("add user message: %v", err)
	}
	if err := history.AddAIMessage(ctx, "hi there"); err != nil {
		t.Fatalf("add ai message: %v", err)
	}

	want := []messages.Message{
		messages.Human("hello"),
		messages.AI("hi there"),
	}
	if !reflect.DeepEqual(store, want) {
		t.Fatalf("store mismatch:\n got %#v\nwant %#v", store, want)
	}
}

func TestBaseChatMessageHistoryString(t *testing.T) {
	history := &BaseChatMessageHistory{
		MessagesFunc: func(context.Context) ([]messages.Message, error) {
			return []messages.Message{
				messages.Human("hello"),
				messages.AI("hi"),
			}, nil
		},
	}
	if got, want := history.String(), "Human: hello\nAI: hi"; got != want {
		t.Fatalf("String = %q, want %q", got, want)
	}

	bare := &BaseChatMessageHistory{}
	if got := bare.String(); got != "" {
		t.Fatalf("String without MessagesFunc = %q, want empty", got)
	}

	failing := &BaseChatMessageHistory{
		MessagesFunc: func(context.Context) ([]messages.Message, error) {
			return nil, errors.New("boom")
		},
	}
	if got := failing.String(); got != "" {
		t.Fatalf("String with failing MessagesFunc = %q, want empty", got)
	}
}

func TestInMemoryChatMessageHistoryConvenienceMethods(t *testing.T) {
	ctx := context.Background()
	history := NewInMemoryChatMessageHistory(messages.System("be nice"))

	if err := history.AddUserMessage(ctx, "hello"); err != nil {
		t.Fatalf("add user message: %v", err)
	}
	if err := history.AddAIMessage(ctx, "hi there"); err != nil {
		t.Fatalf("add ai message: %v", err)
	}

	got, err := history.Messages(ctx)
	if err != nil {
		t.Fatalf("messages: %v", err)
	}
	want := []messages.Message{
		messages.System("be nice"),
		messages.Human("hello"),
		messages.AI("hi there"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("messages mismatch:\n got %#v\nwant %#v", got, want)
	}

	if got, want := history.String(), "System: be nice\nHuman: hello\nAI: hi there"; got != want {
		t.Fatalf("String = %q, want %q", got, want)
	}
}

func TestInMemoryChatMessageHistoryClonesInitialMessages(t *testing.T) {
	initial := []messages.Message{messages.Human("original")}
	history := NewInMemoryChatMessageHistory(initial...)

	initial[0] = messages.Human("mutated")
	got, err := history.Messages(context.Background())
	if err != nil {
		t.Fatalf("messages: %v", err)
	}
	if len(got) != 1 || got[0].Content != "original" {
		t.Fatalf("history was affected by caller mutation: %#v", got)
	}
}

func TestBufferString(t *testing.T) {
	tests := []struct {
		name string
		in   []messages.Message
		want string
	}{
		{name: "empty", in: nil, want: ""},
		{
			name: "all roles",
			in: []messages.Message{
				messages.System("sys"),
				messages.Human("hi"),
				messages.AI("hello"),
				messages.Tool("call-1", "result"),
			},
			want: "System: sys\nHuman: hi\nAI: hello\nTool: result",
		},
		{
			name: "unknown role uses raw role string",
			in:   []messages.Message{{Role: messages.Role("function"), Content: "payload"}},
			want: "function: payload",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := BufferString(tc.in); got != tc.want {
				t.Fatalf("BufferString = %q, want %q", got, tc.want)
			}
		})
	}
}
