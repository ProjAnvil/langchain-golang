package callbacks

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/outputs"
)

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

type failingHandler struct{ err error }

func (h failingHandler) HandleEvent(context.Context, Event) error {
	return h.err
}

type stringerChunk struct{ text string }

func (c stringerChunk) String() string { return c.text }

func TestManagerEmitsEvents(t *testing.T) {
	recorder := NewRecorder()
	manager := NewManager(recorder)

	err := manager.Emit(context.Background(), Event{
		Kind: EventChatModelStart,
		Name: "fake",
		Tags: []string{"unit"},
		Metadata: map[string]any{
			"model": "fake",
		},
	})
	if err != nil {
		t.Fatalf("emit: %v", err)
	}

	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("events: got %d want 1", len(events))
	}
	if events[0].Kind != EventChatModelStart {
		t.Fatalf("kind: got %q", events[0].Kind)
	}
	if events[0].Timestamp.IsZero() {
		t.Fatal("expected timestamp")
	}
}

func TestRecorderReturnsCopies(t *testing.T) {
	recorder := NewRecorder()
	err := recorder.HandleEvent(context.Background(), Event{
		Kind:     EventToolStart,
		Tags:     []string{"original"},
		Metadata: map[string]any{"source": "test"},
	})
	if err != nil {
		t.Fatalf("handle event: %v", err)
	}

	events := recorder.Events()
	events[0].Tags[0] = "changed"
	events[0].Metadata["source"] = "changed"

	again := recorder.Events()
	if again[0].Tags[0] != "original" {
		t.Fatalf("tag was mutated: %q", again[0].Tags[0])
	}
	if again[0].Metadata["source"] != "test" {
		t.Fatalf("metadata was mutated: %v", again[0].Metadata["source"])
	}
}

func TestManagerAppliesInheritedConfig(t *testing.T) {
	recorder := NewRecorder()
	manager := NewManager(recorder).
		WithTags("root", "tenant").
		WithMetadata(map[string]any{
			"tenant":   "acme",
			"override": "manager",
		}).
		Child("parent-run")
	tags := []string{"event"}
	metadata := map[string]any{
		"override": "event",
		"request":  "123",
	}

	err := manager.Emit(context.Background(), Event{
		Kind:     EventToolStart,
		RunID:    "child-run",
		Tags:     tags,
		Metadata: metadata,
	})
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	tags[0] = "mutated"
	metadata["request"] = "mutated"

	event := recorder.Events()[0]
	if event.RunID != "child-run" || event.ParentID != "parent-run" {
		t.Fatalf("run fields: %#v", event)
	}
	wantTags := []string{"root", "tenant", "event"}
	if strings.Join(event.Tags, ",") != strings.Join(wantTags, ",") {
		t.Fatalf("tags: got %#v want %#v", event.Tags, wantTags)
	}
	if event.Metadata["tenant"] != "acme" {
		t.Fatalf("missing inherited metadata: %#v", event.Metadata)
	}
	if event.Metadata["override"] != "event" {
		t.Fatalf("event metadata should override manager metadata: %#v", event.Metadata)
	}
	if event.Metadata["request"] != "123" {
		t.Fatalf("event metadata was not copied: %#v", event.Metadata)
	}
}

func TestManagerPreservesExplicitParentID(t *testing.T) {
	recorder := NewRecorder()
	manager := NewManager(recorder).WithParentRunID("manager-parent")
	if err := manager.Emit(context.Background(), Event{
		Kind:     EventToolStart,
		ParentID: "event-parent",
	}); err != nil {
		t.Fatal(err)
	}
	event := recorder.Events()[0]
	if event.ParentID != "event-parent" {
		t.Fatalf("parent id: got %q want event-parent", event.ParentID)
	}
}

func TestStdOutAndStreamingHandlers(t *testing.T) {
	var stdout bytes.Buffer
	manager := NewManager(
		NewStdOutHandler(&stdout),
		NewStreamingStdOutHandler(&stdout),
	)
	if err := manager.Emit(context.Background(), Event{Kind: EventToolStart, Name: "search"}); err != nil {
		t.Fatal(err)
	}
	if err := manager.Emit(context.Background(), Event{Kind: EventChatModelStream, Chunk: "tok"}); err != nil {
		t.Fatal(err)
	}
	if err := manager.Emit(context.Background(), Event{Kind: EventToolEnd, Name: "search"}); err != nil {
		t.Fatal(err)
	}
	got := stdout.String()
	for _, want := range []string{"Entering tool search", "tok", "Finished tool search"} {
		if !strings.Contains(got, want) {
			t.Fatalf("stdout %q missing %q", got, want)
		}
	}
}

func TestFileHandler(t *testing.T) {
	path := filepath.Join(t.TempDir(), "callbacks.txt")
	handler, err := NewFileHandler(path, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := handler.HandleEvent(context.Background(), Event{Kind: EventRetrieverStart, Name: "docs"}); err != nil {
		t.Fatal(err)
	}
	if err := handler.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "Entering retriever docs") {
		t.Fatalf("file output = %q", string(data))
	}
	if err := handler.HandleEvent(context.Background(), Event{Kind: EventRetrieverEnd}); err == nil {
		t.Fatal("expected closed file error")
	}
}

func TestUsageMetadataHandlerAggregatesByModel(t *testing.T) {
	handler := NewUsageMetadataHandler()
	msg := messages.AI("ok")
	msg.ResponseMetadata = map[string]any{"model_name": "fake"}
	msg.UsageMetadata = messages.UsageMetadata{InputTokens: 2, OutputTokens: 3, TotalTokens: 5}
	if err := handler.HandleEvent(context.Background(), Event{Kind: EventChatModelEnd, Output: msg}); err != nil {
		t.Fatal(err)
	}
	if err := handler.HandleEvent(context.Background(), Event{Kind: EventChatModelEnd, Output: msg}); err != nil {
		t.Fatal(err)
	}
	usage := handler.Usage()["fake"]
	if usage.InputTokens != 4 || usage.OutputTokens != 6 || usage.TotalTokens != 10 {
		t.Fatalf("unexpected usage: %#v", usage)
	}
	usageMap := handler.Usage()
	usageMap["fake"] = messages.UsageMetadata{}
	if handler.Usage()["fake"].TotalTokens != 10 {
		t.Fatal("usage map was not copied")
	}
}

func TestNestedRunFieldsPreserved(t *testing.T) {
	recorder := NewRecorder()
	manager := NewManager(recorder)
	err := manager.Emit(context.Background(), Event{
		Kind:     EventToolStart,
		RunID:    "child",
		ParentID: "parent",
	})
	if err != nil {
		t.Fatal(err)
	}
	event := recorder.Events()[0]
	if event.RunID != "child" || event.ParentID != "parent" {
		t.Fatalf("run fields not preserved: %#v", event)
	}
}

func TestManagerEmitReturnsHandlerError(t *testing.T) {
	recorder := NewRecorder()
	want := errors.New("handler failed")
	manager := NewManager(failingHandler{err: want}, recorder)

	err := manager.Emit(context.Background(), Event{Kind: EventToolStart})
	if !errors.Is(err, want) {
		t.Fatalf("emit error: got %v want %v", err, want)
	}
	if got := len(recorder.Events()); got != 0 {
		t.Fatalf("later handlers should not run after error, got %d events", got)
	}
}

func TestManagerNestedAsHandler(t *testing.T) {
	recorder := NewRecorder()
	inner := NewManager(recorder).WithParentRunID("inner-parent")
	outer := NewManager(inner).WithTags("outer")

	if err := outer.Emit(context.Background(), Event{Kind: EventLLMStart, Name: "m"}); err != nil {
		t.Fatal(err)
	}
	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("events: got %d want 1", len(events))
	}
	if events[0].ParentID != "inner-parent" {
		t.Fatalf("inner manager parent id not applied: %#v", events[0])
	}
	if strings.Join(events[0].Tags, ",") != "outer" {
		t.Fatalf("outer tags not applied: %#v", events[0].Tags)
	}
}

func TestContextWithManagerRoundTrip(t *testing.T) {
	if _, ok := ManagerFromContext(context.Background()); ok {
		t.Fatal("expected no manager in background context")
	}

	manager := NewManager(NewRecorder()).WithTags("ctx")
	ctx := ContextWithManager(context.Background(), manager)
	got, ok := ManagerFromContext(ctx)
	if !ok {
		t.Fatal("expected manager in context")
	}
	if got.Empty() {
		t.Fatal("expected manager with handlers")
	}
}

func TestManagerEmpty(t *testing.T) {
	if !NewManager().Empty() {
		t.Fatal("manager without handlers should be empty")
	}
	if NewManager(NewRecorder()).Empty() {
		t.Fatal("manager with handlers should not be empty")
	}
}

func TestWithMetadataEmptyIsNoop(t *testing.T) {
	recorder := NewRecorder()
	manager := NewManager(recorder).
		WithMetadata(map[string]any{"a": 1}).
		WithMetadata(nil).
		WithMetadata(map[string]any{})

	if err := manager.Emit(context.Background(), Event{Kind: EventToolStart}); err != nil {
		t.Fatal(err)
	}
	event := recorder.Events()[0]
	if event.Metadata["a"] != 1 || len(event.Metadata) != 1 {
		t.Fatalf("metadata changed by empty merge: %#v", event.Metadata)
	}
}

func TestStdOutHandlerNilWriterDefaults(t *testing.T) {
	handler := NewStdOutHandler(nil)
	if handler.Writer != os.Stdout {
		t.Fatalf("writer: got %#v want os.Stdout", handler.Writer)
	}
	streaming := NewStreamingStdOutHandler(nil)
	if streaming.Writer != os.Stdout {
		t.Fatalf("writer: got %#v want os.Stdout", streaming.Writer)
	}

	// Zero-value handlers fall back to os.Stdout without writing anything for
	// ignored event kinds.
	if err := (StdOutHandler{}).HandleEvent(context.Background(), Event{Kind: EventKind("custom")}); err != nil {
		t.Fatal(err)
	}
	if err := (StreamingStdOutHandler{}).HandleEvent(context.Background(), Event{Kind: EventLLMStream, Chunk: nil}); err != nil {
		t.Fatal(err)
	}
}

func TestStdOutHandlerErrorEvents(t *testing.T) {
	for _, kind := range []EventKind{EventChatModelError, EventLLMError, EventToolError, EventRetrieverError} {
		var buf bytes.Buffer
		handler := NewStdOutHandler(&buf)
		err := handler.HandleEvent(context.Background(), Event{Kind: kind, Name: "thing", Error: "boom"})
		if err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		if !strings.Contains(buf.String(), "error: boom") {
			t.Fatalf("%s: output %q missing error text", kind, buf.String())
		}
	}
}

func TestStdOutHandlerWriteError(t *testing.T) {
	handler := NewStdOutHandler(failingWriter{})
	err := handler.HandleEvent(context.Background(), Event{Kind: EventToolStart, Name: "search"})
	if err == nil {
		t.Fatal("expected write error")
	}
}

func TestStreamingHandlerFormatsChunks(t *testing.T) {
	cases := []struct {
		name  string
		chunk any
		want  string
	}{
		{name: "nil", chunk: nil, want: ""},
		{name: "string", chunk: "tok", want: "tok"},
		{name: "message", chunk: messages.AI("hello"), want: "hello"},
		{name: "stringer", chunk: stringerChunk{text: "s"}, want: "s"},
		{name: "other", chunk: 42, want: "42"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			handler := NewStreamingStdOutHandler(&buf)
			err := handler.HandleEvent(context.Background(), Event{Kind: EventChatModelStream, Chunk: tc.chunk})
			if err != nil {
				t.Fatal(err)
			}
			if buf.String() != tc.want {
				t.Fatalf("output: got %q want %q", buf.String(), tc.want)
			}
		})
	}
}

func TestStreamingHandlerIgnoresNonStreamAndPropagatesErrors(t *testing.T) {
	var buf bytes.Buffer
	handler := NewStreamingStdOutHandler(&buf)
	if err := handler.HandleEvent(context.Background(), Event{Kind: EventToolStart, Chunk: "ignored"}); err != nil {
		t.Fatal(err)
	}
	if buf.Len() != 0 {
		t.Fatalf("unexpected output %q", buf.String())
	}

	failing := NewStreamingStdOutHandler(failingWriter{})
	if err := failing.HandleEvent(context.Background(), Event{Kind: EventLLMStream, Chunk: "tok"}); err == nil {
		t.Fatal("expected write error")
	}
}

func TestFileHandlerAppendMode(t *testing.T) {
	path := filepath.Join(t.TempDir(), "callbacks.txt")

	first, err := NewFileHandler(path, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.HandleEvent(context.Background(), Event{Kind: EventToolStart, Name: "one"}); err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	// Closing again is a no-op.
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	second, err := NewFileHandler(path, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := second.HandleEvent(context.Background(), Event{Kind: EventToolEnd, Name: "two"}); err != nil {
		t.Fatal(err)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if !strings.Contains(content, "Entering tool one") || !strings.Contains(content, "Finished tool two") {
		t.Fatalf("file output = %q", content)
	}
}

func TestFileHandlerOpenError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing", "callbacks.txt")
	if _, err := NewFileHandler(path, false); err == nil {
		t.Fatal("expected open error")
	}
}

func TestFileHandlerNilReceiver(t *testing.T) {
	var handler *FileHandler
	if err := handler.Close(); err != nil {
		t.Fatal(err)
	}
	if err := handler.HandleEvent(context.Background(), Event{Kind: EventToolStart}); err == nil {
		t.Fatal("expected error for nil handler")
	}
}

func TestUsageMetadataHandlerModelNameFallbacks(t *testing.T) {
	usage := messages.UsageMetadata{InputTokens: 1, OutputTokens: 2, TotalTokens: 3}

	withModelKey := messages.AI("ok")
	withModelKey.ResponseMetadata = map[string]any{"model": "fallback-model"}
	withModelKey.UsageMetadata = usage

	fromEventMetadata := messages.AI("ok")
	fromEventMetadata.UsageMetadata = usage

	handler := NewUsageMetadataHandler()
	events := []Event{
		{Kind: EventToolEnd, Output: withModelKey},      // ignored: not a chat model end
		{Kind: EventChatModelEnd, Output: withModelKey}, // "model" metadata key
		{Kind: EventChatModelEnd, Output: fromEventMetadata, Metadata: map[string]any{"model_name": "event-model"}},
		{Kind: EventChatModelEnd, Output: fromEventMetadata},    // no model name: skipped
		{Kind: EventChatModelEnd, Output: messages.Human("hi")}, // non-AI role: skipped
	}
	for _, event := range events {
		if err := handler.HandleEvent(context.Background(), event); err != nil {
			t.Fatal(err)
		}
	}

	zeroUsage := messages.AI("ok")
	zeroUsage.ResponseMetadata = map[string]any{"model_name": "zero"}
	if err := handler.HandleEvent(context.Background(), Event{Kind: EventChatModelEnd, Output: zeroUsage}); err != nil {
		t.Fatal(err)
	}

	got := handler.Usage()
	if got["fallback-model"] != usage {
		t.Fatalf("fallback-model usage: %#v", got["fallback-model"])
	}
	if got["event-model"] != usage {
		t.Fatalf("event-model usage: %#v", got["event-model"])
	}
	if _, ok := got["zero"]; ok {
		t.Fatal("zero usage should be skipped")
	}
	if len(got) != 2 {
		t.Fatalf("unexpected models: %#v", got)
	}
}

func TestUsageMetadataHandlerOutputShapes(t *testing.T) {
	usage := messages.UsageMetadata{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}
	msg := messages.AI("ok")
	msg.ResponseMetadata = map[string]any{"model_name": "fake"}
	msg.UsageMetadata = usage

	handler := NewUsageMetadataHandler()
	events := []Event{
		{Kind: EventChatModelEnd, Output: []messages.Message{msg, messages.Human("ignored")}},
		{Kind: EventChatModelEnd, Output: outputs.NewChatGeneration(msg, nil)},
		{Kind: EventChatModelEnd, Output: outputs.ChatResult{Generations: []outputs.ChatGeneration{outputs.NewChatGeneration(msg, nil)}}},
		{Kind: EventChatModelEnd, Output: "not a message"}, // unsupported: ignored
	}
	for _, event := range events {
		if err := handler.HandleEvent(context.Background(), event); err != nil {
			t.Fatal(err)
		}
	}

	got := handler.Usage()["fake"]
	if got.InputTokens != 3 || got.OutputTokens != 3 || got.TotalTokens != 6 {
		t.Fatalf("usage: %#v", got)
	}
}

func TestWithMetadataMergesOntoExisting(t *testing.T) {
	recorder := NewRecorder()
	manager := NewManager(recorder).
		WithMetadata(map[string]any{"first": 1, "shared": "old"}).
		WithMetadata(map[string]any{"second": 2, "shared": "new"})

	if err := manager.Emit(context.Background(), Event{Kind: EventToolStart}); err != nil {
		t.Fatal(err)
	}
	metadata := recorder.Events()[0].Metadata
	if metadata["first"] != 1 || metadata["second"] != 2 || metadata["shared"] != "new" {
		t.Fatalf("merged metadata: %#v", metadata)
	}
}
