package callbacks

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

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
