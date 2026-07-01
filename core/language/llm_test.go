package language

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/prompts"
	"github.com/projanvil/langchain-golang/core/runnables"
)

func TestFakeLLMInvokeBatchAndProfile(t *testing.T) {
	model := NewFakeLLM(
		WithLLMResponses("one", "two"),
		WithLLMModelProfile(map[string]any{"name": "fake-llm"}),
	)

	got, err := model.Invoke(context.Background(), "hello")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "one" {
		t.Fatalf("invoke = %q", got)
	}
	batch, err := NewFakeLLM().Batch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if !reflect.DeepEqual(batch, []string{"fake response: a", "fake response: b"}) {
		t.Fatalf("batch = %#v", batch)
	}
	profile := model.ModelProfile()
	if profile["name"] != "fake-llm" {
		t.Fatalf("profile: %#v", profile)
	}
	profile["name"] = "changed"
	if model.ModelProfile()["name"] != "fake-llm" {
		t.Fatal("profile was not copied")
	}
}

func TestFakeLLMCallbacks(t *testing.T) {
	recorder := callbacks.NewRecorder()
	model := NewFakeLLM()

	got, err := model.Invoke(
		context.Background(),
		"hello",
		runnables.WithName("fake-llm"),
		runnables.WithRunID("run-1"),
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "fake response: hello" {
		t.Fatalf("invoke = %q", got)
	}

	events := recorder.Events()
	if len(events) != 2 {
		t.Fatalf("events: got %d want 2", len(events))
	}
	if events[0].Kind != callbacks.EventLLMStart || events[1].Kind != callbacks.EventLLMEnd {
		t.Fatalf("events: %+v", events)
	}
	if events[0].Name != "fake-llm" || events[0].RunID != "run-1" {
		t.Fatalf("identity: %+v", events[0])
	}
}

func TestFakeLLMStreamCallbacks(t *testing.T) {
	recorder := callbacks.NewRecorder()
	model := NewFakeLLM(WithLLMStreamChunks("he", "llo"))
	stream, err := model.Stream(
		context.Background(),
		"hello",
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var chunks []string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		chunks = append(chunks, chunk)
	}
	if !reflect.DeepEqual(chunks, []string{"he", "llo"}) {
		t.Fatalf("chunks = %#v", chunks)
	}
	events := recorder.Events()
	want := []callbacks.EventKind{
		callbacks.EventLLMStart,
		callbacks.EventLLMStream,
		callbacks.EventLLMStream,
		callbacks.EventLLMEnd,
	}
	if len(events) != len(want) {
		t.Fatalf("events: got %d want %d: %+v", len(events), len(want), events)
	}
	for i := range want {
		if events[i].Kind != want[i] {
			t.Fatalf("event[%d]: got %q want %q", i, events[i].Kind, want[i])
		}
	}
}

func TestFakeLLMRateLimiterErrorPreventsStartEvent(t *testing.T) {
	recorder := callbacks.NewRecorder()
	wantErr := errors.New("rate limited")
	model := NewFakeLLM(WithLLMRateLimiter(&recordingLimiter{err: wantErr}))
	_, err := model.Invoke(
		context.Background(),
		"hello",
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
	if len(recorder.Events()) != 0 {
		t.Fatalf("unexpected events: %#v", recorder.Events())
	}
}

func TestPromptValueConversions(t *testing.T) {
	stringValue := prompts.StringPromptValue{Text: "hello"}
	text, err := PromptValueString(stringValue)
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("string prompt = %q", text)
	}
	msgs, err := PromptValueMessages(stringValue)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 || msgs[0].Role != messages.RoleHuman || msgs[0].Content != "hello" {
		t.Fatalf("string prompt messages: %#v", msgs)
	}

	chatValue := prompts.ChatPromptValue{Messages: []messages.Message{
		messages.System("rules"),
		messages.Human("question"),
	}}
	text, err = PromptValueString(chatValue)
	if err != nil {
		t.Fatal(err)
	}
	if text != "System: rules\nHuman: question" {
		t.Fatalf("chat prompt string = %q", text)
	}
	msgs, err = PromptValueMessages(chatValue)
	if err != nil {
		t.Fatal(err)
	}
	msgs[0].Content = "mutated"
	again, err := PromptValueMessages(chatValue)
	if err != nil {
		t.Fatal(err)
	}
	if again[0].Content != "rules" {
		t.Fatal("chat prompt messages were not copied")
	}
	if _, err := PromptValueString(42); err == nil {
		t.Fatal("expected unsupported string conversion error")
	}
	if _, err := PromptValueMessages(42); err == nil {
		t.Fatal("expected unsupported messages conversion error")
	}
}
