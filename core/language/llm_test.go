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

func TestFakeLLMSchemas(t *testing.T) {
	model := NewFakeLLM()

	input := model.InputSchema()
	if input["type"] != "string" || input["description"] != "text prompt" {
		t.Fatalf("input schema: %#v", input)
	}
	output := model.OutputSchema()
	if output["type"] != "string" || output["description"] != "text completion" {
		t.Fatalf("output schema: %#v", output)
	}
}

func TestFakeLLMDefaultModelProfile(t *testing.T) {
	profile := NewFakeLLM().ModelProfile()
	if profile["text_inputs"] != true || profile["text_outputs"] != true {
		t.Fatalf("profile: %#v", profile)
	}
	if len(profile) != 2 {
		t.Fatalf("unexpected profile keys: %#v", profile)
	}
}

func TestFakeLLMStreamFallsBackToInvoke(t *testing.T) {
	model := NewFakeLLM(WithLLMResponses("streamed response"))

	stream, err := model.Stream(context.Background(), "hello")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	chunk, ok, err := stream.Next(context.Background())
	if err != nil {
		t.Fatalf("next: %v", err)
	}
	if !ok || chunk != "streamed response" {
		t.Fatalf("chunk: ok=%v value=%q", ok, chunk)
	}
	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected exhausted stream, ok=%v err=%v", ok, err)
	}
}

func TestFakeLLMStreamRateLimiterError(t *testing.T) {
	wantErr := errors.New("rate limited")
	model := NewFakeLLM(
		WithLLMRateLimiter(&recordingLimiter{err: wantErr}),
		WithLLMStreamChunks("chunk"),
	)

	_, err := model.Stream(context.Background(), "hello")
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}

	// Without configured chunks, Stream delegates to Invoke, which must surface
	// the same limiter error.
	noChunks := NewFakeLLM(WithLLMRateLimiter(&recordingLimiter{err: wantErr}))
	_, err = noChunks.Stream(context.Background(), "hello")
	if !errors.Is(err, wantErr) {
		t.Fatalf("fallback stream err=%v, want %v", err, wantErr)
	}
}

func TestFakeLLMStreamWithoutCallbacks(t *testing.T) {
	model := NewFakeLLM(WithLLMStreamChunks("a", "b"))

	stream, err := model.Stream(context.Background(), "hello")
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
	if !reflect.DeepEqual(chunks, []string{"a", "b"}) {
		t.Fatalf("chunks = %#v", chunks)
	}
}

func TestFakeLLMInvokeCallbackError(t *testing.T) {
	wantErr := errors.New("callback failed")

	startModel := NewFakeLLM()
	_, err := startModel.Invoke(
		context.Background(),
		"hello",
		runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
			kind: callbacks.EventLLMStart,
			err:  wantErr,
		})),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("start err=%v, want %v", err, wantErr)
	}

	endModel := NewFakeLLM()
	_, err = endModel.Invoke(
		context.Background(),
		"hello",
		runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
			kind: callbacks.EventLLMEnd,
			err:  wantErr,
		})),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("end err=%v, want %v", err, wantErr)
	}
}

func TestFakeLLMStreamStartCallbackError(t *testing.T) {
	wantErr := errors.New("callback failed")
	model := NewFakeLLM(WithLLMStreamChunks("chunk"))

	_, err := model.Stream(
		context.Background(),
		"hello",
		runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
			kind: callbacks.EventLLMStart,
			err:  wantErr,
		})),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
}

func TestLLMCallbackStreamErrorPaths(t *testing.T) {
	wantErr := errors.New("boom")

	t.Run("inner stream error emits error event", func(t *testing.T) {
		recorder := callbacks.NewRecorder()
		stream := newLLMCallbackStream(
			runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(recorder))),
			errorStringStream{err: wantErr},
		)

		_, ok, err := stream.Next(context.Background())
		if !errors.Is(err, wantErr) || ok {
			t.Fatalf("next: ok=%v err=%v", ok, err)
		}

		events := recorder.Events()
		if len(events) != 1 || events[0].Kind != callbacks.EventLLMError {
			t.Fatalf("events: %+v", events)
		}
		if events[0].Error != wantErr.Error() {
			t.Fatalf("event error: got %q want %q", events[0].Error, wantErr)
		}
	})

	t.Run("end event failure propagates once", func(t *testing.T) {
		stream := newLLMCallbackStream(
			runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
				kind: callbacks.EventLLMEnd,
				err:  wantErr,
			}))),
			runnables.NewSliceStream([]string{}),
		)

		_, ok, err := stream.Next(context.Background())
		if !errors.Is(err, wantErr) || ok {
			t.Fatalf("next: ok=%v err=%v", ok, err)
		}
		// The end event must only be emitted once; a second Next is a clean stop.
		if _, ok, err := stream.Next(context.Background()); err != nil || ok {
			t.Fatalf("second next: ok=%v err=%v", ok, err)
		}
	})

	t.Run("stream event failure propagates", func(t *testing.T) {
		stream := newLLMCallbackStream(
			runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
				kind: callbacks.EventLLMStream,
				err:  wantErr,
			}))),
			runnables.NewSliceStream([]string{"chunk"}),
		)

		_, ok, err := stream.Next(context.Background())
		if !errors.Is(err, wantErr) || ok {
			t.Fatalf("next: ok=%v err=%v", ok, err)
		}
	})
}

func TestPromptValueConversionsFromMessages(t *testing.T) {
	single := messages.Human("just one")
	text, err := PromptValueString(single)
	if err != nil {
		t.Fatal(err)
	}
	if text != "just one" {
		t.Fatalf("message string = %q", text)
	}
	msgs, err := PromptValueMessages(single)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 || msgs[0].Role != messages.RoleHuman || msgs[0].Content != "just one" {
		t.Fatalf("message messages: %#v", msgs)
	}

	list := []messages.Message{
		messages.System("rules"),
		messages.Human("question"),
	}
	text, err = PromptValueString(list)
	if err != nil {
		t.Fatal(err)
	}
	if text != "System: rules\nHuman: question" {
		t.Fatalf("list string = %q", text)
	}
	msgs, err = PromptValueMessages(list)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 {
		t.Fatalf("list messages: %#v", msgs)
	}
	// The conversion must clone so callers cannot mutate the source slice.
	msgs[0].Content = "mutated"
	if list[0].Content != "rules" {
		t.Fatal("source messages were mutated through the converted slice")
	}
}

// errorStringStream fails on the first Next call.
type errorStringStream struct {
	err error
}

func (s errorStringStream) Next(context.Context) (string, bool, error) {
	return "", false, s.err
}

func (s errorStringStream) Close() error { return nil }
