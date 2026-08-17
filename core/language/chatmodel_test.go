package language

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/streamevents"
	"github.com/projanvil/langchain-golang/core/tools"
)

func TestFakeChatModelInvokeEchoesLastMessage(t *testing.T) {
	model := NewFakeChatModel()

	got, err := model.Invoke(context.Background(), []messages.Message{
		messages.System("be concise"),
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Role != messages.RoleAI {
		t.Fatalf("role: got %q want %q", got.Role, messages.RoleAI)
	}
	if got.Content != "fake response: hello" {
		t.Fatalf("content: got %q", got.Content)
	}
	if got.UsageMetadata.TotalTokens == 0 {
		t.Fatal("expected usage metadata")
	}
}

func TestFakeChatModelBindTools(t *testing.T) {
	model := NewFakeChatModel(WithCapabilities(ChatModelCapabilities{
		ToolCalling: true,
		Streaming:   true,
	}))
	adder, err := tools.NewFunc(
		"adder",
		"adds integers",
		schema.Object(map[string]schema.Schema{
			"a": schema.Integer("left side"),
			"b": schema.Integer("right side"),
		}, "a", "b"),
		func(_ context.Context, _ map[string]any) (tools.Result, error) {
			return tools.Result{Content: "3"}, nil
		},
	)
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}

	bound, err := model.BindTools([]tools.Tool{adder})
	if err != nil {
		t.Fatalf("bind tools: %v", err)
	}

	fake, ok := bound.(*FakeChatModel)
	if !ok {
		t.Fatalf("bound model type: %T", bound)
	}
	if len(fake.BoundTools()) != 1 {
		t.Fatalf("bound tools: got %d want 1", len(fake.BoundTools()))
	}
}

func TestFakeChatModelBindToolsUnsupported(t *testing.T) {
	model := NewFakeChatModel()
	_, err := model.BindTools([]tools.Tool{
		mustNoopTool(t),
	})
	if err == nil {
		t.Fatal("expected unsupported tool calling error")
	}
}

func TestFakeChatModelInvokeCallbacks(t *testing.T) {
	recorder := callbacks.NewRecorder()
	model := NewFakeChatModel()

	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hello")},
		runnables.WithName("fake-chat"),
		runnables.WithRunID("run-1"),
		runnables.WithTags("unit"),
		runnables.WithMetadata("provider", "fake"),
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	events := recorder.Events()
	if len(events) != 2 {
		t.Fatalf("events: got %d want 2", len(events))
	}
	if events[0].Kind != callbacks.EventChatModelStart {
		t.Fatalf("start kind: got %q", events[0].Kind)
	}
	if events[1].Kind != callbacks.EventChatModelEnd {
		t.Fatalf("end kind: got %q", events[1].Kind)
	}
	if events[0].Name != "fake-chat" || events[0].RunID != "run-1" {
		t.Fatalf("event identity: %+v", events[0])
	}
	if events[0].Metadata["provider"] != "fake" {
		t.Fatalf("metadata: %+v", events[0].Metadata)
	}
}

func TestFakeChatModelStreamCallbacks(t *testing.T) {
	recorder := callbacks.NewRecorder()
	model := NewFakeChatModel(
		WithStreamChunks(
			messages.AI("hel"),
			messages.AI("lo"),
		),
	)

	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hello")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	for {
		_, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
	}

	events := recorder.Events()
	if len(events) != 4 {
		t.Fatalf("events: got %d want 4", len(events))
	}
	want := []callbacks.EventKind{
		callbacks.EventChatModelStart,
		callbacks.EventChatModelStream,
		callbacks.EventChatModelStream,
		callbacks.EventChatModelEnd,
	}
	for i := range want {
		if events[i].Kind != want[i] {
			t.Fatalf("event[%d]: got %q want %q", i, events[i].Kind, want[i])
		}
	}
}

func TestFakeChatModelRateLimiter(t *testing.T) {
	limiter := &recordingLimiter{}
	model := NewFakeChatModel(WithRateLimiter(limiter))
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hello")}); err != nil {
		t.Fatal(err)
	}
	if limiter.calls != 1 || !limiter.blocking {
		t.Fatalf("limiter calls=%d blocking=%v", limiter.calls, limiter.blocking)
	}

	streamModel := NewFakeChatModel(
		WithRateLimiter(limiter),
		WithStreamChunks(messages.AI("chunk")),
	)
	stream, err := streamModel.Stream(context.Background(), []messages.Message{messages.Human("hello")})
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Close()
	if limiter.calls != 2 {
		t.Fatalf("limiter calls after stream=%d, want 2", limiter.calls)
	}
}

func TestFakeChatModelRateLimiterErrorPreventsStartEvent(t *testing.T) {
	recorder := callbacks.NewRecorder()
	wantErr := errors.New("rate limited")
	model := NewFakeChatModel(WithRateLimiter(&recordingLimiter{err: wantErr}))
	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hello")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
	if len(recorder.Events()) != 0 {
		t.Fatalf("unexpected events: %#v", recorder.Events())
	}
}

func TestChatModelCapabilitiesModelProfile(t *testing.T) {
	profile := (ChatModelCapabilities{
		ToolCalling:      true,
		ToolChoice:       true,
		StructuredOutput: true,
		ImageInputs:      true,
		ImageURLs:        true,
		Streaming:        true,
	}).ModelProfile()
	if profile["tool_calling"] != true ||
		profile["tool_choice"] != true ||
		profile["structured_output"] != true ||
		profile["image_inputs"] != true ||
		profile["image_url_inputs"] != true ||
		profile["tool_call_streaming"] != true {
		t.Fatalf("profile: %#v", profile)
	}
}

func TestFakeChatModelExplicitModelProfile(t *testing.T) {
	model := NewFakeChatModel(WithModelProfile(map[string]any{
		"name":             "Fake",
		"max_input_tokens": 128,
	}))
	profile := model.ModelProfile()
	if profile["name"] != "Fake" || profile["max_input_tokens"] != 128 {
		t.Fatalf("profile: %#v", profile)
	}
	profile["name"] = "Changed"
	if model.ModelProfile()["name"] != "Fake" {
		t.Fatal("profile was not copied")
	}
}

func TestStreamEventsFallbackTextProjection(t *testing.T) {
	model := NewFakeChatModel(
		WithStreamChunks(
			messages.AI("hel"),
			messages.AI("lo"),
		),
	)

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	if got := stream.Text(); got != "hello" {
		t.Fatalf("text: got %q", got)
	}
	if got := stream.TextDeltas(); len(got) != 2 || got[0] != "hel" || got[1] != "lo" {
		t.Fatalf("text deltas: %+v", got)
	}
	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if output.Content != "hello" {
		t.Fatalf("output content: %+v", output)
	}
	if events := stream.Events(); len(events) != 6 {
		t.Fatalf("protocol events: got %d want 6: %+v", len(events), events)
	}
}

func TestStreamEventsPreservesUserCallbacks(t *testing.T) {
	recorder := callbacks.NewRecorder()
	model := NewFakeChatModel(
		WithStreamChunks(messages.AI("ok")),
	)

	_, err := StreamEvents(
		context.Background(),
		model,
		[]messages.Message{messages.Human("hello")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	events := recorder.Events()
	if len(events) != 3 {
		t.Fatalf("events: got %d want 3: %+v", len(events), events)
	}
	if events[0].Kind != callbacks.EventChatModelStart ||
		events[1].Kind != callbacks.EventChatModelStream ||
		events[2].Kind != callbacks.EventChatModelEnd {
		t.Fatalf("events: %+v", events)
	}
}

func TestStreamEventsFallbackMalformedToolCallChunk(t *testing.T) {
	chunk := messages.AI("")
	chunk.ContentBlocks = []messages.ContentBlock{messages.ParseContentBlock(map[string]any{
		"type":  "tool_call_chunk",
		"id":    "call_1",
		"name":  "search",
		"args":  `{"q": `,
		"index": 0,
	})}
	model := NewFakeChatModel(WithStreamChunks(chunk))

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("search"),
	})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if len(output.ToolCalls) != 0 || len(output.InvalidToolCalls) != 1 {
		t.Fatalf("tool calls: valid=%+v invalid=%+v", output.ToolCalls, output.InvalidToolCalls)
	}
	if output.InvalidToolCalls[0].Name != "search" {
		t.Fatalf("invalid tool call: %+v", output.InvalidToolCalls[0])
	}
}

func TestStreamEventsUsesNativeProtocolEvents(t *testing.T) {
	model := protocolFakeChatModel{}

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	if got := stream.Text(); got != "native" {
		t.Fatalf("text: got %q", got)
	}
	events := stream.Events()
	if len(events) != 5 {
		t.Fatalf("events: got %d want 5: %+v", len(events), events)
	}
	if events[0].Event != streamevents.EventMessageStart ||
		events[4].Event != streamevents.EventMessageFinish {
		t.Fatalf("protocol events: %+v", events)
	}
}

type protocolFakeChatModel struct{}

func (m protocolFakeChatModel) Invoke(context.Context, []messages.Message, ...runnables.Option) (messages.Message, error) {
	return messages.AI("native"), nil
}

func (m protocolFakeChatModel) Batch(context.Context, [][]messages.Message, ...runnables.Option) ([]messages.Message, error) {
	return []messages.Message{messages.AI("native")}, nil
}

func (m protocolFakeChatModel) Stream(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	cfg := runnables.NewConfig(opts...)
	if err := emitChatEvent(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return nil, err
	}
	events := []streamevents.Event{
		{Event: streamevents.EventMessageStart},
		{
			Event: streamevents.EventContentBlockStart,
			Index: 0,
			Content: messages.ParseContentBlock(map[string]any{
				"type": "text",
				"text": "",
			}),
		},
		{
			Event: streamevents.EventContentBlockDelta,
			Index: 0,
			Delta: messages.ParseContentBlock(map[string]any{
				"type": "text-delta",
				"text": "native",
			}),
		},
		{
			Event: streamevents.EventContentBlockFinish,
			Index: 0,
			Content: messages.ParseContentBlock(map[string]any{
				"type": "text",
				"text": "native",
			}),
		},
		{Event: streamevents.EventMessageFinish, Output: messages.AI("native")},
	}
	for _, event := range events {
		if err := cfg.Callbacks.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelProtocol, Chunk: event}); err != nil {
			return nil, err
		}
	}
	return runnables.NewSliceStream([]messages.Message{messages.AI("legacy should be ignored")}), nil
}

func (m protocolFakeChatModel) InputSchema() schema.Schema {
	return schema.Schema{"type": "array"}
}

func (m protocolFakeChatModel) OutputSchema() schema.Schema {
	return schema.Schema{"type": "object"}
}

func (m protocolFakeChatModel) BindTools([]tools.Tool) (ChatModel, error) {
	return m, nil
}

func (m protocolFakeChatModel) Capabilities() ChatModelCapabilities {
	return ChatModelCapabilities{Streaming: true}
}

func TestFakeChatModelBindToolsProducesIndependentCopy(t *testing.T) {
	model := NewFakeChatModel(
		WithCapabilities(ChatModelCapabilities{ToolCalling: true}),
		WithResponses(
			messages.AI("original-1"),
			messages.AI("original-2"),
		),
	)

	bound, err := model.BindTools([]tools.Tool{mustNoopTool(t)})
	if err != nil {
		t.Fatalf("bind tools: %v", err)
	}
	boundFake, ok := bound.(*FakeChatModel)
	if !ok {
		t.Fatalf("bound model type: %T", bound)
	}

	// Advancing the bound copy's response cursor must not move the original's.
	if _, err := bound.Invoke(context.Background(), []messages.Message{messages.Human("a")}); err != nil {
		t.Fatalf("bound invoke: %v", err)
	}
	if boundFake.responseIdx != 1 {
		t.Fatalf("bound cursor: got %d want 1", boundFake.responseIdx)
	}
	if model.responseIdx != 0 {
		t.Fatalf("original cursor: got %d want 0", model.responseIdx)
	}

	// The original still serves its own configured responses.
	got, err := model.Invoke(context.Background(), []messages.Message{messages.Human("b")})
	if err != nil {
		t.Fatalf("original invoke: %v", err)
	}
	if got.Content != "original-1" {
		t.Fatalf("original content: got %q want %q", got.Content, "original-1")
	}
}

func TestFakeChatModelBoundAndOriginalAreConcurrentlySafe(t *testing.T) {
	model := NewFakeChatModel(
		WithCapabilities(ChatModelCapabilities{ToolCalling: true}),
		WithResponses(messages.AI("ok")),
	)
	bound, err := model.BindTools([]tools.Tool{mustNoopTool(t)})
	if err != nil {
		t.Fatalf("bind tools: %v", err)
	}

	input := []messages.Message{messages.Human("hi")}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 50; i++ {
			if _, err := model.Invoke(context.Background(), input); err != nil {
				t.Errorf("original invoke: %v", err)
				return
			}
		}
	}()
	for i := 0; i < 50; i++ {
		if _, err := bound.Invoke(context.Background(), input); err != nil {
			t.Fatalf("bound invoke: %v", err)
		}
	}
	<-done
}

func mustNoopTool(t *testing.T) tools.Tool {
	t.Helper()
	tool, err := tools.NewFunc(
		"noop",
		"does nothing",
		schema.Object(map[string]schema.Schema{}),
		func(_ context.Context, _ map[string]any) (tools.Result, error) {
			return tools.Result{}, nil
		},
	)
	if err != nil {
		t.Fatalf("new noop tool: %v", err)
	}
	return tool
}

type recordingLimiter struct {
	calls    int
	blocking bool
	err      error
}

func (l *recordingLimiter) Acquire(_ context.Context, blocking bool) (bool, error) {
	l.calls++
	l.blocking = blocking
	if l.err != nil {
		return false, l.err
	}
	return true, nil
}

func TestFakeChatModelBatch(t *testing.T) {
	// Batch runs the invokes concurrently, so per-input outputs must come from
	// the deterministic echo path rather than the shared response cursor.
	model := NewFakeChatModel()

	got, err := model.Batch(context.Background(), [][]messages.Message{
		{messages.Human("a")},
		{messages.Human("b")},
		{messages.Human("c")},
	})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	want := []string{"fake response: a", "fake response: b", "fake response: c"}
	if len(got) != len(want) {
		t.Fatalf("batch len: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Content != want[i] {
			t.Fatalf("batch[%d]: got %q want %q", i, got[i].Content, want[i])
		}
	}
}

func TestFakeChatModelBatchPropagatesErrors(t *testing.T) {
	wantErr := errors.New("rate limited")
	// Batch invokes concurrently, so use a stateless limiter to stay race-free.
	model := NewFakeChatModel(WithRateLimiter(staticErrorLimiter{err: wantErr}))

	_, err := model.Batch(context.Background(), [][]messages.Message{
		{messages.Human("a")},
		{messages.Human("b")},
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
}

func TestFakeChatModelSchemasAndCapabilities(t *testing.T) {
	model := NewFakeChatModel(WithCapabilities(ChatModelCapabilities{
		ToolCalling: true,
		Streaming:   true,
	}))

	input := model.InputSchema()
	if input["type"] != "array" {
		t.Fatalf("input schema: %#v", input)
	}
	output := model.OutputSchema()
	if output["type"] != "object" {
		t.Fatalf("output schema: %#v", output)
	}
	props, ok := output["properties"].(map[string]any)
	if !ok || props["role"] == nil || props["content"] == nil {
		t.Fatalf("output schema properties: %#v", output)
	}
	capabilities := model.Capabilities()
	if !capabilities.ToolCalling || !capabilities.Streaming {
		t.Fatalf("capabilities: %+v", capabilities)
	}
}

func TestFakeChatModelModelProfileFromCapabilities(t *testing.T) {
	model := NewFakeChatModel(WithCapabilities(ChatModelCapabilities{
		ToolCalling: true,
		Streaming:   true,
	}))

	profile := model.ModelProfile()
	if profile["tool_calling"] != true || profile["tool_call_streaming"] != true {
		t.Fatalf("profile: %#v", profile)
	}
}

func TestFakeChatModelStreamFallsBackToInvoke(t *testing.T) {
	model := NewFakeChatModel(WithResponses(messages.AI("streamed response")))

	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hello")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	chunk, ok, err := stream.Next(context.Background())
	if err != nil {
		t.Fatalf("next: %v", err)
	}
	if !ok || chunk.Content != "streamed response" {
		t.Fatalf("chunk: ok=%v content=%q", ok, chunk.Content)
	}
	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected exhausted stream, ok=%v err=%v", ok, err)
	}
}

func TestFakeChatModelStreamRateLimiterError(t *testing.T) {
	wantErr := errors.New("rate limited")
	model := NewFakeChatModel(
		WithRateLimiter(&recordingLimiter{err: wantErr}),
		WithStreamChunks(messages.AI("chunk")),
	)

	_, err := model.Stream(context.Background(), []messages.Message{messages.Human("hello")})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}

	// Without configured chunks, Stream delegates to Invoke, which must surface
	// the same limiter error.
	noChunks := NewFakeChatModel(WithRateLimiter(&recordingLimiter{err: wantErr}))
	_, err = noChunks.Stream(context.Background(), []messages.Message{messages.Human("hello")})
	if !errors.Is(err, wantErr) {
		t.Fatalf("fallback stream err=%v, want %v", err, wantErr)
	}
}

func TestFakeChatModelInvokeCallbackError(t *testing.T) {
	wantErr := errors.New("callback failed")

	startModel := NewFakeChatModel(WithResponses(messages.AI("ok")))
	_, err := startModel.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hello")},
		runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
			kind: callbacks.EventChatModelStart,
			err:  wantErr,
		})),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("start err=%v, want %v", err, wantErr)
	}

	endHandler := failOnKindHandler{kind: callbacks.EventChatModelEnd, err: wantErr}

	// End-event failure on the configured-response path.
	endModel := NewFakeChatModel(WithResponses(messages.AI("ok")))
	_, err = endModel.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hello")},
		runnables.WithCallbacks(callbacks.NewManager(endHandler)),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("end err=%v, want %v", err, wantErr)
	}

	// End-event failure on the echo path (no configured responses).
	echoModel := NewFakeChatModel()
	_, err = echoModel.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hello")},
		runnables.WithCallbacks(callbacks.NewManager(endHandler)),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("echo end err=%v, want %v", err, wantErr)
	}
}

func TestFakeChatModelStreamStartCallbackError(t *testing.T) {
	wantErr := errors.New("callback failed")
	model := NewFakeChatModel(WithStreamChunks(messages.AI("chunk")))

	_, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hello")},
		runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
			kind: callbacks.EventChatModelStart,
			err:  wantErr,
		})),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
}

func TestCallbackStreamErrorPaths(t *testing.T) {
	wantErr := errors.New("boom")

	t.Run("inner stream error emits error event", func(t *testing.T) {
		recorder := callbacks.NewRecorder()
		stream := newCallbackStream(
			context.Background(),
			runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(recorder))),
			errorMessageStream{err: wantErr},
		)

		_, ok, err := stream.Next(context.Background())
		if !errors.Is(err, wantErr) || ok {
			t.Fatalf("next: ok=%v err=%v", ok, err)
		}

		events := recorder.Events()
		if len(events) != 1 || events[0].Kind != callbacks.EventChatModelError {
			t.Fatalf("events: %+v", events)
		}
		if events[0].Error != wantErr.Error() {
			t.Fatalf("event error: got %q want %q", events[0].Error, wantErr)
		}
	})

	t.Run("end event failure propagates once", func(t *testing.T) {
		stream := newCallbackStream(
			context.Background(),
			runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
				kind: callbacks.EventChatModelEnd,
				err:  wantErr,
			}))),
			runnables.NewSliceStream([]messages.Message{}),
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
		stream := newCallbackStream(
			context.Background(),
			runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(failOnKindHandler{
				kind: callbacks.EventChatModelStream,
				err:  wantErr,
			}))),
			runnables.NewSliceStream([]messages.Message{messages.AI("chunk")}),
		)

		_, ok, err := stream.Next(context.Background())
		if !errors.Is(err, wantErr) || ok {
			t.Fatalf("next: ok=%v err=%v", ok, err)
		}
	})
}

func TestStreamEventsModelStreamError(t *testing.T) {
	wantErr := errors.New("stream unavailable")
	model := stubChatModel{streamErr: wantErr}

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("hello"),
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
	if _, outErr := stream.Output(); !errors.Is(outErr, wantErr) {
		t.Fatalf("output err=%v, want %v", outErr, wantErr)
	}
}

func TestStreamEventsStreamNextError(t *testing.T) {
	wantErr := errors.New("chunk decode failed")
	model := stubChatModel{nextErr: wantErr}

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("hello"),
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v, want %v", err, wantErr)
	}
	if _, outErr := stream.Output(); !errors.Is(outErr, wantErr) {
		t.Fatalf("output err=%v, want %v", outErr, wantErr)
	}
}

func TestStreamEventsFallbackContentBlocks(t *testing.T) {
	chunk := messages.AI("")
	chunk.ContentBlocks = []messages.ContentBlock{
		messages.TextBlock{Text: "block text"},
		messages.TextBlock{Text: ""}, // empty text blocks produce no delta
		messages.ReasoningBlock{Reasoning: "step one", Index: 1},
		messages.ReasoningBlock{Reasoning: "step two", Index: float64(2)},
		messages.ParseContentBlock(map[string]any{
			"type": "tool_call_chunk",
			"id":   "call_1",
			"name": "search",
			"args": `{"q":"go"}`,
		}),
		messages.NonStandardContentBlock{Type: "custom_block", Value: map[string]any{"foo": "bar"}},
		messages.NonStandardContentBlock{Type: "", Value: map[string]any{"ignored": true}},
	}
	model := NewFakeChatModel(WithStreamChunks(chunk))

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	if got := stream.Text(); got != "block text" {
		t.Fatalf("text: got %q", got)
	}
	if got := stream.Reasoning(); got != "step onestep two" {
		t.Fatalf("reasoning: got %q", got)
	}
	if got := stream.ToolCalls(); len(got) != 1 || got[0].Name != "search" {
		t.Fatalf("tool calls: %+v", got)
	}

	var sawTextDelta, sawReasoningDelta, sawCustomFinish bool
	for _, event := range stream.Events() {
		switch event.Event {
		case streamevents.EventContentBlockDelta:
			if block, ok := event.Delta.(messages.NonStandardContentBlock); ok {
				if block.Type == "text-delta" {
					sawTextDelta = true
				}
				if block.Type == "reasoning-delta" {
					sawReasoningDelta = true
				}
			}
		case streamevents.EventContentBlockFinish:
			if block, ok := event.Content.(messages.NonStandardContentBlock); ok && block.Type == "custom_block" {
				sawCustomFinish = true
			}
		}
	}
	if !sawTextDelta || !sawReasoningDelta || !sawCustomFinish {
		t.Fatalf("events: textDelta=%v reasoningDelta=%v customFinish=%v",
			sawTextDelta, sawReasoningDelta, sawCustomFinish)
	}
}

func TestStreamEventsFallbackToolCalls(t *testing.T) {
	chunk := messages.AI("")
	chunk.ToolCalls = []messages.ToolCall{{
		ID:   "call_1",
		Name: "search",
		Args: map[string]any{"q": "go"},
	}}
	chunk.InvalidToolCalls = []messages.ToolCall{{
		ID:   "call_2",
		Name: "broken",
		Args: map[string]any{"raw": `{"q":`},
	}}
	model := NewFakeChatModel(WithStreamChunks(chunk))

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("search"),
	})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	if got := stream.ToolCalls(); len(got) != 1 || got[0].Name != "search" {
		t.Fatalf("tool calls: %+v", got)
	}
	if got := stream.InvalidToolCalls(); len(got) != 1 || got[0].Name != "broken" {
		t.Fatalf("invalid tool calls: %+v", got)
	}
	// No text was streamed, so the fallback must not emit a text finish block.
	for _, event := range stream.Events() {
		if event.Event == streamevents.EventContentBlockFinish {
			if _, ok := event.Content.(messages.TextBlock); ok {
				t.Fatalf("unexpected text finish block: %+v", event)
			}
		}
	}
	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if output.Content != "" {
		t.Fatalf("output content: got %q want empty", output.Content)
	}
}

func TestStreamEventsEmptyStream(t *testing.T) {
	model := stubChatModel{}

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	events := stream.Events()
	if len(events) != 1 || events[0].Event != streamevents.EventMessageFinish {
		t.Fatalf("events: %+v", events)
	}
}

func TestStreamEventsIgnoresMalformedProtocolEvent(t *testing.T) {
	model := badProtocolChatModel{chunks: []messages.Message{messages.AI("legacy")}}

	stream, err := StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	// A protocol event whose chunk is not a streamevents.Event must be ignored,
	// so the legacy chunk bridge still projects the text.
	if got := stream.Text(); got != "legacy" {
		t.Fatalf("text: got %q", got)
	}
}

// failOnKindHandler returns its error for events of a single kind.
type failOnKindHandler struct {
	kind callbacks.EventKind
	err  error
}

func (h failOnKindHandler) HandleEvent(_ context.Context, event callbacks.Event) error {
	if event.Kind == h.kind {
		return h.err
	}
	return nil
}

// errorMessageStream fails on the first Next call.
type errorMessageStream struct {
	err error
}

func (s errorMessageStream) Next(context.Context) (messages.Message, bool, error) {
	return messages.Message{}, false, s.err
}

func (s errorMessageStream) Close() error { return nil }

// stubChatModel streams fixed chunks (or fails) without emitting callbacks.
type stubChatModel struct {
	chunks    []messages.Message
	streamErr error
	nextErr   error
}

func (m stubChatModel) Invoke(context.Context, []messages.Message, ...runnables.Option) (messages.Message, error) {
	return messages.AI("stub"), nil
}

func (m stubChatModel) Batch(_ context.Context, inputs [][]messages.Message, _ ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i := range out {
		out[i] = messages.AI("stub")
	}
	return out, nil
}

func (m stubChatModel) Stream(
	_ context.Context,
	_ []messages.Message,
	_ ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	if m.streamErr != nil {
		return nil, m.streamErr
	}
	if m.nextErr != nil {
		return errorMessageStream{err: m.nextErr}, nil
	}
	return runnables.NewSliceStream(m.chunks), nil
}

func (m stubChatModel) InputSchema() schema.Schema {
	return schema.Schema{"type": "array"}
}

func (m stubChatModel) OutputSchema() schema.Schema {
	return schema.Schema{"type": "object"}
}

func (m stubChatModel) BindTools([]tools.Tool) (ChatModel, error) {
	return m, nil
}

func (m stubChatModel) Capabilities() ChatModelCapabilities {
	return ChatModelCapabilities{Streaming: true}
}

// badProtocolChatModel emits a protocol event with a chunk of the wrong type
// before streaming legacy chunks.
type badProtocolChatModel struct {
	chunks []messages.Message
}

func (m badProtocolChatModel) Invoke(context.Context, []messages.Message, ...runnables.Option) (messages.Message, error) {
	return messages.AI("stub"), nil
}

func (m badProtocolChatModel) Batch(_ context.Context, inputs [][]messages.Message, _ ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i := range out {
		out[i] = messages.AI("stub")
	}
	return out, nil
}

func (m badProtocolChatModel) Stream(
	ctx context.Context,
	_ []messages.Message,
	opts ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	cfg := runnables.NewConfig(opts...)
	if err := cfg.Callbacks.Emit(ctx, callbacks.Event{
		Kind:  callbacks.EventChatModelProtocol,
		Chunk: "not-a-protocol-event",
	}); err != nil {
		return nil, err
	}
	return runnables.NewSliceStream(m.chunks), nil
}

func (m badProtocolChatModel) InputSchema() schema.Schema {
	return schema.Schema{"type": "array"}
}

func (m badProtocolChatModel) OutputSchema() schema.Schema {
	return schema.Schema{"type": "object"}
}

func (m badProtocolChatModel) BindTools([]tools.Tool) (ChatModel, error) {
	return m, nil
}

func (m badProtocolChatModel) Capabilities() ChatModelCapabilities {
	return ChatModelCapabilities{Streaming: true}
}

func TestFakeChatModelStreamWithoutCallbacks(t *testing.T) {
	model := NewFakeChatModel(WithStreamChunks(
		messages.AI("he"),
		messages.AI("llo"),
	))

	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hello")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var text string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		text += chunk.Content
	}
	if text != "hello" {
		t.Fatalf("streamed text: got %q want %q", text, "hello")
	}
}

// staticErrorLimiter always fails Acquire with the same error and keeps no
// mutable state, so it is safe under concurrent Batch invokes.
type staticErrorLimiter struct {
	err error
}

func (l staticErrorLimiter) Acquire(context.Context, bool) (bool, error) {
	return false, l.err
}
