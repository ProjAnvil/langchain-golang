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
	chunk.ContentBlocks = []messages.ContentBlock{{
		"type":  "tool_call_chunk",
		"id":    "call_1",
		"name":  "search",
		"args":  `{"q": `,
		"index": 0,
	}}
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
			Content: messages.ContentBlock{
				"type": "text",
				"text": "",
			},
		},
		{
			Event: streamevents.EventContentBlockDelta,
			Index: 0,
			Delta: messages.ContentBlock{
				"type": "text-delta",
				"text": "native",
			},
		},
		{
			Event: streamevents.EventContentBlockFinish,
			Index: 0,
			Content: messages.ContentBlock{
				"type": "text",
				"text": "native",
			},
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
