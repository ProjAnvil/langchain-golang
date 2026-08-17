package ollama

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/streamevents"
)

// failOnKindHandler fails every callback event of a configured kind.
type failOnKindHandler struct {
	kind callbacks.EventKind
}

func (h failOnKindHandler) HandleEvent(_ context.Context, event callbacks.Event) error {
	if event.Kind == h.kind {
		return fmt.Errorf("injected failure on %s", h.kind)
	}
	return nil
}

// failOnProtocolHandler fails protocol events with a configured event name,
// optionally restricted to one content-block index (-1 matches any index).
type failOnProtocolHandler struct {
	event streamevents.EventName
	index int
}

func (h failOnProtocolHandler) HandleEvent(_ context.Context, event callbacks.Event) error {
	if event.Kind != callbacks.EventChatModelProtocol {
		return nil
	}
	protocolEvent, ok := event.Chunk.(streamevents.Event)
	if !ok || protocolEvent.Event != h.event {
		return nil
	}
	if h.index >= 0 && protocolEvent.Index != h.index {
		return nil
	}
	return fmt.Errorf("injected failure on %s index %d", h.event, protocolEvent.Index)
}

func newNDJSONServer(t *testing.T, lines ...string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, line := range lines {
			_, _ = fmt.Fprint(w, line+"\n")
		}
	}))
}

// drainStream consumes the stream until it ends or errors, returning the first
// error encountered.
func drainStream(stream runnables.Stream[messages.Message]) error {
	for {
		_, ok, err := stream.Next(context.Background())
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
	}
}

func TestChatModelStreamRequestMarshalError(t *testing.T) {
	model := NewChatModel(
		modelconfig.WithBaseURL("http://localhost:1"),
		modelconfig.WithExtra(topKKey, make(chan int)),
	)
	_, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected marshal error")
	}
}

func TestChatModelStreamInvalidBaseURL(t *testing.T) {
	model := NewChatModel(modelconfig.WithBaseURL("http://invalid host"))
	_, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected request construction error")
	}
}

func TestChatModelStreamTransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := server.URL
	server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(url))
	_, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected transport error")
	}
}

func TestChatModelStreamEndsWithoutDoneChunk(t *testing.T) {
	server := newNDJSONServer(t,
		"",
		`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"partial"},"done":false}`,
	)
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var content string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		content += chunk.Content
	}
	if content != "partial" {
		t.Fatalf("content: got %q want %q", content, "partial")
	}

	// The stream finalizes gracefully: the open text block is finished and the
	// chat-model end event fires.
	endEvents := filterEvents(recorder.Events(), callbacks.EventChatModelEnd)
	if len(endEvents) != 1 {
		t.Fatalf("end events: got %d want 1", len(endEvents))
	}
	output, ok := endEvents[0].Output.(messages.Message)
	if !ok || output.Content != "partial" {
		t.Fatalf("end output: %+v", endEvents[0].Output)
	}

	// Next after the stream is done keeps reporting completion.
	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("next after done: ok=%v err=%v", ok, err)
	}
}

func TestChatModelStreamNextWithCanceledContext(t *testing.T) {
	server := newNDJSONServer(t,
		`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"hi"},"done":false}`,
	)
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := stream.Next(ctx); err == nil {
		t.Fatal("expected context error")
	}

	errorEvents := filterEvents(recorder.Events(), callbacks.EventChatModelError)
	if len(errorEvents) != 1 {
		t.Fatalf("error events: got %d want 1", len(errorEvents))
	}
}

func TestChatModelStreamScannerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		// One NDJSON line larger than the scanner's 1MB max token size.
		_, _ = fmt.Fprint(w, strings.Repeat("a", 1024*1024+1)+"\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	if err := drainStream(stream); err == nil {
		t.Fatal("expected scanner error")
	}
	errorEvents := filterEvents(recorder.Events(), callbacks.EventChatModelError)
	if len(errorEvents) != 1 {
		t.Fatalf("error events: got %d want 1", len(errorEvents))
	}
}

func TestChatModelStreamReasoningLevelString(t *testing.T) {
	server := newNDJSONServer(t,
		`{"model":"deepseek-r1","created_at":"t1","message":{"role":"assistant","thinking":"hmm"},"done":false}`,
		`{"model":"deepseek-r1","created_at":"t2","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`,
	)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		WithReasoning("low"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var reasoning string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		for _, block := range chunk.ContentBlocks {
			if text, ok := messages.BlockToMap(block)["reasoning"].(string); ok {
				reasoning += text
			}
		}
	}
	if reasoning != "hmm" {
		t.Fatalf("reasoning: got %q want %q", reasoning, "hmm")
	}
}

func TestChatModelStreamReasoningFalseStringDisablesThinking(t *testing.T) {
	server := newNDJSONServer(t,
		`{"model":"deepseek-r1","created_at":"t1","message":{"role":"assistant","thinking":"hmm"},"done":false}`,
		`{"model":"deepseek-r1","created_at":"t2","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`,
	)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		WithReasoning("false"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var reasoning string
	var content string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		content += chunk.Content
		for _, block := range chunk.ContentBlocks {
			if text, ok := messages.BlockToMap(block)["reasoning"].(string); ok {
				reasoning += text
			}
		}
	}
	if reasoning != "" {
		t.Fatalf("reasoning should be suppressed: got %q", reasoning)
	}
	if content != "ok" {
		t.Fatalf("content: got %q want %q", content, "ok")
	}
}

// TestChatModelStreamProtocolEmitFailures drives every streamed step into its
// callback-error branch with handlers that fail on specific protocol events.
func TestChatModelStreamProtocolEmitFailures(t *testing.T) {
	textLines := []string{
		`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"hi"},"done":false}`,
		`{"model":"llama3","created_at":"t2","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`,
	}
	reasoningLines := []string{
		`{"model":"deepseek-r1","created_at":"t1","message":{"role":"assistant","thinking":"hmm"},"done":false}`,
		`{"model":"deepseek-r1","created_at":"t2","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`,
	}
	toolLines := []string{
		`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"add","arguments":{"a":1}}}]},"done":true,"done_reason":"stop"}`,
	}

	cases := []struct {
		name      string
		lines     []string
		reasoning bool
		handler   failOnProtocolHandler
	}{
		{"message start before text", textLines, false, failOnProtocolHandler{event: streamevents.EventMessageStart, index: -1}},
		{"text block start", textLines, false, failOnProtocolHandler{event: streamevents.EventContentBlockStart, index: textBlockIndex}},
		{"text delta", textLines, false, failOnProtocolHandler{event: streamevents.EventContentBlockDelta, index: textBlockIndex}},
		{"text block finish", textLines, false, failOnProtocolHandler{event: streamevents.EventContentBlockFinish, index: textBlockIndex}},
		{"message finish", textLines, false, failOnProtocolHandler{event: streamevents.EventMessageFinish, index: -1}},
		{"message start before reasoning", reasoningLines, true, failOnProtocolHandler{event: streamevents.EventMessageStart, index: -1}},
		{"reasoning block start", reasoningLines, true, failOnProtocolHandler{event: streamevents.EventContentBlockStart, index: reasoningBlockIndex}},
		{"reasoning delta", reasoningLines, true, failOnProtocolHandler{event: streamevents.EventContentBlockDelta, index: reasoningBlockIndex}},
		{"reasoning block finish", reasoningLines, true, failOnProtocolHandler{event: streamevents.EventContentBlockFinish, index: reasoningBlockIndex}},
		{"message start before tool call", toolLines, false, failOnProtocolHandler{event: streamevents.EventMessageStart, index: -1}},
		{"tool call block start", toolLines, false, failOnProtocolHandler{event: streamevents.EventContentBlockStart, index: firstToolBase}},
		{"tool call block finish", toolLines, false, failOnProtocolHandler{event: streamevents.EventContentBlockFinish, index: firstToolBase}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := newNDJSONServer(t, tc.lines...)
			defer server.Close()

			opts := []modelconfig.Option{modelconfig.WithBaseURL(server.URL)}
			if tc.reasoning {
				opts = append(opts, WithReasoning(true))
			}
			model := NewChatModel(opts...)
			stream, err := model.Stream(
				context.Background(),
				[]messages.Message{messages.Human("hi")},
				runnables.WithCallbacks(callbacks.NewManager(tc.handler)),
			)
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			defer stream.Close()

			if err := drainStream(stream); err == nil {
				t.Fatal("expected injected protocol failure")
			}
		})
	}
}

// TestChatModelStreamChunkEmitFailures covers the EventChatModelStream error
// branches of the text, reasoning, and tool-call steps.
func TestChatModelStreamChunkEmitFailures(t *testing.T) {
	cases := []struct {
		name      string
		lines     []string
		reasoning bool
	}{
		{"text chunk", []string{
			`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"hi"},"done":false}`,
		}, false},
		{"reasoning chunk", []string{
			`{"model":"deepseek-r1","created_at":"t1","message":{"role":"assistant","thinking":"hmm"},"done":false}`,
		}, true},
		{"tool call chunk", []string{
			`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"add","arguments":{"a":1}}}]},"done":true,"done_reason":"stop"}`,
		}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := newNDJSONServer(t, tc.lines...)
			defer server.Close()

			opts := []modelconfig.Option{modelconfig.WithBaseURL(server.URL)}
			if tc.reasoning {
				opts = append(opts, WithReasoning(true))
			}
			model := NewChatModel(opts...)
			stream, err := model.Stream(
				context.Background(),
				[]messages.Message{messages.Human("hi")},
				runnables.WithCallbacks(callbacks.NewManager(
					failOnKindHandler{kind: callbacks.EventChatModelStream},
				)),
			)
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			defer stream.Close()

			if err := drainStream(stream); err == nil {
				t.Fatal("expected injected stream failure")
			}
		})
	}
}

func TestChatModelStreamFinalizeErrorAtEOF(t *testing.T) {
	// The stream ends without a done chunk; finalizing the still-open text
	// block fails, so the EOF path must surface the callback error.
	server := newNDJSONServer(t,
		`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"hi"},"done":false}`,
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(
			failOnProtocolHandler{event: streamevents.EventContentBlockFinish, index: textBlockIndex},
		)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	if err := drainStream(stream); err == nil {
		t.Fatal("expected finalize error at EOF")
	}
}
