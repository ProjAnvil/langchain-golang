package gemini

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// newStreamServer serves SSE frames for streamGenerateContent requests.
func newStreamServer(t *testing.T, frames ...string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "streamGenerateContent") {
			t.Errorf("stream path: %q", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, frame := range frames {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
		}
	}))
}

func TestStreamTextChunks(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}`,
		`{"candidates":[{"content":{"parts":[{"text":" world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}`,
	)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	chunks := drainStream(t, stream)
	if len(chunks) != 2 {
		t.Fatalf("chunks: %d (%+v)", len(chunks), chunks)
	}
	if chunks[0].Content != "Hello" || chunks[1].Content != " world" {
		t.Errorf("chunk texts: %+v", chunks)
	}
	// Usage lands only on the terminal chunk (the API reports cumulative
	// counts per response; re-emitting them would overcount).
	if chunks[0].UsageMetadata != (messages.UsageMetadata{}) {
		t.Errorf("intermediate chunk carries usage: %+v", chunks[0].UsageMetadata)
	}
	if chunks[1].UsageMetadata.InputTokens != 3 || chunks[1].UsageMetadata.TotalTokens != 5 {
		t.Errorf("terminal usage: %+v", chunks[1].UsageMetadata)
	}
}

func TestStreamToolCallsComplete(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[
			{"functionCall":{"name":"magic_function","args":{"input":3}}}
		]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6}}`,
	)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	chunks := drainStream(t, stream)
	var complete *messages.ToolCall
	for _, chunk := range chunks {
		for _, call := range chunk.ToolCalls {
			if call.Name == "magic_function" && call.ID != "" && call.Args["input"] != nil {
				complete = new(call)
			}
		}
	}
	if complete == nil {
		t.Fatalf("no complete tool call surfaced on any chunk: %+v", chunks)
	}
	if input, ok := complete.Args["input"].(float64); !ok || input != 3 {
		t.Errorf("tool call args: %v", complete.Args)
	}
}

// TestStreamTerminalUsageOnlyChunk covers streams whose final SSE response
// carries usage after the finish-carrying chunk: a terminal usage-only chunk
// is appended before the stream ends (the anthropic stream_usage shape).
func TestStreamTerminalUsageOnlyChunk(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}`,
		`{"candidates":[],"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":2,"totalTokenCount":10}}`,
	)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	chunks := drainStream(t, stream)
	if len(chunks) != 2 {
		t.Fatalf("chunks: %d (%+v)", len(chunks), chunks)
	}
	last := chunks[len(chunks)-1]
	if last.UsageMetadata.InputTokens != 8 || last.UsageMetadata.TotalTokens != 10 {
		t.Errorf("terminal usage chunk: %+v", last.UsageMetadata)
	}
}

func TestStreamCancellationTerminates(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":" world"}]}},"finishReason":"STOP"}]}`,
	)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	ctx, cancel := context.WithCancel(t.Context())
	stream, err := model.Stream(ctx, []messages.Message{messages.Human("hi")})
	if err != nil {
		cancel()
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()
	cancel()

	terminated := make(chan struct{})
	go func() {
		defer close(terminated)
		for {
			_, ok, err := stream.Next(ctx)
			if err != nil || !ok {
				return
			}
		}
	}()
	select {
	case <-terminated:
	case <-time.After(10 * time.Second):
		t.Fatalf("stream did not terminate after context cancellation")
	}
}

// TestStreamCloseIdempotent: Close is safe to call repeatedly (the anthropic
// adapter's contract) — a drain-then-Close or defer-Close consumer must not
// panic on the second call.
func TestStreamCloseIdempotent(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`,
	)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	_ = drainStream(t, stream)
	if err := stream.Close(); err != nil {
		t.Fatalf("close after drain: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}

func TestStreamHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"code":404,"message":"model not found","status":"NOT_FOUND"}}`, http.StatusNotFound)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream call: %v", err)
	}
	defer func() { _ = stream.Close() }()
	if _, _, err := stream.Next(t.Context()); err == nil {
		t.Fatalf("expected stream error, got none")
	} else if !strings.Contains(err.Error(), "404") && !strings.Contains(err.Error(), "model not found") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestStreamCloseBeforeDrain(t *testing.T) {
	blocking := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Send headers and the first frame so Stream() returns, then hold the
		// body open — Close must cancel the in-flight request and end the
		// stream without waiting for the rest.
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"first\"}]}}]}\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		<-blocking
		_, _ = fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"late\"}]}}]}\n\n")
	}))
	defer server.Close()
	defer close(blocking)

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, ok, err := stream.Next(t.Context()); err != nil || ok {
		t.Fatalf("next after close: ok=%v err=%v", ok, err)
	}
}

func TestStreamCallbacks(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":" there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":2,"totalTokenCount":4}}`,
	)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	recorder := callbacks.NewRecorder()
	stream, err := model.Stream(
		t.Context(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()
	_ = drainStream(t, stream)

	var sawStart, sawStream, sawEnd bool
	var endEvent *callbacks.Event
	for _, event := range recorder.Events() {
		switch event.Kind {
		case callbacks.EventChatModelStart:
			sawStart = true
		case callbacks.EventChatModelStream:
			sawStream = true
		case callbacks.EventChatModelEnd:
			sawEnd = true
			endEvent = new(event)
		}
	}
	if !sawStart || !sawStream || !sawEnd {
		t.Fatalf("missing events: start=%v stream=%v end=%v (%d events)",
			sawStart, sawStream, sawEnd, len(recorder.Events()))
	}
	output, ok := endEvent.Output.(messages.Message)
	if !ok {
		t.Fatalf("chat_model_end output: %T", endEvent.Output)
	}
	if output.Content != "Hello there" {
		t.Errorf("aggregated end output: %q", output.Content)
	}
	if output.UsageMetadata.TotalTokens != 4 {
		t.Errorf("aggregated end usage: %+v", output.UsageMetadata)
	}
}

// TestStreamUnconfiguredClient: a zero-value ChatModel fails Stream with the
// client-configuration error before any request.
func TestStreamUnconfiguredClient(t *testing.T) {
	_, err := ChatModel{}.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "client is not configured") {
		t.Fatalf("zero-value stream err = %v, want unconfigured client", err)
	}
}

// TestStreamRejectsUnknownRole: a malformed history fails Stream setup and
// the error callback event is emitted.
func TestStreamRejectsUnknownRole(t *testing.T) {
	recorder := callbacks.NewRecorder()
	server := newStreamServer(t) // never reached
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("k"))
	odd := messages.Human("hi")
	odd.Role = messages.Role("alien")
	_, err := model.Stream(
		t.Context(),
		[]messages.Message{odd},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err == nil || !strings.Contains(err.Error(), `unexpected message role "alien"`) {
		t.Fatalf("stream err = %v, want role error", err)
	}
	var errorEvents []callbacks.Event
	for _, event := range recorder.Events() {
		if event.Kind == callbacks.EventChatModelError {
			errorEvents = append(errorEvents, event)
		}
	}
	if len(errorEvents) != 1 {
		t.Fatalf("error events: %+v", errorEvents)
	}
}

// TestStreamRejectsMalformedToolSchema: a bound tool whose schema cannot
// convert fails Stream setup the same way.
func TestStreamRejectsMalformedToolSchema(t *testing.T) {
	server := newStreamServer(t) // never reached
	defer server.Close()

	base, err := coretools.FromFunc("broken", "schema carrier",
		func(context.Context) (any, error) { return nil, nil })
	if err != nil {
		t.Fatalf("FromFunc() error = %v", err)
	}
	poisoned := &schemaOverrider{Tool: base, schema: schema.Schema{"type": "widget"}}
	bound, err := NewChatModel(
		modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("k"),
	).BindTools([]coretools.Tool{poisoned})
	if err != nil {
		t.Fatalf("BindTools() error = %v", err)
	}
	_, err = bound.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), `tool "broken" schema`) {
		t.Fatalf("stream err = %v, want tool schema error", err)
	}
}

// TestStreamStartCallbackError: a failing OnChatModelStart handler aborts
// Stream before any request.
func TestStreamStartCallbackError(t *testing.T) {
	server := newStreamServer(t) // never reached
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("k"))
	_, err := model.Stream(
		t.Context(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(callbackStubHandler{
			failOn: callbacks.EventChatModelStart, err: errCallbackStub,
		})),
	)
	if !errors.Is(err, errCallbackStub) {
		t.Fatalf("stream start callback error should propagate: %v", err)
	}
}

// TestStreamChunkCallbackError: a failing stream-chunk handler surfaces on
// the Next call that tried to emit it.
func TestStreamChunkCallbackError(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":" world"}]}},"finishReason":"STOP"}]}`,
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("k"))
	stream, err := model.Stream(
		t.Context(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(callbackStubHandler{
			failOn: callbacks.EventChatModelStream, err: errCallbackStub,
		})),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()
	if _, _, err := stream.Next(t.Context()); err == nil {
		t.Fatal("first Next must surface the chunk callback error")
	}
}

// TestStreamEndCallbackError: a failing OnChatModelEnd handler surfaces on
// the terminal Next call.
func TestStreamEndCallbackError(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`,
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("k"))
	stream, err := model.Stream(
		t.Context(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(callbackStubHandler{
			failOn: callbacks.EventChatModelEnd, err: errCallbackStub,
		})),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()
	if _, ok, err := stream.Next(t.Context()); err != nil || !ok {
		t.Fatalf("content chunk: ok=%v err=%v", ok, err)
	}
	if _, _, err := stream.Next(t.Context()); !errors.Is(err, errCallbackStub) {
		t.Fatalf("terminal Next err = %v, want end callback error", err)
	}
}

// TestStreamModelVersionMetadataAndSilentFinish: responses carrying
// modelVersion stamp the aggregate output's metadata; a finish-only response
// with nothing to say ends the stream silently without yielding a chunk.
func TestStreamModelVersionMetadataAndSilentFinish(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"finishReason":"STOP"}],"modelVersion":"gemini-test","responseId":"resp_s"}`,
		`{"candidates":[{"finishReason":"STOP"}],"modelVersion":"gemini-test","responseId":"resp_s"}`,
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("k"))
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	genaiStream := stream.(*genaiStream)
	defer func() { _ = stream.Close() }()

	chunks := drainStream(t, stream)
	if len(chunks) != 1 || chunks[0].Content != "partial" {
		t.Fatalf("chunks = %+v, want the single content chunk", chunks)
	}
	if genaiStream.output.ID != "resp_s" {
		t.Fatalf("aggregate ID = %q, want resp_s", genaiStream.output.ID)
	}
	if genaiStream.output.ResponseMetadata["model"] != "gemini-test" ||
		genaiStream.output.ResponseMetadata["model_provider"] != "gemini" {
		t.Fatalf("aggregate metadata = %v", genaiStream.output.ResponseMetadata)
	}
}

// TestStreamSkipsEmptyFrames: bookkeeping-only frames (nil parts, empty
// candidates, empty text) yield no chunks but do not end the stream.
func TestStreamSkipsEmptyFrames(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[null,{"text":""}]}}]}`,
		`{"candidates":[{}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"real"}]},"finishReason":"STOP"}]}`,
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("k"))
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	chunks := drainStream(t, stream)
	if len(chunks) != 1 || chunks[0].Content != "real" {
		t.Fatalf("chunks = %+v, want only the real text chunk", chunks)
	}
}

// TestStreamPendingUsageChunkCallbackError: the deferred terminal usage-only
// chunk (usage arriving after the finish-carrying frame) goes through the
// same chunk callback, and a failing handler surfaces there.
func TestStreamPendingUsageChunkCallbackError(t *testing.T) {
	server := newStreamServer(t,
		`{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}`,
		`{"candidates":[],"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":2,"totalTokenCount":10}}`,
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("k"))
	stream, err := model.Stream(
		t.Context(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(callbackStubHandler{
			failOn: callbacks.EventChatModelStream, err: errCallbackStub,
		})),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer func() { _ = stream.Close() }()
	// The content chunk fails the callback first; the stream is dead after.
	if _, _, err := stream.Next(t.Context()); !errors.Is(err, errCallbackStub) {
		t.Fatalf("content chunk err = %v, want callback error", err)
	}
}

// TestGenaiStreamPendingChunkCallbackError: with a healthy content chunk but
// a failing stream handler, the failure lands exactly on the deferred
// usage-only chunk's Next call.
func TestGenaiStreamPendingChunkCallbackError(t *testing.T) {
	s := &genaiStream{
		cfg:    runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(callbackStubHandler{failOn: callbacks.EventChatModelStream, err: errCallbackStub}))),
		output: messages.AI(""),
		pending: []messages.Message{func() messages.Message {
			m := messages.AI("")
			m.UsageMetadata = messages.UsageMetadata{TotalTokens: 3}
			return m
		}()},
	}
	if _, _, err := s.Next(t.Context()); !errors.Is(err, errCallbackStub) {
		t.Fatalf("pending chunk err = %v, want callback error", err)
	}
}

// TestGenaiStreamHandleResponseNil: a nil SSE response frame is skipped
// without yielding and without touching the aggregate output.
func TestGenaiStreamHandleResponseNil(t *testing.T) {
	s := &genaiStream{output: messages.AI("")}
	chunk, yield := s.handleResponse(nil)
	if yield {
		t.Fatalf("nil response must not yield, got %+v", chunk)
	}
}
