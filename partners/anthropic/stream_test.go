package anthropic

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// sse formats one server-sent event frame.
func sse(event, data string) string {
	return "event: " + event + "\ndata: " + data + "\n\n"
}

// streamServer serves the given pre-rendered SSE frames.
func streamServer(frames ...string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, frame := range frames {
			_, _ = fmt.Fprint(w, frame)
		}
	}))
}

const streamMessageStart = `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}`

// drainStream reads the stream to completion and returns the emitted chunks.
func drainStream(t *testing.T, stream runnables.Stream[messages.Message]) []messages.Message {
	t.Helper()
	defer stream.Close()
	var chunks []messages.Message
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			return chunks
		}
		chunks = append(chunks, chunk)
	}
}

// finalOutput extracts the terminal message from the recorder's
// EventChatModelEnd event.
func finalOutput(t *testing.T, recorder *callbacks.Recorder) messages.Message {
	t.Helper()
	for _, event := range recorder.Events() {
		if event.Kind == callbacks.EventChatModelEnd {
			if msg, ok := event.Output.(messages.Message); ok {
				return msg
			}
		}
	}
	t.Fatal("no EventChatModelEnd with message output recorded")
	return messages.Message{}
}

func streamWithRecorder(t *testing.T, server *httptest.Server) (runnables.Stream[messages.Message], *callbacks.Recorder) {
	t.Helper()
	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
		runnables.WithMetadata("origin", "test"),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	return stream, recorder
}

func TestStreamToolUseBlock(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"search"}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"weather\"}"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	chunks := drainStream(t, stream)

	// tool_use start, both input_json deltas, and stop each surface a chunk.
	if len(chunks) != 4 {
		t.Fatalf("chunks: %+v", chunks)
	}
	startBlock := messages.BlockToMap(chunks[0].ContentBlocks[0])
	if startBlock["type"] != "tool_use" || startBlock["name"] != "search" {
		t.Fatalf("start chunk block: %+v", startBlock)
	}
	if len(chunks[3].ToolCalls) != 1 || chunks[3].ToolCalls[0].Args["q"] != "weather" {
		t.Fatalf("stop chunk tool call: %+v", chunks[3].ToolCalls)
	}

	output := finalOutput(t, recorder)
	if len(output.ToolCalls) != 1 {
		t.Fatalf("output tool calls: %+v", output.ToolCalls)
	}
	call := output.ToolCalls[0]
	if call.ID != "toolu_1" || call.Name != "search" || call.Args["q"] != "weather" {
		t.Fatalf("output tool call: %+v", call)
	}
	if output.UsageMetadata.OutputTokens != 7 || output.UsageMetadata.TotalTokens != 10 {
		t.Fatalf("output usage: %+v", output.UsageMetadata)
	}
	if output.ResponseMetadata["stop_reason"] != "tool_use" {
		t.Fatalf("output metadata: %+v", output.ResponseMetadata)
	}
}

func TestStreamToolUseStartWithInput(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_2","name":"lookup","input":{"a":1}}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	drainStream(t, stream)

	output := finalOutput(t, recorder)
	if len(output.ToolCalls) != 1 || output.ToolCalls[0].Args["a"] != float64(1) {
		t.Fatalf("output tool calls: %+v", output.ToolCalls)
	}
}

func TestStreamRedactedThinkingBlock(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"ZW5j","id":"rt_1"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	drainStream(t, stream)

	output := finalOutput(t, recorder)
	if len(output.ContentBlocks) != 1 {
		t.Fatalf("output blocks: %+v", output.ContentBlocks)
	}
	bm := messages.BlockToMap(output.ContentBlocks[0])
	if bm["type"] != "reasoning" || bm["data"] != "ZW5j" || bm["id"] != "rt_1" {
		t.Fatalf("redacted thinking block: %+v", bm)
	}
}

func TestStreamThinkingStartPrefilled(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"initial","signature":"sig_0"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	drainStream(t, stream)

	output := finalOutput(t, recorder)
	if len(output.ContentBlocks) != 1 {
		t.Fatalf("output blocks: %+v", output.ContentBlocks)
	}
	bm := messages.BlockToMap(output.ContentBlocks[0])
	if bm["reasoning"] != "initial" || bm["signature"] != "sig_0" {
		t.Fatalf("thinking block: %+v", bm)
	}
}

func TestStreamFinishOpenBlocksOnMessageStop(t *testing.T) {
	// message_stop without content_block_stop must flush the open text block.
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	chunks := drainStream(t, stream)

	if len(chunks) != 1 || chunks[0].Content != "hi" {
		t.Fatalf("chunks: %+v", chunks)
	}
	output := finalOutput(t, recorder)
	if output.Content != "hi" {
		t.Fatalf("output content: %q", output.Content)
	}
	if len(output.ContentBlocks) != 1 || messages.BlockToMap(output.ContentBlocks[0])["text"] != "hi" {
		t.Fatalf("output blocks not flushed: %+v", output.ContentBlocks)
	}
}

func TestStreamErrorEvent(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`),
	)
	defer server.Close()

	stream, _ := streamWithRecorder(t, server)
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "Overloaded") {
		t.Fatalf("stream error event: %v", err)
	}
}

func TestStreamErrorEventWithoutMessage(t *testing.T) {
	server := streamServer(
		sse("error", `{"type":"error","error":{"type":"api_error"}}`),
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	_, _, err = stream.Next(context.Background())
	if err == nil || err.Error() != "anthropic stream error" {
		t.Fatalf("stream error event without message: %v", err)
	}
}

func TestStreamDoneMarker(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		"event: message_stop\ndata: [DONE]\n\n",
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	chunks := drainStream(t, stream)
	if len(chunks) != 0 {
		t.Fatalf("chunks: %+v", chunks)
	}
	if output := finalOutput(t, recorder); output.ID != "msg_1" {
		t.Fatalf("output: %+v", output)
	}
}

func TestStreamInvalidEventJSON(t *testing.T) {
	server := streamServer("event: message_start\ndata: {not json}\n\n")
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	_, _, err = stream.Next(context.Background())
	if err == nil || !strings.Contains(err.Error(), "decode anthropic stream event") {
		t.Fatalf("invalid event json: %v", err)
	}
}

func TestStreamUnknownEventsIgnored(t *testing.T) {
	server := streamServer(
		sse("ping", `{"type":"ping"}`),
		sse("message_start", streamMessageStart),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"mystery_delta","text":"x"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":5}`),
		sse("message_delta", `{"type":"message_delta","delta":{},"usage":{}}`),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		// data payload without "type": the event name line drives dispatch.
		sse("message_stop", `{}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	chunks := drainStream(t, stream)
	if len(chunks) != 1 || chunks[0].Content != "ok" {
		t.Fatalf("chunks: %+v", chunks)
	}
	if output := finalOutput(t, recorder); output.Content != "ok" {
		t.Fatalf("output: %+v", output)
	}
}

func TestStreamNextWithCanceledContext(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := stream.Next(canceled); err == nil {
		t.Fatal("Next with canceled context should fail")
	}
}

func TestStreamEOFWithoutMessageStop(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	chunks := drainStream(t, stream)
	if len(chunks) != 1 || chunks[0].Content != "partial" {
		t.Fatalf("chunks: %+v", chunks)
	}
	if output := finalOutput(t, recorder); output.ID != "msg_1" {
		t.Fatalf("output: %+v", output)
	}
}

func TestStreamBuildRequestError(t *testing.T) {
	bad := messages.Human("")
	bad.ContentBlocks = []messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "image"}),
	}
	model := NewChatModel(modelconfig.WithBaseURL("http://127.0.0.1:1"), modelconfig.WithModel("m"))
	if _, err := model.Stream(context.Background(), []messages.Message{bad}); err == nil {
		t.Fatal("stream with invalid content block should fail")
	}
}

func TestStreamTransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	url := server.URL
	server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(url), modelconfig.WithModel("m"))
	if _, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("stream to a dead server should fail")
	}
}

// stubHandler fails events of a single kind, to exercise callback error paths.
type stubHandler struct {
	failOn callbacks.EventKind
	err    error
}

func (h stubHandler) HandleEvent(_ context.Context, event callbacks.Event) error {
	if event.Kind == h.failOn {
		return h.err
	}
	return nil
}

var errStub = errors.New("stub callback failure")

func TestStreamProtocolCallbackError(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(stubHandler{failOn: callbacks.EventChatModelProtocol, err: errStub})),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); !errors.Is(err, errStub) {
		t.Fatalf("protocol callback error should propagate: %v", err)
	}
}

func TestStreamChunkCallbackError(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`),
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(stubHandler{failOn: callbacks.EventChatModelStream, err: errStub})),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); !errors.Is(err, errStub) {
		t.Fatalf("stream callback error should propagate: %v", err)
	}
}

func TestStreamFlushOpenToolAndThinkingBlocks(t *testing.T) {
	// message_stop with tool and thinking blocks still open must flush both.
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"search"}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}`),
		sse("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"hmm"}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	drainStream(t, stream)

	output := finalOutput(t, recorder)
	if len(output.ToolCalls) != 1 || output.ToolCalls[0].Args["q"] != "x" {
		t.Fatalf("flushed tool call: %+v", output.ToolCalls)
	}
	foundReasoning := false
	for _, block := range output.ContentBlocks {
		if messages.BlockToMap(block)["reasoning"] == "hmm" {
			foundReasoning = true
		}
	}
	if !foundReasoning {
		t.Fatalf("flushed thinking block missing: %+v", output.ContentBlocks)
	}
}

func TestStreamScannerError(t *testing.T) {
	// A truncated body (Content-Length larger than what is sent) makes the
	// stream reader fail mid-scan.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Content-Length", "10000")
		_, _ = fmt.Fprint(w, sse("message_start", streamMessageStart))
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); err == nil {
		t.Fatal("truncated stream body should fail")
	}
}

func TestStreamInvalidBaseURL(t *testing.T) {
	model := NewChatModel(modelconfig.WithBaseURL("://invalid-url"), modelconfig.WithModel("m"))
	if _, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("stream with invalid base URL should fail")
	}
}

// failAfterHandler fails the (after+1)-th callback event of a kind, so tests
// can force an error at each successive emit site along the stream pipeline.
type failAfterHandler struct {
	kind  callbacks.EventKind
	after int
	count int
	err   error
}

func (h *failAfterHandler) HandleEvent(_ context.Context, event callbacks.Event) error {
	if event.Kind == h.kind {
		h.count++
		if h.count > h.after {
			return h.err
		}
	}
	return nil
}

// richStreamFrames exercises every content-block kind in a deterministic
// order: text, tool_use, thinking, redacted_thinking.
func richStreamFrames() []string {
	return []string{
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"search"}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":1}`),
		sse("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"t"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":2}`),
		sse("content_block_start", `{"type":"content_block_start","index":3,"content_block":{"type":"redacted_thinking","data":"ZW5j"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":3}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	}
}

func TestStreamProtocolCallbackFailureAtEachStep(t *testing.T) {
	// Protocol events for richStreamFrames: message start, then one event per
	// content-block start/delta/stop, then message finish (13 total). Failing
	// at each successive event exercises every emitProtocol error branch.
	for step := 1; step <= 13; step++ {
		t.Run(fmt.Sprintf("step_%d", step), func(t *testing.T) {
			server := streamServer(richStreamFrames()...)
			defer server.Close()

			handler := &failAfterHandler{kind: callbacks.EventChatModelProtocol, after: step - 1, err: errStub}
			model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
			stream, err := model.Stream(
				context.Background(),
				[]messages.Message{messages.Human("hi")},
				runnables.WithCallbacks(callbacks.NewManager(handler)),
			)
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			defer stream.Close()
			for {
				if _, _, err := stream.Next(context.Background()); err != nil {
					if !errors.Is(err, errStub) {
						t.Fatalf("unexpected error: %v", err)
					}
					return
				}
			}
		})
	}
}

func TestStreamChunkCallbackFailureAtEachStep(t *testing.T) {
	// Stream chunks for richStreamFrames: text delta, tool start, tool delta,
	// thinking delta, tool stop (5 total).
	for step := 1; step <= 5; step++ {
		t.Run(fmt.Sprintf("step_%d", step), func(t *testing.T) {
			server := streamServer(richStreamFrames()...)
			defer server.Close()

			handler := &failAfterHandler{kind: callbacks.EventChatModelStream, after: step - 1, err: errStub}
			model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
			stream, err := model.Stream(
				context.Background(),
				[]messages.Message{messages.Human("hi")},
				runnables.WithCallbacks(callbacks.NewManager(handler)),
			)
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			defer stream.Close()
			for {
				if _, _, err := stream.Next(context.Background()); err != nil {
					if !errors.Is(err, errStub) {
						t.Fatalf("unexpected error: %v", err)
					}
					return
				}
			}
		})
	}
}

func TestStreamEndCallbackError(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(stubHandler{failOn: callbacks.EventChatModelEnd, err: errStub})),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); !errors.Is(err, errStub) {
		t.Fatalf("end callback error should propagate: %v", err)
	}
}

func TestStreamFlushCallbackError(t *testing.T) {
	// Flushing an open text block at message_stop can also surface a callback
	// error from finishOpenBlocks.
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	handler := &failAfterHandler{kind: callbacks.EventChatModelProtocol, after: 2, err: errStub}
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(handler)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	for {
		if _, _, err := stream.Next(context.Background()); err != nil {
			if !errors.Is(err, errStub) {
				t.Fatalf("unexpected error: %v", err)
			}
			return
		}
	}
}

func TestStreamWithoutCallbacks(t *testing.T) {
	// With no callbacks registered, emitStream/emitProtocol short-circuit.
	server := streamServer(richStreamFrames()...)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunks := drainStream(t, stream)
	if len(chunks) == 0 {
		t.Fatal("expected chunks without callbacks")
	}
}

func TestStreamEmptyEventIgnored(t *testing.T) {
	// A blank line with no accumulated data is skipped.
	server := streamServer(
		"event: noop\n\n",
		sse("message_start", streamMessageStart),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	drainStream(t, stream)
	if output := finalOutput(t, recorder); output.ID != "msg_1" {
		t.Fatalf("output: %+v", output)
	}
}

func TestStreamDoneMarkerEndCallbackError(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		"event: message_stop\ndata: [DONE]\n\n",
	)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(stubHandler{failOn: callbacks.EventChatModelEnd, err: errStub})),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); !errors.Is(err, errStub) {
		t.Fatalf("end callback error on [DONE] should propagate: %v", err)
	}
}

func TestStreamEOFEndCallbackError(t *testing.T) {
	server := streamServer(sse("message_start", streamMessageStart))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(stubHandler{failOn: callbacks.EventChatModelEnd, err: errStub})),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); !errors.Is(err, errStub) {
		t.Fatalf("end callback error on EOF should propagate: %v", err)
	}
}

func TestStreamMessageDeltaWithoutMessageStart(t *testing.T) {
	// Without message_start the accumulated output has no ResponseMetadata;
	// message_delta must initialize it before recording the stop reason.
	server := streamServer(
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	drainStream(t, stream)
	if output := finalOutput(t, recorder); output.ResponseMetadata["stop_reason"] != "end_turn" {
		t.Fatalf("output metadata: %+v", output.ResponseMetadata)
	}
}

func TestStreamFlushToolBlockCallbackError(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"search"}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	handler := &failAfterHandler{kind: callbacks.EventChatModelProtocol, after: 2, err: errStub}
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(handler)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	for {
		if _, _, err := stream.Next(context.Background()); err != nil {
			if !errors.Is(err, errStub) {
				t.Fatalf("unexpected error: %v", err)
			}
			return
		}
	}
}

func TestStreamFlushThinkingBlockCallbackError(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"t"}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	handler := &failAfterHandler{kind: callbacks.EventChatModelProtocol, after: 2, err: errStub}
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(handler)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	for {
		if _, _, err := stream.Next(context.Background()); err != nil {
			if !errors.Is(err, errStub) {
				t.Fatalf("unexpected error: %v", err)
			}
			return
		}
	}
}

func TestStreamUnmarshalableToolArgs(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{
		ID:   "toolu_1",
		Name: "search",
		Args: map[string]any{"bad": func() {}},
	}}
	model := NewChatModel(modelconfig.WithBaseURL("http://127.0.0.1:1"), modelconfig.WithModel("m"))
	if _, err := model.Stream(context.Background(), []messages.Message{ai}); err == nil {
		t.Fatal("stream with unmarshalable tool args should fail")
	}
}
