package openai

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
	"github.com/projanvil/langchain-golang/core/streamevents"
)

// failOnNthMatch is a callback handler that returns an error on the n-th
// (1-based) event matching match, letting tests drive the emit/emitStream/
// emitProtocol error branches at precise points in a stream.
type failOnNthMatch struct {
	match func(callbacks.Event) bool
	n     int
	seen  int
}

func (f *failOnNthMatch) HandleEvent(_ context.Context, event callbacks.Event) error {
	if f.match(event) {
		f.seen++
		if f.seen == f.n {
			return errors.New("callback boom")
		}
	}
	return nil
}

func onKind(kind callbacks.EventKind) func(callbacks.Event) bool {
	return func(e callbacks.Event) bool { return e.Kind == kind }
}

func onProtocolEvent(name streamevents.EventName) func(callbacks.Event) bool {
	return func(e callbacks.Event) bool {
		if e.Kind != callbacks.EventChatModelProtocol {
			return false
		}
		streamEvent, ok := e.Chunk.(streamevents.Event)
		return ok && streamEvent.Event == name
	}
}

func sseServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, body)
	}))
	t.Cleanup(server.Close)
	return server
}

const (
	sseTextDelta     = `data: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}` + "\n\n"
	sseTextDone      = `data: {"type":"response.output_text.done","output_index":0}` + "\n\n"
	sseCompleted     = `data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-test","output":[],"usage":{}}}` + "\n\n"
	sseFuncAdded     = `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"adder","arguments":""}}` + "\n\n"
	sseFuncArgsDelta = `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"a\":2}"}` + "\n\n"
	sseFuncDone      = `data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"adder","arguments":"{\"a\":2}"}}` + "\n\n"
	sseFuncDoneBad   = `data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"adder","arguments":"{bad json}"}}` + "\n\n"
	sseWSAdded       = `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}}` + "\n\n"
	sseWSDone        = `data: {"type":"response.output_item.done","output_index":0,"item":{"id":"ws_1","type":"web_search_call","status":"completed"}}` + "\n\n"
	sseReasonDelta   = `data: {"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"thinking"}` + "\n\n"
	sseRefusal       = `data: {"type":"response.refusal.done","refusal":"no"}` + "\n\n"
)

// TestResponseStreamEmitErrors drives the error-return branches of emit,
// emitStream, and emitProtocol with a handler that fails at a chosen event.
func TestResponseStreamEmitErrors(t *testing.T) {
	cases := []struct {
		name  string
		sse   string
		match func(callbacks.Event) bool
		n     int
	}{
		{"text delta emitStream", sseTextDelta, onKind(callbacks.EventChatModelStream), 1},
		{"text delta protocol start", sseTextDelta, onProtocolEvent(streamevents.EventMessageStart), 1},
		{"text done finishTextBlock", sseTextDelta + sseTextDone, onProtocolEvent(streamevents.EventContentBlockFinish), 1},
		{"function call added protocol", sseFuncAdded, onProtocolEvent(streamevents.EventContentBlockStart), 1},
		{"function call added emitStream", sseFuncAdded, onKind(callbacks.EventChatModelStream), 1},
		{"args delta protocol", sseFuncArgsDelta, onProtocolEvent(streamevents.EventContentBlockDelta), 1},
		{"args delta emitStream", sseFuncArgsDelta, onKind(callbacks.EventChatModelStream), 1},
		{"protocol item start protocol", sseWSAdded, onProtocolEvent(streamevents.EventContentBlockStart), 1},
		{"protocol item start emitStream", sseWSAdded, onKind(callbacks.EventChatModelStream), 1},
		{"protocol item finish protocol", sseWSAdded + sseWSDone, onProtocolEvent(streamevents.EventContentBlockFinish), 1},
		{"protocol item finish emitStream", sseWSAdded + sseWSDone, onKind(callbacks.EventChatModelStream), 2},
		{"function call finish protocol", sseFuncAdded + sseFuncDone, onProtocolEvent(streamevents.EventContentBlockFinish), 1},
		{"invalid function call finish protocol", sseFuncAdded + sseFuncDoneBad, onProtocolEvent(streamevents.EventContentBlockFinish), 1},
		{"function call finish emitStream", sseFuncAdded + sseFuncDone, onKind(callbacks.EventChatModelStream), 2},
		{"reasoning delta protocol", sseReasonDelta, onProtocolEvent(streamevents.EventContentBlockDelta), 1},
		{"reasoning delta emitStream", sseReasonDelta, onKind(callbacks.EventChatModelStream), 1},
		{"refusal emitStream", sseRefusal, onKind(callbacks.EventChatModelStream), 1},
		{"completed finishOpenTextBlocks", sseTextDelta + sseCompleted, onProtocolEvent(streamevents.EventContentBlockFinish), 1},
		{"completed message finish", sseCompleted, onProtocolEvent(streamevents.EventMessageFinish), 1},
		{"completed end event", sseCompleted, onKind(callbacks.EventChatModelEnd), 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := sseServer(t, tc.sse)
			model := NewChatModel(
				modelconfig.WithBaseURL(server.URL),
				modelconfig.WithModel("gpt-test"),
			)
			handler := &failOnNthMatch{match: tc.match, n: tc.n}
			stream, err := model.Stream(
				context.Background(),
				[]messages.Message{messages.Human("hi")},
				runnables.WithCallbacks(callbacks.NewManager(handler)),
			)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			defer stream.Close()

			var streamErr error
			for i := 0; i < 20; i++ {
				_, ok, err := stream.Next(context.Background())
				if err != nil {
					streamErr = err
					break
				}
				if !ok {
					break
				}
			}
			if streamErr == nil || !strings.Contains(streamErr.Error(), "callback boom") {
				t.Fatalf("expected callback boom error, got %v", streamErr)
			}
		})
	}
}

func TestChatModelCallbackStartAndEndErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(responsesAPIBody))
	}))
	defer server.Close()

	t.Run("invoke start", func(t *testing.T) {
		model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
		handler := &failOnNthMatch{match: onKind(callbacks.EventChatModelStart), n: 1}
		_, err := model.Invoke(
			context.Background(),
			[]messages.Message{messages.Human("hi")},
			runnables.WithCallbacks(callbacks.NewManager(handler)),
		)
		if err == nil || !strings.Contains(err.Error(), "callback boom") {
			t.Fatalf("expected callback boom, got %v", err)
		}
	})

	t.Run("invoke end", func(t *testing.T) {
		model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
		handler := &failOnNthMatch{match: onKind(callbacks.EventChatModelEnd), n: 1}
		_, err := model.Invoke(
			context.Background(),
			[]messages.Message{messages.Human("hi")},
			runnables.WithCallbacks(callbacks.NewManager(handler)),
		)
		if err == nil || !strings.Contains(err.Error(), "callback boom") {
			t.Fatalf("expected callback boom, got %v", err)
		}
	})

	t.Run("stream start", func(t *testing.T) {
		server := sseServer(t, sseCompleted)
		model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
		handler := &failOnNthMatch{match: onKind(callbacks.EventChatModelStart), n: 1}
		stream, err := model.Stream(
			context.Background(),
			[]messages.Message{messages.Human("hi")},
			runnables.WithCallbacks(callbacks.NewManager(handler)),
		)
		if err == nil || !strings.Contains(err.Error(), "callback boom") {
			if stream != nil {
				_ = stream.Close()
			}
			t.Fatalf("expected callback boom, got %v", err)
		}
	})
}

// TestResponseStreamDoneMarkerAndBlankLines covers the [DONE] sentinel and
// blank-line flushing with no accumulated data in the Responses SSE stream.
func TestResponseStreamDoneMarkerAndBlankLines(t *testing.T) {
	server := sseServer(t, "\n\ndata: [DONE]\n\n")
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected immediate stream end, ok=%v err=%v", ok, err)
	}
}

// TestResponseStreamTextDoneWithoutDelta covers finishTextBlock being called
// for an index that never started.
func TestResponseStreamTextDoneWithoutDelta(t *testing.T) {
	server := sseServer(t, sseTextDone+sseCompleted)
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected stream end without chunks, ok=%v err=%v", ok, err)
	}
}

// TestResponseStreamOutputItemDoneNonProtocol covers output_item.done for an
// item type that is neither a function call nor a protocol item.
func TestResponseStreamOutputItemDoneNonProtocol(t *testing.T) {
	server := sseServer(t,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant"}}`+"\n\n"+
			sseCompleted,
	)
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected stream end without chunks, ok=%v err=%v", ok, err)
	}
}

func TestResponseStreamBadBaseURL(t *testing.T) {
	model := NewChatModel(
		modelconfig.WithBaseURL("://bad-url"),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected request construction error")
	}
}

func TestResponseStreamTransportError(t *testing.T) {
	model := NewChatModel(
		modelconfig.WithBaseURL("http://127.0.0.1:1"),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected transport error")
	}
}

// errReadCloser fails on Read to drive the scanner-error branches.
type errReadCloser struct{}

func (errReadCloser) Read([]byte) (int, error) { return 0, errors.New("read boom") }
func (errReadCloser) Close() error             { return nil }

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func failingBodyHTTPClient() *http.Client {
	return &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			Status:     "200 OK",
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       errReadCloser{},
			Request:    req,
		}, nil
	})}
}

func TestResponseStreamScannerError(t *testing.T) {
	model := NewChatModel(
		modelconfig.WithBaseURL("http://stream.test"),
		modelconfig.WithModel("gpt-test"),
		func(c *modelconfig.Config) { c.HTTPClient = failingBodyHTTPClient() },
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	_, _, err = stream.Next(context.Background())
	if err == nil || !strings.Contains(err.Error(), "read boom") {
		t.Fatalf("expected read boom, got %v", err)
	}
}

func TestChatCompletionsStreamScannerError(t *testing.T) {
	model := NewChatModel(
		modelconfig.WithBaseURL("http://stream.test"),
		modelconfig.WithModel("gpt-test"),
		func(c *modelconfig.Config) { c.HTTPClient = failingBodyHTTPClient() },
	).WithChatCompletions()
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	_, _, err = stream.Next(context.Background())
	if err == nil || !strings.Contains(err.Error(), "read boom") {
		t.Fatalf("expected read boom, got %v", err)
	}
}

func TestChatCompletionsStreamTransportError(t *testing.T) {
	model := NewChatModel(
		modelconfig.WithBaseURL("http://127.0.0.1:1"),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected transport error")
	}
}

// TestChatCompletionsStreamEmitErrors drives the emitStream error branches of
// the Chat Completions stream.
func TestChatCompletionsStreamEmitErrors(t *testing.T) {
	cases := []struct {
		name string
		sse  string
	}{
		{"content chunk", "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"},
		{"reasoning chunk", "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"r\"}}]}\n\n"},
		{"final tool call chunk", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]}}]}\n\ndata: [DONE]\n\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := sseServer(t, tc.sse)
			model := NewChatModel(
				modelconfig.WithBaseURL(server.URL),
				modelconfig.WithModel("gpt-test"),
			).WithChatCompletions()
			handler := &failOnNthMatch{match: onKind(callbacks.EventChatModelStream), n: 1}
			stream, err := model.Stream(
				context.Background(),
				[]messages.Message{messages.Human("hi")},
				runnables.WithCallbacks(callbacks.NewManager(handler)),
			)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			defer stream.Close()

			_, _, err = stream.Next(context.Background())
			if err == nil || !strings.Contains(err.Error(), "callback boom") {
				t.Fatalf("expected callback boom, got %v", err)
			}
		})
	}
}

// TestChatCompletionsStreamSparseToolCallIndex covers the finalToolCallChunk
// path where streamed tool call indices leave gaps below the highest index.
func TestChatCompletionsStreamSparseToolCallIndex(t *testing.T) {
	t.Run("gaps below highest index are skipped", func(t *testing.T) {
		server := sseServer(t,
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":2,\"id\":\"call_2\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]}}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]}}]}\n\n"+
				"data: [DONE]\n\n",
		)
		model := NewChatModel(
			modelconfig.WithBaseURL(server.URL),
			modelconfig.WithModel("gpt-test"),
		).WithChatCompletions()
		stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		defer stream.Close()

		// Index 1 never streamed: finalToolCallChunk iterates 0..len-1, so the
		// missing slot is skipped — but the call at index 2 is also dropped
		// (same lossy behavior locked by the second subtest).
		chunk, ok, err := stream.Next(context.Background())
		if err != nil || !ok {
			t.Fatalf("Next: ok=%v err=%v", ok, err)
		}
		if len(chunk.ToolCalls) != 1 || chunk.ToolCalls[0].ID != "call_0" {
			t.Fatalf("tool calls = %#v", chunk.ToolCalls)
		}
	})

	t.Run("single high index yields no final chunk", func(t *testing.T) {
		server := sseServer(t,
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":2,\"id\":\"call_2\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]}}]}\n\n"+
				"data: [DONE]\n\n",
		)
		model := NewChatModel(
			modelconfig.WithBaseURL(server.URL),
			modelconfig.WithModel("gpt-test"),
		).WithChatCompletions()
		stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		defer stream.Close()

		// NOTE: finalToolCallChunk iterates 0..len(toolCalls)-1, so a lone
		// tool call at index 2 is never assembled — the stream ends without a
		// final chunk. This locks the current (lossy) behavior; it looks like a
		// production bug worth reporting rather than covering up.
		if _, ok, err := stream.Next(context.Background()); err != nil || ok {
			t.Fatalf("expected stream end without chunk, ok=%v err=%v", ok, err)
		}
	})
}

// TestResponseStreamEndEmitErrors covers the EventChatModelEnd emit error
// branches at stream end (both the [DONE] sentinel and scanner EOF).
func TestResponseStreamEndEmitErrors(t *testing.T) {
	cases := []struct {
		name string
		sse  string
	}{
		{"done marker", "data: [DONE]\n\n"},
		{"scanner EOF", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := sseServer(t, tc.sse)
			model := NewChatModel(
				modelconfig.WithBaseURL(server.URL),
				modelconfig.WithModel("gpt-test"),
			)
			handler := &failOnNthMatch{match: onKind(callbacks.EventChatModelEnd), n: 1}
			stream, err := model.Stream(
				context.Background(),
				[]messages.Message{messages.Human("hi")},
				runnables.WithCallbacks(callbacks.NewManager(handler)),
			)
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			defer stream.Close()

			_, _, err = stream.Next(context.Background())
			if err == nil || !strings.Contains(err.Error(), "callback boom") {
				t.Fatalf("expected callback boom, got %v", err)
			}
		})
	}
}

func TestCloneMetadataNil(t *testing.T) {
	if got := cloneMetadata(nil); got != nil {
		t.Fatalf("cloneMetadata(nil) = %#v, want nil", got)
	}
}

func TestContentBlockFromOutputItemDefaultsType(t *testing.T) {
	// An item whose raw payload lacks a "type" key gets item.Type filled in.
	block := contentBlockFromOutputItem(outputItem{Type: "custom_type"}, 3)
	m := messages.BlockToMap(block)
	if m["type"] != "custom_type" {
		t.Fatalf("type = %v, want custom_type", m["type"])
	}
	if m["index"] != 3 {
		t.Fatalf("index = %v, want 3", m["index"])
	}
}
