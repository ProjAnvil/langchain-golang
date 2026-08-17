package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
)

const responsesAPIBody = `{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`

func TestChatModelBatch(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(responsesAPIBody))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	outputs, err := model.Batch(context.Background(), [][]messages.Message{
		{messages.Human("one")},
		{messages.Human("two")},
	})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if len(outputs) != 2 || outputs[0].Content != "ok" || outputs[1].Content != "ok" {
		t.Fatalf("outputs = %#v", outputs)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("server calls = %d, want 2", got)
	}
}

func TestChatModelSchemasAndLLMType(t *testing.T) {
	model := NewChatModel(modelconfig.WithModel("gpt-test"))
	if model.LLMType() != "openai-chat" {
		t.Fatalf("LLMType = %q, want openai-chat", model.LLMType())
	}
	out := model.OutputSchema()
	if out["type"] != "object" {
		t.Fatalf("OutputSchema = %#v", out)
	}
	in := model.InputSchema()
	if in["type"] != "array" {
		t.Fatalf("InputSchema = %#v", in)
	}
}

func TestChatModelEmptyResponseIsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Gateways sometimes wrap errors in HTTP 200 bodies that decode into an
		// all-zero payload; Invoke must surface that loudly.
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "response parsed but empty") {
		t.Fatalf("expected empty-response error, got %v", err)
	}
}

func TestChatModelInvokeErrorEmitsErrorCallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadGateway)
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
		modelconfig.WithMaxRetries(0),
	)
	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err == nil {
		t.Fatal("expected invoke error")
	}
	events := recorder.Events()
	if len(events) != 2 {
		t.Fatalf("events = %d, want 2 (start + error)", len(events))
	}
	if events[0].Kind != callbacks.EventChatModelStart || events[1].Kind != callbacks.EventChatModelError {
		t.Fatalf("events = %+v", events)
	}
	if events[1].Error == "" {
		t.Fatal("error event must carry the error message")
	}
}

func TestChatModelStreamErrorEmitsErrorCallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected stream construction error")
	}
	events := recorder.Events()
	if len(events) != 2 || events[1].Kind != callbacks.EventChatModelError {
		t.Fatalf("events = %+v", events)
	}
}

func TestChatModelCallbackMetadataPropagated(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(responsesAPIBody))
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
		runnables.WithMetadata("request_id", "req-1"),
	)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	events := recorder.Events()
	if len(events) == 0 {
		t.Fatal("expected events")
	}
	if events[0].Metadata["request_id"] != "req-1" {
		t.Fatalf("metadata = %#v", events[0].Metadata)
	}
}

func TestOutputItemUnmarshalJSONError(t *testing.T) {
	// Valid JSON that cannot decode into an outputItem exercises the decode
	// error branch (note: json.Unmarshal rejects malformed JSON itself before
	// UnmarshalJSON runs, so the input must be well-formed).
	var item outputItem
	if err := json.Unmarshal([]byte(`"just a string"`), &item); err == nil {
		t.Fatal("expected unmarshal error")
	}
}

func TestResponseStreamNextCanceledContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_text.delta","delta":"hi"}`+"\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := stream.Next(ctx); err == nil {
		t.Fatal("expected context error")
	}
}

func TestResponseStreamMalformedEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {bad json}\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	_, _, err = stream.Next(context.Background())
	if err == nil || !strings.Contains(err.Error(), "decode openai stream event") {
		t.Fatalf("expected decode error, got %v", err)
	}
}

func TestResponseStreamRefusal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			`data: {"type":"response.refusal.done","refusal":"cannot help with that"}`+"\n\n"+
				`data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-test","output":[],"usage":{}}}`+"\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok {
		t.Fatalf("Next: ok=%v err=%v", ok, err)
	}
	if len(chunk.ContentBlocks) != 1 {
		t.Fatalf("content blocks = %#v", chunk.ContentBlocks)
	}
	block := messages.BlockToMap(chunk.ContentBlocks[0])
	if block["type"] != "refusal" || block["refusal"] != "cannot help with that" {
		t.Fatalf("refusal block = %#v", block)
	}
}

// TestResponseStreamIgnoredEvents exercises the non-chunk branches: empty text
// deltas, unknown event types, and non-protocol output items are skipped.
func TestResponseStreamIgnoredEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			`data: {"type":"response.output_text.delta","delta":""}`+"\n\n"+
				`data: {"type":"response.some_future_event","foo":1}`+"\n\n"+
				`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant"}}`+"\n\n"+
				`data: {"type":"response.output_text.delta","delta":"real"}`+"\n\n"+
				`data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-test","output":[],"usage":{}}}`+"\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	var chunks []messages.Message
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 1 || chunks[0].Content != "real" {
		t.Fatalf("chunks = %#v, want a single 'real' chunk", chunks)
	}
}

func TestResponseStreamFailedEventWithoutMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"response.failed"}`+"\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	_, _, err = stream.Next(context.Background())
	if err == nil || err.Error() != "openai stream error" {
		t.Fatalf("expected generic stream error, got %v", err)
	}
}

// TestResponseStreamCompletedWithoutResponse covers response.completed events
// that carry no response payload while a text block is still open.
func TestResponseStreamCompletedWithoutResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			`data: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}`+"\n\n"+
				`data: {"type":"response.completed"}`+"\n\n",
		)
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk.Content != "hi" {
		t.Fatalf("chunk = %+v ok=%v err=%v", chunk, ok, err)
	}
	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected stream end, ok=%v err=%v", ok, err)
	}
	// The still-open text block must be finished before the message finish.
	endEvents := filterEvents(recorder.Events(), callbacks.EventChatModelEnd)
	if len(endEvents) != 1 {
		t.Fatalf("end events = %+v", endEvents)
	}
}

// TestResponseStreamFunctionCallWithoutArguments covers the empty-arguments
// path of streamToolCall.toToolCall.
func TestResponseStreamFunctionCallWithoutArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"noop","arguments":""}}`+"\n\n"+
				`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"noop","arguments":""}}`+"\n\n"+
				`data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-test","output":[],"usage":{}}}`+"\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	var last messages.Message
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		last = chunk
	}
	if len(last.ToolCalls) != 1 {
		t.Fatalf("tool calls = %#v", last.ToolCalls)
	}
	if last.ToolCalls[0].ID != "call_1" || last.ToolCalls[0].Name != "noop" || last.ToolCalls[0].Args != nil {
		t.Fatalf("tool call = %#v", last.ToolCalls[0])
	}
}

func TestChatCompletionsStreamNextCanceledContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := stream.Next(ctx); err == nil {
		t.Fatal("expected context error")
	}
}

func TestChatCompletionsStreamMalformedEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {bad json}\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	_, _, err = stream.Next(context.Background())
	if err == nil || !strings.Contains(err.Error(), "decode openai chat completions stream event") {
		t.Fatalf("expected decode error, got %v", err)
	}
}

func TestChatCompletionsStreamInvalidToolCallArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{bad json}\"}}]}}]}\n\n"+
				"data: [DONE]\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok {
		t.Fatalf("Next: ok=%v err=%v", ok, err)
	}
	if len(chunk.ToolCalls) != 0 || len(chunk.InvalidToolCalls) != 1 {
		t.Fatalf("tool calls = %#v invalid = %#v", chunk.ToolCalls, chunk.InvalidToolCalls)
	}
	if chunk.InvalidToolCalls[0].ID != "call_1" || chunk.InvalidToolCalls[0].Name != "search" {
		t.Fatalf("invalid tool call = %#v", chunk.InvalidToolCalls[0])
	}
}

// TestChatCompletionsStreamEOFWithoutDone covers the scanner-EOF path: the
// server closes the connection without a terminal [DONE] event, and non-data
// SSE lines (comments) and choice-less events are skipped along the way.
func TestChatCompletionsStreamEOFWithoutDone(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			": keep-alive comment\n\n"+
				"data: {\"choices\":[]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk.Content != "hi" {
		t.Fatalf("chunk = %+v ok=%v err=%v", chunk, ok, err)
	}
	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected stream end after EOF, ok=%v err=%v", ok, err)
	}
	// A subsequent Next stays done.
	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected stream to stay done, ok=%v err=%v", ok, err)
	}
}

func TestChatCompletionsStreamNon2xx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected error for 400 response")
	}
}

func TestChatCompletionsStreamBadBaseURL(t *testing.T) {
	model := NewChatModel(
		modelconfig.WithBaseURL("://bad-url"),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected request construction error")
	}
}

// TestChatCompletionsRequestSamplingParamsAndToolCalls locks the request
// mapping for temperature/max_tokens and for assistant messages that carry
// tool calls (the multi-turn function-calling shape).
func TestChatCompletionsRequestSamplingParamsAndToolCalls(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
		modelconfig.WithTemperature(0.7),
		modelconfig.WithMaxTokens(64),
	).WithChatCompletions()

	ai := messages.AI("let me search")
	ai.ToolCalls = []messages.ToolCall{{ID: "call_1", Name: "search", Args: map[string]any{"q": "hi"}}}
	_, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("search please"),
		ai,
		messages.Tool("call_1", "result"),
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	if gotBody["temperature"] != 0.7 {
		t.Fatalf("temperature = %v, want 0.7", gotBody["temperature"])
	}
	if gotBody["max_tokens"] != float64(64) {
		t.Fatalf("max_tokens = %v, want 64", gotBody["max_tokens"])
	}
	msgs, ok := gotBody["messages"].([]any)
	if !ok || len(msgs) != 3 {
		t.Fatalf("messages = %v", gotBody["messages"])
	}
	assistant, _ := msgs[1].(map[string]any)
	toolCalls, ok := assistant["tool_calls"].([]any)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("assistant tool_calls = %v", assistant["tool_calls"])
	}
	call, _ := toolCalls[0].(map[string]any)
	if call["id"] != "call_1" || call["type"] != "function" {
		t.Fatalf("tool call = %v", call)
	}
	fn, _ := call["function"].(map[string]any)
	if fn["name"] != "search" || fn["arguments"] != `{"q":"hi"}` {
		t.Fatalf("tool call function = %v", fn)
	}
}
