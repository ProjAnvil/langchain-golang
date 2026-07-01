package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/streamevents"
)

func TestChatModelStreamTextDeltas(t *testing.T) {
	var got chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"hel"},"done":false}`+"\n")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t2","message":{"role":"assistant","content":"lo"},"done":false}`+"\n")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":2,"eval_count":2}`+"\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("llama3"),
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

	if !got.Stream {
		t.Fatal("expected streaming request")
	}
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
	if content != "hello" {
		t.Fatalf("content: got %q want %q", content, "hello")
	}

	events := filterEvents(recorder.Events(), callbacks.EventChatModelStart, callbacks.EventChatModelStream, callbacks.EventChatModelEnd)
	if len(events) != 4 {
		t.Fatalf("events: got %d want 4: %+v", len(events), events)
	}
	if events[0].Kind != callbacks.EventChatModelStart ||
		events[1].Kind != callbacks.EventChatModelStream ||
		events[2].Kind != callbacks.EventChatModelStream ||
		events[3].Kind != callbacks.EventChatModelEnd {
		t.Fatalf("event kinds: %+v", events)
	}
}

func TestChatModelStreamProtocolEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"hel"},"done":false}`+"\n")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t2","message":{"role":"assistant","content":"lo"},"done":false}`+"\n")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":2,"eval_count":2}`+"\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("llama3"),
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

	events := filterEvents(recorder.Events(), callbacks.EventChatModelProtocol)
	if len(events) != 6 {
		t.Fatalf("protocol events: got %d want 6: %+v", len(events), events)
	}
	want := []streamevents.EventName{
		streamevents.EventMessageStart,
		streamevents.EventContentBlockStart,
		streamevents.EventContentBlockDelta,
		streamevents.EventContentBlockDelta,
		streamevents.EventContentBlockFinish,
		streamevents.EventMessageFinish,
	}
	for i := range want {
		got, ok := events[i].Chunk.(streamevents.Event)
		if !ok {
			t.Fatalf("event[%d] chunk type: %T", i, events[i].Chunk)
		}
		if got.Event != want[i] {
			t.Fatalf("event[%d]: got %q want %q", i, got.Event, want[i])
		}
	}
	finish := events[4].Chunk.(streamevents.Event)
	if finish.Content["text"] != "hello" {
		t.Fatalf("finish content: %+v", finish.Content)
	}
	end := filterEvents(recorder.Events(), callbacks.EventChatModelEnd)
	if len(end) != 1 {
		t.Fatalf("end events: got %d want 1", len(end))
	}
	output, ok := end[0].Output.(messages.Message)
	if !ok {
		t.Fatalf("end output type: %T", end[0].Output)
	}
	if output.UsageMetadata.TotalTokens != 4 {
		t.Fatalf("usage: %+v", output.UsageMetadata)
	}
}

func filterEvents(events []callbacks.Event, kinds ...callbacks.EventKind) []callbacks.Event {
	want := map[callbacks.EventKind]bool{}
	for _, kind := range kinds {
		want[kind] = true
	}
	var out []callbacks.Event
	for _, event := range events {
		if want[event.Kind] {
			out = append(out, event)
		}
	}
	return out
}

func TestChatModelStreamMultipleToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"add","arguments":{"a":1,"b":2}}},{"function":{"name":"mul","arguments":{"x":3,"y":4}}}]},"done":true,"done_reason":"stop"}`+"\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("compute")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var last messages.Message
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		last = chunk
	}

	if len(last.ToolCalls) != 2 {
		t.Fatalf("tool calls: got %d want 2: %+v", len(last.ToolCalls), last.ToolCalls)
	}
	names := []string{last.ToolCalls[0].Name, last.ToolCalls[1].Name}
	if names[0] != "add" || names[1] != "mul" {
		t.Fatalf("tool call order: %+v", names)
	}

	var finishes []streamevents.Event
	for _, event := range filterEvents(recorder.Events(), callbacks.EventChatModelProtocol) {
		protocolEvent, ok := event.Chunk.(streamevents.Event)
		if ok && protocolEvent.Event == streamevents.EventContentBlockFinish {
			finishes = append(finishes, protocolEvent)
		}
	}
	if len(finishes) != 2 {
		t.Fatalf("tool finish events: got %d want 2: %+v", len(finishes), finishes)
	}
	if finishes[0].Content["name"] != "add" || finishes[1].Content["name"] != "mul" {
		t.Fatalf("finish names: %+v %+v", finishes[0].Content, finishes[1].Content)
	}
	if finishes[0].Index == finishes[1].Index {
		t.Fatalf("tool finish indices collide: %d", finishes[0].Index)
	}
}

func TestChatModelStreamToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"add","arguments":{"a":2,"b":3}}}]},"done":true,"done_reason":"stop","prompt_eval_count":2,"eval_count":1}`+"\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("llama3"),
	)
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("add")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var last messages.Message
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		last = chunk
	}

	if len(last.ToolCalls) != 1 {
		t.Fatalf("tool calls: got %d want 1: %+v", len(last.ToolCalls), last.ToolCalls)
	}
	call := last.ToolCalls[0]
	if call.Name != "add" || call.ID == "" {
		t.Fatalf("tool call identity: %+v", call)
	}
	if call.Args["a"].(float64) != 2 || call.Args["b"].(float64) != 3 {
		t.Fatalf("tool call args: %+v", call.Args)
	}

	protocol := filterEvents(recorder.Events(), callbacks.EventChatModelProtocol)
	names := make([]streamevents.EventName, 0, len(protocol))
	for _, event := range protocol {
		protocolEvent, ok := event.Chunk.(streamevents.Event)
		if !ok {
			t.Fatalf("protocol chunk type: %T", event.Chunk)
		}
		names = append(names, protocolEvent.Event)
	}
	want := []streamevents.EventName{
		streamevents.EventMessageStart,
		streamevents.EventContentBlockStart,
		streamevents.EventContentBlockFinish,
		streamevents.EventMessageFinish,
	}
	if len(names) != len(want) {
		t.Fatalf("protocol events: got %v want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Fatalf("protocol event[%d]: got %q want %q", i, names[i], want[i])
		}
	}
	end := filterEvents(recorder.Events(), callbacks.EventChatModelEnd)
	if len(end) != 1 {
		t.Fatalf("end events: got %d want 1", len(end))
	}
	output, ok := end[0].Output.(messages.Message)
	if !ok || len(output.ToolCalls) != 1 {
		t.Fatalf("end output: %+v", end[0].Output)
	}
}

func TestChatModelStreamReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprint(w, `{"model":"deepseek-r1","created_at":"t1","message":{"role":"assistant","content":"","thinking":"counting"},"done":false}`+"\n")
		_, _ = fmt.Fprint(w, `{"model":"deepseek-r1","created_at":"t2","message":{"role":"assistant","content":"three","thinking":"...rs"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`+"\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("deepseek-r1"),
		WithReasoning(true),
	)
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("how many r")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var content string
	var reasoning string
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
			if r, ok := block["reasoning"].(string); ok {
				reasoning += r
			}
		}
	}

	if content != "three" {
		t.Fatalf("content: got %q want %q", content, "three")
	}
	if reasoning != "counting...rs" {
		t.Fatalf("reasoning: got %q want %q", reasoning, "counting...rs")
	}

	var reasoningFinish streamevents.Event
	for _, event := range filterEvents(recorder.Events(), callbacks.EventChatModelProtocol) {
		protocolEvent, ok := event.Chunk.(streamevents.Event)
		if ok && protocolEvent.Event == streamevents.EventContentBlockFinish && protocolEvent.Index == reasoningBlockIndex {
			reasoningFinish = protocolEvent
		}
	}
	if reasoningFinish.Event == "" {
		t.Fatal("missing reasoning finish event")
	}
	if reasoningFinish.Content["reasoning"] != "counting...rs" {
		t.Fatalf("reasoning finish: %+v", reasoningFinish.Content)
	}
}

func TestChatModelStreamSkipsLoadOnlyChunk(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":""},"done":true,"done_reason":"load"}`+"\n")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t2","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`+"\n")
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
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
	if content != "hi" {
		t.Fatalf("content: got %q want %q", content, "hi")
	}
}

func TestChatModelStreamHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"model loading"}`))
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithMaxRetries(0),
	)
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err == nil {
		stream.Close()
		t.Fatal("expected stream error")
	}

	events := recorder.Events()
	if len(events) != 2 {
		t.Fatalf("events: got %d want 2: %+v", len(events), events)
	}
	if events[0].Kind != callbacks.EventChatModelStart || events[1].Kind != callbacks.EventChatModelError {
		t.Fatalf("event kinds: %+v", events)
	}
}

func TestChatModelStreamMalformedChunk(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"ok"},"done":false}`+"\n")
		_, _ = fmt.Fprint(w, `{not valid json}` + "\n")
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

	var streamErr error
	for {
		_, ok, err := stream.Next(context.Background())
		if err != nil {
			streamErr = err
			break
		}
		if !ok {
			break
		}
	}
	if streamErr == nil {
		t.Fatal("expected stream error on malformed chunk")
	}
	events := recorder.Events()
	if len(events) == 0 || events[len(events)-1].Kind != callbacks.EventChatModelError {
		t.Fatalf("error event: %+v", events)
	}
}
