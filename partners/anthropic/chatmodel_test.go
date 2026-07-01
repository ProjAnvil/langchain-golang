package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/streamevents"
	"github.com/projanvil/langchain-golang/core/tools"
)

func TestChatModelInvokeMessagesAPI(t *testing.T) {
	var got requestPayload
	var apiKey string
	var version string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiKey = r.Header.Get("x-api-key")
		version = r.Header.Get("anthropic-version")
		if r.URL.Path != "/messages" {
			t.Fatalf("path: got %q", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = fmt.Fprint(w, `{
			"id":"msg_123",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"stop_reason":"end_turn",
			"content":[{"type":"text","text":"Hello from Claude"}],
			"usage":{"input_tokens":2,"output_tokens":3}
		}`)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
		modelconfig.WithAPIKey("test-key"),
		modelconfig.WithMaxTokens(17),
	)
	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.System("be concise"),
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if apiKey != "test-key" || version != "2023-06-01" {
		t.Fatalf("headers: key=%q version=%q", apiKey, version)
	}
	if got.Model != "claude-test" || got.MaxTokens != 17 || got.System != "be concise" {
		t.Fatalf("request: %+v", got)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" || got.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("messages: %+v", got.Messages)
	}
	if response.Content != "Hello from Claude" || response.ID != "msg_123" {
		t.Fatalf("response: %+v", response)
	}
	if response.UsageMetadata.InputTokens != 2 || response.UsageMetadata.TotalTokens != 5 {
		t.Fatalf("usage: %+v", response.UsageMetadata)
	}
}

func TestChatModelInvokeToolUse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `{
			"id":"msg_tools",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"stop_reason":"tool_use",
			"content":[
				{"type":"text","text":"I'll search."},
				{"type":"tool_use","id":"toolu_123","name":"search","input":{"q":"weather"}}
			],
			"usage":{"input_tokens":4,"output_tokens":5}
		}`)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
	)
	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("weather"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(response.ToolCalls) != 1 {
		t.Fatalf("tool calls: %+v", response.ToolCalls)
	}
	call := response.ToolCalls[0]
	if call.ID != "toolu_123" || call.Name != "search" || call.Args["q"] != "weather" {
		t.Fatalf("tool call: %+v", call)
	}
	if response.ResponseMetadata["stop_reason"] != "tool_use" {
		t.Fatalf("metadata: %+v", response.ResponseMetadata)
	}
}

func TestChatModelBindToolsRequest(t *testing.T) {
	var got requestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = fmt.Fprint(w, `{
			"id":"msg_123",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"content":[{"type":"text","text":"ok"}],
			"usage":{}
		}`)
	}))
	defer server.Close()

	tool, err := tools.NewFunc(
		"search",
		"searches",
		schema.Object(map[string]schema.Schema{
			"q": schema.String("query"),
		}, "q"),
		func(_ context.Context, _ map[string]any) (tools.Result, error) {
			return tools.Result{Content: "ok"}, nil
		},
	)
	if err != nil {
		t.Fatalf("tool: %v", err)
	}
	base := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
	)
	bound, err := base.BindTools([]tools.Tool{tool})
	if err != nil {
		t.Fatalf("bind tools: %v", err)
	}
	_, err = bound.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(got.Tools) != 1 || got.Tools[0].Name != "search" || got.Tools[0].InputSchema["type"] != "object" {
		t.Fatalf("tools: %+v", got.Tools)
	}
}

func TestChatModelStreamTextProtocol(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: message_start\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":2,"output_tokens":0}}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_stop\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: message_delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: message_stop\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
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
		t.Fatalf("content: %q", content)
	}

	events := filterEvents(recorder.Events(), callbacks.EventChatModelProtocol)
	if len(events) != 6 {
		t.Fatalf("protocol events: got %d want 6: %+v", len(events), events)
	}
	finish := events[4].Chunk.(streamevents.Event)
	if finish.Event != streamevents.EventContentBlockFinish || finish.Content["text"] != "hello" {
		t.Fatalf("finish: %+v", finish)
	}
}

func TestChatModelStreamEventsProjection(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: message_start\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":2,"output_tokens":0}}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_stop\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: message_stop\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
	)
	stream, err := language.StreamEvents(context.Background(), model, []messages.Message{
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	if stream.Text() != "hi" {
		t.Fatalf("text: %q", stream.Text())
	}
	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if output.Content != "hi" || output.ID != "msg_123" {
		t.Fatalf("output: %+v", output)
	}
}

func TestChatModelCapabilities(t *testing.T) {
	caps := NewChatModel().Capabilities()
	if !caps.ToolCalling || !caps.Streaming || !caps.UsageMetadata {
		t.Fatalf("capabilities: %+v", caps)
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
