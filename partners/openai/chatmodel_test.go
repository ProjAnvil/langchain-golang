package openai

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
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/streamevents"
	"github.com/projanvil/langchain-golang/core/structuredoutput"
	"github.com/projanvil/langchain-golang/core/tools"
)

func TestChatModelInvokeResponsesAPI(t *testing.T) {
	var got requestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("path: got %q", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("method: got %q", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("authorization header: got %q", r.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"Hello from OpenAI"}]
			}],
			"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
		modelconfig.WithModel("gpt-test"),
		modelconfig.WithTemperature(0.1),
		modelconfig.WithMaxTokens(32),
	)

	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.System("Be concise"),
		messages.Human("Say hello"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Model != "gpt-test" {
		t.Fatalf("model: got %q", got.Model)
	}
	if got.Instructions != "Be concise" {
		t.Fatalf("instructions: got %q", got.Instructions)
	}
	if len(got.Input) != 1 || got.Input[0].Role != "user" || got.Input[0].Content != "Say hello" {
		t.Fatalf("input: %+v", got.Input)
	}
	if got.Temperature == nil || *got.Temperature != 0.1 {
		t.Fatalf("temperature: %v", got.Temperature)
	}
	if got.MaxOutputTokens == nil || *got.MaxOutputTokens != 32 {
		t.Fatalf("max output tokens: %v", got.MaxOutputTokens)
	}
	if response.Content != "Hello from OpenAI" {
		t.Fatalf("content: got %q", response.Content)
	}
	if response.UsageMetadata.TotalTokens != 5 {
		t.Fatalf("usage: %+v", response.UsageMetadata)
	}
	if response.ResponseMetadata["model"] != "gpt-test" {
		t.Fatalf("metadata: %+v", response.ResponseMetadata)
	}
}

func TestChatModelRequestMapping(t *testing.T) {
	var got requestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Test-Header") != "custom-value" {
			t.Fatalf("custom header: got %q", r.Header.Get("X-Test-Header"))
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
		modelconfig.WithHeader("X-Test-Header", "custom-value"),
	)
	_, err := model.Invoke(context.Background(), []messages.Message{
		messages.System("first instruction"),
		messages.System("second instruction"),
		messages.Human("hello"),
		messages.AI("hi"),
		messages.Tool("call_123", "tool result"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Instructions != "first instruction\nsecond instruction" {
		t.Fatalf("instructions: got %q", got.Instructions)
	}
	if len(got.Input) != 3 {
		t.Fatalf("input length: got %d want 3: %+v", len(got.Input), got.Input)
	}
	wantRoles := []string{"user", "assistant", "tool"}
	wantContent := []string{"hello", "hi", "tool result"}
	for i := range wantRoles {
		if got.Input[i].Role != wantRoles[i] || got.Input[i].Content != wantContent[i] {
			t.Fatalf("input[%d]: got %+v want role=%q content=%q", i, got.Input[i], wantRoles[i], wantContent[i])
		}
	}
}

func TestChatModelParsesMultipleResponseOutputs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"resp_multi",
			"model":"gpt-test",
			"output":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello "}]},
				{"type":"message","role":"user","content":[{"type":"output_text","text":"ignored"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"there"}]},
				{"type":"function_call","call_id":"call_123","name":"lookup","arguments":"{\"query\":\"weather\"}"}
			],
			"usage":{"input_tokens":4,"output_tokens":5,"total_tokens":9}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if response.ID != "resp_multi" {
		t.Fatalf("id: got %q", response.ID)
	}
	if response.Content != "Hello there" {
		t.Fatalf("content: got %q", response.Content)
	}
	if len(response.ToolCalls) != 1 {
		t.Fatalf("tool calls: %+v", response.ToolCalls)
	}
	if response.ToolCalls[0].Args["query"] != "weather" {
		t.Fatalf("tool call args: %+v", response.ToolCalls[0].Args)
	}
	if response.UsageMetadata.InputTokens != 4 ||
		response.UsageMetadata.OutputTokens != 5 ||
		response.UsageMetadata.TotalTokens != 9 {
		t.Fatalf("usage: %+v", response.UsageMetadata)
	}
}

func TestChatModelBindTools(t *testing.T) {
	var got requestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[],
			"usage":{}
		}`))
	}))
	defer server.Close()

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

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	bound, err := model.BindTools([]tools.Tool{adder})
	if err != nil {
		t.Fatalf("bind tools: %v", err)
	}
	_, err = bound.Invoke(context.Background(), []messages.Message{
		messages.Human("use a tool"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(got.Tools) != 1 {
		t.Fatalf("tools: got %d want 1", len(got.Tools))
	}
	if got.Tools[0].Type != "function" || got.Tools[0].Name != "adder" {
		t.Fatalf("tool spec: %+v", got.Tools[0])
	}
}

func TestChatModelCallbacks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"ok"}]
			}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hello")},
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
		t.Fatalf("start event: %+v", events[0])
	}
	if events[1].Kind != callbacks.EventChatModelEnd {
		t.Fatalf("end event: %+v", events[1])
	}
}

func TestChatModelParsesToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[{
				"type":"function_call",
				"call_id":"call_123",
				"name":"adder",
				"arguments":"{\"a\":2,\"b\":3}"
			}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("add two numbers"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(response.ToolCalls) != 1 {
		t.Fatalf("tool calls: got %d want 1", len(response.ToolCalls))
	}
	toolCall := response.ToolCalls[0]
	if toolCall.ID != "call_123" || toolCall.Name != "adder" {
		t.Fatalf("tool call identity: %+v", toolCall)
	}
	if toolCall.Args["a"].(float64) != 2 || toolCall.Args["b"].(float64) != 3 {
		t.Fatalf("tool call args: %+v", toolCall.Args)
	}
}

func TestChatModelInvalidToolCallArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[{
				"type":"function_call",
				"call_id":"call_123",
				"name":"adder",
				"arguments":"{bad json}"
			}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("add two numbers"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(response.ToolCalls) != 0 {
		t.Fatalf("tool calls: got %d want 0", len(response.ToolCalls))
	}
	if len(response.InvalidToolCalls) != 1 {
		t.Fatalf("invalid tool calls: got %d want 1", len(response.InvalidToolCalls))
	}
}

func TestChatModelStructuredOutputRequest(t *testing.T) {
	var got requestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"{\"name\":\"Ada\"}"}]
			}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithStructuredOutput(
		"person",
		schema.Object(map[string]schema.Schema{
			"name": schema.String("person name"),
		}, "name"),
		true,
	)

	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("extract person"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Text == nil {
		t.Fatal("expected text config")
	}
	if got.Text.Format.Type != "json_schema" {
		t.Fatalf("format type: got %q", got.Text.Format.Type)
	}
	if got.Text.Format.Name != "person" {
		t.Fatalf("format name: got %q", got.Text.Format.Name)
	}
	if !got.Text.Format.Strict {
		t.Fatal("expected strict schema")
	}
	if got.Text.Format.Schema["type"] != "object" {
		t.Fatalf("schema: %+v", got.Text.Format.Schema)
	}
	if response.Content != `{"name":"Ada"}` {
		t.Fatalf("content: got %q", response.Content)
	}
}

func TestChatModelTypedStructuredOutput(t *testing.T) {
	var got requestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"{\"name\":\"Ada\",\"age\":37}"}]
			}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	runnable, err := structuredoutput.BindJSON[ChatModel, openAIPerson](
		model,
		"person",
		schema.Object(map[string]schema.Schema{
			"name": schema.String("person name"),
			"age":  schema.Integer("person age"),
		}, "name", "age"),
		true,
	)
	if err != nil {
		t.Fatalf("bind json: %v", err)
	}

	response, err := runnable.Invoke(context.Background(), []messages.Message{
		messages.Human("extract person"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Text == nil {
		t.Fatal("expected text config")
	}
	if got.Text.Format.Type != "json_schema" ||
		got.Text.Format.Name != "person" ||
		!got.Text.Format.Strict {
		t.Fatalf("format: %+v", got.Text.Format)
	}
	if response.Name != "Ada" || response.Age != 37 {
		t.Fatalf("typed response: %+v", response)
	}
}

type openAIPerson struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestChatModelStreamTextDeltas(t *testing.T) {
	var got requestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_text.delta","delta":"hel"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_text.delta","delta":"lo"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.completed","response":{"id":"resp_123","model":"gpt-test","output":[],"usage":{}}}`+"\n\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
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

	if !got.Stream {
		t.Fatal("expected stream request")
	}
	if content != "hello" {
		t.Fatalf("content: got %q", content)
	}
	events := filterEvents(recorder.Events(), callbacks.EventChatModelStart, callbacks.EventChatModelStream, callbacks.EventChatModelEnd)
	if len(events) != 4 {
		t.Fatalf("events: got %d want 4", len(events))
	}
	if events[0].Kind != callbacks.EventChatModelStart ||
		events[1].Kind != callbacks.EventChatModelStream ||
		events[2].Kind != callbacks.EventChatModelStream ||
		events[3].Kind != callbacks.EventChatModelEnd {
		t.Fatalf("events: %+v", events)
	}
}

func TestChatModelStreamEventNameFallbackAndCompletedOutput(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"hi"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"id":"resp_done","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`+"\n\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
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

	chunk, ok, err := stream.Next(context.Background())
	if err != nil {
		t.Fatalf("next delta: %v", err)
	}
	if !ok || chunk.Content != "hi" {
		t.Fatalf("delta chunk: ok=%v chunk=%+v", ok, chunk)
	}
	_, ok, err = stream.Next(context.Background())
	if err != nil {
		t.Fatalf("next completed: %v", err)
	}
	if ok {
		t.Fatal("completed event should not return a stream chunk")
	}

	events := filterEvents(recorder.Events(), callbacks.EventChatModelStart, callbacks.EventChatModelStream, callbacks.EventChatModelEnd)
	if len(events) != 3 {
		t.Fatalf("events: got %d want 3", len(events))
	}
	end := events[2]
	if end.Kind != callbacks.EventChatModelEnd {
		t.Fatalf("end event: %+v", end)
	}
	output, ok := end.Output.(messages.Message)
	if !ok {
		t.Fatalf("end output type: got %T", end.Output)
	}
	if output.Content != "hi" ||
		output.ID != "resp_done" ||
		output.UsageMetadata.TotalTokens != 2 {
		t.Fatalf("end output: %+v", output)
	}
}

func TestChatModelStreamProtocolEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_text.delta","output_index":0,"delta":"hel"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_text.delta","output_index":0,"delta":"lo"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_text.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_text.done","output_index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.completed","response":{"id":"resp_123","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{}}}`+"\n\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
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
}

func TestChatModelStreamFunctionCallDeltas(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_123","type":"function_call","call_id":"call_123","name":"adder","arguments":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"a\":2,"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"b\":3}"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.function_call_arguments.done","output_index":0,"name":"adder","arguments":"{\"a\":2,\"b\":3}"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_123","type":"function_call","call_id":"call_123","name":"adder","arguments":"{\"a\":2,\"b\":3}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.completed","response":{"id":"resp_123","model":"gpt-test","output":[],"usage":{}}}`+"\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{
		messages.Human("add two numbers"),
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var chunks []messages.Message
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

	if len(chunks) != 4 {
		t.Fatalf("chunks: got %d want 4: %+v", len(chunks), chunks)
	}
	if len(chunks[0].ContentBlocks) != 1 {
		t.Fatalf("added chunk blocks: %+v", chunks[0].ContentBlocks)
	}
	if chunks[0].ContentBlocks[0]["name"] != "adder" ||
		chunks[0].ContentBlocks[0]["call_id"] != "call_123" {
		t.Fatalf("added chunk block: %+v", chunks[0].ContentBlocks[0])
	}
	if chunks[1].ContentBlocks[0]["arguments"] != `{"a":2,` ||
		chunks[2].ContentBlocks[0]["arguments"] != `"b":3}` {
		t.Fatalf("argument chunks: %+v %+v", chunks[1], chunks[2])
	}
	if len(chunks[3].ToolCalls) != 1 {
		t.Fatalf("final tool calls: %+v", chunks[3].ToolCalls)
	}
	toolCall := chunks[3].ToolCalls[0]
	if toolCall.ID != "call_123" || toolCall.Name != "adder" {
		t.Fatalf("tool call identity: %+v", toolCall)
	}
	if toolCall.Args["a"].(float64) != 2 || toolCall.Args["b"].(float64) != 3 {
		t.Fatalf("tool call args: %+v", toolCall.Args)
	}
}

func TestChatModelStreamInvalidFunctionCallArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_bad","name":"adder","arguments":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{bad json}"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_bad","name":"adder","arguments":"{bad json}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.completed","response":{"id":"resp_123","model":"gpt-test","output":[],"usage":{}}}`+"\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(context.Background(), []messages.Message{
		messages.Human("add two numbers"),
	})
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

	if len(last.ToolCalls) != 0 {
		t.Fatalf("tool calls: %+v", last.ToolCalls)
	}
	if len(last.InvalidToolCalls) != 1 {
		t.Fatalf("invalid tool calls: %+v", last.InvalidToolCalls)
	}
	if last.InvalidToolCalls[0].ID != "call_bad" || last.InvalidToolCalls[0].Name != "adder" {
		t.Fatalf("invalid tool call identity: %+v", last.InvalidToolCalls[0])
	}
}

func TestChatModelStreamAdditionalResponsesItemsProtocolEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"ws_123","type":"web_search_call","status":"in_progress"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.done","output_index":0,"item":{"id":"ws_123","type":"web_search_call","status":"completed"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fs_123","type":"file_search_call","queries":["python code"]}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.done","output_index":1,"item":{"id":"fs_123","type":"file_search_call","queries":["python code"],"results":[{"filename":"example.py"}]}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.added","output_index":2,"item":{"id":"ci_123","type":"code_interpreter_call","code":"print(1)"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.done","output_index":2,"item":{"id":"ci_123","type":"code_interpreter_call","code":"print(1)","outputs":[{"type":"logs","logs":"1"}]}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.added","output_index":3,"item":{"id":"ct_123","type":"custom_tool_call","call_id":"call_custom","name":"grammar","input":"abc"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.done","output_index":3,"item":{"id":"ct_123","type":"custom_tool_call","call_id":"call_custom","name":"grammar","input":"abc"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.completed","response":{"id":"resp_123","model":"gpt-test","output":[],"usage":{}}}`+"\n\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("use tools")},
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

	var finishes []streamevents.Event
	for _, event := range filterEvents(recorder.Events(), callbacks.EventChatModelProtocol) {
		chunk, ok := event.Chunk.(streamevents.Event)
		if !ok || chunk.Event != streamevents.EventContentBlockFinish {
			continue
		}
		finishes = append(finishes, chunk)
	}
	if len(finishes) != 4 {
		t.Fatalf("finish protocol events: got %d want 4: %+v", len(finishes), finishes)
	}
	wantTypes := []string{"web_search_call", "file_search_call", "code_interpreter_call", "custom_tool_call"}
	for i, want := range wantTypes {
		if got := finishes[i].Content["type"]; got != want {
			t.Fatalf("finish[%d] type: got %v want %q (%+v)", i, got, want, finishes[i].Content)
		}
	}
	if finishes[1].Content["queries"] == nil || finishes[2].Content["outputs"] == nil || finishes[3].Content["input"] != "abc" {
		t.Fatalf("finish payloads not preserved: %+v", finishes)
	}
}

func TestChatModelStreamReasoningFinishProtocolEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_123","type":"reasoning","summary":[]}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"reasoning block"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":" one"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_text.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.reasoning_summary_text.done","output_index":0,"summary_index":0,"text":"reasoning block one"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_part.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.reasoning_summary_part.done","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"reasoning block one"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_123","type":"reasoning","summary":[{"type":"summary_text","text":"reasoning block one"}]}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"type":"response.completed","response":{"id":"resp_123","model":"gpt-test","output":[],"usage":{}}}`+"\n\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("think")},
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

	var finish streamevents.Event
	for _, event := range filterEvents(recorder.Events(), callbacks.EventChatModelProtocol) {
		chunk, ok := event.Chunk.(streamevents.Event)
		if ok && chunk.Event == streamevents.EventContentBlockFinish {
			finish = chunk
		}
	}
	if finish.Event != streamevents.EventContentBlockFinish {
		t.Fatal("missing reasoning finish event")
	}
	if finish.Content["type"] != "reasoning" || finish.Content["reasoning"] != "reasoning block one" {
		t.Fatalf("finish content: %+v", finish.Content)
	}
}

func TestChatModelStreamErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: error\n")
		_, _ = fmt.Fprint(w, `data: {"type":"error","error":{"message":"boom","code":"server_error"}}`+"\n\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
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

	_, _, err = stream.Next(context.Background())
	if err == nil {
		t.Fatal("expected stream error")
	}
	if err.Error() != "openai stream error: boom" {
		t.Fatalf("error: got %q", err.Error())
	}
	events := recorder.Events()
	if len(events) != 2 {
		t.Fatalf("events: got %d want 2", len(events))
	}
	if events[1].Kind != callbacks.EventChatModelError {
		t.Fatalf("error event: %+v", events[1])
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
