package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/structuredoutput"
	"github.com/projanvil/langchain-golang/core/tools"
)

func TestChatModelInvokeNonStreaming(t *testing.T) {
	var got chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Fatalf("path: got %q want /api/chat", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("method: got %q", r.Method)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Fatalf("content-type: got %q", ct)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"model":"llama3",
			"created_at":"2024-01-01T00:00:00Z",
			"message":{"role":"assistant","content":"Hello from Ollama"},
			"done":true,
			"done_reason":"stop",
			"prompt_eval_count":5,
			"eval_count":3
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("llama3"),
	)

	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.System("Be concise"),
		messages.Human("Say hello"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Model != "llama3" {
		t.Fatalf("model: got %q", got.Model)
	}
	if got.Stream {
		t.Fatal("expected non-streaming request")
	}
	if len(got.Messages) != 2 {
		t.Fatalf("messages: got %d want 2", len(got.Messages))
	}
	if got.Messages[0].Role != "system" || got.Messages[0].Content != "Be concise" {
		t.Fatalf("system message: %+v", got.Messages[0])
	}
	if got.Messages[1].Role != "user" || got.Messages[1].Content != "Say hello" {
		t.Fatalf("user message: %+v", got.Messages[1])
	}
	if response.Content != "Hello from Ollama" {
		t.Fatalf("content: got %q", response.Content)
	}
	if response.UsageMetadata.InputTokens != 5 {
		t.Fatalf("input tokens: got %d want 5", response.UsageMetadata.InputTokens)
	}
	if response.UsageMetadata.OutputTokens != 3 {
		t.Fatalf("output tokens: got %d want 3", response.UsageMetadata.OutputTokens)
	}
	if response.UsageMetadata.TotalTokens != 8 {
		t.Fatalf("total tokens: got %d want 8", response.UsageMetadata.TotalTokens)
	}
	if response.ResponseMetadata["model"] != "llama3" {
		t.Fatalf("metadata model: %+v", response.ResponseMetadata)
	}
}

func TestChatModelRequestMapping(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("llama3"),
		modelconfig.WithHeader("X-Custom", "value"),
	)
	_, err := model.Invoke(context.Background(), []messages.Message{
		messages.System("first instruction"),
		messages.System("second instruction"),
		messages.Human("hello"),
		messages.AI("hi"),
		messages.Tool("call_1", "tool result"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(got.Messages) != 5 {
		t.Fatalf("messages: got %d want 5: %+v", len(got.Messages), got.Messages)
	}
	wantRoles := []string{"system", "system", "user", "assistant", "tool"}
	wantContent := []string{"first instruction", "second instruction", "hello", "hi", "tool result"}
	for i := range wantRoles {
		if got.Messages[i].Role != wantRoles[i] || got.Messages[i].Content != wantContent[i] {
			t.Fatalf("message[%d]: got role=%q content=%q want role=%q content=%q",
				i, got.Messages[i].Role, got.Messages[i].Content, wantRoles[i], wantContent[i])
		}
	}
	if got.Messages[4].ToolCallID != "call_1" {
		t.Fatalf("tool_call_id: got %q", got.Messages[4].ToolCallID)
	}
	if got.Options != nil {
		t.Fatalf("expected no options when none configured: %+v", got.Options)
	}
}

func TestChatModelMapsAIToolCallsIntoRequest(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	ai := messages.AI("I will call a tool")
	ai.ToolCalls = []messages.ToolCall{{
		ID:   "call_abc",
		Name: "search",
		Args: map[string]any{"query": "weather"},
	}}
	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	_, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("use a tool"),
		ai,
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(got.Messages[1].ToolCalls) != 1 {
		t.Fatalf("ai tool calls: %+v", got.Messages[1].ToolCalls)
	}
	tc := got.Messages[1].ToolCalls[0]
	if tc.Type != "function" || tc.ID != "call_abc" || tc.Function.Name != "search" {
		t.Fatalf("tool call shape: %+v", tc)
	}
	if tc.Function.Arguments.(map[string]any)["query"] != "weather" {
		t.Fatalf("tool call arguments: %+v", tc.Function.Arguments)
	}
}

func TestChatModelMapsImagesFromContentBlocks(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	msg := messages.Human("")
	msg.ContentBlocks = []messages.ContentBlock{
		{"type": "text", "text": "describe this"},
		{"type": "image", "base64": "data:image/png;base64,aGVsbG8="},
		{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,c3Rhcg=="}},
	}
	_, err := model.Invoke(context.Background(), []messages.Message{msg})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(got.Messages) != 1 {
		t.Fatalf("messages: got %d want 1", len(got.Messages))
	}
	if got.Messages[0].Content != "describe this" {
		t.Fatalf("content: got %q", got.Messages[0].Content)
	}
	if len(got.Messages[0].Images) != 2 {
		t.Fatalf("images: got %d want 2: %+v", len(got.Messages[0].Images), got.Messages[0].Images)
	}
	if got.Messages[0].Images[0] != "aGVsbG8=" || got.Messages[0].Images[1] != "c3Rhcg==" {
		t.Fatalf("image payloads: %+v", got.Messages[0].Images)
	}
}

func TestChatModelTemperatureAndMaxTokensOptions(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{"model":"llama3","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithTemperature(0.5),
		modelconfig.WithMaxTokens(42),
	)
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Options["temperature"] != 0.5 {
		t.Fatalf("temperature option: got %v", got.Options["temperature"])
	}
	if optionNumber(t, got.Options, "num_predict") != 42 {
		t.Fatalf("num_predict option: got %v", got.Options["num_predict"])
	}
}

func TestChatModelSamplingOptions(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{"model":"llama3","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		WithTopP(0.9),
		WithTopK(40),
		WithNumCtx(2048),
		WithSeed(7),
		WithStop([]string{"END"}),
		WithKeepAlive("5m"),
	)
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Options["top_p"] != 0.9 {
		t.Fatalf("top_p: got %v", got.Options["top_p"])
	}
	if optionNumber(t, got.Options, "top_k") != 40 {
		t.Fatalf("top_k: got %v", got.Options["top_k"])
	}
	if optionNumber(t, got.Options, "num_ctx") != 2048 {
		t.Fatalf("num_ctx: got %v", got.Options["num_ctx"])
	}
	if optionNumber(t, got.Options, "seed") != 7 {
		t.Fatalf("seed: got %v", got.Options["seed"])
	}
	if stop, _ := got.Options["stop"].([]any); len(stop) != 1 || stop[0] != "END" {
		t.Fatalf("stop: got %v", got.Options["stop"])
	}
	if got.KeepAlive != "5m" {
		t.Fatalf("keep_alive: got %v", got.KeepAlive)
	}
}

func TestChatModelParsesToolCallsWithDictArguments(t *testing.T) {
	server := newChatServer(t, nil, `{
		"model":"llama3",
		"message":{"role":"assistant","content":"","tool_calls":[
			{"function":{"name":"add","arguments":{"a":2,"b":3}}}
		]},
		"done":true,"done_reason":"stop"
	}`)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	response, err := model.Invoke(context.Background(), []messages.Message{messages.Human("add")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(response.ToolCalls) != 1 {
		t.Fatalf("tool calls: got %d want 1: %+v", len(response.ToolCalls), response.ToolCalls)
	}
	call := response.ToolCalls[0]
	if call.Name != "add" || call.ID == "" {
		t.Fatalf("tool call identity: %+v", call)
	}
	if call.Args["a"].(float64) != 2 || call.Args["b"].(float64) != 3 {
		t.Fatalf("tool call args: %+v", call.Args)
	}
}

func TestChatModelParsesToolCallsWithStringArguments(t *testing.T) {
	server := newChatServer(t, nil, `{
		"model":"llama3",
		"message":{"role":"assistant","content":"","tool_calls":[
			{"function":{"name":"add","arguments":"{\"a\":2,\"b\":3}"}}
		]},
		"done":true,"done_reason":"stop"
	}`)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	response, err := model.Invoke(context.Background(), []messages.Message{messages.Human("add")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(response.ToolCalls) != 1 || response.ToolCalls[0].Args["a"].(float64) != 2 {
		t.Fatalf("tool calls: %+v", response.ToolCalls)
	}
}

func TestChatModelParsesInvalidToolCallArguments(t *testing.T) {
	server := newChatServer(t, nil, `{
		"model":"llama3",
		"message":{"role":"assistant","content":"","tool_calls":[
			{"function":{"name":"add","arguments":"{bad json}"}}
		]},
		"done":true,"done_reason":"stop"
	}`)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	response, err := model.Invoke(context.Background(), []messages.Message{messages.Human("add")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(response.ToolCalls) != 0 {
		t.Fatalf("tool calls: got %d want 0", len(response.ToolCalls))
	}
	if len(response.InvalidToolCalls) != 1 || response.InvalidToolCalls[0].Name != "add" {
		t.Fatalf("invalid tool calls: %+v", response.InvalidToolCalls)
	}
}

func TestChatModelBindTools(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	adder, err := tools.NewFunc(
		"add",
		"adds integers",
		schema.Object(map[string]schema.Schema{
			"a": schema.Integer("left"),
			"b": schema.Integer("right"),
		}, "a", "b"),
		func(_ context.Context, _ map[string]any) (tools.Result, error) {
			return tools.Result{Content: "3"}, nil
		},
	)
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	bound, err := model.BindTools([]tools.Tool{adder})
	if err != nil {
		t.Fatalf("bind tools: %v", err)
	}
	_, err = bound.Invoke(context.Background(), []messages.Message{messages.Human("add")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(got.Tools) != 1 {
		t.Fatalf("tools: got %d want 1", len(got.Tools))
	}
	if got.Tools[0].Type != "function" || got.Tools[0].Function.Name != "add" {
		t.Fatalf("tool spec: %+v", got.Tools[0])
	}
	if got.Tools[0].Function.Parameters["type"] != "object" {
		t.Fatalf("tool parameters: %+v", got.Tools[0].Function.Parameters)
	}
}

func TestChatModelStructuredOutput(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{"model":"llama3","message":{"role":"assistant","content":"{\"name\":\"Ada\"}"},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL)).WithStructuredOutput(
		"person",
		schema.Object(map[string]schema.Schema{
			"name": schema.String("person name"),
		}, "name"),
		true,
	)
	response, err := model.Invoke(context.Background(), []messages.Message{messages.Human("extract")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	format, ok := got.Format.(map[string]any)
	if !ok {
		t.Fatalf("format: got %T want map", got.Format)
	}
	if format["type"] != "object" {
		t.Fatalf("format type: got %v", format["type"])
	}
	if response.Content != `{"name":"Ada"}` {
		t.Fatalf("content: got %q", response.Content)
	}
}

func TestChatModelTypedStructuredOutput(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{"model":"llama3","message":{"role":"assistant","content":"{\"name\":\"Ada\",\"age\":37}"},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	runnable, err := structuredoutput.BindJSON[ChatModel, ollamaPerson](
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

	result, err := runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if result.Name != "Ada" || result.Age != 37 {
		t.Fatalf("typed result: %+v", result)
	}
	if _, ok := got.Format.(map[string]any); !ok {
		t.Fatalf("format not set: %+v", got.Format)
	}
}

type ollamaPerson struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestChatModelReasoning(t *testing.T) {
	var got chatRequest
	server := newChatServer(t, &got, `{
		"model":"deepseek-r1",
		"message":{"role":"assistant","content":"three","thinking":"let me count..."},
		"done":true,"done_reason":"stop"
	}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("deepseek-r1"),
		WithReasoning(true),
	)
	response, err := model.Invoke(context.Background(), []messages.Message{messages.Human("how many r")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Think != true {
		t.Fatalf("think param: got %v want true", got.Think)
	}
	if response.Content != "three" {
		t.Fatalf("content: got %q", response.Content)
	}
	if response.AdditionalKwargs["reasoning_content"] != "let me count..." {
		t.Fatalf("reasoning content: %+v", response.AdditionalKwargs)
	}
}

func TestChatModelCallbacks(t *testing.T) {
	server := newChatServer(t, nil, `{"model":"llama3","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`)
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	events := recorder.Events()
	if len(events) != 2 {
		t.Fatalf("events: got %d want 2", len(events))
	}
	if events[0].Kind != callbacks.EventChatModelStart || events[1].Kind != callbacks.EventChatModelEnd {
		t.Fatalf("event kinds: %+v", events)
	}
}

func TestChatModelRetriesOnServerError(t *testing.T) {
	var attempts int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte(`{"model":"llama3","message":{"role":"assistant","content":"recovered"},"done":true,"done_reason":"stop"}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithMaxRetries(3),
		modelconfig.WithRetryDelay(0),
	)
	response, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if attempts != 3 {
		t.Fatalf("attempts: got %d want 3", attempts)
	}
	if response.Content != "recovered" {
		t.Fatalf("content: got %q", response.Content)
	}
}

func TestChatModelInvokeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"model not found"}`))
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected error")
	}
}

func newChatServer(t *testing.T, got *chatRequest, response string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("path: got %q want /api/chat", r.URL.Path)
		}
		if got != nil {
			if err := json.NewDecoder(r.Body).Decode(got); err != nil {
				t.Errorf("decode request: %v", err)
			}
		}
		_, _ = w.Write([]byte(response))
	}))
}

// optionNumber reads a numeric Ollama option from the decoded request. JSON
// numbers in a map[string]any decode to float64, so all numeric assertions go
// through this helper rather than comparing against untyped int constants.
func optionNumber(t *testing.T, options map[string]any, key string) float64 {
	t.Helper()
	value, ok := options[key].(float64)
	if !ok {
		t.Fatalf("%s option: unexpected type %T value %v", key, options[key], options[key])
	}
	return value
}
