package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// newGeminiTestServer spins a fake Gemini API server. The handler runs before
// the canned response is written, with the decoded request body and request
// metadata available for assertions.
func newGeminiTestServer(
	t *testing.T,
	handler func(t *testing.T, r *http.Request, body map[string]any),
	response string,
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
			return
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode body: %v (raw: %s)", err, raw)
			return
		}
		handler(t, r, body)
		_, _ = fmt.Fprint(w, response)
	}))
}

func TestChatModelInvoke(t *testing.T) {
	server := newGeminiTestServer(t, func(t *testing.T, r *http.Request, body map[string]any) {
		if want := "/v1beta/models/gemini-test:generateContent"; r.URL.Path != want {
			t.Errorf("path: got %q want %q", r.URL.Path, want)
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("x-goog-api-key: got %q", got)
		}
		system, _ := body["systemInstruction"].(map[string]any)
		if system == nil {
			t.Fatalf("systemInstruction missing: %v", body)
		}
		sysParts, _ := system["parts"].([]any)
		if len(sysParts) != 1 {
			t.Fatalf("systemInstruction parts: %v", sysParts)
		}
		if part := sysParts[0].(map[string]any); part["text"] != "be concise" {
			t.Errorf("systemInstruction text: %v", part)
		}

		contents, _ := body["contents"].([]any)
		if len(contents) != 1 {
			t.Fatalf("contents: %v", contents)
		}
		content := contents[0].(map[string]any)
		if content["role"] != "user" {
			t.Errorf("content role: %v", content["role"])
		}
		parts, _ := content["parts"].([]any)
		if len(parts) != 1 || parts[0].(map[string]any)["text"] != "hello" {
			t.Errorf("content parts: %v", parts)
		}

		generation, _ := body["generationConfig"].(map[string]any)
		if generation == nil || generation["temperature"] != 0.7 || generation["maxOutputTokens"] != float64(17) {
			t.Errorf("generationConfig: %v", generation)
		}
		if _, has := body["tools"]; has {
			t.Errorf("unexpected tools in request: %v", body["tools"])
		}
	}, `{
		"candidates":[{
			"content":{"role":"model","parts":[{"text":"Hello from Gemini"}]},
			"finishReason":"STOP"
		}],
		"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},
		"modelVersion":"gemini-test",
		"responseId":"resp_123"
	}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
		modelconfig.WithModel("gemini-test"),
		modelconfig.WithTemperature(0.7),
		modelconfig.WithMaxTokens(17),
	)
	response, err := model.Invoke(t.Context(), []messages.Message{
		messages.System("be concise"),
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if response.Content != "Hello from Gemini" {
		t.Errorf("content: %q", response.Content)
	}
	if response.ID != "resp_123" {
		t.Errorf("id: %q", response.ID)
	}
	if response.UsageMetadata.InputTokens != 5 ||
		response.UsageMetadata.OutputTokens != 2 ||
		response.UsageMetadata.TotalTokens != 7 {
		t.Errorf("usage: %+v", response.UsageMetadata)
	}
	if response.ResponseMetadata["finish_reason"] != "STOP" ||
		response.ResponseMetadata["model"] != "gemini-test" ||
		response.ResponseMetadata["model_provider"] != "gemini" {
		t.Errorf("response metadata: %v", response.ResponseMetadata)
	}
}

func TestChatModelDefaultModelAndBaseURL(t *testing.T) {
	server := newGeminiTestServer(t, func(t *testing.T, r *http.Request, _ map[string]any) {
		if want := "/v1beta/models/gemini-2.5-flash:generateContent"; r.URL.Path != want {
			t.Errorf("default model path: got %q want %q", r.URL.Path, want)
		}
	}, `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke with defaults: %v", err)
	}
}

func TestChatModelInvokeToolCall(t *testing.T) {
	server := newGeminiTestServer(t, func(t *testing.T, _ *http.Request, _ map[string]any) {},
		`{
		"candidates":[{
			"content":{"role":"model","parts":[
				{"text":"Let me check."},
				{"functionCall":{"name":"search","args":{"q":"weather"}}}
			]},
			"finishReason":"STOP"
		}],
		"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":5,"totalTokenCount":9}
	}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
		modelconfig.WithModel("gemini-test"),
	)
	response, err := model.Invoke(t.Context(), []messages.Message{messages.Human("weather?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if response.Content != "Let me check." {
		t.Errorf("content: %q", response.Content)
	}
	if len(response.ToolCalls) != 1 {
		t.Fatalf("tool calls: %+v", response.ToolCalls)
	}
	call := response.ToolCalls[0]
	if call.Name != "search" || call.ID == "" {
		t.Errorf("tool call: %+v", call)
	}
	if call.Args["q"] != "weather" {
		t.Errorf("tool call args: %v", call.Args)
	}
}

func TestChatModelInvokeThoughtsAndCacheUsage(t *testing.T) {
	// Python's _response_to_result (chat_models.py:2042-2070): output tokens
	// include thoughts; cache_read lands in input_token_details.
	server := newGeminiTestServer(t, func(t *testing.T, _ *http.Request, _ map[string]any) {},
		`{
		"candidates":[{"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP"}],
		"usageMetadata":{
			"promptTokenCount":10,"candidatesTokenCount":3,"thoughtsTokenCount":4,
			"cachedContentTokenCount":6,"totalTokenCount":17
		}
	}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	response, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	usage := response.UsageMetadata
	if usage.InputTokens != 10 || usage.OutputTokens != 7 || usage.TotalTokens != 17 {
		t.Errorf("usage: %+v", usage)
	}
	if usage.InputTokenDetails == nil || usage.InputTokenDetails.CacheReadInputTokens != 6 {
		t.Errorf("input token details: %+v", usage.InputTokenDetails)
	}
	if usage.OutputTokenDetails == nil || usage.OutputTokenDetails.ReasoningOutputTokens != 4 {
		t.Errorf("output token details: %+v", usage.OutputTokenDetails)
	}
}

func TestChatModelInvokeNoCandidatesError(t *testing.T) {
	server := newGeminiTestServer(t, func(t *testing.T, _ *http.Request, _ map[string]any) {},
		`{"promptFeedback":{"blockReason":"SAFETY"}}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "no candidates") || !strings.Contains(err.Error(), "SAFETY") {
		t.Fatalf("expected no-candidates error with block reason, got %v", err)
	}
}

func TestChatModelMissingAPIKey(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")
	model := NewChatModel()
	_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "api key") {
		t.Fatalf("expected api key error, got %v", err)
	}
}

func TestChatModelEnvKeyFallback(t *testing.T) {
	// The genai SDK reads GEMINI_API_KEY / GOOGLE_API_KEY when no explicit
	// key is configured; the key reaches the server via x-goog-api-key.
	t.Setenv("GEMINI_API_KEY", "env-key")
	server := newGeminiTestServer(t, func(t *testing.T, r *http.Request, _ map[string]any) {
		if got := r.Header.Get("x-goog-api-key"); got != "env-key" {
			t.Errorf("x-goog-api-key: got %q want env-key", got)
		}
	}, `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL))
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke with env key: %v", err)
	}
}

func TestChatModelBatch(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	responses, err := model.Batch(t.Context(), [][]messages.Message{
		{messages.Human("first")},
		{messages.Human("second")},
	})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if len(responses) != 2 {
		t.Fatalf("responses: %d", len(responses))
	}
	if requests != 2 {
		t.Errorf("requests: %d want 2", requests)
	}
}

func TestChatModelCapabilitiesAndLLMType(t *testing.T) {
	model := NewChatModel(modelconfig.WithAPIKey("k"))
	caps := model.Capabilities()
	for _, enabled := range []bool{
		caps.ToolCalling, caps.ToolChoice, caps.StructuredOutput,
		caps.ImageInputs, caps.ImageURLs, caps.AudioInputs,
		caps.PDFInputs, caps.VideoInputs, caps.UsageMetadata, caps.Streaming,
	} {
		if !enabled {
			t.Fatalf("capability not declared: %+v", caps)
		}
	}
	if caps.JSONMode {
		t.Errorf("JSONMode should not be declared (not exposed as its own toggle)")
	}
	if model.LLMType() != "chat-google-generative-ai" {
		t.Errorf("llm type: %q", model.LLMType())
	}
	if _, ok := any(model).(language.ToolBinder); !ok {
		t.Errorf("model does not implement language.ToolBinder")
	}
}

// invokeStream drains a stream with the test's context.
func drainStream(t *testing.T, stream interface {
	Next(context.Context) (messages.Message, bool, error)
	Close() error
}) []messages.Message {
	t.Helper()
	var chunks []messages.Message
	for {
		chunk, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("stream next: %v", err)
		}
		if !ok {
			return chunks
		}
		chunks = append(chunks, chunk)
	}
}
