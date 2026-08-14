package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestTextModelInvoke(t *testing.T) {
	var gotPath string
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"cmpl_123",
			"object":"text_completion",
			"model":"gpt-3.5-turbo-instruct",
			"choices":[{"text":"completion text","index":0,"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
		}`))
	}))
	defer server.Close()

	model := NewTextModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
	)
	response, err := model.Invoke(context.Background(), "hello")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if response != "completion text" {
		t.Fatalf("response: got %q want %q", response, "completion text")
	}
	if gotPath != "/completions" {
		t.Fatalf("path: got %q want %q", gotPath, "/completions")
	}
}

func TestTextModelRequestUsesPromptField(t *testing.T) {
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"choices":[{"text":"ok"}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewTextModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
	)
	if _, err := model.Invoke(context.Background(), "the prompt"); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if _, ok := gotBody["prompt"]; !ok {
		t.Fatalf("request body missing prompt field: %v", gotBody)
	}
	if prompt, ok := gotBody["prompt"].(string); !ok || prompt != "the prompt" {
		t.Fatalf("prompt: got %v want %q", gotBody["prompt"], "the prompt")
	}
	if _, ok := gotBody["messages"]; ok {
		t.Fatalf("request body must not use messages field: %v", gotBody)
	}
}

func TestTextModelNon2xxSurfacesError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer server.Close()

	model := NewTextModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
	)
	if _, err := model.Invoke(context.Background(), "hello"); err == nil {
		t.Fatal("expected non-2xx error")
	}
}
