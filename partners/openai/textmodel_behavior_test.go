package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func textModelServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server
}

func TestTextModelBatch(t *testing.T) {
	server := textModelServer(t, `{"choices":[{"text":"done"}],"usage":{}}`)

	model := NewTextModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
	)
	outputs, err := model.Batch(context.Background(), []string{"one", "two"})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if len(outputs) != 2 || outputs[0] != "done" || outputs[1] != "done" {
		t.Fatalf("outputs = %#v", outputs)
	}
}

func TestTextModelStream(t *testing.T) {
	server := textModelServer(t, `{"choices":[{"text":"streamed"}],"usage":{}}`)

	model := NewTextModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
	)
	stream, err := model.Stream(context.Background(), "prompt")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk != "streamed" {
		t.Fatalf("chunk = %q ok=%v err=%v", chunk, ok, err)
	}
	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected stream end, ok=%v err=%v", ok, err)
	}
}

func TestTextModelSchemasAndProfile(t *testing.T) {
	model := NewTextModel()
	if model.InputSchema() == nil || model.OutputSchema() == nil {
		t.Fatal("schemas must be non-nil")
	}
	profile := model.ModelProfile()
	if profile["text_inputs"] != true || profile["text_outputs"] != true {
		t.Fatalf("ModelProfile = %#v", profile)
	}
}

func TestTextModelRequestIncludesSamplingParams(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"choices":[{"text":"ok"}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewTextModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
		modelconfig.WithMaxTokens(16),
		modelconfig.WithTemperature(0.3),
	)
	if _, err := model.Invoke(context.Background(), "prompt"); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotBody["max_tokens"] != float64(16) {
		t.Fatalf("max_tokens = %v, want 16", gotBody["max_tokens"])
	}
	if gotBody["temperature"] != 0.3 {
		t.Fatalf("temperature = %v, want 0.3", gotBody["temperature"])
	}
}

func TestTextModelNoChoicesError(t *testing.T) {
	server := textModelServer(t, `{"choices":[],"usage":{}}`)

	model := NewTextModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
	)
	_, err := model.Invoke(context.Background(), "prompt")
	if err == nil || !strings.Contains(err.Error(), "no choices") {
		t.Fatalf("expected no choices error, got %v", err)
	}
}

func TestTextModelUsesDefaultModel(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"choices":[{"text":"ok"}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewTextModel(modelconfig.WithBaseURL(server.URL))
	if _, err := model.Invoke(context.Background(), "prompt"); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotBody["model"] != "gpt-3.5-turbo-instruct" {
		t.Fatalf("model = %v, want gpt-3.5-turbo-instruct default", gotBody["model"])
	}
}
