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

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/tools"
)

const azureChatCompletionBody = `{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`

func azureChatServer(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(handler))
	t.Cleanup(server.Close)
	return server
}

func TestAzureClientFromEnv(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "https://env.example")
	t.Setenv("AZURE_OPENAI_API_KEY", "env-key")
	t.Setenv("AZURE_OPENAI_AD_TOKEN", "env-ad")
	t.Setenv("AZURE_OPENAI_API_VERSION", "env-version")

	filled := azureClient{}.fromEnv()
	if filled.endpoint != "https://env.example" ||
		filled.apiKey != "env-key" ||
		filled.adToken != "env-ad" ||
		filled.apiVersion != "env-version" {
		t.Fatalf("fromEnv did not fill from environment: %#v", filled)
	}

	// Explicit values win over the environment.
	kept := azureClient{endpoint: "https://explicit.example", apiKey: "explicit-key"}.fromEnv()
	if kept.endpoint != "https://explicit.example" || kept.apiKey != "explicit-key" {
		t.Fatalf("fromEnv overwrote explicit values: %#v", kept)
	}
	if kept.adToken != "env-ad" || kept.apiVersion != "env-version" {
		t.Fatalf("fromEnv did not fill remaining fields: %#v", kept)
	}
}

func TestAzureChatModelBatch(t *testing.T) {
	var calls atomic.Int64
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(azureChatCompletionBody))
	})

	model := NewAzureChatModel(server.URL, "dep", "2024-01-01", "az-key")
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

func TestAzureChatModelMeta(t *testing.T) {
	model := NewAzureChatModel("https://example", "dep", "2024-01-01", "az-key")
	if model.LLMType() != "azure-openai-chat" {
		t.Fatalf("LLMType = %q, want azure-openai-chat", model.LLMType())
	}
	if !model.Capabilities().ToolCalling {
		t.Fatalf("Capabilities = %#v, want ToolCalling", model.Capabilities())
	}
	if model.InputSchema() == nil || model.OutputSchema() == nil {
		t.Fatal("schemas must be non-nil")
	}
}

func TestAzureChatModelBindTools(t *testing.T) {
	var gotBody map[string]any
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(azureChatCompletionBody))
	})

	tool, err := tools.FromFunc("search", "search the web", func(ctx context.Context, args struct{ Q string }) (tools.Result, error) {
		return tools.Result{Content: args.Q}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	model := NewAzureChatModel(server.URL, "dep", "2024-01-01", "az-key")
	bound, err := model.BindTools([]tools.Tool{tool})
	if err != nil {
		t.Fatalf("BindTools: %v", err)
	}
	if _, err := bound.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	toolsList, ok := gotBody["tools"].([]any)
	if !ok || len(toolsList) != 1 {
		t.Fatalf("tools = %v", gotBody["tools"])
	}
	entry, _ := toolsList[0].(map[string]any)
	fn, ok := entry["function"].(map[string]any)
	if !ok || fn["name"] != "search" {
		t.Fatalf("tools[0] must nest the descriptor under function, got %v", entry)
	}
}

func TestAzureChatModelCustomHeaders(t *testing.T) {
	var gotHeader string
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Custom")
		_, _ = w.Write([]byte(azureChatCompletionBody))
	})

	model := NewAzureChatModel(server.URL, "dep", "2024-01-01", "az-key",
		modelconfig.WithHeader("X-Custom", "custom-value"),
	)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotHeader != "custom-value" {
		t.Fatalf("X-Custom = %q, want custom-value", gotHeader)
	}
}

func TestAzureChatModelInvokeError(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})

	model := NewAzureChatModel(server.URL, "dep", "2024-01-01", "az-key",
		modelconfig.WithMaxRetries(0),
	)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected error for 500 response")
	}
}

func TestAzureChatModelStreamHTTPError(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadRequest)
	})

	model := NewAzureChatModel(server.URL, "dep", "2024-01-01", "az-key")
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected error for 400 response")
	}
}

func TestAzureChatModelStreamTransportError(t *testing.T) {
	// Nothing listens on 127.0.0.1:1, so the HTTP round trip fails fast.
	model := NewAzureChatModel("http://127.0.0.1:1", "dep", "2024-01-01", "az-key")
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected transport error")
	}
}

func TestAzureChatModelStreamBadEndpointURL(t *testing.T) {
	model := NewAzureChatModel("://bad-url", "dep", "2024-01-01", "az-key")
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		if stream != nil {
			_ = stream.Close()
		}
		t.Fatal("expected request construction error")
	}
}

func TestAzureChatModelStreamReadsChunks(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("api-key") != "az-key" {
			t.Errorf("api-key = %q", r.Header.Get("api-key"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n"+
				"data: [DONE]\n\n",
		)
	})

	model := NewAzureChatModel(server.URL, "dep", "2024-01-01", "az-key")
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	var content string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		content += chunk.Content
	}
	if content != "ab" {
		t.Fatalf("content = %q, want ab", content)
	}
}

func TestAzureEmbeddingsEmbedQuery(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.5,0.6]}]}`))
	})

	e := NewAzureEmbeddings(server.URL, "emb-dep", "2024-01-01", "az-key")
	vector, err := e.EmbedQuery(context.Background(), "query")
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if len(vector) != 2 || vector[0] != 0.5 {
		t.Fatalf("vector = %#v", vector)
	}
}

func TestAzureEmbeddingsOptionalParams(t *testing.T) {
	var got embeddingRequestPayload
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1]}]}`))
	})

	e := NewAzureEmbeddings(server.URL, "emb-dep", "2024-01-01", "az-key",
		modelconfig.WithModel("text-embedding-3-small"),
		WithEmbeddingDimensions(128),
		WithEmbeddingEncodingFormat("base64"),
	)
	if _, err := e.EmbedDocuments(context.Background(), []string{"hello"}); err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if got.Dimensions == nil || *got.Dimensions != 128 {
		t.Fatalf("dimensions = %v, want 128", got.Dimensions)
	}
	if got.EncodingFormat != "base64" {
		t.Fatalf("encoding_format = %q, want base64", got.EncodingFormat)
	}
}

func TestAzureEmbeddingsIndexOutOfRange(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":5,"embedding":[0.1]}]}`))
	})

	e := NewAzureEmbeddings(server.URL, "emb-dep", "2024-01-01", "az-key")
	_, err := e.EmbedDocuments(context.Background(), []string{"only"})
	if err == nil || !strings.Contains(err.Error(), "index out of range") {
		t.Fatalf("expected index out of range error, got %v", err)
	}
}

func TestAzureEmbeddingsCountMismatch(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1]}]}`))
	})

	e := NewAzureEmbeddings(server.URL, "emb-dep", "2024-01-01", "az-key")
	_, err := e.EmbedDocuments(context.Background(), []string{"one", "two"})
	if err == nil || !strings.Contains(err.Error(), "count mismatch") {
		t.Fatalf("expected count mismatch error, got %v", err)
	}
}

func TestAzureEmbeddingsEmptyDocuments(t *testing.T) {
	e := NewAzureEmbeddings("http://127.0.0.1:1", "emb-dep", "2024-01-01", "az-key")
	vectors, err := e.EmbedDocuments(context.Background(), nil)
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if vectors != nil {
		t.Fatalf("vectors = %#v, want nil", vectors)
	}
}

func TestAzureEmbeddingsEmbedQueryError(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})

	e := NewAzureEmbeddings(server.URL, "emb-dep", "2024-01-01", "az-key",
		modelconfig.WithMaxRetries(0),
	)
	if _, err := e.EmbedQuery(context.Background(), "query"); err == nil {
		t.Fatal("expected error for 500 response")
	}
}

func TestAzureTextModelBatch(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"text":"done"}]}`))
	})

	m := NewAzureTextModel(server.URL, "txt-dep", "2024-01-01", "az-key")
	outputs, err := m.Batch(context.Background(), []string{"one", "two"})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if len(outputs) != 2 || outputs[0] != "done" || outputs[1] != "done" {
		t.Fatalf("outputs = %#v", outputs)
	}
}

func TestAzureTextModelStream(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"text":"streamed"}]}`))
	})

	m := NewAzureTextModel(server.URL, "txt-dep", "2024-01-01", "az-key")
	stream, err := m.Stream(context.Background(), "prompt")
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

func TestAzureTextModelMeta(t *testing.T) {
	m := NewAzureTextModel("https://example", "txt-dep", "2024-01-01", "az-key")
	if m.InputSchema() == nil || m.OutputSchema() == nil {
		t.Fatal("schemas must be non-nil")
	}
	profile := m.ModelProfile()
	if profile["text_inputs"] != true || profile["text_outputs"] != true {
		t.Fatalf("ModelProfile = %#v", profile)
	}
}

func TestAzureTextModelNoChoices(t *testing.T) {
	server := azureChatServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[]}`))
	})

	m := NewAzureTextModel(server.URL, "txt-dep", "2024-01-01", "az-key",
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
	)
	_, err := m.Invoke(context.Background(), "prompt")
	if err == nil || !strings.Contains(err.Error(), "no choices") {
		t.Fatalf("expected no choices error, got %v", err)
	}
}
