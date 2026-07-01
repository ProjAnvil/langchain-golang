package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestEmbeddingsEmbedDocuments(t *testing.T) {
	var got embedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Fatalf("path: got %q want /api/embed", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("method: got %q", r.Method)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"model":"nomic-embed-text",
			"embeddings":[[0.1,0.2,0.3],[0.4,0.5,0.6]],
			"total_duration":123456
		}`))
	}))
	defer server.Close()

	embeddings := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("nomic-embed-text"),
	)
	vectors, err := embeddings.EmbedDocuments(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}

	if len(vectors) != 2 {
		t.Fatalf("vectors: got %d want 2", len(vectors))
	}
	if len(vectors[0]) != 3 || vectors[0][0] != 0.1 {
		t.Fatalf("vector[0]: %+v", vectors[0])
	}
	if len(vectors[1]) != 3 || vectors[1][2] != 0.6 {
		t.Fatalf("vector[1]: %+v", vectors[1])
	}
	if len(got.Input) != 2 || got.Input[0] != "hello" || got.Input[1] != "world" {
		t.Fatalf("input: %+v", got.Input)
	}
	if got.Model != "nomic-embed-text" {
		t.Fatalf("model: got %q", got.Model)
	}
}

func TestEmbeddingsEmbedQuery(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"nomic-embed-text","embeddings":[[0.7,0.8,0.9]]}`))
	}))
	defer server.Close()

	embeddings := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	vector, err := embeddings.EmbedQuery(context.Background(), "query text")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if len(vector) != 3 || vector[0] != 0.7 {
		t.Fatalf("vector: %+v", vector)
	}
}

func TestEmbeddingsDimensionsAndKeepAlive(t *testing.T) {
	var got embedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"model":"nomic-embed-text","embeddings":[[0.1]]}`))
	}))
	defer server.Close()

	embeddings := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("nomic-embed-text"),
		WithEmbeddingDimensions(768),
		WithKeepAlive("10m"),
	)
	_, err := embeddings.EmbedDocuments(context.Background(), []string{"hi"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}

	if got.Dimensions == nil || *got.Dimensions != 768 {
		t.Fatalf("dimensions: got %v", got.Dimensions)
	}
	if got.KeepAlive != "10m" {
		t.Fatalf("keep_alive: got %v", got.KeepAlive)
	}
}

func TestEmbeddingsEmptyInputReturnsNil(t *testing.T) {
	embeddings := NewEmbeddings(modelconfig.WithBaseURL("http://unreachable.invalid"))
	vectors, err := embeddings.EmbedDocuments(context.Background(), nil)
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	if vectors != nil {
		t.Fatalf("vectors: got %v want nil", vectors)
	}
}

func TestEmbeddingsPropagatesError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"model not found"}`))
	}))
	defer server.Close()

	embeddings := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithMaxRetries(0),
	)
	_, err := embeddings.EmbedDocuments(context.Background(), []string{"hi"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbeddingsCountMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"nomic-embed-text","embeddings":[[0.1]]}`))
	}))
	defer server.Close()

	embeddings := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	_, err := embeddings.EmbedDocuments(context.Background(), []string{"a", "b"})
	if err == nil {
		t.Fatal("expected count mismatch error")
	}
}
