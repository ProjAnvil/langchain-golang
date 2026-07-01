package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	coreembeddings "github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/standardtests"
)

func TestEmbeddingsEmbedDocuments(t *testing.T) {
	var got embeddingRequestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
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
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[
				{"object":"embedding","index":1,"embedding":[0.3,0.4]},
				{"object":"embedding","index":0,"embedding":[0.1,0.2]}
			],
			"usage":{"prompt_tokens":4,"total_tokens":4}
		}`))
	}))
	defer server.Close()

	model := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
		modelconfig.WithModel("text-embedding-3-small"),
	)
	vectors, err := model.EmbedDocuments(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}

	if got.Model != "text-embedding-3-small" {
		t.Fatalf("model: got %q", got.Model)
	}
	if len(got.Input) != 2 || got.Input[0] != "first" || got.Input[1] != "second" {
		t.Fatalf("input: %+v", got.Input)
	}
	if len(vectors) != 2 {
		t.Fatalf("vectors: got %d want 2", len(vectors))
	}
	if vectors[0][0] != 0.1 || vectors[1][0] != 0.3 {
		t.Fatalf("vectors were not ordered by response index: %+v", vectors)
	}
}

func TestEmbeddingsUsesDefaultModel(t *testing.T) {
	var got embeddingRequestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":0,"embedding":[0.1]}],
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`))
	}))
	defer server.Close()

	model := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	_, err := model.EmbedDocuments(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	if got.Model != "text-embedding-3-small" {
		t.Fatalf("model: got %q", got.Model)
	}
}

func TestEmbeddingsOptionalRequestParameters(t *testing.T) {
	var got embeddingRequestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`))
	}))
	defer server.Close()

	model := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		WithEmbeddingDimensions(256),
		WithEmbeddingEncodingFormat("float"),
	)
	_, err := model.EmbedDocuments(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}

	if got.Dimensions == nil || *got.Dimensions != 256 {
		t.Fatalf("dimensions: got %v", got.Dimensions)
	}
	if got.EncodingFormat != "float" {
		t.Fatalf("encoding format: got %q", got.EncodingFormat)
	}
}

func TestEmbeddingsEmptyDocumentsDoesNotCallAPI(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		t.Fatal("unexpected API call")
	}))
	defer server.Close()

	model := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	vectors, err := model.EmbedDocuments(context.Background(), nil)
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	if vectors != nil {
		t.Fatalf("vectors: got %+v want nil", vectors)
	}
	if called {
		t.Fatal("server was called")
	}
}

func TestEmbeddingsResponseCountMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":0,"embedding":[0.1]}],
			"usage":{"prompt_tokens":2,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	model := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	_, err := model.EmbedDocuments(context.Background(), []string{"first", "second"})
	if err == nil {
		t.Fatal("expected count mismatch error")
	}
	if err.Error() != "embedding count mismatch: got 1 want 2" {
		t.Fatalf("error: got %q", err.Error())
	}
}

func TestEmbeddingsResponseIndexOutOfRange(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":2,"embedding":[0.1]}],
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`))
	}))
	defer server.Close()

	model := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	_, err := model.EmbedDocuments(context.Background(), []string{"only"})
	if err == nil {
		t.Fatal("expected index error")
	}
	if err.Error() != "embedding index out of range: 2" {
		t.Fatalf("error: got %q", err.Error())
	}
}

func TestEmbeddingsEmbedQuery(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],
			"usage":{"prompt_tokens":2,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	model := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	vector, err := model.EmbedQuery(context.Background(), "query")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if len(vector) != 3 {
		t.Fatalf("vector dimensions: got %d want 3", len(vector))
	}
}

func TestEmbeddingsStandardBasics(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request embeddingRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		response := embeddingResponsePayload{
			Object: "list",
			Model:  request.Model,
			Data:   make([]embeddingDataPayload, len(request.Input)),
			Usage:  embeddingUsagePayload{PromptTokens: len(request.Input), TotalTokens: len(request.Input)},
		}
		for i := range request.Input {
			response.Data[i] = embeddingDataPayload{
				Object:    "embedding",
				Index:     i,
				Embedding: []float64{float64(i + 1), float64(i + 2)},
			}
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	standardtests.RunEmbeddingsBasics(t, func(t testing.TB) coreembeddings.Embeddings {
		t.Helper()
		return NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	})
}
