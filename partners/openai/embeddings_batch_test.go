package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// Mirrors libs/partners/openai/tests/unit_tests/embeddings/test_base.py
// ::test_embed_documents_with_custom_chunk_size_no_check_ctx_length
// (embeddings/base.py simple batching path, chunk_size default 1000).

// batchRecordingServer returns one embedding per input (vector [index+1] so
// order across batches is observable) and records each request's input texts.
// The returned func snapshots the recorded inputs.
func batchRecordingServer(t *testing.T) (func() [][]string, *httptest.Server) {
	t.Helper()
	var mu sync.Mutex
	var inputs [][]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request embeddingRequestPayload
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		mu.Lock()
		inputs = append(inputs, request.Input)
		mu.Unlock()
		response := embeddingResponsePayload{Object: "list", Model: request.Model}
		for i := range request.Input {
			response.Data = append(response.Data, embeddingDataPayload{
				Object:    "embedding",
				Index:     i,
				Embedding: []float64{float64(i + 1)},
			})
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)
	snapshot := func() [][]string {
		mu.Lock()
		defer mu.Unlock()
		out := make([][]string, len(inputs))
		copy(out, inputs)
		return out
	}
	return snapshot, server
}

func TestEmbeddingsChunkSizeSplitsRequests(t *testing.T) {
	snapshot, server := batchRecordingServer(t)
	model := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		WithEmbeddingChunkSize(2),
	)
	texts := []string{"a", "b", "c", "d", "e"}
	vectors, err := model.EmbedDocuments(context.Background(), texts)
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	got := snapshot()
	if len(got) != 3 {
		t.Fatalf("requests = %d, want 3 (2+2+1)", len(got))
	}
	wantLens := []int{2, 2, 1}
	for i, want := range wantLens {
		if len(got[i]) != want {
			t.Fatalf("request %d inputs = %v, want length %d", i, got[i], want)
		}
	}
	if got[0][0] != "a" || got[1][0] != "c" || got[2][0] != "e" {
		t.Fatalf("batch boundaries wrong: %v", got)
	}
	if len(vectors) != 5 {
		t.Fatalf("vectors = %d, want 5", len(vectors))
	}
	// Per-batch index is relative to the batch; values must concatenate in order.
	for i, vector := range vectors {
		want := float64(i%2 + 1)
		if len(vector) != 1 || vector[0] != want {
			t.Fatalf("vectors[%d] = %v, want [%v]", i, vector, want)
		}
	}
}

func TestEmbeddingsChunkSizeDefault1000(t *testing.T) {
	snapshot, server := batchRecordingServer(t)
	model := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	texts := make([]string, 1001)
	for i := range texts {
		texts[i] = fmt.Sprintf("doc-%d", i)
	}
	vectors, err := model.EmbedDocuments(context.Background(), texts)
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	got := snapshot()
	if len(got) != 2 || len(got[0]) != 1000 || len(got[1]) != 1 {
		t.Fatalf("requests = %v lengths, want [1000 1]", []int{len(got), len(got[0]), len(got[1])})
	}
	if len(vectors) != 1001 {
		t.Fatalf("vectors = %d, want 1001", len(vectors))
	}
}

func TestEmbeddingsChunkSizeZeroOrNegativeFallsBackTo1000(t *testing.T) {
	snapshot, server := batchRecordingServer(t)
	model := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		WithEmbeddingChunkSize(0),
	)
	texts := make([]string, 3)
	for i := range texts {
		texts[i] = fmt.Sprintf("doc-%d", i)
	}
	if _, err := model.EmbedDocuments(context.Background(), texts); err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if got := snapshot(); len(got) != 1 || len(got[0]) != 3 {
		t.Fatalf("requests = %v, want single request (0 falls back to default 1000)", got)
	}
}

func TestAzureEmbeddingsChunkSizeSplitsRequests(t *testing.T) {
	snapshot, server := batchRecordingServer(t)
	model := NewAzureEmbeddings(server.URL, "emb-dep", "2024-01-01", "az-key",
		modelconfig.WithModel("text-embedding-3-small"),
		WithEmbeddingChunkSize(1),
	)
	vectors, err := model.EmbedDocuments(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	got := snapshot()
	if len(got) != 2 || len(got[0]) != 1 || len(got[1]) != 1 {
		t.Fatalf("requests = %v, want two single-text requests", got)
	}
	if len(vectors) != 2 {
		t.Fatalf("vectors = %d, want 2", len(vectors))
	}
}
