package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
)

// EmbeddingsFactory creates a fresh embeddings implementation for standard
// tests.
type EmbeddingsFactory func(t testing.TB) embeddings.Embeddings

// RunEmbeddingsBasics verifies behavior expected from every embeddings
// integration.
func RunEmbeddingsBasics(t *testing.T, factory EmbeddingsFactory) {
	t.Helper()

	t.Run("embed query", func(t *testing.T) {
		model := factory(t)
		vector, err := model.EmbedQuery(context.Background(), "hello world")
		if err != nil {
			t.Fatalf("embed query: %v", err)
		}
		if len(vector) == 0 {
			t.Fatal("expected non-empty vector")
		}
	})

	t.Run("embed documents", func(t *testing.T) {
		model := factory(t)
		vectors, err := model.EmbedDocuments(context.Background(), []string{
			"first",
			"second",
		})
		if err != nil {
			t.Fatalf("embed documents: %v", err)
		}
		if len(vectors) != 2 {
			t.Fatalf("vectors: got %d want 2", len(vectors))
		}
		if len(vectors[0]) == 0 || len(vectors[1]) == 0 {
			t.Fatal("expected non-empty vectors")
		}
		if len(vectors[0]) != len(vectors[1]) {
			t.Fatalf("inconsistent dimensions: %d != %d", len(vectors[0]), len(vectors[1]))
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		model := factory(t)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := model.EmbedQuery(ctx, "hello"); err == nil {
			t.Fatal("expected canceled query error")
		}
	})
}
