package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/retrievers"
)

// RetrieverFactory creates a fresh retriever for standard tests.
type RetrieverFactory func(t testing.TB) retrievers.Retriever

// RunRetrieverBasics verifies behavior expected from retrievers.
func RunRetrieverBasics(t *testing.T, factory RetrieverFactory) {
	t.Helper()

	t.Run("retrieve documents", func(t *testing.T) {
		retriever := factory(t)
		docs, err := retriever.GetRelevantDocuments(context.Background(), "alpha")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) == 0 {
			t.Fatal("expected documents")
		}
		docs[0].Metadata["mutated"] = true
		again, err := factory(t).GetRelevantDocuments(context.Background(), "alpha")
		if err != nil {
			t.Fatalf("retrieve again: %v", err)
		}
		if _, ok := again[0].Metadata["mutated"]; ok {
			t.Fatal("retriever returned shared metadata")
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := factory(t).GetRelevantDocuments(ctx, "alpha"); err == nil {
			t.Fatal("expected canceled context error")
		}
	})
}
