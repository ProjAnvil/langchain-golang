package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// VectorStoreFactory creates a fresh vector store for standard tests.
type VectorStoreFactory func(t testing.TB) vectorstores.VectorStore

type filterSearcher interface {
	SimilaritySearchWithScoreFilter(context.Context, string, int, vectorstores.Filter) ([]vectorstores.SearchResult, error)
}

type mmrSearcher interface {
	MaxMarginalRelevanceSearch(context.Context, string, int, int, float64, vectorstores.Filter) ([]documents.Document, error)
}

// RunVectorStoreBasics verifies behavior expected from every vector store
// integration.
func RunVectorStoreBasics(t *testing.T, factory VectorStoreFactory) {
	t.Helper()

	t.Run("add and get by ids", func(t *testing.T) {
		store := factory(t)
		ids, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("alpha beta", nil).WithID("alpha"),
			documents.New("gamma delta", nil).WithID("gamma"),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		if len(ids) != 2 {
			t.Fatalf("ids: got %d want 2", len(ids))
		}

		docs, err := store.GetByIDs(context.Background(), []string{"alpha", "missing"})
		if err != nil {
			t.Fatalf("get by ids: %v", err)
		}
		if len(docs) != 1 {
			t.Fatalf("docs: got %d want 1", len(docs))
		}
		if docs[0].ID != "alpha" {
			t.Fatalf("doc id: got %q want alpha", docs[0].ID)
		}
	})

	t.Run("similarity search", func(t *testing.T) {
		store := factory(t)
		_, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("alpha beta", nil),
			documents.New("gamma delta", nil),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}

		results, err := store.SimilaritySearchWithScore(context.Background(), "alpha", 1)
		if err != nil {
			t.Fatalf("similarity search: %v", err)
		}
		if len(results) != 1 {
			t.Fatalf("results: got %d want 1", len(results))
		}
		if results[0].Document.PageContent != "alpha beta" {
			t.Fatalf("top result: got %q", results[0].Document.PageContent)
		}
	})

	t.Run("delete", func(t *testing.T) {
		store := factory(t)
		_, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("delete me", nil).WithID("delete-me"),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		if err := store.Delete(context.Background(), []string{"delete-me"}); err != nil {
			t.Fatalf("delete: %v", err)
		}
		docs, err := store.GetByIDs(context.Background(), []string{"delete-me"})
		if err != nil {
			t.Fatalf("get by ids: %v", err)
		}
		if len(docs) != 0 {
			t.Fatalf("docs after delete: got %d want 0", len(docs))
		}
	})

	t.Run("optional add texts", func(t *testing.T) {
		store := factory(t)
		adder, ok := store.(vectorstores.TextAdder)
		if !ok {
			t.Skip("vector store does not expose AddTexts")
		}
		metadata := map[string]any{"source": "standard"}
		ids, err := adder.AddTexts(
			context.Background(),
			[]string{"alpha beta", "gamma delta"},
			[]map[string]any{metadata},
			[]string{"alpha-text", "gamma-text"},
		)
		if err != nil {
			t.Fatalf("add texts: %v", err)
		}
		if len(ids) != 2 || ids[0] != "alpha-text" || ids[1] != "gamma-text" {
			t.Fatalf("ids: got %v", ids)
		}
		metadata["source"] = "mutated"

		docs, err := store.GetByIDs(context.Background(), ids)
		if err != nil {
			t.Fatalf("get by ids: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
		if docs[0].PageContent != "alpha beta" || docs[0].Metadata["source"] != "standard" {
			t.Fatalf("first doc: %#v", docs[0])
		}
	})

	t.Run("retriever adapter", func(t *testing.T) {
		store := factory(t)
		_, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("alpha beta", nil),
			documents.New("gamma delta", nil),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}

		retriever := retrievers.NewVectorStoreRetriever(store, 1)
		docs, err := retriever.GetRelevantDocuments(context.Background(), "alpha")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 1 {
			t.Fatalf("docs: got %d want 1", len(docs))
		}
	})

	t.Run("optional filter search", func(t *testing.T) {
		store := factory(t)
		searcher, ok := store.(filterSearcher)
		if !ok {
			t.Skip("vector store does not expose filtered search")
		}
		_, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("alpha beta", map[string]any{"group": "a"}),
			documents.New("alpha gamma", map[string]any{"group": "b"}),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		results, err := searcher.SimilaritySearchWithScoreFilter(context.Background(), "alpha", 2, func(doc documents.Document) bool {
			return doc.Metadata["group"] == "b"
		})
		if err != nil {
			t.Fatalf("filtered search: %v", err)
		}
		if len(results) != 1 || results[0].Document.Metadata["group"] != "b" {
			t.Fatalf("unexpected filtered results: %#v", results)
		}
	})

	t.Run("optional mmr search", func(t *testing.T) {
		store := factory(t)
		searcher, ok := store.(mmrSearcher)
		if !ok {
			t.Skip("vector store does not expose MMR search")
		}
		_, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("alpha beta", nil),
			documents.New("alpha gamma", nil),
			documents.New("delta epsilon", nil),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		docs, err := searcher.MaxMarginalRelevanceSearch(context.Background(), "alpha", 2, 3, 0.5, nil)
		if err != nil {
			t.Fatalf("mmr search: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
	})
}
