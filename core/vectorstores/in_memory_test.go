package vectorstores

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

func TestInMemorySimilaritySearch(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(32))
	ids, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", map[string]any{"rank": 1}),
		documents.New("gamma delta", map[string]any{"rank": 2}),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}
	if len(ids) != 2 {
		t.Fatalf("ids: got %d want 2", len(ids))
	}

	results, err := store.SimilaritySearchWithScore(context.Background(), "alpha", 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("results: got %d want 1", len(results))
	}
	if results[0].Document.PageContent != "alpha beta" {
		t.Fatalf("top result: got %q", results[0].Document.PageContent)
	}
	if results[0].Score <= 0 {
		t.Fatalf("score should be positive: %f", results[0].Score)
	}
}

func TestInMemoryGetAndDelete(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(16))
	ids, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("keep", nil).WithID("keep"),
		documents.New("delete", nil).WithID("delete"),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}
	if ids[0] != "keep" || ids[1] != "delete" {
		t.Fatalf("ids: got %v", ids)
	}

	if err := store.Delete(context.Background(), []string{"delete"}); err != nil {
		t.Fatalf("delete: %v", err)
	}

	docs, err := store.GetByIDs(context.Background(), []string{"keep", "delete"})
	if err != nil {
		t.Fatalf("get by ids: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs: got %d want 1", len(docs))
	}
	if docs[0].ID != "keep" {
		t.Fatalf("remaining doc: got %q", docs[0].ID)
	}
}

func TestInMemoryAddTexts(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(16))
	metadata := map[string]any{"source": "unit"}
	ids, err := store.AddTexts(
		context.Background(),
		[]string{"alpha beta", "gamma delta"},
		[]map[string]any{metadata},
		[]string{"alpha-id", "gamma-id"},
	)
	if err != nil {
		t.Fatalf("add texts: %v", err)
	}
	if ids[0] != "alpha-id" || ids[1] != "gamma-id" {
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
	if docs[0].PageContent != "alpha beta" || docs[0].Metadata["source"] != "unit" {
		t.Fatalf("first doc: %#v", docs[0])
	}
	if docs[1].PageContent != "gamma delta" {
		t.Fatalf("second doc content: %q", docs[1].PageContent)
	}
	if len(docs[1].Metadata) != 0 {
		t.Fatalf("missing metadata should become empty map: %#v", docs[1].Metadata)
	}
}

func TestInMemoryFilterAndVectorSearch(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(32))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", map[string]any{"group": "a"}),
		documents.New("alpha gamma", map[string]any{"group": "b"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	results, err := store.SimilaritySearchWithScoreFilter(context.Background(), "alpha", 2, func(doc documents.Document) bool {
		return doc.Metadata["group"] == "b"
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Document.Metadata["group"] != "b" {
		t.Fatalf("unexpected filtered results: %#v", results)
	}
	vector, err := embeddings.NewFake(32).EmbedQuery(context.Background(), "alpha")
	if err != nil {
		t.Fatal(err)
	}
	docs, err := store.SimilaritySearchByVector(context.Background(), vector, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs len = %d, want 1", len(docs))
	}
}

func TestMaximalMarginalRelevance(t *testing.T) {
	indices := MaximalMarginalRelevance(
		[]float64{1, 0},
		[][]float64{{1, 0}, {0.9, 0.1}, {0, 1}},
		0.5,
		2,
	)
	if len(indices) != 2 || indices[0] != 0 {
		t.Fatalf("unexpected MMR indices: %v", indices)
	}
}

func TestInMemoryMaxMarginalRelevanceSearch(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(32))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
		documents.New("alpha gamma", nil),
		documents.New("delta epsilon", nil),
	})
	if err != nil {
		t.Fatal(err)
	}
	docs, err := store.MaxMarginalRelevanceSearch(context.Background(), "alpha", 2, 3, 0.5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs len = %d, want 2", len(docs))
	}
}

func TestRelevanceScoreHelpers(t *testing.T) {
	if CosineRelevanceScore(0.25) != 0.75 {
		t.Fatal("unexpected cosine relevance")
	}
	if MaxInnerProductRelevanceScore(-0.5) != 0.5 {
		t.Fatal("unexpected inner product relevance")
	}
}
