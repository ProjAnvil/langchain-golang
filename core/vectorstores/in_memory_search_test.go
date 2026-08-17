package vectorstores

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

func canceledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

// shortEmbedder returns fewer vectors than requested to trigger the
// embedding-count mismatch guard in AddDocuments.
type shortEmbedder struct{}

func (shortEmbedder) EmbedDocuments(_ context.Context, _ []string) ([][]float64, error) {
	return [][]float64{{1, 0}}, nil
}

func (shortEmbedder) EmbedQuery(_ context.Context, _ string) ([]float64, error) {
	return []float64{1, 0}, nil
}

func TestInMemorySimilaritySearchWrapper(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(16))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
		documents.New("gamma delta", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}

	docs, err := store.SimilaritySearch(context.Background(), "alpha", 2)
	if err != nil {
		t.Fatalf("similarity search: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2", len(docs))
	}
	if docs[0].PageContent != "alpha beta" {
		t.Fatalf("top doc: got %q", docs[0].PageContent)
	}

	if _, err := store.SimilaritySearch(canceledContext(), "alpha", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled error, got %v", err)
	}
}

func TestInMemoryAddDocumentsErrors(t *testing.T) {
	docs := []documents.Document{documents.New("alpha", nil), documents.New("beta", nil)}

	if _, err := NewInMemory(nil).AddDocuments(context.Background(), docs); err == nil {
		t.Fatal("expected error for nil embedder")
	}

	if _, err := NewInMemory(embeddings.NewFake(8)).AddDocuments(canceledContext(), docs); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled error, got %v", err)
	}

	if _, err := NewInMemory(shortEmbedder{}).AddDocuments(context.Background(), docs); err == nil {
		t.Fatal("expected embedding count mismatch error")
	}
}

func TestInMemorySearchWithScoreFilterEdgeCases(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(16))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}

	results, err := store.SimilaritySearchWithScoreFilter(context.Background(), "alpha", 0, nil)
	if err != nil {
		t.Fatalf("k<=0 search: %v", err)
	}
	if results != nil {
		t.Fatalf("k<=0 should return nil results, got %#v", results)
	}

	if _, err := NewInMemory(nil).SimilaritySearchWithScoreFilter(context.Background(), "q", 1, nil); err == nil {
		t.Fatal("expected error for nil embedder")
	}

	if _, err := store.SimilaritySearchWithScoreFilter(canceledContext(), "q", 1, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled error, got %v", err)
	}
}

func TestInMemorySearchByVectorEdgeCases(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(16))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}
	vector, err := embeddings.NewFake(16).EmbedQuery(context.Background(), "alpha")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}

	if _, err := store.SimilaritySearchWithScoreByVector(canceledContext(), vector, 1, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled error, got %v", err)
	}

	results, err := store.SimilaritySearchWithScoreByVector(context.Background(), vector, -1, nil)
	if err != nil {
		t.Fatalf("k<=0 search: %v", err)
	}
	if results != nil {
		t.Fatalf("k<=0 should return nil results, got %#v", results)
	}

	if _, err := store.SimilaritySearchByVector(canceledContext(), vector, 1, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled error, got %v", err)
	}
}

func TestInMemoryRelevanceScoresThreshold(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(16))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
		documents.New("gamma delta", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}

	// A high threshold keeps only near-identical matches.
	threshold := 0.99
	results, err := store.SimilaritySearchWithRelevanceScores(context.Background(), "alpha beta", 2, &threshold)
	if err != nil {
		t.Fatalf("relevance search: %v", err)
	}
	if len(results) != 1 || results[0].Document.PageContent != "alpha beta" {
		t.Fatalf("unexpected thresholded results: %#v", results)
	}
	for _, result := range results {
		if result.Score < threshold {
			t.Fatalf("score %f below threshold %f", result.Score, threshold)
		}
	}

	// A zero threshold keeps everything.
	zero := 0.0
	results, err = store.SimilaritySearchWithRelevanceScores(context.Background(), "alpha", 2, &zero)
	if err != nil {
		t.Fatalf("relevance search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("docs: got %d want 2", len(results))
	}

	if _, err := store.SimilaritySearchWithRelevanceScores(canceledContext(), "q", 1, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled error, got %v", err)
	}
}

func TestInMemoryMMREdgeCases(t *testing.T) {
	if _, err := NewInMemory(nil).MaxMarginalRelevanceSearch(context.Background(), "q", 1, 1, 0.5, nil); err == nil {
		t.Fatal("expected error for nil embedder")
	}

	store := NewInMemory(embeddings.NewFake(16))
	if _, err := store.MaxMarginalRelevanceSearch(canceledContext(), "q", 1, 1, 0.5, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled error, got %v", err)
	}

	docs := make([]documents.Document, 0, 6)
	for _, text := range []string{"alpha beta", "alpha gamma", "beta delta", "gamma delta", "delta epsilon", "epsilon zeta"} {
		docs = append(docs, documents.New(text, nil))
	}
	if _, err := store.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("add documents: %v", err)
	}

	vector, err := embeddings.NewFake(16).EmbedQuery(context.Background(), "alpha")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}

	if _, err := store.MaxMarginalRelevanceSearchByVector(canceledContext(), vector, 1, 1, 0.5, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled error, got %v", err)
	}

	// Zero k and fetchK fall back to the defaults k=4, fetchK=20.
	selected, err := store.MaxMarginalRelevanceSearchByVector(context.Background(), vector, 0, 0, 0.5, nil)
	if err != nil {
		t.Fatalf("mmr by vector: %v", err)
	}
	if len(selected) != 4 {
		t.Fatalf("default k: got %d want 4", len(selected))
	}
}

// TestInMemorySearchSkipsStaleSequenceIDs covers the defensive branch where an
// ID in idSequence no longer has a stored document.
func TestInMemorySearchSkipsStaleSequenceIDs(t *testing.T) {
	store := NewInMemory(embeddings.NewFake(16))
	ids, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
		documents.New("gamma delta", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}

	// Simulate a stale sequence entry without compacting (bypasses Delete).
	delete(store.documents, ids[0])
	delete(store.vectors, ids[0])

	vector, err := embeddings.NewFake(16).EmbedQuery(context.Background(), "alpha")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	results, err := store.SimilaritySearchWithScoreByVector(context.Background(), vector, 5, nil)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 || results[0].Document.PageContent != "gamma delta" {
		t.Fatalf("unexpected results: %#v", results)
	}
}

func TestCosineEdgeCases(t *testing.T) {
	if got := cosine(nil, nil); got != 0 {
		t.Fatalf("empty vectors: got %f", got)
	}
	if got := cosine([]float64{1, 0}, []float64{1}); got != 0 {
		t.Fatalf("mismatched lengths: got %f", got)
	}
	if got := cosine([]float64{0, 0}, []float64{1, 0}); got != 0 {
		t.Fatalf("zero norm: got %f", got)
	}
	if got := cosine([]float64{1, 0}, []float64{1, 0}); got != 1 {
		t.Fatalf("identical vectors: got %f", got)
	}
}

func TestRelevanceScoreHelpersExtended(t *testing.T) {
	if got := EuclideanRelevanceScore(0.0); got != 1.0 {
		t.Fatalf("euclidean relevance at zero distance: got %f", got)
	}
	if got := EuclideanRelevanceScore(1.4142135623730951); got != 0.0 {
		t.Fatalf("euclidean relevance at max distance: got %f", got)
	}
	if got := MaxInnerProductRelevanceScore(0.25); got != 0.75 {
		t.Fatalf("positive inner product relevance: got %f", got)
	}
	if got := MaxInnerProductRelevanceScore(-0.5); got != 0.5 {
		t.Fatalf("negative inner product relevance: got %f", got)
	}
}

func TestMaximalMarginalRelevanceEdgeCases(t *testing.T) {
	embeds := [][]float64{{1, 0}, {0, 1}}

	if got := MaximalMarginalRelevance([]float64{1, 0}, embeds, 0.5, 0); got != nil {
		t.Fatalf("k<=0: got %v", got)
	}
	if got := MaximalMarginalRelevance([]float64{1, 0}, nil, 0.5, 2); got != nil {
		t.Fatalf("empty embeddings: got %v", got)
	}

	// k larger than the embedding count is clamped to the count.
	if got := MaximalMarginalRelevance([]float64{1, 0}, embeds, 0.5, 10); len(got) != 2 {
		t.Fatalf("clamped k: got %v", got)
	}

	// lambdaMult is clamped into [0, 1]; both extremes must still select all
	// embeddings without panicking.
	if got := MaximalMarginalRelevance([]float64{1, 0}, embeds, -1, 2); len(got) != 2 {
		t.Fatalf("negative lambda: got %v", got)
	}
	if got := MaximalMarginalRelevance([]float64{1, 0}, embeds, 2, 2); len(got) != 2 {
		t.Fatalf("lambda above one: got %v", got)
	}

	// lambdaMult=0 favors diversity: after picking the most similar doc, the
	// next pick should be the least redundant one.
	got := MaximalMarginalRelevance(
		[]float64{1, 0},
		[][]float64{{1, 0}, {0.9, 0.1}, {0, 1}},
		0,
		2,
	)
	if len(got) != 2 || got[0] != 0 || got[1] != 2 {
		t.Fatalf("diversity selection: got %v", got)
	}
}
