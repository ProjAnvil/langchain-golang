package vectorstores_test

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// TestAsRetrieverMMR verifies that a retriever built via AsRetriever with
// search_type="mmr" dispatches to the vector store's MMR search. The chosen
// vectors make MMR order ([A, C]) differ from plain similarity order ([A, B]).
func TestAsRetrieverMMR(t *testing.T) {
	store := vectorstores.NewInMemory(embeddings.Static{
		DocumentVectors: [][]float64{{1, 0}, {0.8, 0.6}, {0, 1}},
		QueryVector:     []float64{1, 0},
	})
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("A", nil),
		documents.New("B", nil),
		documents.New("C", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}

	retriever, err := retrievers.AsRetriever(
		store,
		retrievers.WithSearchType("mmr"),
		retrievers.WithSearchKwargs(map[string]any{"k": 2, "fetch_k": 3, "lambda_mult": 0.1}),
	)
	if err != nil {
		t.Fatalf("as retriever: %v", err)
	}

	docs, err := retriever.GetRelevantDocuments(context.Background(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2", len(docs))
	}
	if docs[0].PageContent != "A" || docs[1].PageContent != "C" {
		t.Fatalf("mmr order: got %q, %q; want A, C", docs[0].PageContent, docs[1].PageContent)
	}
}

// TestAsRetrieverSimilarityScoreThreshold verifies that search_type
// "similarity_score_threshold" filters out documents below the relevance
// threshold.
func TestAsRetrieverSimilarityScoreThreshold(t *testing.T) {
	store := vectorstores.NewInMemory(embeddings.NewFake(32))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
		documents.New("gamma delta", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}

	retriever, err := retrievers.AsRetriever(
		store,
		retrievers.WithSearchType("similarity_score_threshold"),
		retrievers.WithSearchKwargs(map[string]any{"k": 2, "score_threshold": 0.5}),
	)
	if err != nil {
		t.Fatalf("as retriever: %v", err)
	}

	docs, err := retriever.GetRelevantDocuments(context.Background(), "alpha")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs: got %d want 1", len(docs))
	}
	if docs[0].PageContent != "alpha beta" {
		t.Fatalf("kept doc: got %q want %q", docs[0].PageContent, "alpha beta")
	}
}

// TestSimilaritySearchWithRelevanceScores verifies that the store returns
// documents paired with relevance scores ordered from most to least similar.
func TestSimilaritySearchWithRelevanceScores(t *testing.T) {
	store := vectorstores.NewInMemory(embeddings.NewFake(32))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
		documents.New("gamma delta", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}

	results, err := store.SimilaritySearchWithRelevanceScores(context.Background(), "alpha", 2, nil)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results: got %d want 2", len(results))
	}
	if results[0].Document.PageContent != "alpha beta" {
		t.Fatalf("top result: got %q", results[0].Document.PageContent)
	}
	if results[0].Score <= 0.5 {
		t.Fatalf("top score should exceed 0.5: %f", results[0].Score)
	}
	if results[1].Score >= results[0].Score {
		t.Fatalf("scores should be ordered descending: %f then %f", results[0].Score, results[1].Score)
	}
}

// TestAsRetrieverValidation verifies constructor validation of search_type and
// required kwargs, mirroring Python's VectorStoreRetriever validation.
func TestAsRetrieverValidation(t *testing.T) {
	store := vectorstores.NewInMemory(embeddings.NewFake(32))

	if _, err := retrievers.AsRetriever(store, retrievers.WithSearchType("bogus")); err == nil {
		t.Fatal("expected error for invalid search_type")
	}
	if _, err := retrievers.AsRetriever(
		store,
		retrievers.WithSearchType("similarity_score_threshold"),
	); err == nil {
		t.Fatal("expected error for similarity_score_threshold without score_threshold")
	}
}
