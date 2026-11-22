package retrievers_test

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

// controllableEmbeddings is a deterministic embeddings.Embeddings test double
// with configurable vectors and errors.
type controllableEmbeddings struct {
	query      []float64
	docVectors [][]float64
	queryErr   error
	docsErr    error
}

func (e controllableEmbeddings) EmbedQuery(_ context.Context, _ string) ([]float64, error) {
	if e.queryErr != nil {
		return nil, e.queryErr
	}
	return append([]float64(nil), e.query...), nil
}

func (e controllableEmbeddings) EmbedDocuments(_ context.Context, _ []string) ([][]float64, error) {
	if e.docsErr != nil {
		return nil, e.docsErr
	}
	out := make([][]float64, len(e.docVectors))
	for i, vector := range e.docVectors {
		out[i] = append([]float64(nil), vector...)
	}
	return out, nil
}

func embeddingsFilterDocs() []documents.Document {
	return []documents.Document{
		documents.New("doc zero", nil),
		documents.New("doc one", nil),
		documents.New("doc two", nil),
	}
}

func TestEmbeddingsFilterTopK(t *testing.T) {
	embedder := controllableEmbeddings{
		query:      []float64{1, 0},
		docVectors: [][]float64{{1, 0}, {0, 1}, {0.9, 0.1}},
	}
	filter, err := retrievers.NewEmbeddingsFilter(
		embedder,
		retrievers.WithEmbeddingsFilterK(2),
	)
	if err != nil {
		t.Fatalf("new embeddings filter: %v", err)
	}
	docs := embeddingsFilterDocs()
	out, err := filter.CompressDocuments(t.Context(), docs, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	// Similarity: doc0 = 1.0, doc2 ~= 0.994, doc1 = 0.0, so top 2 are
	// doc0 and doc2 in descending similarity order.
	if got := docContents(out); len(got) != 2 || got[0] != "doc zero" || got[1] != "doc two" {
		t.Fatalf("compressed = %v, want [doc zero doc two]", got)
	}
	for _, doc := range out {
		if _, ok := doc.Metadata["query_similarity_score"].(float64); !ok {
			t.Fatalf("compressed doc missing query_similarity_score: %v", doc.Metadata)
		}
	}
	// Inputs must not gain the score metadata.
	for _, doc := range docs {
		if _, ok := doc.Metadata["query_similarity_score"]; ok {
			t.Fatalf("input document was mutated: %v", doc.Metadata)
		}
	}
}

func TestEmbeddingsFilterThresholdKeepsInputOrder(t *testing.T) {
	// No top-k cap: with vectors doc0 = 0.6 and doc1 = 1.0 similarity, both
	// exceed the 0.5 threshold. Without a top-k sort the survivors must stay
	// in input order (doc0, doc1), not similarity order (doc1, doc0).
	embedder := controllableEmbeddings{
		query:      []float64{1, 0},
		docVectors: [][]float64{{0.6, 0.8}, {1, 0}, {0, 1}},
	}
	filter, err := retrievers.NewEmbeddingsFilter(
		embedder,
		retrievers.WithEmbeddingsFilterK(0),
		retrievers.WithEmbeddingsFilterSimilarityThreshold(0.5),
	)
	if err != nil {
		t.Fatalf("new embeddings filter: %v", err)
	}
	out, err := filter.CompressDocuments(t.Context(), embeddingsFilterDocs(), "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if got := docContents(out); len(got) != 2 || got[0] != "doc zero" || got[1] != "doc one" {
		t.Fatalf("compressed = %v, want [doc zero doc one] in input order", got)
	}
}

func TestEmbeddingsFilterCombinedTopKAndThreshold(t *testing.T) {
	embedder := controllableEmbeddings{
		query:      []float64{1, 0},
		docVectors: [][]float64{{1, 0}, {0, 1}, {0.9, 0.1}},
	}
	filter, err := retrievers.NewEmbeddingsFilter(
		embedder,
		retrievers.WithEmbeddingsFilterK(1),
		retrievers.WithEmbeddingsFilterSimilarityThreshold(0.5),
	)
	if err != nil {
		t.Fatalf("new embeddings filter: %v", err)
	}
	out, err := filter.CompressDocuments(t.Context(), embeddingsFilterDocs(), "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if got := docContents(out); len(got) != 1 || got[0] != "doc zero" {
		t.Fatalf("compressed = %v, want [doc zero]", got)
	}
}

func TestEmbeddingsFilterConstructorValidation(t *testing.T) {
	if _, err := retrievers.NewEmbeddingsFilter(nil); err == nil {
		t.Fatal("nil embedder must error")
	}
	if _, err := retrievers.NewEmbeddingsFilter(
		controllableEmbeddings{},
		retrievers.WithEmbeddingsFilterK(0),
	); err == nil {
		t.Fatal("k disabled without a threshold must error")
	}
}

func TestEmbeddingsFilterEmptyInput(t *testing.T) {
	embedder := controllableEmbeddings{
		query:      []float64{1, 0},
		docVectors: [][]float64{},
	}
	filter, err := retrievers.NewEmbeddingsFilter(embedder)
	if err != nil {
		t.Fatalf("new embeddings filter: %v", err)
	}
	out, err := filter.CompressDocuments(t.Context(), nil, "q")
	if err != nil {
		t.Fatalf("compress empty: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("docs = %v, want empty", docContents(out))
	}
}

func TestEmbeddingsFilterEmbeddingErrors(t *testing.T) {
	wantErr := errors.New("embedding failed")
	cases := []struct {
		name     string
		embedder controllableEmbeddings
	}{
		{
			name:     "query embedding error",
			embedder: controllableEmbeddings{queryErr: wantErr},
		},
		{
			name:     "document embedding error",
			embedder: controllableEmbeddings{query: []float64{1, 0}, docsErr: wantErr},
		},
		{
			name: "embedding count mismatch",
			embedder: controllableEmbeddings{
				query:      []float64{1, 0},
				docVectors: [][]float64{{1, 0}}, // one vector for three docs
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			filter, err := retrievers.NewEmbeddingsFilter(tc.embedder)
			if err != nil {
				t.Fatalf("new embeddings filter: %v", err)
			}
			if _, err := filter.CompressDocuments(t.Context(), embeddingsFilterDocs(), "q"); err == nil {
				t.Fatal("compress should return an error")
			}
		})
	}

	// A query embedding error must wrap the sentinel errors.Is target.
	filter, err := retrievers.NewEmbeddingsFilter(controllableEmbeddings{queryErr: wantErr})
	if err != nil {
		t.Fatalf("new embeddings filter: %v", err)
	}
	if _, err := filter.CompressDocuments(t.Context(), embeddingsFilterDocs(), "q"); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}

func TestEmbeddingsFilterCanceledContext(t *testing.T) {
	embedder := controllableEmbeddings{
		query:      []float64{1, 0},
		docVectors: [][]float64{{1, 0}, {0, 1}, {0.9, 0.1}},
	}
	filter, err := retrievers.NewEmbeddingsFilter(embedder)
	if err != nil {
		t.Fatalf("new embeddings filter: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := filter.CompressDocuments(ctx, embeddingsFilterDocs(), "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled compress error = %v, want context.Canceled", err)
	}
}
