package embeddings

import (
	"context"
	"errors"
	"testing"
)

func canceledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

func TestFakeEmbeddingsDefaultsAndDimensions(t *testing.T) {
	if got := NewFake(0).Dimensions(); got != 8 {
		t.Fatalf("zero dimensions: got %d want 8", got)
	}
	if got := NewFake(-3).Dimensions(); got != 8 {
		t.Fatalf("negative dimensions: got %d want 8", got)
	}
	if got := NewFake(4).Dimensions(); got != 4 {
		t.Fatalf("dimensions: got %d want 4", got)
	}
}

func TestFakeEmbeddingsEmbedDocuments(t *testing.T) {
	model := NewFake(8)

	docs, err := model.EmbedDocuments(context.Background(), []string{"alpha beta", "alpha"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("documents: got %d want 2", len(docs))
	}
	for i, doc := range docs {
		if len(doc) != 8 {
			t.Fatalf("document %d dimensions: got %d want 8", i, len(doc))
		}
	}
	// Deterministic: same token set must produce the same vector.
	again, err := model.EmbedDocuments(context.Background(), []string{"alpha beta"})
	if err != nil {
		t.Fatalf("embed documents again: %v", err)
	}
	for i := range docs[0] {
		if docs[0][i] != again[0][i] {
			t.Fatalf("vector[%d]: got %f want %f", i, again[0][i], docs[0][i])
		}
	}

	// Empty input yields an empty (but non-nil) result without error.
	empty, err := model.EmbedDocuments(context.Background(), nil)
	if err != nil {
		t.Fatalf("embed empty: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("empty documents: got %d want 0", len(empty))
	}
}

func TestFakeEmbeddingsContextCancellation(t *testing.T) {
	model := NewFake(8)

	if _, err := model.EmbedQuery(canceledContext(), "hello"); !errors.Is(err, context.Canceled) {
		t.Fatalf("query canceled: got %v want context.Canceled", err)
	}
	if _, err := model.EmbedDocuments(canceledContext(), []string{"a", "b"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("documents canceled: got %v want context.Canceled", err)
	}
}

func TestDeterministicFakeDefaultsAndDocuments(t *testing.T) {
	defaulted := NewDeterministicFake(0)
	vec, err := defaulted.EmbedQuery(context.Background(), "x")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if len(vec) != 8 {
		t.Fatalf("zero size: got %d want 8", len(vec))
	}

	model := NewDeterministicFake(6)
	docs, err := model.EmbedDocuments(context.Background(), []string{"one", "two", "one"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	if len(docs) != 3 {
		t.Fatalf("documents: got %d want 3", len(docs))
	}
	for i, doc := range docs {
		if len(doc) != 6 {
			t.Fatalf("document %d dimensions: got %d want 6", i, len(doc))
		}
	}
	// Same text maps to the same vector within one batch.
	for i := range docs[0] {
		if docs[0][i] != docs[2][i] {
			t.Fatalf("duplicate text differs at %d: %f != %f", i, docs[0][i], docs[2][i])
		}
	}

	if _, err := model.EmbedQuery(canceledContext(), "x"); !errors.Is(err, context.Canceled) {
		t.Fatalf("query canceled: got %v want context.Canceled", err)
	}
	if _, err := model.EmbedDocuments(canceledContext(), []string{"x"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("documents canceled: got %v want context.Canceled", err)
	}
}

func TestRandomFakeDefaultsAndCancellation(t *testing.T) {
	model := NewRandomFake(0)
	vec, err := model.EmbedQuery(context.Background(), "x")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if len(vec) != 8 {
		t.Fatalf("zero size: got %d want 8", len(vec))
	}

	if _, err := model.EmbedQuery(canceledContext(), "x"); !errors.Is(err, context.Canceled) {
		t.Fatalf("query canceled: got %v want context.Canceled", err)
	}
	if _, err := model.EmbedDocuments(canceledContext(), []string{"x"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("documents canceled: got %v want context.Canceled", err)
	}
}

func TestRandomFakeZeroValueLazilyInitializes(t *testing.T) {
	var model RandomFakeEmbedding
	vec, err := model.EmbedQuery(context.Background(), "x")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if len(vec) != 0 {
		t.Fatalf("zero-value size: got %d want 0", len(vec))
	}

	var sized = RandomFakeEmbedding{size: 3}
	docs, err := sized.EmbedDocuments(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	if len(docs) != 2 || len(docs[0]) != 3 || len(docs[1]) != 3 {
		t.Fatalf("documents: %#v", docs)
	}
}

func TestStaticEmbeddingsShortVectorListAndCancellation(t *testing.T) {
	model := Static{
		DocumentVectors: [][]float64{{1, 2}},
		QueryVector:     []float64{5, 6},
	}

	// More texts than configured vectors: extra slots stay nil.
	docs, err := model.EmbedDocuments(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	if len(docs) != 3 {
		t.Fatalf("documents: got %d want 3", len(docs))
	}
	if docs[1] != nil || docs[2] != nil {
		t.Fatalf("expected nil vectors beyond configured list: %#v", docs)
	}

	if _, err := model.EmbedQuery(canceledContext(), "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("query canceled: got %v want context.Canceled", err)
	}
	if _, err := model.EmbedDocuments(canceledContext(), []string{"a"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("documents canceled: got %v want context.Canceled", err)
	}
}

func TestFakeEmbeddingsNormalizeEmptyText(t *testing.T) {
	model := NewFake(8)
	// Text with no tokens keeps the zero vector (normalize's sum == 0 branch).
	vec, err := model.EmbedQuery(context.Background(), "   ")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if len(vec) != 8 {
		t.Fatalf("dimensions: got %d want 8", len(vec))
	}
	for i, v := range vec {
		if v != 0 {
			t.Fatalf("vector[%d]: got %f want 0", i, v)
		}
	}
}
