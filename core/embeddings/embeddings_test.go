package embeddings

import (
	"context"
	"testing"
)

func TestFakeEmbeddingsAreDeterministic(t *testing.T) {
	model := NewFake(16)

	first, err := model.EmbedQuery(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("embed first: %v", err)
	}
	second, err := model.EmbedQuery(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("embed second: %v", err)
	}

	if len(first) != 16 {
		t.Fatalf("dimensions: got %d want 16", len(first))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("vector[%d]: got %f want %f", i, first[i], second[i])
		}
	}
}

func TestDeterministicFakeEmbedding(t *testing.T) {
	model := NewDeterministicFake(6)
	first, err := model.EmbedQuery(context.Background(), "same")
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.EmbedQuery(context.Background(), "same")
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 6 || len(second) != 6 {
		t.Fatalf("unexpected dimensions: %d %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("vectors differ at %d: %f != %f", i, first[i], second[i])
		}
	}
}

func TestRandomFakeEmbeddingPythonFakeParity(t *testing.T) {
	model := NewRandomFake(5)
	first, err := model.EmbedQuery(context.Background(), "same")
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.EmbedQuery(context.Background(), "same")
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 5 || len(second) != 5 {
		t.Fatalf("dimensions: %d %d", len(first), len(second))
	}
	equal := true
	for i := range first {
		if first[i] != second[i] {
			equal = false
			break
		}
	}
	if equal {
		t.Fatalf("random fake returned identical vectors: %#v", first)
	}
	docs, err := model.EmbedDocuments(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 || len(docs[0]) != 5 || len(docs[1]) != 5 {
		t.Fatalf("documents: %#v", docs)
	}
}

func TestStaticEmbeddingsCopiesVectors(t *testing.T) {
	model := Static{
		DocumentVectors: [][]float64{{1, 2}, {3, 4}},
		QueryVector:     []float64{5, 6},
	}
	docs, err := model.EmbedDocuments(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	query, err := model.EmbedQuery(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	docs[0][0] = 99
	query[0] = 99
	againDocs, _ := model.EmbedDocuments(context.Background(), []string{"a"})
	againQuery, _ := model.EmbedQuery(context.Background(), "q")
	if againDocs[0][0] != 1 || againQuery[0] != 5 {
		t.Fatal("static embeddings returned internal vectors")
	}
}
