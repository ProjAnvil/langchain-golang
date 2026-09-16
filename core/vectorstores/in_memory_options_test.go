package vectorstores

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

var _ OptionSearcher = (*InMemory)(nil)

func newFilterTestStore(t *testing.T) *InMemory {
	t.Helper()
	store := NewInMemory(embeddings.NewFake(16))
	_, err := store.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha one", map[string]any{"group": "a", "page": 1}).WithID("one"),
		documents.New("alpha two", map[string]any{"group": "a", "page": 2}).WithID("two"),
		documents.New("alpha three", map[string]any{"group": "b", "page": 3}).WithID("three"),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}
	return store
}

func assertGroup(t *testing.T, docs []documents.Document, wantGroup string) {
	t.Helper()
	if len(docs) == 0 {
		t.Fatalf("expected documents, got none")
	}
	for _, doc := range docs {
		if doc.Metadata["group"] != wantGroup {
			t.Fatalf("doc %q group: got %#v want %q", doc.ID, doc.Metadata["group"], wantGroup)
		}
	}
}

func TestInMemorySimilaritySearchWithOptions(t *testing.T) {
	t.Run("eq filter keeps only matching group", func(t *testing.T) {
		store := newFilterTestStore(t)
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", SearchOptions{
			K:      3,
			Filter: map[string]any{"group": map[string]any{FilterEq: "a"}},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
		assertGroup(t, docs, "a")
	})

	t.Run("shorthand equality", func(t *testing.T) {
		store := newFilterTestStore(t)
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", SearchOptions{
			K:      3,
			Filter: map[string]any{"group": "b"},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 1 || docs[0].ID != "three" {
			t.Fatalf("docs: %#v", docs)
		}
	})

	t.Run("in filter", func(t *testing.T) {
		store := newFilterTestStore(t)
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", SearchOptions{
			K:      3,
			Filter: map[string]any{"group": map[string]any{FilterIn: []any{"b"}}},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 1 || docs[0].ID != "three" {
			t.Fatalf("docs: %#v", docs)
		}
	})

	t.Run("gt filter on numeric metadata", func(t *testing.T) {
		store := newFilterTestStore(t)
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", SearchOptions{
			K:      3,
			Filter: map[string]any{"page": map[string]any{FilterGt: 2}},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 1 || docs[0].ID != "three" {
			t.Fatalf("docs: %#v", docs)
		}
	})

	t.Run("combined fields are ANDed", func(t *testing.T) {
		store := newFilterTestStore(t)
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", SearchOptions{
			K: 3,
			Filter: map[string]any{
				"group": "a",
				"page":  map[string]any{FilterBetween: []any{2, 5}},
			},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 1 || docs[0].ID != "two" {
			t.Fatalf("docs: %#v", docs)
		}
	})

	t.Run("invalid filter errors", func(t *testing.T) {
		store := newFilterTestStore(t)
		_, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", SearchOptions{
			K:      3,
			Filter: map[string]any{"group": map[string]any{"$bogus": "a"}},
		})
		if err == nil {
			t.Fatal("expected error for invalid filter")
		}
	})

	t.Run("score threshold keeps near-identical docs", func(t *testing.T) {
		store := NewInMemory(embeddings.NewFake(16))
		_, err := store.AddDocuments(t.Context(), []documents.Document{
			documents.New("alpha one", nil),
			documents.New("gamma two", nil),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha one", SearchOptions{
			K:              2,
			ScoreThreshold: 0.99,
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 1 || docs[0].PageContent != "alpha one" {
			t.Fatalf("docs: %#v", docs)
		}
	})

	t.Run("zero K falls back to default 4", func(t *testing.T) {
		store := newFilterTestStore(t)
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", SearchOptions{})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 3 {
			t.Fatalf("docs: got %d want 3 (all, default k=4)", len(docs))
		}
	})

	t.Run("canceled context", func(t *testing.T) {
		store := newFilterTestStore(t)
		if _, err := store.SimilaritySearchWithOptions(canceledContext(), "alpha", SearchOptions{K: 1}); err == nil {
			t.Fatal("expected canceled error")
		}
	})
}

func TestInMemoryMMRSearchWithOptions(t *testing.T) {
	t.Run("filter restricts mmr candidates", func(t *testing.T) {
		store := newFilterTestStore(t)
		docs, err := store.MMRSearchWithOptions(t.Context(), "alpha", SearchOptions{
			K:      2,
			FetchK: 3,
			Filter: map[string]any{"group": "a"},
		})
		if err != nil {
			t.Fatalf("mmr: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
		assertGroup(t, docs, "a")
	})

	t.Run("invalid filter errors", func(t *testing.T) {
		store := newFilterTestStore(t)
		_, err := store.MMRSearchWithOptions(t.Context(), "alpha", SearchOptions{
			K:      2,
			Filter: map[string]any{"group": map[string]any{FilterBetween: []any{1}}},
		})
		if err == nil {
			t.Fatal("expected error for invalid filter")
		}
	})

	t.Run("defaults with empty options", func(t *testing.T) {
		store := newFilterTestStore(t)
		docs, err := store.MMRSearchWithOptions(t.Context(), "alpha", SearchOptions{})
		if err != nil {
			t.Fatalf("mmr: %v", err)
		}
		if len(docs) != 3 {
			t.Fatalf("docs: got %d want 3 (k default 4 clamped to store size)", len(docs))
		}
	})
}
