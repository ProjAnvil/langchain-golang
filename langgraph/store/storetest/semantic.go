package storetest

import (
	"context"
	"math"
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/langgraph/store"
)

// runeEmbedder is the deterministic Embeddings fake injected by RunSemantic:
// each rune of the text increments bucket int(rune)%dims. The suite's texts
// are built only from 'a'-'h' (runes 97..104), which occupy distinct buckets
// mod 8, so expected cosine similarities and orderings are computable by hand:
//
//	cos("aa", "aaa") = 1; cos("aa", "aab") = 2/sqrt(5); cos("aa", "bbb") = 0.
type runeEmbedder struct {
	dims int
}

var _ embeddings.Embeddings = runeEmbedder{}

// EmbedDocuments embeds all documents.
func (e runeEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	vectors := make([][]float64, len(texts))
	for i, text := range texts {
		vectors[i] = e.embed(text)
	}
	return vectors, nil
}

// EmbedQuery embeds one query.
func (e runeEmbedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return e.embed(text), nil
}

func (e runeEmbedder) embed(text string) []float64 {
	vector := make([]float64, e.dims)
	for _, r := range text {
		if r < 0 {
			r = -r
		}
		vector[int(r)%e.dims]++
	}
	return vector
}

// RunSemantic executes the OPTIONAL semantic-search conformance suite as
// subtests of t. newIndexedStore must return an EMPTY store.Store that indexes
// the "text" value field with the supplied deterministic embedder — for the
// InMemoryStore: store.NewInMemoryStoreWithIndex(store.IndexConfig{Embed:
// embed, Fields: []string{"text"}}), mirroring Python's
// InMemoryStore(index={"embed": ..., "fields": ["text"]}).
//
// The suite asserts the semantic contract layered on top of Run: Query results
// ranked by cosine similarity descending (with exact expected scores under the
// deterministic embedder), Filter narrowing candidates before ranking, and a
// re-Put recomputing the item's vectors. Implementations without semantic
// support simply do not invoke this suite (Run already covers the
// query-ignored-without-index behavior).
func RunSemantic(t *testing.T, newIndexedStore func(t *testing.T, embed embeddings.Embeddings) store.Store) {
	t.Helper()
	embed := runeEmbedder{dims: 8}
	t.Run("query_ranks_by_similarity", func(t *testing.T) { testSemanticRanking(t, newIndexedStore, embed) })
	t.Run("query_combines_with_filter", func(t *testing.T) { testSemanticFilter(t, newIndexedStore, embed) })
	t.Run("put_update_recomputes_vectors", func(t *testing.T) { testSemanticUpdate(t, newIndexedStore, embed) })
}

// testSemanticRanking: Put embeds the indexed field; Search with a Query
// returns the candidates ranked by cosine similarity, descending, with the
// scores carried on SearchItem.Score.
func testSemanticRanking(t *testing.T, newIndexedStore func(t *testing.T, embed embeddings.Embeddings) store.Store, embedder embeddings.Embeddings) {
	t.Helper()
	ctx := context.Background()
	s := newIndexedStore(t, embedder)
	for key, text := range map[string]string{"k1": "aaa", "k2": "bbb", "k3": "aab"} {
		if err := s.Put(ctx, []string{"docs"}, key, map[string]any{"text": text}, nil); err != nil {
			t.Fatalf("Put %s: %v", key, err)
		}
	}

	results, err := s.Search(ctx, []string{"docs"}, store.SearchOptions{Query: "aa"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got := keysOf(results); got != "k1,k3,k2" {
		t.Fatalf("Search(query) keys = %q, want %q (cosine descending)", got, "k1,k3,k2")
	}
	wantScores := []float64{1, 2 / math.Sqrt(5), 0}
	for i, want := range wantScores {
		if math.Abs(results[i].Score-want) > 1e-9 {
			t.Errorf("results[%d].Score = %v, want %v", i, results[i].Score, want)
		}
	}
}

// testSemanticFilter: the filter narrows the candidate set first; cosine
// ranking orders whatever survives (the best-matching item can be excluded by
// the filter).
func testSemanticFilter(t *testing.T, newIndexedStore func(t *testing.T, embed embeddings.Embeddings) store.Store, embedder embeddings.Embeddings) {
	t.Helper()
	ctx := context.Background()
	s := newIndexedStore(t, embedder)
	puts := []struct {
		key, kind, text string
	}{
		{"k1", "x", "bbb"},
		{"k2", "x", "aab"},
		{"k3", "y", "aaa"}, // the best match for "aa", but kind=y
	}
	for _, p := range puts {
		if err := s.Put(ctx, []string{"docs"}, p.key, map[string]any{"kind": p.kind, "text": p.text}, nil); err != nil {
			t.Fatalf("Put %s: %v", p.key, err)
		}
	}

	results, err := s.Search(ctx, []string{"docs"}, store.SearchOptions{
		Query:  "aa",
		Filter: map[string]any{"kind": "x"},
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got := keysOf(results); got != "k2,k1" {
		t.Fatalf("Search(query, filter kind=x) keys = %q, want %q (k3 excluded by the filter)", got, "k2,k1")
	}
}

// testSemanticUpdate: re-Putting an item re-embeds its indexed fields; the old
// text no longer ranks (score 0) and the new text scores maximally.
func testSemanticUpdate(t *testing.T, newIndexedStore func(t *testing.T, embed embeddings.Embeddings) store.Store, embedder embeddings.Embeddings) {
	t.Helper()
	ctx := context.Background()
	s := newIndexedStore(t, embedder)
	ns := []string{"docs"}

	if err := s.Put(ctx, ns, "k", map[string]any{"text": "aaa"}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Put(ctx, ns, "k", map[string]any{"text": "bbb"}, nil); err != nil {
		t.Fatalf("Put (update): %v", err)
	}

	results, err := s.Search(ctx, ns, store.SearchOptions{Query: "aa"})
	if err != nil {
		t.Fatalf("Search(aa): %v", err)
	}
	if len(results) != 1 || results[0].Key != "k" || results[0].Score != 0 {
		t.Fatalf("after updating aaa→bbb, Search(aa) = %+v, want [k] with score 0 (vector recomputed)", results)
	}
	results, err = s.Search(ctx, ns, store.SearchOptions{Query: "bb"})
	if err != nil {
		t.Fatalf("Search(bb): %v", err)
	}
	if len(results) != 1 || results[0].Key != "k" || math.Abs(results[0].Score-1) > 1e-9 {
		t.Fatalf("after updating aaa→bbb, Search(bb) = %+v, want [k] with score 1", results)
	}
}
