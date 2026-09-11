package store

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
)

// fakeEmbedder is a deterministic embeddings.Embeddings for semantic-store
// tests: each rune of the text increments bucket int(r)%dims. The tests use
// texts built only from 'a'-'h' (runes 97..104), which occupy distinct buckets
// mod 8, so expected cosine similarities are computable by hand:
//
//	cos("aa", "aaa") = 1; cos("aa", "aab") = 2/sqrt(5); cos("aa", "bbb") = 0.
//
// It also records every document text passed to EmbedDocuments and can inject
// EmbedDocuments / EmbedQuery failures and vector-count mismatches.
type fakeEmbedder struct {
	dims     int
	mu       sync.Mutex
	texts    []string
	docErr   error
	docShort bool // return one vector fewer than texts (count mismatch)
	queryErr error
}

var _ embeddings.Embeddings = (*fakeEmbedder)(nil)

// EmbedDocuments embeds all documents, recording the texts.
func (f *fakeEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.mu.Lock()
	f.texts = append(f.texts, texts...)
	f.mu.Unlock()
	if f.docErr != nil {
		return nil, f.docErr
	}
	vectors := make([][]float64, len(texts))
	for i, text := range texts {
		vectors[i] = f.embed(text)
	}
	if f.docShort && len(vectors) > 0 {
		vectors = vectors[:len(vectors)-1]
	}
	return vectors, nil
}

// EmbedQuery embeds one query.
func (f *fakeEmbedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if f.queryErr != nil {
		return nil, f.queryErr
	}
	return f.embed(text), nil
}

func (f *fakeEmbedder) embed(text string) []float64 {
	vector := make([]float64, f.dims)
	for _, r := range text {
		if r < 0 {
			r = -r
		}
		vector[int(r)%f.dims]++
	}
	return vector
}

// recorded returns the document texts embedded so far, in order.
func (f *fakeEmbedder) recorded() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.texts...)
}

// newIndexedTestStore builds an InMemoryStore indexing the "text" field with a
// recording 8-dim fakeEmbedder, mirroring Python's
// InMemoryStore(index={"embed": ..., "fields": ["text"]}).
func newIndexedTestStore(t *testing.T) (*InMemoryStore, *fakeEmbedder) {
	t.Helper()
	e := &fakeEmbedder{dims: 8}
	return NewInMemoryStoreWithIndex(IndexConfig{Embed: e, Fields: []string{"text"}}), e
}

// TestIndexedSearchRanksBySimilarity: Put embeds the indexed field and Search
// with a Query ranks candidates by cosine similarity descending (with
// limit/offset slicing the ranked list), mirroring Python's _batch_search.
func TestIndexedSearchRanksBySimilarity(t *testing.T) {
	ctx := context.Background()
	s, _ := newIndexedTestStore(t)
	for key, text := range map[string]string{"k1": "aaa", "k2": "bbb", "k3": "aab"} {
		if err := s.Put(ctx, []string{"docs"}, key, map[string]any{"text": text}, nil); err != nil {
			t.Fatalf("Put %s: %v", key, err)
		}
	}

	got, err := s.Search(ctx, []string{"docs"}, SearchOptions{Query: "aa"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if want := []string{"k1", "k3", "k2"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Fatalf("ranked keys = %v, want %v", keysOfItems(got), want)
	}
	wantScores := []float64{1, 2 / math.Sqrt(5), 0}
	for i, w := range wantScores {
		if math.Abs(got[i].Score-w) > 1e-9 {
			t.Errorf("results[%d].Score = %v, want %v", i, got[i].Score, w)
		}
	}

	// Limit truncates the ranked list.
	got, err = s.Search(ctx, []string{"docs"}, SearchOptions{Query: "aa", Limit: 2})
	if err != nil {
		t.Fatalf("Search limit=2: %v", err)
	}
	if want := []string{"k1", "k3"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Errorf("Search(limit=2) = %v, want %v", keysOfItems(got), want)
	}

	// Offset skips within the ranked list.
	got, err = s.Search(ctx, []string{"docs"}, SearchOptions{Query: "aa", Offset: 1})
	if err != nil {
		t.Fatalf("Search offset=1: %v", err)
	}
	if want := []string{"k3", "k2"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Errorf("Search(offset=1) = %v, want %v", keysOfItems(got), want)
	}
}

// TestIndexedSearchFilterCombinesWithQuery: the filter narrows candidates
// first; cosine ranking orders what survives, mirroring Python's _filter_items
// followed by _batch_search.
func TestIndexedSearchFilterCombinesWithQuery(t *testing.T) {
	ctx := context.Background()
	s, _ := newIndexedTestStore(t)
	puts := []struct {
		key  string
		kind string
		text string
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

	// Without the filter, k3 ranks first.
	got, err := s.Search(ctx, []string{"docs"}, SearchOptions{Query: "aa"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if want := []string{"k3", "k2", "k1"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Fatalf("unfiltered ranking = %v, want %v", keysOfItems(got), want)
	}

	// With the filter, k3 is excluded and the rest are ranked.
	got, err = s.Search(ctx, []string{"docs"}, SearchOptions{
		Query:  "aa",
		Filter: map[string]any{"kind": "x"},
	})
	if err != nil {
		t.Fatalf("Search with filter: %v", err)
	}
	if want := []string{"k2", "k1"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Errorf("filtered ranking = %v, want %v (k3 excluded by filter)", keysOfItems(got), want)
	}
}

// TestQueryIgnoredWithoutIndex: a store built without an index accepts a Query
// without error and ignores it — results keep the deterministic (namespace,
// key) order with zero scores. Python parity: InMemoryStore(index=None) drops
// the query in _batch_search's else branch rather than raising.
func TestQueryIgnoredWithoutIndex(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	for _, key := range []string{"b", "a", "c"} {
		if err := s.Put(ctx, []string{"ns"}, key, map[string]any{"text": key}, nil); err != nil {
			t.Fatalf("Put %s: %v", key, err)
		}
	}
	got, err := s.Search(ctx, []string{"ns"}, SearchOptions{Query: "anything"})
	if err != nil {
		t.Fatalf("Search with query on a non-indexed store: %v", err)
	}
	if want := []string{"a", "b", "c"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Errorf("Search(query, no index) = %v, want %v in deterministic order (query ignored)", keysOfItems(got), want)
	}
	for _, r := range got {
		if r.Score != 0 {
			t.Errorf("result %s scored %v on a non-indexed store, want 0", r.Key, r.Score)
		}
	}
}

// TestIndexedPutUpdateRecomputesVectors: re-Putting an item re-embeds its
// indexed fields, replacing the old vectors (the previous text no longer
// ranks).
func TestIndexedPutUpdateRecomputesVectors(t *testing.T) {
	ctx := context.Background()
	s, _ := newIndexedTestStore(t)
	ns := []string{"docs"}

	if err := s.Put(ctx, ns, "k", map[string]any{"text": "aaa"}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.Search(ctx, ns, SearchOptions{Query: "aa"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].Key != "k" || math.Abs(got[0].Score-1) > 1e-9 {
		t.Fatalf("after Put(aaa), Search(aa) = %+v, want [k] with score 1", got)
	}

	if err := s.Put(ctx, ns, "k", map[string]any{"text": "bbb"}, nil); err != nil {
		t.Fatalf("Put (update): %v", err)
	}
	got, err = s.Search(ctx, ns, SearchOptions{Query: "aa"})
	if err != nil {
		t.Fatalf("Search after update: %v", err)
	}
	if len(got) != 1 || got[0].Key != "k" || got[0].Score != 0 {
		t.Fatalf("after update to bbb, Search(aa) = %+v, want [k] with score 0 (old vector replaced)", got)
	}
	got, err = s.Search(ctx, ns, SearchOptions{Query: "bb"})
	if err != nil {
		t.Fatalf("Search after update: %v", err)
	}
	if len(got) != 1 || got[0].Key != "k" || math.Abs(got[0].Score-1) > 1e-9 {
		t.Fatalf("after update to bbb, Search(bb) = %+v, want [k] with score 1", got)
	}
}

// TestIndexedSearchScorelessFill: items whose fields were never embedded (the
// indexed field is absent) are appended after the ranked results with score 0,
// and only when the ranked window is shorter than Limit — mirroring Python's
// _batch_search corner case.
func TestIndexedSearchScorelessFill(t *testing.T) {
	ctx := context.Background()
	s, _ := newIndexedTestStore(t)
	if err := s.Put(ctx, []string{"docs"}, "k1", map[string]any{"text": "aaa"}, nil); err != nil {
		t.Fatalf("Put k1: %v", err)
	}
	if err := s.Put(ctx, []string{"docs"}, "k2", map[string]any{"note": "no text field"}, nil); err != nil {
		t.Fatalf("Put k2: %v", err)
	}
	if err := s.Put(ctx, []string{"docs"}, "k3", map[string]any{"text": "bbb"}, nil); err != nil {
		t.Fatalf("Put k3: %v", err)
	}

	// Default limit (10) > ranked count (2): the scoreless k2 fills the tail.
	got, err := s.Search(ctx, []string{"docs"}, SearchOptions{Query: "aa"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if want := []string{"k1", "k3", "k2"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Fatalf("Search(query) = %v, want %v (k2 filled last with score 0)", keysOfItems(got), want)
	}
	if got[2].Score != 0 {
		t.Errorf("filled item k2 score = %v, want 0", got[2].Score)
	}

	// Limit=2 caps the ranked results; the scoreless item does NOT fill.
	got, err = s.Search(ctx, []string{"docs"}, SearchOptions{Query: "aa", Limit: 2})
	if err != nil {
		t.Fatalf("Search limit=2: %v", err)
	}
	if want := []string{"k1", "k3"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Errorf("Search(limit=2) = %v, want %v (no scoreless fill at capacity)", keysOfItems(got), want)
	}
}

// TestIndexedPutEmbedErrorPropagates: an EmbedDocuments failure (or a vector
// count mismatch) fails the Put and leaves the item UNSTORED — mirroring
// Python's batch, which embeds before applying put ops.
func TestIndexedPutEmbedErrorPropagates(t *testing.T) {
	ctx := context.Background()
	s, e := newIndexedTestStore(t)

	e.docErr = errors.New("embed boom")
	err := s.Put(ctx, []string{"docs"}, "k", map[string]any{"text": "aaa"}, nil)
	if err == nil || !strings.Contains(err.Error(), "embed boom") {
		t.Fatalf("Put with failing embedder = %v, want the embed error", err)
	}
	if item, err := s.Get(ctx, []string{"docs"}, "k"); err != nil || item != nil {
		t.Errorf("item stored despite embed failure: item=%v err=%v, want nil,nil", item, err)
	}

	e.docErr = nil
	e.docShort = true
	if err := s.Put(ctx, []string{"docs"}, "k", map[string]any{"text": "aaa", "title": "bb"}, nil); err == nil {
		t.Fatal("Put with a vector-count mismatch succeeded; want an error")
	}
	if item, err := s.Get(ctx, []string{"docs"}, "k"); err != nil || item != nil {
		t.Errorf("item stored despite count mismatch: item=%v err=%v, want nil,nil", item, err)
	}
}

// TestIndexedSearchEmbedErrorPropagates: an EmbedQuery failure fails the
// Search (Python raises through the embed futures); without a Query the
// embedder is not consulted at all.
func TestIndexedSearchEmbedErrorPropagates(t *testing.T) {
	ctx := context.Background()
	s, e := newIndexedTestStore(t)
	if err := s.Put(ctx, []string{"docs"}, "k", map[string]any{"text": "aaa"}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}

	e.queryErr = errors.New("query boom")
	if _, err := s.Search(ctx, []string{"docs"}, SearchOptions{Query: "aa"}); err == nil {
		t.Fatal("Search with failing EmbedQuery succeeded; want an error")
	}

	// No query: no embedding, no error.
	got, err := s.Search(ctx, []string{"docs"}, SearchOptions{})
	if err != nil {
		t.Fatalf("Search without query: %v", err)
	}
	if len(got) != 1 || got[0].Key != "k" {
		t.Errorf("Search without query = %v, want [k]", keysOfItems(got))
	}
	if n := len(e.recorded()); n != 1 {
		t.Errorf("embedder recorded %d texts after a query-less Search, want 1 (query embedding only)", n)
	}
}

// TestIndexedPutIndexOverride: Put's index argument overrides the configured
// fields for that call — nil uses the config, a non-nil slice embeds exactly
// those fields, and an empty non-nil slice embeds nothing (standing in for
// Python's put(..., index=False)).
func TestIndexedPutIndexOverride(t *testing.T) {
	ctx := context.Background()
	s, e := newIndexedTestStore(t)

	if err := s.Put(ctx, []string{"docs"}, "k1", map[string]any{"text": "aaa", "title": "hhh"}, nil); err != nil {
		t.Fatalf("Put k1: %v", err)
	}
	if err := s.Put(ctx, []string{"docs"}, "k2", map[string]any{"text": "bbb", "title": "ccc"}, []string{"title"}); err != nil {
		t.Fatalf("Put k2: %v", err)
	}
	if err := s.Put(ctx, []string{"docs"}, "k3", map[string]any{"text": "ddd"}, []string{}); err != nil {
		t.Fatalf("Put k3: %v", err)
	}

	if want := []string{"aaa", "ccc"}; !reflect.DeepEqual(e.recorded(), want) {
		t.Fatalf("embedded texts = %v, want %v (k3 skipped, k2 title-only)", e.recorded(), want)
	}

	// k2's title vector ranks it first for a title-ish query.
	got, err := s.Search(ctx, []string{"docs"}, SearchOptions{Query: "cc"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) == 0 || got[0].Key != "k2" {
		t.Fatalf("Search(cc) first = %+v, want k2 (title-indexed)", got)
	}
	// k2's text was NOT embedded: a query matching only its text cannot rank
	// it first (k1 wins the deterministic tie at score 0).
	got, err = s.Search(ctx, []string{"docs"}, SearchOptions{Query: "bb"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) == 0 || got[0].Key != "k1" {
		t.Fatalf("Search(bb) first = %+v, want k1 (k2's text field was not indexed)", got)
	}
}

// TestIndexedDefaultFieldIsWholeValue: an IndexConfig without Fields embeds
// the sorted-key JSON of the whole value, mirroring Python's default
// fields=["$"] → json.dumps(value, sort_keys=True).
func TestIndexedDefaultFieldIsWholeValue(t *testing.T) {
	ctx := context.Background()
	e := &fakeEmbedder{dims: 8}
	s := NewInMemoryStoreWithIndex(IndexConfig{Embed: e})
	if err := s.Put(ctx, []string{"docs"}, "k", map[string]any{"text": "aaa"}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if want := []string{`{"text":"aaa"}`}; !reflect.DeepEqual(e.recorded(), want) {
		t.Fatalf("embedded texts = %v, want %v (JSON of the whole value)", e.recorded(), want)
	}
	// The JSON-encoded value is searchable: querying its exact text scores 1.
	got, err := s.Search(ctx, []string{"docs"}, SearchOptions{Query: `{"text":"aaa"}`})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].Key != "k" || math.Abs(got[0].Score-1) > 1e-9 {
		t.Fatalf("Search(exact JSON) = %+v, want [k] with score 1", got)
	}
}

// TestNewInMemoryStoreWithIndexRequiresEmbed: constructing an indexed store
// without an embedder panics, mirroring the ValueError Python's
// ensure_embeddings raises for embed=None.
func TestNewInMemoryStoreWithIndexRequiresEmbed(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("NewInMemoryStoreWithIndex without Embed did not panic")
		}
	}()
	_ = NewInMemoryStoreWithIndex(IndexConfig{Dims: 8})
}

// TestIndexedStoreStillPassesBaseContract: an indexed store must satisfy the
// plain Store contract too — in particular a query-less Search keeps the
// deterministic (namespace, key) order.
func TestIndexedStoreStillPassesBaseContract(t *testing.T) {
	ctx := context.Background()
	s, _ := newIndexedTestStore(t)
	for _, key := range []string{"b", "a"} {
		if err := s.Put(ctx, []string{"ns"}, key, map[string]any{"text": key + key + key}, nil); err != nil {
			t.Fatalf("Put %s: %v", key, err)
		}
	}
	got, err := s.Search(ctx, []string{"ns"}, SearchOptions{})
	if err != nil {
		t.Fatalf("Search without query: %v", err)
	}
	if want := []string{"a", "b"}; !reflect.DeepEqual(keysOfItems(got), want) {
		t.Errorf("query-less Search on indexed store = %v, want %v", keysOfItems(got), want)
	}
	for _, r := range got {
		if r.Score != 0 {
			t.Errorf("query-less Search result %s score = %v, want 0", r.Key, r.Score)
		}
	}
}
