package redisvector

import (
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
	"github.com/projanvil/langchain-golang/standardtests"
)

// e2eAddr returns the address of a Redis Stack server (RediSearch + RedisJSON
// modules), or "" when e2e tests should skip (CI runs without docker).
func e2eAddr() string {
	return strings.TrimSpace(os.Getenv("REDIS_TEST_ADDR"))
}

var e2ePrefixCounter atomic.Int64

func e2ePrefix(t testing.TB) string {
	t.Helper()
	sanitized := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			return r
		default:
			return '_'
		}
	}, t.Name())
	return sanitized + "_" + strconv.FormatInt(e2ePrefixCounter.Add(1), 10)
}

// e2eOptions are the options every e2e store uses: a deterministic fake
// embedder plus the metadata fields the filter suites exercise.
func e2eOptions() []Option {
	return []Option{
		WithEmbedder(embeddings.NewFake(8)),
		WithMetadataField("group", MetadataTag),
		WithMetadataField("page", MetadataNumeric),
		WithMetadataField("name", MetadataTag),
		WithMetadataField("active", MetadataTag),
	}
}

func newE2EStore(t *testing.T, extra ...Option) *Store {
	t.Helper()
	addr := e2eAddr()
	if addr == "" {
		t.Skip("REDIS_TEST_ADDR not set; skipping redisvector e2e (requires Redis Stack)")
	}
	opts := append([]Option{WithAddr(addr)}, e2eOptions()...)
	opts = append(opts, extra...)
	store, err := New(t.Context(), e2ePrefix(t), opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(store.Close)
	return store
}

// TestE2EConformance runs the shared vector store conformance suite (including
// the declarative filter subtests) against a real Redis Stack server.
func TestE2EConformance(t *testing.T) {
	addr := e2eAddr()
	if addr == "" {
		t.Skip("REDIS_TEST_ADDR not set; skipping redisvector e2e (requires Redis Stack)")
	}
	standardtests.RunVectorStoreBasics(t, func(t testing.TB) vectorstores.VectorStore {
		opts := append([]Option{WithAddr(addr)}, e2eOptions()...)
		store, err := New(t.Context(), e2ePrefix(t), opts...)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		t.Cleanup(store.Close)
		return store
	})
}

// TestE2EFilterOperators exercises the declarative DSL operators that map to
// RediSearch clauses against a real server.
func TestE2EFilterOperators(t *testing.T) {
	store := newE2EStore(t)

	_, err := store.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha one", map[string]any{
			"group": "a", "page": 1, "name": "alice",
		}).WithID("one"),
		documents.New("alpha two", map[string]any{
			"group": "b", "page": 2, "name": "bob", "active": true,
		}).WithID("two"),
		documents.New("alpha three", map[string]any{
			"group": "c", "page": 3, "name": "carol", "active": false,
		}).WithID("three"),
		documents.New("alpha four", map[string]any{
			"group": "a", "page": 4, "name": "alvin",
		}).WithID("four"),
	})
	if err != nil {
		t.Fatalf("seed: %v", err)
	}

	tests := []struct {
		name    string
		filter  map[string]any
		wantIDs []string
	}{
		{"eq tag shorthand", map[string]any{"group": "a"}, []string{"four", "one"}},
		{"eq numeric", map[string]any{"page": map[string]any{vectorstores.FilterEq: 2}}, []string{"two"}},
		{"eq bool", map[string]any{"active": map[string]any{vectorstores.FilterEq: true}}, []string{"two"}},
		{"ne tag", map[string]any{"group": map[string]any{vectorstores.FilterNe: "a"}}, []string{"three", "two"}},
		{"gt numeric", map[string]any{"page": map[string]any{vectorstores.FilterGt: 2}}, []string{"four", "three"}},
		{"gte numeric", map[string]any{"page": map[string]any{vectorstores.FilterGte: 3}}, []string{"four", "three"}},
		{"lt numeric", map[string]any{"page": map[string]any{vectorstores.FilterLt: 3}}, []string{"one", "two"}},
		{"lte numeric", map[string]any{"page": map[string]any{vectorstores.FilterLte: 2}}, []string{"one", "two"}},
		{"in tag", map[string]any{"group": map[string]any{vectorstores.FilterIn: []any{"a", "b"}}}, []string{"four", "one", "two"}},
		{"nin tag", map[string]any{"group": map[string]any{vectorstores.FilterNin: []any{"a", "b"}}}, []string{"three"}},
		{"between numeric", map[string]any{"page": map[string]any{vectorstores.FilterBetween: []any{2, 3}}}, []string{"three", "two"}},
		{"exists true", map[string]any{"active": map[string]any{vectorstores.FilterExists: true}}, []string{"three", "two"}},
		{"exists false", map[string]any{"active": map[string]any{vectorstores.FilterExists: false}}, []string{"four", "one"}},
		{
			"multiple fields and",
			map[string]any{"group": "a", "page": map[string]any{vectorstores.FilterGt: 2}},
			[]string{"four"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
				K:      10,
				Filter: tt.filter,
			})
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			gotIDs := make([]string, 0, len(docs))
			for _, doc := range docs {
				gotIDs = append(gotIDs, doc.ID)
			}
			if !equalIDSlices(gotIDs, tt.wantIDs) {
				t.Fatalf("ids: got %v want %v", gotIDs, tt.wantIDs)
			}
		})
	}

	t.Run("unsupported like errors", func(t *testing.T) {
		if _, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
			K:      10,
			Filter: map[string]any{"name": map[string]any{vectorstores.FilterLike: "al%"}},
		}); err == nil {
			t.Fatal("expected error for $like on Redis")
		}
	})

	t.Run("delete with filter", func(t *testing.T) {
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "a"}); err != nil {
			t.Fatalf("DeleteWithFilter: %v", err)
		}
		docs, err := store.GetByIDs(t.Context(), []string{"one", "four"})
		if err != nil {
			t.Fatalf("GetByIDs: %v", err)
		}
		if len(docs) != 0 {
			t.Fatalf("docs after filtered delete: %d want 0", len(docs))
		}
	})
}

// TestE2EDistanceMetrics runs a search round-trip with the L2 metric to cover
// the alternative DISTANCE_METRIC mapping.
func TestE2EDistanceMetrics(t *testing.T) {
	store := newE2EStore(t, WithDistanceMetric(DistanceL2))
	if _, err := store.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha beta", nil),
		documents.New("gamma delta", nil),
	}); err != nil {
		t.Fatalf("add: %v", err)
	}
	results, err := store.SimilaritySearchWithScore(t.Context(), "alpha", 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 || results[0].Document.PageContent != "alpha beta" {
		t.Fatalf("top result: %#v", results)
	}
}

func equalIDSlices(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	counts := map[string]int{}
	for _, id := range got {
		counts[id]++
	}
	for _, id := range want {
		counts[id]--
		if counts[id] < 0 {
			return false
		}
	}
	return true
}
