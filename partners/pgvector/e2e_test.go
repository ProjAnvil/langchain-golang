package pgvector

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

// e2eDSN returns the connection string for a real Postgres with the pgvector
// extension, or "" when e2e tests should skip (CI runs without docker).
func e2eDSN() string {
	return strings.TrimSpace(os.Getenv("PGVECTOR_TEST_DSN"))
}

var e2eCollectionCounter atomic.Int64

// e2eCollectionName derives a fresh, identifier-safe table name per subtest so
// parallel e2e runs do not collide.
func e2eCollectionName(t testing.TB) string {
	t.Helper()
	sanitized := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			return r
		default:
			return '_'
		}
	}, t.Name())
	return sanitized + "_" + strconv.FormatInt(e2eCollectionCounter.Add(1), 10)
}

func newE2EStore(t *testing.T) *Store {
	t.Helper()
	dsn := e2eDSN()
	if dsn == "" {
		t.Skip("PGVECTOR_TEST_DSN not set; skipping pgvector e2e (requires a Postgres with the pgvector extension)")
	}
	store, err := New(t.Context(), e2eCollectionName(t),
		WithDSN(dsn),
		WithEmbedder(embeddings.NewFake(8)),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(store.Close)
	return store
}

// TestE2EConformance runs the shared vector store conformance suite (including
// the declarative filter subtests) against a real Postgres with pgvector.
func TestE2EConformance(t *testing.T) {
	dsn := e2eDSN()
	if dsn == "" {
		t.Skip("PGVECTOR_TEST_DSN not set; skipping pgvector e2e (requires a Postgres with the pgvector extension)")
	}
	standardtests.RunVectorStoreBasics(t, func(t testing.TB) vectorstores.VectorStore {
		store, err := New(t.Context(), e2eCollectionName(t),
			WithDSN(dsn),
			WithEmbedder(embeddings.NewFake(8)),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		t.Cleanup(store.Close)
		return store
	})
}

// TestE2EFilterOperators exercises every DSL operator against jsonb SQL
// translation on a real database.
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
		{"eq shorthand", map[string]any{"group": "a"}, []string{"four", "one"}},
		{"eq numeric", map[string]any{"page": map[string]any{vectorstores.FilterEq: 2}}, []string{"two"}},
		{"eq bool", map[string]any{"active": map[string]any{vectorstores.FilterEq: true}}, []string{"two"}},
		{"ne", map[string]any{"group": map[string]any{vectorstores.FilterNe: "a"}}, []string{"three", "two"}},
		{"gt", map[string]any{"page": map[string]any{vectorstores.FilterGt: 2}}, []string{"four", "three"}},
		{"gte", map[string]any{"page": map[string]any{vectorstores.FilterGte: 3}}, []string{"four", "three"}},
		{"lt", map[string]any{"page": map[string]any{vectorstores.FilterLt: 3}}, []string{"one", "two"}},
		{"lte", map[string]any{"page": map[string]any{vectorstores.FilterLte: 2}}, []string{"one", "two"}},
		{"in", map[string]any{"group": map[string]any{vectorstores.FilterIn: []any{"a", "b"}}}, []string{"four", "one", "two"}},
		{"nin", map[string]any{"group": map[string]any{vectorstores.FilterNin: []any{"a", "b"}}}, []string{"three"}},
		{"between", map[string]any{"page": map[string]any{vectorstores.FilterBetween: []any{2, 3}}}, []string{"three", "two"}},
		{"exists true", map[string]any{"active": map[string]any{vectorstores.FilterExists: true}}, []string{"three", "two"}},
		{"exists false", map[string]any{"active": map[string]any{vectorstores.FilterExists: false}}, []string{"four", "one"}},
		{"like", map[string]any{"name": map[string]any{vectorstores.FilterLike: "al%"}}, []string{"four", "one"}},
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

// TestE2EDistanceMetrics runs conformance with the L2 and inner-product
// distance metrics to cover the alternative operators and opclasses.
func TestE2EDistanceMetrics(t *testing.T) {
	for _, metric := range []DistanceMetric{DistanceL2, DistanceIP} {
		t.Run(string(metric), func(t *testing.T) {
			dsn := e2eDSN()
			if dsn == "" {
				t.Skip("PGVECTOR_TEST_DSN not set; skipping pgvector e2e")
			}
			store, err := New(t.Context(), e2eCollectionName(t),
				WithDSN(dsn),
				WithEmbedder(embeddings.NewFake(8)),
				WithDistanceMetric(metric),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			t.Cleanup(store.Close)

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
		})
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
