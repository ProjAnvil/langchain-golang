package chroma

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// marshalEqual compares two values through their JSON encoding; encoding/json
// writes map keys in sorted order, so this is a stable deep-equality check
// for the decoded-JSON shapes produced by translateFilter.
func marshalEqual(a, b any) bool {
	encodedA, errA := json.Marshal(a)
	encodedB, errB := json.Marshal(b)
	return errA == nil && errB == nil && string(encodedA) == string(encodedB)
}

func containsStr(s, substr string) bool {
	return strings.Contains(s, substr)
}

func TestTranslateFilter(t *testing.T) {
	tests := []struct {
		name    string
		filter  map[string]any
		want    map[string]any
		wantErr bool
	}{
		{
			name:   "nil filter",
			filter: nil,
			want:   nil,
		},
		{
			name:   "empty filter",
			filter: map[string]any{},
			want:   nil,
		},
		{
			name:   "shorthand equality",
			filter: map[string]any{"color": "red"},
			want:   map[string]any{"color": map[string]any{"$eq": "red"}},
		},
		{
			name:   "eq explicit",
			filter: map[string]any{"color": map[string]any{vectorstores.FilterEq: "red"}},
			want:   map[string]any{"color": map[string]any{"$eq": "red"}},
		},
		{
			name:   "ne",
			filter: map[string]any{"color": map[string]any{vectorstores.FilterNe: "red"}},
			want:   map[string]any{"color": map[string]any{"$ne": "red"}},
		},
		{
			name:   "gt",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterGt: 2}},
			want:   map[string]any{"page": map[string]any{"$gt": 2}},
		},
		{
			name:   "gte",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterGte: 2}},
			want:   map[string]any{"page": map[string]any{"$gte": 2}},
		},
		{
			name:   "lt",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterLt: 2}},
			want:   map[string]any{"page": map[string]any{"$lt": 2}},
		},
		{
			name:   "lte",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterLte: 2}},
			want:   map[string]any{"page": map[string]any{"$lte": 2}},
		},
		{
			name:   "in",
			filter: map[string]any{"color": map[string]any{vectorstores.FilterIn: []any{"red", "blue"}}},
			want:   map[string]any{"color": map[string]any{"$in": []any{"red", "blue"}}},
		},
		{
			name:   "nin",
			filter: map[string]any{"color": map[string]any{vectorstores.FilterNin: []any{"red"}}},
			want:   map[string]any{"color": map[string]any{"$nin": []any{"red"}}},
		},
		{
			name:   "between decomposes to and of gte and lte",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterBetween: []any{1, 5}}},
			want: map[string]any{"$and": []any{
				map[string]any{"page": map[string]any{"$gte": 1}},
				map[string]any{"page": map[string]any{"$lte": 5}},
			}},
		},
		{
			name: "multiple fields join with and",
			filter: map[string]any{
				"color": "red",
				"page":  map[string]any{vectorstores.FilterGt: 2},
			},
			want: map[string]any{"$and": []any{
				map[string]any{"color": map[string]any{"$eq": "red"}},
				map[string]any{"page": map[string]any{"$gt": 2}},
			}},
		},
		{
			name:    "exists is unsupported",
			filter:  map[string]any{"color": map[string]any{vectorstores.FilterExists: true}},
			wantErr: true,
		},
		{
			name:    "like is unsupported",
			filter:  map[string]any{"color": map[string]any{vectorstores.FilterLike: "%red%"}},
			wantErr: true,
		},
		{
			name:    "invalid operator rejected by shared validation",
			filter:  map[string]any{"color": map[string]any{"$bogus": "red"}},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := translateFilter(tt.filter)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("translateFilter(%#v): expected error", tt.filter)
				}
				return
			}
			if err != nil {
				t.Fatalf("translateFilter(%#v): %v", tt.filter, err)
			}
			if !equalJSONMaps(got, tt.want) {
				t.Fatalf("translateFilter: got %#v want %#v", got, tt.want)
			}
		})
	}
}

// equalJSONMaps compares decoded-JSON-shaped maps (nested slices instead of
// typed ones) so the test can express expectations as plain literals.
func equalJSONMaps(a, b map[string]any) bool {
	return marshalEqual(a, b)
}

func TestTranslateFilterUnsupportedErrorMentionsOperator(t *testing.T) {
	_, err := translateFilter(map[string]any{"color": map[string]any{vectorstores.FilterExists: true}})
	if err == nil {
		t.Fatal("expected error")
	}
	if !containsStr(err.Error(), "$exists") {
		t.Fatalf("error should name the unsupported operator: %v", err)
	}
}

func TestStoreSimilaritySearchWithOptions(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store, err := New(
		t.Context(),
		"langchain",
		embeddings.NewFake(16),
		WithBaseURL(server.URL),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	_, err = store.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha one", map[string]any{"group": "a", "page": 1}).WithID("one"),
		documents.New("alpha two", map[string]any{"group": "a", "page": 2}).WithID("two"),
		documents.New("alpha three", map[string]any{"group": "b", "page": 3}).WithID("three"),
	})
	if err != nil {
		t.Fatalf("add: %v", err)
	}

	t.Run("eq filter is translated to a where clause", func(t *testing.T) {
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
			K:      3,
			Filter: map[string]any{"group": "a"},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
		for _, doc := range docs {
			if doc.Metadata["group"] != "a" {
				t.Fatalf("unexpected doc: %#v", doc)
			}
		}
		var payload queryRequest
		server.decodeLastBody(t, &payload)
		want := map[string]any{"group": map[string]any{"$eq": "a"}}
		if !marshalEqual(payload.Where, want) {
			t.Fatalf("where clause: got %#v want %#v", payload.Where, want)
		}
	})

	t.Run("in and gt filters", func(t *testing.T) {
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
			K:      3,
			Filter: map[string]any{"group": map[string]any{vectorstores.FilterIn: []any{"b"}}},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 1 || docs[0].ID != "three" {
			t.Fatalf("docs: %#v", docs)
		}

		docs, err = store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
			K:      3,
			Filter: map[string]any{"page": map[string]any{vectorstores.FilterGt: 2}},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 1 || docs[0].ID != "three" {
			t.Fatalf("docs: %#v", docs)
		}
	})

	t.Run("between decomposes into and-clause", func(t *testing.T) {
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
			K:      3,
			Filter: map[string]any{"page": map[string]any{vectorstores.FilterBetween: []any{2, 5}}},
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
		var payload queryRequest
		server.decodeLastBody(t, &payload)
		want := map[string]any{"$and": []any{
			map[string]any{"page": map[string]any{"$gte": 2}},
			map[string]any{"page": map[string]any{"$lte": 5}},
		}}
		if !marshalEqual(payload.Where, want) {
			t.Fatalf("where clause: got %#v want %#v", payload.Where, want)
		}
	})

	t.Run("unsupported operators error", func(t *testing.T) {
		for _, filter := range []map[string]any{
			{"group": map[string]any{vectorstores.FilterExists: true}},
			{"group": map[string]any{vectorstores.FilterLike: "%a%"}},
		} {
			if _, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
				K:      3,
				Filter: filter,
			}); err == nil {
				t.Fatalf("expected error for filter %#v", filter)
			}
		}
	})

	t.Run("score threshold applies client-side", func(t *testing.T) {
		// The fake server returns cosine distances, so the exact-match doc has
		// relevance 1.0 while the others fall below 0.99.
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha one", vectorstores.SearchOptions{
			K:              3,
			ScoreThreshold: 0.99,
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 1 || docs[0].ID != "one" {
			t.Fatalf("docs: %#v", docs)
		}
	})

	t.Run("zero K defaults to 4", func(t *testing.T) {
		docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(docs) != 3 {
			t.Fatalf("docs: got %d want 3", len(docs))
		}
	})
}

func TestStoreMMRSearchWithOptions(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store, err := New(
		t.Context(),
		"langchain",
		embeddings.NewFake(16),
		WithBaseURL(server.URL),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	_, err = store.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha one", map[string]any{"group": "a", "page": 1}).WithID("one"),
		documents.New("alpha two", map[string]any{"group": "a", "page": 2}).WithID("two"),
		documents.New("alpha three", map[string]any{"group": "b", "page": 3}).WithID("three"),
	})
	if err != nil {
		t.Fatalf("add: %v", err)
	}

	docs, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
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
	for _, doc := range docs {
		if doc.Metadata["group"] != "a" {
			t.Fatalf("unexpected doc: %#v", doc)
		}
	}

	// The prefetch query inside MMR must carry the translated filter.
	var payload queryRequest
	server.decodeLastBody(t, &payload)
	want := map[string]any{"group": map[string]any{"$eq": "a"}}
	if !marshalEqual(payload.Where, want) {
		t.Fatalf("where clause: got %#v want %#v", payload.Where, want)
	}
}
