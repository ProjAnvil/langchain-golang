package redisvector

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/vectorstores"
	"github.com/redis/go-redis/v9"
)

func newFilterStore() *Store {
	return &Store{
		metadataFields: map[string]MetadataFieldType{
			"group": MetadataTag,
			"page":  MetadataNumeric,
		},
	}
}

func TestSimilaritySearchWithOptions_RejectsInvalidFilter(t *testing.T) {
	store, _ := newFakeStore(t)
	if _, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
		Filter: map[string]any{"group": map[string]any{"$bogus": "a"}},
	}); err == nil {
		t.Fatal("expected invalid filter error")
	}
}

func TestSimilaritySearchWithOptions_KnnsSearchFailure(t *testing.T) {
	store, _ := newFakeStore(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := store.SimilaritySearchWithOptions(ctx, "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
		t.Fatal("expected knn search failure")
	}
}

// K and FetchK must fall back to the langchain defaults (4 and 20) when unset.
func TestMMRSearchWithOptions_DefaultsKAndFetchK(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(searchReplyRESP2("0.1", "0.9"))...)

	docs, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{})
	if err != nil {
		t.Fatalf("MMRSearchWithOptions: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2", len(docs))
	}
	// The KNN k and LIMIT must both use the default fetch_k of 20.
	call := client.lastCall()
	if got := call[2].(string); got != "*=>[KNN 20 @embedding $BLOB AS __embedding_score]" {
		t.Fatalf("KNN expression: got %q", got)
	}
	if got := call[7]; got != 20 {
		t.Fatalf("LIMIT: got %v want 20", got)
	}
}

func TestMMRSearchWithOptions_Failures(t *testing.T) {
	t.Run("invalid_filter", func(t *testing.T) {
		store, _ := newFakeStore(t)
		if _, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
			Filter: map[string]any{"group": map[string]any{"$bogus": "a"}},
		}); err == nil {
			t.Fatal("expected invalid filter error")
		}
	})

	t.Run("embedding_failure", func(t *testing.T) {
		store, _ := newFakeStore(t)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := store.MMRSearchWithOptions(ctx, "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
			t.Fatal("expected embedding failure")
		}
	})

	t.Run("filter_rendering_failure", func(t *testing.T) {
		store, _ := newFakeStore(t)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		_, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
			Filter: map[string]any{"group": map[string]any{vectorstores.FilterLike: "%a%"}},
		})
		if err == nil {
			t.Fatal("expected filter rendering failure")
		}
	})

	t.Run("search_reply_failure", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(fakeReply{err: errors.New("search blew up")})...)
		if _, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
			t.Fatal("expected search failure")
		}
	})
}

func TestDeleteWithFilter_Failures(t *testing.T) {
	t.Run("invalid_filter", func(t *testing.T) {
		store, _ := newFakeStore(t)
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": map[string]any{"$bogus": "a"}}); err == nil {
			t.Fatal("expected invalid filter error")
		}
	})

	t.Run("count_search_failure", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(fakeReply{err: errors.New("search blew up")})...)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err == nil {
			t.Fatal("expected count search failure")
		}
	})

	t.Run("unexpected_count_reply", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(fakeReply{value: "OK"})...)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err == nil {
			t.Fatal("expected unexpected count reply error")
		}
	})

	t.Run("zero_total_is_a_noop", func(t *testing.T) {
		store, client := newFakeStore(t, withDefaultSchema(fakeReply{value: int64(0)})...)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err != nil {
			t.Fatalf("DeleteWithFilter: %v", err)
		}
		if client.callCount() != 3 { // FT.INFO, FT.CREATE, count search
			t.Fatalf("no id fetch or DEL expected, got %d calls", client.callCount())
		}
	})

	t.Run("id_search_failure", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(
			fakeReply{value: []any{int64(2)}},
			fakeReply{err: errors.New("search blew up")},
		)...)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err == nil {
			t.Fatal("expected id search failure")
		}
	})

	t.Run("malformed_id_reply", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(
			fakeReply{value: []any{int64(1)}},
			// The flat id's fields payload is malformed JSON.
			fakeReply{value: []any{int64(1), "docs:one", []any{"$", "not-json"}}},
		)...)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err == nil {
			t.Fatal("expected reply parse failure")
		}
	})

	t.Run("empty_id_hits_is_a_noop", func(t *testing.T) {
		store, client := newFakeStore(t, withDefaultSchema(
			fakeReply{value: []any{int64(2)}},
			fakeReply{value: []any{int64(0)}},
		)...)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err != nil {
			t.Fatalf("DeleteWithFilter: %v", err)
		}
		if client.lastCall()[0] != "FT.SEARCH" {
			t.Fatalf("DEL must not run without hits, got %v", client.lastCall())
		}
	})

	t.Run("del_failure", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(
			fakeReply{value: []any{int64(1)}},
			fakeReply{value: []any{int64(1), "docs:one"}},
			fakeReply{err: errors.New("DEL failed")},
		)...)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err == nil {
			t.Fatal("expected DEL failure")
		}
	})
}

// Rendering error branches that ValidateFilter blocks stay reachable through
// direct condition rendering.
func TestBuildConditionClause_RejectsInvalidOperands(t *testing.T) {
	store := newFilterStore()

	tests := []struct {
		name      string
		condition any
	}{
		{"numeric_between_with_string_bounds", map[string]any{vectorstores.FilterBetween: []any{"a", "c"}}},
		{"numeric_field_with_bogus_operator", map[string]any{"$bogus": 1}},
		{"numeric_in_with_non_numeric_item", map[string]any{vectorstores.FilterIn: []any{true}}},
		{"numeric_in_with_empty_list", map[string]any{vectorstores.FilterIn: []any{}}},
		{"tag_ne_with_numeric_operand", map[string]any{vectorstores.FilterNe: 3}},
		{"tag_in_with_empty_list", map[string]any{vectorstores.FilterIn: []any{}}},
		{"tag_nin_with_non_list", map[string]any{vectorstores.FilterNin: "scalar"}},
		{"tag_in_with_numeric_item", map[string]any{vectorstores.FilterIn: []any{1, 2}}},
		{"tag_field_with_bogus_operator", map[string]any{"$bogus": "a"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			field := "group"
			if tt.name == "numeric_between_with_string_bounds" ||
				tt.name == "numeric_field_with_bogus_operator" ||
				tt.name == "numeric_in_with_non_numeric_item" ||
				tt.name == "numeric_in_with_empty_list" {
				field = "page"
			}
			if _, err := store.buildConditionClause(field, tt.condition); err == nil {
				t.Fatal("expected rendering error")
			}
		})
	}
}

func TestTagValue_RejectsNonTagOperands(t *testing.T) {
	if _, err := tagValue(struct{}{}); err == nil {
		t.Fatal("expected error for non-scalar tag operand")
	}
	if _, err := tagValue(7); err == nil {
		t.Fatal("expected error for numeric operand on tag field")
	}
}

func TestBetweenBounds_RejectsInvalidBounds(t *testing.T) {
	if _, err := betweenBounds([]any{1, 2, 3}); err == nil {
		t.Fatal("expected error for three bounds")
	}
	if _, err := betweenBounds([]any{1, "c"}); err == nil {
		t.Fatal("expected error for non-numeric bound")
	}
}

func TestEscapeTag(t *testing.T) {
	if got := escapeTag(`a b|c{d}e\f`); got != `a\ b\|c\{d\}e\\f` {
		t.Fatalf("escapeTag: got %q", got)
	}
}

func TestReplyFirstInt(t *testing.T) {
	tests := []struct {
		name  string
		raw   any
		want  int
		valid bool
	}{
		{"bare_int64", int64(7), 7, true},
		{"array_head_int64", []any{int64(3), "docs:one"}, 3, true},
		{"array_head_not_int", []any{"x"}, 0, false},
		{"empty_array", []any{}, 0, false},
		{"resp3_total_results", map[any]any{"total_results": int64(9)}, 9, true},
		{"resp3_missing_total", map[any]any{"results": []any{}}, 0, false},
		{"unexpected_type", "OK", 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := replyFirstInt(tt.raw)
			if ok != tt.valid || got != tt.want {
				t.Fatalf("replyFirstInt(%#v): got (%d, %v) want (%d, %v)", tt.raw, got, ok, tt.want, tt.valid)
			}
		})
	}
}

func TestNumericValue(t *testing.T) {
	tests := []struct {
		name  string
		value any
		want  float64
	}{
		{"int", 7, 7},
		{"int8", int8(7), 7},
		{"int16", int16(7), 7},
		{"int32", int32(7), 7},
		{"int64", int64(7), 7},
		{"uint", uint(7), 7},
		{"uint8", uint8(7), 7},
		{"uint16", uint16(7), 7},
		{"uint32", uint32(7), 7},
		{"uint64", uint64(7), 7},
		{"float32", float32(1.5), 1.5},
		{"float64", 1.5, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := numericValue(tt.value)
			if !ok || got != tt.want {
				t.Fatalf("numericValue(%v): got (%v, %v) want (%v, true)", tt.value, got, ok, tt.want)
			}
		})
	}
	t.Run("non_numeric", func(t *testing.T) {
		if _, ok := numericValue("7"); ok {
			t.Fatal("string must not be numeric")
		}
		if _, ok := numericValue(struct{}{}); ok {
			t.Fatal("struct must not be numeric")
		}
	})
}

func TestToAnySlice(t *testing.T) {
	t.Run("any_slice_passthrough", func(t *testing.T) {
		got, ok := toAnySlice([]any{"a", 1})
		if !ok || len(got) != 2 || got[0] != "a" || got[1] != 1 {
			t.Fatalf("got %v ok %v", got, ok)
		}
	})
	t.Run("typed_string_slice_widens", func(t *testing.T) {
		got, ok := toAnySlice([]string{"a", "b"})
		if !ok || len(got) != 2 || got[0] != "a" || got[1] != "b" {
			t.Fatalf("got %v ok %v", got, ok)
		}
	})
	t.Run("typed_array_widens", func(t *testing.T) {
		got, ok := toAnySlice([2]int{3, 4})
		if !ok || len(got) != 2 || got[0] != 3 || got[1] != 4 {
			t.Fatalf("got %v ok %v", got, ok)
		}
	})
	t.Run("non_slice_rejected", func(t *testing.T) {
		if got, ok := toAnySlice("scalar"); ok {
			t.Fatalf("scalar must not widen, got %v", got)
		}
		if got, ok := toAnySlice(nil); ok {
			t.Fatalf("nil must not widen, got %v", got)
		}
	})
}

// Sanity guard that redis.Nil is still classified as a missing key, since the
// GetByIDs skip logic depends on it.
func TestIsRedisNil(t *testing.T) {
	if !isRedisNil(redis.Nil) {
		t.Fatal("redis.Nil must be classified nil")
	}
	if isRedisNil(errors.New("other")) {
		t.Fatal("other errors must not be classified nil")
	}
}
