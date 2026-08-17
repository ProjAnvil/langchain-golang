package store

import (
	"context"
	"fmt"
	"reflect"
	"testing"
)

// TestListNamespacesEmptyStore: listing an empty store returns no namespaces
// and no error.
func TestListNamespacesEmptyStore(t *testing.T) {
	s := NewInMemoryStore()
	got, err := s.ListNamespaces(context.Background(), ListNamespacesOptions{})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("ListNamespaces on empty store = %v, want empty", got)
	}
}

// TestListNamespacesReturnsDefensiveCopy: the returned namespace slices must be
// copies — mutating them must not corrupt the store's internal namespaces.
func TestListNamespacesReturnsDefensiveCopy(t *testing.T) {
	s := NewInMemoryStore()
	if err := s.Put(context.Background(), []string{"a", "b"}, "k", map[string]any{"v": 1}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.ListNamespaces(context.Background(), ListNamespacesOptions{})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("ListNamespaces = %v, want one namespace", got)
	}
	got[0][0] = "MUTATED"

	again, err := s.ListNamespaces(context.Background(), ListNamespacesOptions{})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if !reflect.DeepEqual(again, [][]string{{"a", "b"}}) {
		t.Errorf("mutation of a returned namespace leaked into the store: %v", again)
	}
}

// TestListNamespacesZeroAndNegativeLimitDefault: a zero (or negative) Limit
// defaults to 100, so a small store is returned in full.
func TestListNamespacesZeroAndNegativeLimitDefault(t *testing.T) {
	s := NewInMemoryStore()
	for i := 0; i < 3; i++ {
		if err := s.Put(context.Background(), []string{fmt.Sprintf("n%d", i)}, "k", map[string]any{}, nil); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	for _, limit := range []int{0, -1} {
		got, err := s.ListNamespaces(context.Background(), ListNamespacesOptions{Limit: limit})
		if err != nil {
			t.Fatalf("ListNamespaces(limit=%d): %v", limit, err)
		}
		if len(got) != 3 {
			t.Errorf("ListNamespaces(limit=%d) returned %d namespaces, want 3 (default 100)", limit, len(got))
		}
	}
}

// TestListNamespacesMaxDepthZero: a zero max_depth truncates every namespace to
// the empty path, deduped to a single empty namespace (Python's ns[:0] then set).
func TestListNamespacesMaxDepthZero(t *testing.T) {
	s := NewInMemoryStore()
	if err := s.Put(context.Background(), []string{"a", "b"}, "k1", map[string]any{}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Put(context.Background(), []string{"c", "d"}, "k2", map[string]any{}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	zero := 0
	got, err := s.ListNamespaces(context.Background(), ListNamespacesOptions{MaxDepth: &zero})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if len(got) != 1 || len(got[0]) != 0 {
		t.Errorf("ListNamespaces(max_depth=0) = %v, want a single empty namespace", got)
	}
}

// TestGetPutDeleteRoundTrip: the basic lifecycle — a missing key reads back
// (nil, nil), a Put makes it retrievable with timestamps and value, and a
// Delete removes it again (Delete of a missing key is a silent no-op).
func TestGetPutDeleteRoundTrip(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	ns := []string{"memories", "user1"}

	item, err := s.Get(ctx, ns, "k1")
	if err != nil {
		t.Fatalf("Get on empty store: %v", err)
	}
	if item != nil {
		t.Fatalf("Get on empty store = %v, want nil", item)
	}

	value := map[string]any{"text": "hello", "score": 42}
	if err := s.Put(ctx, ns, "k1", value, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	item, err = s.Get(ctx, ns, "k1")
	if err != nil {
		t.Fatalf("Get after Put: %v", err)
	}
	if item == nil {
		t.Fatal("Get after Put returned nil item")
	}
	if !reflect.DeepEqual(item.Namespace, ns) {
		t.Errorf("item.Namespace = %v, want %v", item.Namespace, ns)
	}
	if item.Key != "k1" {
		t.Errorf("item.Key = %q, want %q", item.Key, "k1")
	}
	if !reflect.DeepEqual(item.Value, value) {
		t.Errorf("item.Value = %v, want %v", item.Value, value)
	}
	if item.CreatedAt.IsZero() || item.UpdatedAt.IsZero() {
		t.Errorf("timestamps should be set on Put: %+v", item)
	}

	// Same key in a different namespace must not be visible.
	other, err := s.Get(ctx, []string{"memories", "user2"}, "k1")
	if err != nil {
		t.Fatalf("Get in other namespace: %v", err)
	}
	if other != nil {
		t.Errorf("Get in other namespace = %v, want nil", other)
	}

	if err := s.Delete(ctx, ns, "k1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	item, err = s.Get(ctx, ns, "k1")
	if err != nil {
		t.Fatalf("Get after Delete: %v", err)
	}
	if item != nil {
		t.Errorf("Get after Delete = %v, want nil", item)
	}

	// Deleting a missing key from a now-pruned bucket is a no-op, not an error.
	if err := s.Delete(ctx, ns, "k1"); err != nil {
		t.Errorf("Delete of missing key from pruned bucket: %v", err)
	}
	// Deleting from a bucket that still holds other keys is also a no-op.
	if err := s.Put(ctx, ns, "k2", map[string]any{}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Delete(ctx, ns, "missing"); err != nil {
		t.Errorf("Delete of missing key in existing bucket: %v", err)
	}
	kept, err := s.Get(ctx, ns, "k2")
	if err != nil || kept == nil {
		t.Errorf("Delete of missing key must not disturb other items: item=%v err=%v", kept, err)
	}
}

// TestDeletePrunesEmptyBucket: deleting the last item of a namespace removes
// the namespace from ListNamespaces.
func TestDeletePrunesEmptyBucket(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	ns := []string{"a", "b"}
	if err := s.Put(ctx, ns, "k", map[string]any{}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Delete(ctx, ns, "k"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	got, err := s.ListNamespaces(ctx, ListNamespacesOptions{})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("ListNamespaces after deleting last item = %v, want empty", got)
	}
}

// TestPutReplacesExisting: a second Put on the same (namespace, key) replaces
// the value entirely (fresh Item, mirroring Python's _apply_put_ops).
func TestPutReplacesExisting(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	ns := []string{"n"}
	if err := s.Put(ctx, ns, "k", map[string]any{"v": 1}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Put(ctx, ns, "k", map[string]any{"v": 2}, nil); err != nil {
		t.Fatalf("Put (replace): %v", err)
	}
	item, err := s.Get(ctx, ns, "k")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !reflect.DeepEqual(item.Value, map[string]any{"v": 2}) {
		t.Errorf("item.Value after replace = %v, want {v: 2}", item.Value)
	}
}

// TestPutDoesNotAliasCallerValue: mutating the caller's value map after Put
// must not change what is stored.
func TestPutDoesNotAliasCallerValue(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	ns := []string{"n"}
	value := map[string]any{"v": 1}
	if err := s.Put(ctx, ns, "k", value, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	value["v"] = 999
	value["extra"] = true

	item, err := s.Get(ctx, ns, "k")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !reflect.DeepEqual(item.Value, map[string]any{"v": 1}) {
		t.Errorf("stored value changed when caller mutated its map: %v", item.Value)
	}
}

// TestPutDoesNotAliasCallerNamespace: mutating the caller's namespace slice
// after Put must not corrupt the store's internal namespace bookkeeping.
func TestPutDoesNotAliasCallerNamespace(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	ns := []string{"a", "b"}
	if err := s.Put(ctx, ns, "k", map[string]any{}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	ns[0] = "MUTATED"

	got, err := s.ListNamespaces(ctx, ListNamespacesOptions{})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if !reflect.DeepEqual(got, [][]string{{"a", "b"}}) {
		t.Errorf("namespace mutation leaked into store: %v", got)
	}
	// The item must still be searchable under the original prefix.
	items, err := s.Search(ctx, []string{"a"}, SearchOptions{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 1 {
		t.Errorf("Search under original prefix returned %d items, want 1", len(items))
	}
}

// TestPutNilValue: a nil value map is stored (and returned) as nil.
func TestPutNilValue(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	if err := s.Put(ctx, []string{"n"}, "k", nil, nil); err != nil {
		t.Fatalf("Put nil value: %v", err)
	}
	item, err := s.Get(ctx, []string{"n"}, "k")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if item == nil {
		t.Fatal("Get after Put(nil value) returned nil item")
	}
	if item.Value != nil {
		t.Errorf("item.Value = %v, want nil", item.Value)
	}
}

// TestSearchEmptyPrefixMatchesAll: an empty namespace prefix matches every
// item, ordered deterministically by (namespace, key).
func TestSearchEmptyPrefixMatchesAll(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	puts := []struct {
		ns  []string
		key string
	}{
		{[]string{"b"}, "k2"},
		{[]string{"a", "y"}, "k1"},
		{[]string{"a", "x"}, "k2"},
		{[]string{"a", "x"}, "k1"},
	}
	for _, p := range puts {
		if err := s.Put(ctx, p.ns, p.key, map[string]any{}, nil); err != nil {
			t.Fatalf("Put(%v, %s): %v", p.ns, p.key, err)
		}
	}
	items, err := s.Search(ctx, nil, SearchOptions{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	want := []struct {
		ns  []string
		key string
	}{
		{[]string{"a", "x"}, "k1"},
		{[]string{"a", "x"}, "k2"},
		{[]string{"a", "y"}, "k1"},
		{[]string{"b"}, "k2"},
	}
	if len(items) != len(want) {
		t.Fatalf("Search returned %d items, want %d", len(items), len(want))
	}
	for i, w := range want {
		if !reflect.DeepEqual(items[i].Namespace, w.ns) || items[i].Key != w.key {
			t.Errorf("items[%d] = (%v, %s), want (%v, %s)", i, items[i].Namespace, items[i].Key, w.ns, w.key)
		}
	}
}

// TestSearchNamespacePrefixMatching: prefix matching is element-wise — a
// prefix segment must match a whole namespace label, and a prefix longer than
// a stored namespace does not match it.
func TestSearchNamespacePrefixMatching(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	for _, ns := range [][]string{{"a", "b"}, {"a", "bc"}, {"ab"}} {
		if err := s.Put(ctx, ns, "k", map[string]any{}, nil); err != nil {
			t.Fatalf("Put(%v): %v", ns, err)
		}
	}

	items, err := s.Search(ctx, []string{"a"}, SearchOptions{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 2 {
		t.Errorf("Search(prefix=[a]) returned %d items, want 2 (element-wise, [ab] excluded)", len(items))
	}

	items, err = s.Search(ctx, []string{"a", "b", "c"}, SearchOptions{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 0 {
		t.Errorf("Search with prefix longer than any namespace returned %d items, want 0", len(items))
	}
}

// TestSearchLimitOffsetAndDefaults: a zero/negative Limit defaults to 10, a
// negative Offset is clamped to 0, and pagination slices the ordered results.
func TestSearchLimitOffsetAndDefaults(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	for i := 0; i < 12; i++ {
		key := fmt.Sprintf("k%02d", i)
		if err := s.Put(ctx, []string{"n"}, key, map[string]any{}, nil); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}

	// Default limit of 10 kicks in for zero and negative limits.
	for _, limit := range []int{0, -3} {
		items, err := s.Search(ctx, []string{"n"}, SearchOptions{Limit: limit})
		if err != nil {
			t.Fatalf("Search(limit=%d): %v", limit, err)
		}
		if len(items) != 10 {
			t.Errorf("Search(limit=%d) returned %d items, want 10 (default)", limit, len(items))
		}
	}

	// Negative offset behaves as 0.
	items, err := s.Search(ctx, []string{"n"}, SearchOptions{Limit: 2, Offset: -5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 2 || items[0].Key != "k00" {
		t.Errorf("Search(offset=-5) = %v, want [k00 k01]", keysOfItems(items))
	}

	// Offset skips leading items.
	items, err = s.Search(ctx, []string{"n"}, SearchOptions{Limit: 3, Offset: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got := keysOfItems(items); !reflect.DeepEqual(got, []string{"k10", "k11"}) {
		t.Errorf("Search(offset=10, limit=3) = %v, want [k10 k11]", got)
	}

	// Offset beyond the result set yields an empty (non-nil) slice.
	items, err = s.Search(ctx, []string{"n"}, SearchOptions{Offset: 100})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if items == nil || len(items) != 0 {
		t.Errorf("Search(offset beyond results) = %v, want empty non-nil slice", items)
	}
}

// TestSearchReturnsDefensiveCopies: mutating a returned item's namespace or
// value must not corrupt the stored item.
func TestSearchReturnsDefensiveCopies(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	ns := []string{"n"}
	if err := s.Put(ctx, ns, "k", map[string]any{"v": 1}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	items, err := s.Search(ctx, ns, SearchOptions{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("Search returned %d items, want 1", len(items))
	}
	items[0].Namespace[0] = "MUTATED"
	items[0].Value["v"] = 999

	item, err := s.Get(ctx, ns, "k")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !reflect.DeepEqual(item.Namespace, ns) {
		t.Errorf("stored namespace mutated via Search result: %v", item.Namespace)
	}
	if !reflect.DeepEqual(item.Value, map[string]any{"v": 1}) {
		t.Errorf("stored value mutated via Search result: %v", item.Value)
	}
}

// TestSearchScalarFilter: a scalar filter value matches by deep equality.
func TestSearchScalarFilter(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	if err := s.Put(ctx, []string{"n"}, "yes", map[string]any{"color": "red", "n": 1}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Put(ctx, []string{"n"}, "no", map[string]any{"color": "blue", "n": 2}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}

	items, err := s.Search(ctx, []string{"n"}, SearchOptions{Filter: map[string]any{"color": "red"}})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 1 || items[0].Key != "yes" {
		t.Errorf("scalar filter returned %v, want [yes]", keysOfItems(items))
	}

	// Multiple filter keys all must hold.
	items, err = s.Search(ctx, []string{"n"}, SearchOptions{Filter: map[string]any{"color": "red", "n": 2}})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 0 {
		t.Errorf("conjunctive filter returned %v, want none", keysOfItems(items))
	}
}

// TestSearchOperatorFilters exercises every comparison operator through
// Search, including the no-match cases for non-numeric operands and unknown
// operators (documented divergence from Python: they narrow to zero results
// instead of raising).
func TestSearchOperatorFilters(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	values := map[string]map[string]any{
		"one":     {"n": 1},
		"two":     {"n": 2},
		"three":   {"n": 3},
		"str":     {"n": "not-a-number"},
		"missing": {"other": 1},
	}
	for k, v := range values {
		if err := s.Put(ctx, []string{"n"}, k, v, nil); err != nil {
			t.Fatalf("Put(%s): %v", k, err)
		}
	}
	search := func(t *testing.T, filter map[string]any) []string {
		t.Helper()
		items, err := s.Search(ctx, []string{"n"}, SearchOptions{Filter: map[string]any{"n": filter}})
		if err != nil {
			t.Fatalf("Search(filter=%v): %v", filter, err)
		}
		return keysOfItems(items)
	}

	cases := []struct {
		name   string
		filter map[string]any
		want   []string
	}{
		{"$eq", map[string]any{"$eq": 2}, []string{"two"}},
		{"$ne", map[string]any{"$ne": 2}, []string{"missing", "one", "str", "three"}},
		{"$gt", map[string]any{"$gt": 1}, []string{"three", "two"}},
		{"$gte", map[string]any{"$gte": 2}, []string{"three", "two"}},
		{"$lt", map[string]any{"$lt": 3}, []string{"one", "two"}},
		{"$lte", map[string]any{"$lte": 2}, []string{"one", "two"}},
		{"combined $gte/$lt", map[string]any{"$gte": 1, "$lt": 3}, []string{"one", "two"}},
		// Non-numeric item values and missing fields never satisfy ordering
		// operators (no-match instead of Python's ValueError).
		{"$gt with non-numeric operand", map[string]any{"$gt": "x"}, nil},
		// Unknown operators match nothing.
		{"unknown operator", map[string]any{"$bogus": 1}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := search(t, tc.filter)
			if len(tc.want) == 0 {
				if len(got) != 0 {
					t.Errorf("filter %v matched %v, want none", tc.filter, got)
				}
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("filter %v matched %v, want %v", tc.filter, got, tc.want)
			}
		})
	}
}

// TestSearchNestedMapAndSliceFilters: a map filter without "$" keys matches a
// nested map recursively; a slice filter matches element-wise.
func TestSearchNestedMapAndSliceFilters(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	if err := s.Put(ctx, []string{"n"}, "match", map[string]any{
		"meta": map[string]any{"city": "paris", "zip": "75001"},
		"tags": []any{"a", "b"},
	}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Put(ctx, []string{"n"}, "nested-scalar", map[string]any{
		"meta": "not-a-map",
		"tags": []any{"a", "b"},
	}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Put(ctx, []string{"n"}, "wrong-len", map[string]any{
		"meta": map[string]any{"city": "paris", "zip": "75001"},
		"tags": []any{"a"},
	}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Put(ctx, []string{"n"}, "wrong-elem", map[string]any{
		"meta": map[string]any{"city": "lyon", "zip": "69001"},
		"tags": []any{"a", "c"},
	}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Put(ctx, []string{"n"}, "not-slice", map[string]any{
		"meta": map[string]any{"city": "paris", "zip": "75001"},
		"tags": "ab",
	}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Nested map filter: partial match against the nested map.
	items, err := s.Search(ctx, []string{"n"}, SearchOptions{
		Filter: map[string]any{"meta": map[string]any{"city": "paris"}},
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got := keysOfItems(items); !reflect.DeepEqual(got, []string{"match", "not-slice", "wrong-len"}) {
		t.Errorf("nested map filter matched %v, want [match not-slice wrong-len]", got)
	}

	// Slice filter: exact element-wise match.
	items, err = s.Search(ctx, []string{"n"}, SearchOptions{
		Filter: map[string]any{"tags": []any{"a", "b"}},
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got := keysOfItems(items); !reflect.DeepEqual(got, []string{"match", "nested-scalar"}) {
		t.Errorf("slice filter matched %v, want [match nested-scalar]", got)
	}

	// Nested map filter against a scalar item value matches nothing.
	items, err = s.Search(ctx, []string{"n"}, SearchOptions{
		Filter: map[string]any{"meta": map[string]any{"city": "not-a-map"}},
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(items) != 0 {
		t.Errorf("nested filter matching a scalar value returned %v, want none", keysOfItems(items))
	}
}

// TestSearchNumericCoercion: ordering operators coerce all Go numeric kinds
// to float64, mirroring Python's float().
func TestSearchNumericCoercion(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	puts := []struct {
		key string
		n   any
	}{
		{"int", int(2)},
		{"int8", int8(2)},
		{"int16", int16(2)},
		{"int32", int32(2)},
		{"int64", int64(2)},
		{"uint", uint(2)},
		{"uint8", uint8(2)},
		{"uint16", uint16(2)},
		{"uint32", uint32(2)},
		{"uint64", uint64(2)},
		{"float32", float32(2)},
		{"float64", float64(2)},
		{"bool", true},
		{"nil", nil},
	}
	for _, p := range puts {
		if err := s.Put(ctx, []string{"n"}, p.key, map[string]any{"n": p.n}, nil); err != nil {
			t.Fatalf("Put(%s): %v", p.key, err)
		}
	}

	items, err := s.Search(ctx, []string{"n"}, SearchOptions{
		Filter: map[string]any{"n": map[string]any{"$gte": int64(2), "$lte": float64(2)}},
		Limit:  100,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	want := []string{"float32", "float64", "int", "int16", "int32", "int64", "int8", "uint", "uint16", "uint32", "uint64", "uint8"}
	if got := keysOfItems(items); !reflect.DeepEqual(got, want) {
		t.Errorf("numeric coercion matched %v, want %v (bool and nil excluded)", got, want)
	}
}

// TestListNamespacesPrefixSuffixMaxDepthOffset covers the remaining
// ListNamespaces branches: prefix/suffix filters, max-depth truncation with
// dedupe, negative max-depth, offset, and offset beyond the result set.
func TestListNamespacesPrefixSuffixMaxDepthOffset(t *testing.T) {
	ctx := context.Background()
	s := NewInMemoryStore()
	for _, ns := range [][]string{
		{"a", "b", "c"},
		{"a", "b", "d"},
		{"a", "x"},
		{"z", "b", "c"},
	} {
		if err := s.Put(ctx, ns, "k", map[string]any{}, nil); err != nil {
			t.Fatalf("Put(%v): %v", ns, err)
		}
	}

	// Prefix filter.
	got, err := s.ListNamespaces(ctx, ListNamespacesOptions{Prefix: []string{"a"}})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	want := [][]string{{"a", "b", "c"}, {"a", "b", "d"}, {"a", "x"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListNamespaces(prefix=[a]) = %v, want %v", got, want)
	}

	// Suffix filter.
	got, err = s.ListNamespaces(ctx, ListNamespacesOptions{Suffix: []string{"b", "c"}})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	want = [][]string{{"a", "b", "c"}, {"z", "b", "c"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListNamespaces(suffix=[b c]) = %v, want %v", got, want)
	}

	// MaxDepth truncates and dedupes.
	one := 1
	got, err = s.ListNamespaces(ctx, ListNamespacesOptions{MaxDepth: &one})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	want = [][]string{{"a"}, {"z"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListNamespaces(max_depth=1) = %v, want %v", got, want)
	}

	// Negative MaxDepth behaves as 0 (truncate to the empty path, deduped).
	neg := -2
	got, err = s.ListNamespaces(ctx, ListNamespacesOptions{MaxDepth: &neg})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if len(got) != 1 || len(got[0]) != 0 {
		t.Errorf("ListNamespaces(max_depth=-2) = %v, want a single empty namespace", got)
	}

	// MaxDepth larger than the namespace leaves it unchanged.
	ten := 10
	got, err = s.ListNamespaces(ctx, ListNamespacesOptions{MaxDepth: &ten, Prefix: []string{"z"}})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	want = [][]string{{"z", "b", "c"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListNamespaces(max_depth=10, prefix=[z]) = %v, want %v", got, want)
	}

	// Limit and Offset paginate the sorted result.
	got, err = s.ListNamespaces(ctx, ListNamespacesOptions{Limit: 2, Offset: 1})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	want = [][]string{{"a", "b", "d"}, {"a", "x"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListNamespaces(limit=2, offset=1) = %v, want %v", got, want)
	}

	// Negative offset behaves as 0.
	got, err = s.ListNamespaces(ctx, ListNamespacesOptions{Offset: -1, Limit: 1})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if len(got) != 1 || !reflect.DeepEqual(got[0], []string{"a", "b", "c"}) {
		t.Errorf("ListNamespaces(offset=-1) = %v, want first namespace", got)
	}

	// Offset beyond the result set yields an empty (non-nil) slice.
	got, err = s.ListNamespaces(ctx, ListNamespacesOptions{Offset: 99})
	if err != nil {
		t.Fatalf("ListNamespaces: %v", err)
	}
	if got == nil || len(got) != 0 {
		t.Errorf("ListNamespaces(offset beyond results) = %v, want empty non-nil slice", got)
	}
}

// keysOfItems extracts the keys of search results for compact comparisons.
func keysOfItems(items []SearchItem) []string {
	if len(items) == 0 {
		return nil
	}
	keys := make([]string, len(items))
	for i, it := range items {
		keys[i] = it.Key
	}
	return keys
}
