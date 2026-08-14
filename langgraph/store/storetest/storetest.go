// Package storetest provides a conformance suite for implementations of the
// store.Store interface, mirroring the shape of checkpoint/savertest: export
// Run(t, newStore) and call it from each implementation's _test.go.
//
// The suite asserts the documented Store contract (Get/Put/Delete/Search) that
// every implementation must satisfy: put/get round-trip, delete, search by
// namespace prefix, search with filter (exact + comparison operators),
// search limit/offset pagination, put-update-replaces, hierarchical
// namespaces, and concurrent put/get. Semantic (query) search is intentionally
// NOT exercised — it is an optional capability the InMemoryStore does not
// provide; implementations that add it should cover ranking in their own
// package tests.
package storetest

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/store"
)

// Run executes the Store contract suite as subtests of t. newStore must return
// a Store backed by EMPTY storage (factories wrapping a shared database must
// truncate all tables between calls).
func Run(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	t.Run("put_get_round_trip", func(t *testing.T) { testPutGetRoundTrip(t, newStore) })
	t.Run("get_missing_returns_nil", func(t *testing.T) { testGetMissingReturnsNil(t, newStore) })
	t.Run("delete", func(t *testing.T) { testDelete(t, newStore) })
	t.Run("delete_missing_is_noop", func(t *testing.T) { testDeleteMissingIsNoop(t, newStore) })
	t.Run("search_by_namespace_prefix", func(t *testing.T) { testSearchByNamespacePrefix(t, newStore) })
	t.Run("search_with_filter", func(t *testing.T) { testSearchWithFilter(t, newStore) })
	t.Run("search_filter_operators", func(t *testing.T) { testSearchFilterOperators(t, newStore) })
	t.Run("search_limit_offset", func(t *testing.T) { testSearchLimitOffset(t, newStore) })
	t.Run("search_empty_prefix_matches_all", func(t *testing.T) { testSearchEmptyPrefixMatchesAll(t, newStore) })
	t.Run("put_updates_existing", func(t *testing.T) { testPutUpdatesExisting(t, newStore) })
	t.Run("namespaces_hierarchical", func(t *testing.T) { testNamespacesHierarchical(t, newStore) })
	t.Run("put_rejects_invalid_namespace", func(t *testing.T) { testPutRejectsInvalidNamespace(t, newStore) })
	t.Run("concurrent_put_get", func(t *testing.T) { testConcurrentPutGet(t, newStore) })
	t.Run("list_namespaces", func(t *testing.T) { testListNamespaces(t, newStore) })
}

// testPutGetRoundTrip: an item Put under (namespace, key) is returned by Get
// value-for-value, with a non-zero Key, Namespace, and timestamps.
func testPutGetRoundTrip(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	ns := []string{"memories", "user1"}
	val := map[string]any{"theme": "dark", "volume": 7}
	if err := s.Put(ctx, ns, "prefs", val, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.Get(ctx, ns, "prefs")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatalf("Get returned nil item")
	}
	if got.Key != "prefs" {
		t.Errorf("Key = %q, want %q", got.Key, "prefs")
	}
	if !reflect.DeepEqual(got.Namespace, ns) {
		t.Errorf("Namespace = %v, want %v", got.Namespace, ns)
	}
	if !reflect.DeepEqual(got.Value, val) {
		t.Errorf("Value = %v, want %v", got.Value, val)
	}
	if got.CreatedAt.IsZero() || got.UpdatedAt.IsZero() {
		t.Errorf("timestamps not set: created=%v updated=%v", got.CreatedAt, got.UpdatedAt)
	}
}

// testGetMissingReturnsNil: Get of an unknown key returns (nil, nil) — not an
// error — and an unknown namespace likewise returns (nil, nil).
func testGetMissingReturnsNil(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	if err := s.Put(ctx, []string{"ns"}, "a", map[string]any{"x": 1}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	// Missing key in an existing namespace.
	got, err := s.Get(ctx, []string{"ns"}, "nope")
	if err != nil || got != nil {
		t.Fatalf("Get missing key: got item=%v err=%v, want nil,nil", got, err)
	}
	// Missing namespace entirely.
	got, err = s.Get(ctx, []string{"nope"}, "a")
	if err != nil || got != nil {
		t.Fatalf("Get missing namespace: got item=%v err=%v, want nil,nil", got, err)
	}
}

// testDelete: Delete removes the item so a subsequent Get returns nil; a sibling
// item in the same namespace survives.
func testDelete(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	ns := []string{"docs"}
	if err := s.Put(ctx, ns, "a", map[string]any{"n": 1}, nil); err != nil {
		t.Fatalf("Put a: %v", err)
	}
	if err := s.Put(ctx, ns, "b", map[string]any{"n": 2}, nil); err != nil {
		t.Fatalf("Put b: %v", err)
	}
	if err := s.Delete(ctx, ns, "a"); err != nil {
		t.Fatalf("Delete a: %v", err)
	}
	got, err := s.Get(ctx, ns, "a")
	if err != nil || got != nil {
		t.Fatalf("Get a after delete: item=%v err=%v, want nil,nil", got, err)
	}
	got, err = s.Get(ctx, ns, "b")
	if err != nil || got == nil {
		t.Fatalf("Get b after deleting a: item=%v err=%v, want the surviving item", got, err)
	}
}

// testDeleteMissingIsNoop: Delete of a key that was never stored is not an
// error (mirrors Python's PutOp(value=None) popping a missing key silently).
func testDeleteMissingIsNoop(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)
	if err := s.Delete(ctx, []string{"ns"}, "never-existed"); err != nil {
		t.Fatalf("Delete missing key: %v", err)
	}
}

// testSearchByNamespacePrefix: Search with a prefix returns only items whose
// namespace starts with that prefix; sibling namespaces under a different root
// are excluded.
func testSearchByNamespacePrefix(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	put := func(ns []string, key string) {
		t.Helper()
		if err := s.Put(ctx, ns, key, map[string]any{"k": key}, nil); err != nil {
			t.Fatalf("Put %v/%s: %v", ns, key, err)
		}
	}
	put([]string{"docs", "v1"}, "a")
	put([]string{"docs", "v2"}, "b")
	put([]string{"notes"}, "c")

	results, err := s.Search(ctx, []string{"docs"}, store.SearchOptions{Limit: 100})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("Search(docs) returned %d items, want 2: %+v", len(results), results)
	}
	for _, r := range results {
		if r.Namespace[0] != "docs" {
			t.Errorf("Search(docs) returned item in namespace %v", r.Namespace)
		}
	}
	// A deeper prefix narrows further.
	results, err = s.Search(ctx, []string{"docs", "v1"}, store.SearchOptions{Limit: 100})
	if err != nil {
		t.Fatalf("Search(docs/v1): %v", err)
	}
	if len(results) != 1 || results[0].Key != "a" {
		t.Fatalf("Search(docs/v1) = %+v, want one item keyed %q", results, "a")
	}
}

// testSearchWithFilter: Search with an exact-match Filter narrows by item value
// fields; nested-map filters recurse.
func testSearchWithFilter(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	put := func(key string, value map[string]any) {
		t.Helper()
		if err := s.Put(ctx, []string{"docs"}, key, value, nil); err != nil {
			t.Fatalf("Put %s: %v", key, err)
		}
	}
	put("a", map[string]any{"type": "report", "status": "active"})
	put("b", map[string]any{"type": "report", "status": "archived"})
	put("c", map[string]any{"type": "memo", "status": "active"})

	results, err := s.Search(ctx, []string{"docs"}, store.SearchOptions{
		Filter: map[string]any{"type": "report", "status": "active"},
		Limit:  100,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 || results[0].Key != "a" {
		t.Fatalf("Search(type=report,status=active) = %+v, want one item keyed %q", results, "a")
	}
}

// testSearchFilterOperators: Filter comparison operators ($gt/$gte/$lt/$lte/$ne)
// match numeric values, mirroring Python's `_apply_operator`.
func testSearchFilterOperators(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	put := func(key string, score int) {
		t.Helper()
		if err := s.Put(ctx, []string{"scores"}, key, map[string]any{"score": score}, nil); err != nil {
			t.Fatalf("Put %s: %v", key, err)
		}
	}
	put("a", 1)
	put("b", 5)
	put("c", 9)

	want := func(filter map[string]any, keys string) {
		t.Helper()
		results, err := s.Search(ctx, []string{"scores"}, store.SearchOptions{Filter: filter, Limit: 100})
		if err != nil {
			t.Fatalf("Search %+v: %v", filter, err)
		}
		got := keysOf(results)
		if got != keys {
			t.Errorf("Search %+v = %q, want %q", filter, got, keys)
		}
	}
	want(map[string]any{"score": map[string]any{"$gt": 4}}, "b,c")
	want(map[string]any{"score": map[string]any{"$gte": 5}}, "b,c")
	want(map[string]any{"score": map[string]any{"$lt": 5}}, "a")
	want(map[string]any{"score": map[string]any{"$lte": 5}}, "a,b")
	want(map[string]any{"score": map[string]any{"$ne": 5}}, "a,c")
}

// testSearchLimitOffset: Search paginates results by Limit/Offset in the
// deterministic (namespace, key) order.
func testSearchLimitOffset(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	for i := 0; i < 5; i++ {
		key := fmt.Sprintf("k%d", i)
		if err := s.Put(ctx, []string{"ns"}, key, map[string]any{"i": i}, nil); err != nil {
			t.Fatalf("Put %s: %v", key, err)
		}
	}
	// Default limit (zero => 10) returns all 5.
	results, err := s.Search(ctx, []string{"ns"}, store.SearchOptions{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if got := keysOf(results); got != "k0,k1,k2,k3,k4" {
		t.Fatalf("Search default limit = %q, want k0,k1,k2,k3,k4", got)
	}
	// Limit=2 returns the first two.
	results, err = s.Search(ctx, []string{"ns"}, store.SearchOptions{Limit: 2})
	if err != nil {
		t.Fatalf("Search limit=2: %v", err)
	}
	if got := keysOf(results); got != "k0,k1" {
		t.Fatalf("Search limit=2 = %q, want k0,k1", got)
	}
	// Offset=2 skips the first two; default limit returns the rest.
	results, err = s.Search(ctx, []string{"ns"}, store.SearchOptions{Offset: 2})
	if err != nil {
		t.Fatalf("Search offset=2: %v", err)
	}
	if got := keysOf(results); got != "k2,k3,k4" {
		t.Fatalf("Search offset=2 = %q, want k2,k3,k4", got)
	}
	// Offset+limit paginates the middle.
	results, err = s.Search(ctx, []string{"ns"}, store.SearchOptions{Offset: 1, Limit: 2})
	if err != nil {
		t.Fatalf("Search offset=1,limit=2: %v", err)
	}
	if got := keysOf(results); got != "k1,k2" {
		t.Fatalf("Search offset=1,limit=2 = %q, want k1,k2", got)
	}
	// Offset past the end returns empty.
	results, err = s.Search(ctx, []string{"ns"}, store.SearchOptions{Offset: 100})
	if err != nil {
		t.Fatalf("Search offset=100: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("Search offset=100 = %+v, want empty", results)
	}
}

// testSearchEmptyPrefixMatchesAll: an empty namespace prefix matches every
// namespace in the store.
func testSearchEmptyPrefixMatchesAll(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	if err := s.Put(ctx, []string{"a"}, "x", map[string]any{"v": 1}, nil); err != nil {
		t.Fatalf("Put a/x: %v", err)
	}
	if err := s.Put(ctx, []string{"b", "c"}, "y", map[string]any{"v": 2}, nil); err != nil {
		t.Fatalf("Put b/c/y: %v", err)
	}
	results, err := s.Search(ctx, nil, store.SearchOptions{Limit: 100})
	if err != nil {
		t.Fatalf("Search empty prefix: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("Search() = %d items, want 2: %+v", len(results), results)
	}
}

// testPutUpdatesExisting: re-Putting the same (namespace, key) replaces the
// value and bumps UpdatedAt; the previous value is gone (mirrors Python's
// _apply_put_ops, which builds a fresh Item).
func testPutUpdatesExisting(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	ns := []string{"ns"}
	if err := s.Put(ctx, ns, "k", map[string]any{"v": "old"}, nil); err != nil {
		t.Fatalf("Put old: %v", err)
	}
	before, err := s.Get(ctx, ns, "k")
	if err != nil || before == nil {
		t.Fatalf("Get before update: item=%v err=%v", before, err)
	}
	// Ensure UpdatedAt advances past the millisecond resolution.
	time.Sleep(2 * time.Millisecond)
	if err := s.Put(ctx, ns, "k", map[string]any{"v": "new"}, nil); err != nil {
		t.Fatalf("Put new: %v", err)
	}
	after, err := s.Get(ctx, ns, "k")
	if err != nil || after == nil {
		t.Fatalf("Get after update: item=%v err=%v", after, err)
	}
	if got := after.Value["v"]; got != "new" {
		t.Errorf("Value after update = %v, want %q", got, "new")
	}
	// UpdatedAt must not go backwards. (CreatedAt is intentionally NOT asserted
	// as preserved: Python's _apply_put_ops builds a fresh Item on every Put.)
	if after.UpdatedAt.Before(before.UpdatedAt) {
		t.Errorf("UpdatedAt went backwards: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
	}
}

// testNamespacesHierarchical: items stored under nested namespaces are each
// addressable by Get at their full path, and a prefix Search at a parent
// namespace returns all descendants.
func testNamespacesHierarchical(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	put := func(ns []string, key, val string) {
		t.Helper()
		if err := s.Put(ctx, ns, key, map[string]any{"text": val}, nil); err != nil {
			t.Fatalf("Put %v/%s: %v", ns, key, err)
		}
	}
	put([]string{"users", "u1", "prefs"}, "theme", "dark")
	put([]string{"users", "u1", "prefs"}, "lang", "en")
	put([]string{"users", "u2", "prefs"}, "theme", "light")

	if got, _ := s.Get(ctx, []string{"users", "u1", "prefs"}, "theme"); got == nil || got.Value["text"] != "dark" {
		t.Fatalf("Get deep item: %+v", got)
	}
	// Prefix at "users" returns everything.
	results, err := s.Search(ctx, []string{"users"}, store.SearchOptions{Limit: 100})
	if err != nil {
		t.Fatalf("Search users: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("Search(users) = %d items, want 3", len(results))
	}
	// Prefix at "users/u1" returns only u1's items.
	results, err = s.Search(ctx, []string{"users", "u1"}, store.SearchOptions{Limit: 100})
	if err != nil {
		t.Fatalf("Search users/u1: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("Search(users/u1) = %d items, want 2", len(results))
	}
}

// testPutRejectsInvalidNamespace: Put rejects empty namespaces, empty labels,
// labels with periods, and the reserved "langgraph" root — mirroring Python's
// _validate_namespace.
func testPutRejectsInvalidNamespace(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	cases := []struct {
		name string
		ns   []string
	}{
		{"empty", nil},
		{"empty_label", []string{"a", ""}},
		{"period_in_label", []string{"a.b"}},
		{"langgraph_root", []string{"langgraph", "x"}},
	}
	for _, c := range cases {
		err := s.Put(ctx, c.ns, "k", map[string]any{"v": 1}, nil)
		if err == nil {
			t.Errorf("Put namespace %s (%v) succeeded; want an error", c.name, c.ns)
			continue
		}
		if !strings.Contains(err.Error(), "invalid namespace") {
			t.Errorf("Put namespace %s error = %v, want an invalid-namespace error", c.name, err)
		}
	}
	// A valid namespace still succeeds (sanity).
	if err := s.Put(ctx, []string{"valid", "ns"}, "k", map[string]any{"v": 1}, nil); err != nil {
		t.Errorf("Put valid namespace: %v", err)
	}
}

// testConcurrentPutGet hammers one Store from many goroutines (distinct
// namespaces + keys) and verifies every written item is readable afterwards,
// and that concurrent Get calls never panic or error.
func testConcurrentPutGet(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	const goroutines = 16
	const putsPerGoroutine = 10
	errs := make(chan error, goroutines*putsPerGoroutine)
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			ns := []string{"ns", fmt.Sprintf("g%d", g)}
			for i := 0; i < putsPerGoroutine; i++ {
				key := fmt.Sprintf("k%d", i)
				val := map[string]any{"g": g, "i": i}
				if err := s.Put(ctx, ns, key, val, nil); err != nil {
					errs <- fmt.Errorf("Put g%d/%s: %w", g, key, err)
					return
				}
				// Concurrent Get of the just-written key.
				if _, err := s.Get(ctx, ns, key); err != nil {
					errs <- fmt.Errorf("Get g%d/%s: %w", g, key, err)
					return
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent: %v", err)
	}
	if t.Failed() {
		return
	}
	// Every written item is now readable.
	for g := 0; g < goroutines; g++ {
		ns := []string{"ns", fmt.Sprintf("g%d", g)}
		results, err := s.Search(ctx, ns, store.SearchOptions{Limit: 100})
		if err != nil {
			t.Fatalf("Search g%d: %v", g, err)
		}
		if len(results) != putsPerGoroutine {
			t.Errorf("g%d: %d items, want %d", g, len(results), putsPerGoroutine)
		}
	}
}

// testListNamespaces: ListNamespaces returns the distinct namespaces in the
// store in deterministic lexicographic order, honoring prefix/suffix filters,
// max_depth truncation (which also dedupes), and offset/limit pagination.
func testListNamespaces(t *testing.T, newStore func(t *testing.T) store.Store) {
	t.Helper()
	ctx := context.Background()
	s := newStore(t)

	// Seed one item per namespace. The store's map iteration order is random,
	// so the assertions below also pin down the deterministic sort.
	seed := [][]string{
		{"a", "b", "c"},
		{"a", "b", "d"},
		{"a", "b", "e", "f"},
		{"a", "c", "d"},
		{"z", "b", "c"},
	}
	for i, ns := range seed {
		if err := s.Put(ctx, ns, fmt.Sprintf("k%d", i), map[string]any{"i": i}, nil); err != nil {
			t.Fatalf("Put %v: %v", ns, err)
		}
	}

	depth := func(d int) *int { return &d }

	cases := []struct {
		name string
		opts store.ListNamespacesOptions
		want [][]string
	}{
		{
			name: "all",
			want: [][]string{
				{"a", "b", "c"},
				{"a", "b", "d"},
				{"a", "b", "e", "f"},
				{"a", "c", "d"},
				{"z", "b", "c"},
			},
		},
		{
			name: "prefix",
			opts: store.ListNamespacesOptions{Prefix: []string{"a", "b"}},
			want: [][]string{
				{"a", "b", "c"},
				{"a", "b", "d"},
				{"a", "b", "e", "f"},
			},
		},
		{
			name: "suffix",
			opts: store.ListNamespacesOptions{Suffix: []string{"d"}},
			want: [][]string{
				{"a", "b", "d"},
				{"a", "c", "d"},
			},
		},
		{
			name: "prefix_and_suffix",
			opts: store.ListNamespacesOptions{Prefix: []string{"a", "b"}, Suffix: []string{"d"}},
			want: [][]string{
				{"a", "b", "d"},
			},
		},
		{
			name: "max_depth_truncates_and_dedupes",
			opts: store.ListNamespacesOptions{Prefix: []string{"a", "b"}, MaxDepth: depth(3)},
			want: [][]string{
				{"a", "b", "c"},
				{"a", "b", "d"},
				{"a", "b", "e"},
			},
		},
		{
			name: "max_depth_dedupes",
			opts: store.ListNamespacesOptions{MaxDepth: depth(2)},
			want: [][]string{
				{"a", "b"},
				{"a", "c"},
				{"z", "b"},
			},
		},
		{
			name: "limit",
			opts: store.ListNamespacesOptions{Limit: 2},
			want: [][]string{
				{"a", "b", "c"},
				{"a", "b", "d"},
			},
		},
		{
			name: "offset",
			opts: store.ListNamespacesOptions{Offset: 2},
			want: [][]string{
				{"a", "b", "e", "f"},
				{"a", "c", "d"},
				{"z", "b", "c"},
			},
		},
		{
			name: "offset_and_limit",
			opts: store.ListNamespacesOptions{Offset: 1, Limit: 2},
			want: [][]string{
				{"a", "b", "d"},
				{"a", "b", "e", "f"},
			},
		},
		{
			name: "offset_past_end",
			opts: store.ListNamespacesOptions{Offset: 100},
			want: [][]string{},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := s.ListNamespaces(ctx, c.opts)
			if err != nil {
				t.Fatalf("ListNamespaces: %v", err)
			}
			if !namespacesEqual(got, c.want) {
				t.Errorf("ListNamespaces(%+v) = %v, want %v", c.opts, got, c.want)
			}
		})
	}
}

// namespacesEqual reports whether two namespace slices-of-slices are equal,
// element by element (ignoring nil-vs-empty for the outer slice).
func namespacesEqual(a, b [][]string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !reflect.DeepEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}

// keysOf joins the keys of the search results (in order) with commas, for
// compact assertion messages.
func keysOf(results []store.SearchItem) string {
	keys := make([]string, len(results))
	for i, r := range results {
		keys[i] = r.Key
	}
	return strings.Join(keys, ",")
}
