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
