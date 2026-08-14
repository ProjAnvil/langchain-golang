// Package store defines the cross-thread BaseStore interface and an in-memory
// implementation, mirroring Python's `langgraph.store.base` (BaseStore, Item,
// SearchItem, and the get/put/delete/search convenience surface) and
// `langgraph.store.memory.InMemoryStore`.
//
// This is a Python->Go port. Python's BaseStore exposes an abstract
// batch/abatch of Op discriminated unions (GetOp/PutOp/SearchOp/...) with
// sync and async convenience methods that delegate to batch. Go collapses
// that abstraction: the Store interface below is the direct, idiomatic
// surface (context.Context carries cancellation; there is no sync/async
// duality and no batch Op type). The semantics of each method mirror the
// corresponding Python convenience method.
//
// See
// https://github.com/langchain-ai/langgraph/blob/main/langgraph/store/base/__init__.py
package store

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Item is a stored key-value pair with metadata, mirroring Python's
// `langgraph.store.base.Item`. Namespace is the hierarchical path locating the
// item (e.g. []string{"memories", "user123"}); Key is the unique identifier
// within that namespace; Value is the stored payload (its keys are filterable
// via SearchOptions.Filter); CreatedAt and UpdatedAt are the timestamps set on
// the most recent Put (mirroring Python, a Put always creates a fresh Item, so
// CreatedAt is NOT preserved across updates — see InMemoryStore.Put).
type Item struct {
	// Namespace is the hierarchical path locating the item.
	Namespace []string
	// Key is the unique identifier within Namespace.
	Key string
	// Value is the stored payload; its top-level keys are filterable.
	Value map[string]any
	// CreatedAt is when the item was first stored (reset on every Put,
	// mirroring Python's _apply_put_ops which builds a fresh Item).
	CreatedAt time.Time
	// UpdatedAt is when the item was last stored.
	UpdatedAt time.Time
}

// SearchItem is an Item returned from Search, carrying a relevance score,
// mirroring Python's `langgraph.store.base.SearchItem`. Score is the similarity
// score when semantic search ranks results; for non-semantic Search it is the
// zero value (Python uses None, which Go represents as 0.0 in a float64).
type SearchItem struct {
	Item
	// Score is the semantic similarity score (0 for non-semantic search).
	Score float64
}

// SearchOptions configures a Search call, mirroring the keyword arguments of
// Python's `BaseStore.search` (namespace_prefix is a separate Store.Search
// parameter, matching Python's positional `namespace_prefix`).
type SearchOptions struct {
	// Query is a natural-language search query for semantic ranking. Requires
	// an index/embeddings configuration on the store implementation; the
	// InMemoryStore in this package does NOT perform semantic ranking (it has
	// no index configured), so Query is accepted but ignored — items matching
	// the namespace prefix and filter are returned unranked, mirroring Python's
	// InMemoryStore(index=None).
	Query string
	// Filter is a set of key/value predicates matched against each item's
	// Value. A scalar filter value matches the corresponding Value field by
	// deep equality; a map filter value whose keys all start with "$" is
	// treated as comparison operators ($eq/$ne/$gt/$gte/$lt/$lte) applied to
	// the Value field; otherwise a map filter value matches a nested map
	// Value field recursively; a slice filter value matches a slice Value
	// field element-wise. Mirrors Python's `_compare_values`.
	Filter map[string]any
	// Limit caps the number of items returned. Defaults to 10 when zero or
	// negative (mirroring Python's default limit=10).
	Limit int
	// Offset is the number of matching items to skip before returning results.
	Offset int
}

// ListNamespacesOptions configures a ListNamespaces call, mirroring the keyword
// arguments of Python's `BaseStore.list_namespaces` (prefix/suffix/max_depth/
// limit/offset).
type ListNamespacesOptions struct {
	// Prefix filters namespaces that start with this path, element-wise (nil
	// or empty matches every namespace). Mirrors Python's `prefix` argument.
	Prefix []string
	// Suffix filters namespaces that end with this path, element-wise (nil or
	// empty matches every namespace). Mirrors Python's `suffix` argument.
	Suffix []string
	// MaxDepth truncates each returned namespace to this many leading segments
	// (truncated duplicates are deduped). nil means no truncation. Mirrors
	// Python's `max_depth` argument, including its truncate-then-dedupe
	// behavior.
	MaxDepth *int
	// Limit caps the number of namespaces returned. Defaults to 100 when zero
	// or negative (mirroring Python's default limit=100; Go uses zero as the
	// "not specified" sentinel).
	Limit int
	// Offset is the number of matching namespaces to skip before returning
	// results.
	Offset int
}

// Store is the cross-thread persistent key-value store interface, mirroring
// Python's `langgraph.store.base.BaseStore` convenience surface (get/put/
// delete/search). Stores enable persistence and memory shared across threads,
// scoped to arbitrary hierarchical namespaces. It is DISTINCT from
// core/stores.BaseStore[V] (the generic typed KV) — this is the langgraph
// semantic store surfaced on Runtime.Store and create_agent(store=...).
type Store interface {
	// Get returns the item stored under (namespace, key), or (nil, nil) when
	// no such item exists. Mirrors Python's BaseStore.get.
	Get(ctx context.Context, namespace []string, key string) (*Item, error)

	// Put stores value under (namespace, key), replacing any existing item
	// (a fresh Item is created with both timestamps set to now, mirroring
	// Python's _apply_put_ops — CreatedAt is NOT preserved across updates).
	// index selects the value fields to index for semantic search; the
	// InMemoryStore in this package does not index, so index is accepted but
	// ignored (mirroring Python's InMemoryStore(index=None)). Mirrors Python's
	// BaseStore.put.
	Put(ctx context.Context, namespace []string, key string, value map[string]any, index []string) error

	// Delete removes the item stored under (namespace, key). It is NOT an
	// error when no such item exists (mirroring Python's BaseStore.delete,
	// which drops a missing key silently). Mirrors Python's BaseStore.delete.
	Delete(ctx context.Context, namespace []string, key string) error

	// Search returns items whose Namespace has namespacePrefix as a prefix
	// (element-wise; an empty prefix matches every namespace), filtered by
	// opts.Filter, ordered deterministically by (namespace, key), then
	// paginated by opts.Offset/opts.Limit. Mirrors Python's BaseStore.search
	// (the non-query path: Query-based semantic ranking is not supported by
	// the InMemoryStore in this package).
	Search(ctx context.Context, namespacePrefix []string, opts SearchOptions) ([]SearchItem, error)

	// ListNamespaces returns the distinct namespaces present in the store,
	// filtered by opts.Prefix (element-wise prefix) and opts.Suffix
	// (element-wise suffix), truncated to opts.MaxDepth when set (truncated
	// duplicates are deduped), sorted lexicographically by namespace segments,
	// then paginated by opts.Offset/opts.Limit (Limit defaults to 100 when
	// zero/negative). Mirrors Python's BaseStore.list_namespaces.
	ListNamespaces(ctx context.Context, opts ListNamespacesOptions) ([][]string, error)
}

// ErrInvalidNamespace is returned by Put when namespace fails validation,
// mirroring Python's InvalidNamespaceError. A namespace is invalid when it is
// empty, contains an empty label, contains a label with ".", or has "langgraph"
// as its root label (the reserved prefix).
var ErrInvalidNamespace = errors.New("store: invalid namespace")

// validateNamespace mirrors Python's `_validate_namespace` (base/__init__.py).
// It is applied on Put (and only Put, matching Python, which validates in
// put/aput but not in get/delete/search).
func validateNamespace(namespace []string) error {
	if len(namespace) == 0 {
		return fmt.Errorf("%w: namespace cannot be empty", ErrInvalidNamespace)
	}
	for _, label := range namespace {
		if label == "" {
			return fmt.Errorf("%w: namespace labels cannot be empty strings; got %v", ErrInvalidNamespace, namespace)
		}
		if strings.Contains(label, ".") {
			return fmt.Errorf("%w: namespace label %q cannot contain periods", ErrInvalidNamespace, label)
		}
	}
	if namespace[0] == "langgraph" {
		return fmt.Errorf("%w: root label cannot be %q", ErrInvalidNamespace, "langgraph")
	}
	return nil
}

// joinNS renders a namespace slice as a stable map key. Segments are joined by
// a NUL byte, which cannot appear in a validated label (labels are non-empty
// and period-free; NUL is simply an out-of-band separator). The joined key is
// only used internally as a map bucket — namespace prefix matching always
// compares the original []string slices element-wise, so the separator never
// affects matching.
func joinNS(namespace []string) string {
	return strings.Join(namespace, "\x00")
}

// namespaceHasPrefix reports whether ns starts with prefix element-wise. An
// empty prefix matches every namespace. Mirrors the prefix check in Python's
// InMemoryStore._filter_items.
func namespaceHasPrefix(ns, prefix []string) bool {
	if len(ns) < len(prefix) {
		return false
	}
	for i, p := range prefix {
		if ns[i] != p {
			return false
		}
	}
	return true
}

// namespaceHasSuffix reports whether ns ends with suffix element-wise. An empty
// suffix matches every namespace. Mirrors the suffix branch of Python's
// `_does_match` (which zips reversed(key) with reversed(path)).
func namespaceHasSuffix(ns, suffix []string) bool {
	if len(ns) < len(suffix) {
		return false
	}
	off := len(ns) - len(suffix)
	for i, s := range suffix {
		if ns[off+i] != s {
			return false
		}
	}
	return true
}
