package store

import (
	"context"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"
)

// InMemoryStore is a dictionary-backed Store that keeps every item in process
// memory, mirroring Python's `langgraph.store.memory.InMemoryStore`. It is safe
// for concurrent use by any number of goroutines.
//
// Documented divergence from Python: the Python InMemoryStore optionally
// indexes items for semantic (vector) search when constructed with an
// `index={dims, embed, fields}` configuration, ranking Search results by cosine
// similarity when a Query is supplied. This Go port does not perform semantic
// search: Query is accepted by Search but ignored (items matching the namespace
// prefix and filter are returned unranked, exactly as Python's
// InMemoryStore(index=None) does). Index/embedding support can be added later
// without changing the Store interface.
//
// TTL (time-to-live) is also out of scope for this port (Python's
// supports_ttl is False by default).
type InMemoryStore struct {
	mu sync.RWMutex
	// data is a two-level map: outer key = joinNS(namespace) buckets the
	// items that share a namespace; inner key = the item key. nsOf remembers
	// the []string namespace for each bucket so Search can match prefixes
	// element-wise without parsing the joined key back apart. Empty buckets
	// are removed on Delete so every bucket in data has at least one item.
	data map[string]map[string]*Item
	nsOf map[string][]string
}

// Compile-time assertion that InMemoryStore satisfies Store.
var _ Store = (*InMemoryStore)(nil)

// NewInMemoryStore returns an empty InMemoryStore.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		data: map[string]map[string]*Item{},
		nsOf: map[string][]string{},
	}
}

// Get returns the item stored under (namespace, key), or (nil, nil) when no
// such item exists. Mirrors Python's InMemoryStore handling of GetOp.
func (s *InMemoryStore) Get(_ context.Context, namespace []string, key string) (*Item, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	bucket := s.data[joinNS(namespace)]
	if bucket == nil {
		return nil, nil
	}
	return bucket[key], nil
}

// Put stores value under (namespace, key), replacing any existing item.
// Mirrors Python's `_apply_put_ops`: a fresh Item is created with both
// CreatedAt and UpdatedAt set to now (CreatedAt is NOT preserved across
// updates). index is accepted but ignored (no semantic indexing in this port,
// mirroring InMemoryStore(index=None)). Returns ErrInvalidNamespace when
// namespace fails validation.
func (s *InMemoryStore) Put(_ context.Context, namespace []string, key string, value map[string]any, _ []string) error {
	if err := validateNamespace(namespace); err != nil {
		return err
	}
	nsCopy := append([]string(nil), namespace...)
	valCopy := cloneValue(value)
	now := time.Now()
	item := &Item{
		Namespace: nsCopy,
		Key:       key,
		Value:     valCopy,
		CreatedAt: now,
		UpdatedAt: now,
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	j := joinNS(namespace)
	bucket := s.data[j]
	if bucket == nil {
		bucket = map[string]*Item{}
		s.data[j] = bucket
		s.nsOf[j] = nsCopy
	}
	bucket[key] = item
	return nil
}

// Delete removes the item stored under (namespace, key). It is NOT an error
// when no such item exists (mirroring Python's PutOp(value=None), which pops a
// missing key silently). Empty namespace buckets are pruned so Search never
// iterates dead buckets.
func (s *InMemoryStore) Delete(_ context.Context, namespace []string, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	j := joinNS(namespace)
	bucket := s.data[j]
	if bucket == nil {
		return nil
	}
	delete(bucket, key)
	if len(bucket) == 0 {
		delete(s.data, j)
		delete(s.nsOf, j)
	}
	return nil
}

// Search returns items whose Namespace has namespacePrefix as a prefix
// (element-wise; an empty prefix matches every namespace), filtered by
// opts.Filter, ordered by (namespace, key), then paginated by
// opts.Offset/opts.Limit. opts.Query is accepted but ignored (no semantic
// ranking; see the InMemoryStore doc comment). A zero/negative Limit defaults
// to 10 (Python parity).
func (s *InMemoryStore) Search(_ context.Context, namespacePrefix []string, opts SearchOptions) ([]SearchItem, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	offset := opts.Offset
	if offset < 0 {
		offset = 0
	}

	s.mu.RLock()
	type matched struct {
		item *Item
	}
	var picks []matched
	for j, bucket := range s.data {
		ns := s.nsOf[j]
		if !namespaceHasPrefix(ns, namespacePrefix) {
			continue
		}
		for _, it := range bucket {
			if opts.Filter != nil && !compareValues(it.Value, opts.Filter) {
				continue
			}
			picks = append(picks, matched{item: it})
		}
	}
	s.mu.RUnlock()

	// Deterministic order: Python returns insertion order for non-query
	// search; Go map iteration is random, so sort by (namespace, key) for a
	// stable, testable ordering.
	sort.Slice(picks, func(i, j int) bool {
		a, b := picks[i].item, picks[j].item
		if c := strings.Compare(joinNS(a.Namespace), joinNS(b.Namespace)); c != 0 {
			return c < 0
		}
		return a.Key < b.Key
	})

	if offset >= len(picks) {
		return []SearchItem{}, nil
	}
	picks = picks[offset:]
	if limit < len(picks) {
		picks = picks[:limit]
	}
	out := make([]SearchItem, len(picks))
	for i, p := range picks {
		// Return defensive copies so callers cannot mutate the stored items.
		out[i] = SearchItem{
			Item: Item{
				Namespace: append([]string(nil), p.item.Namespace...),
				Key:       p.item.Key,
				Value:     cloneValue(p.item.Value),
				CreatedAt: p.item.CreatedAt,
				UpdatedAt: p.item.UpdatedAt,
			},
		}
	}
	return out, nil
}

// cloneValue returns a shallow copy of m so the store does not alias the
// caller's map (a Put must not be affected by the caller later mutating its
// value). Nested maps/slices are still shared — mirroring Python, which stores
// the dict reference directly; callers should not mutate nested structures
// after Put.
func cloneValue(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// compareValues mirrors Python's `_compare_values` (memory/__init__.py): a map
// filter whose keys all start with "$" applies comparison operators; a map
// filter otherwise matches a nested map recursively; a slice filter matches a
// slice element-wise; anything else is deep equality.
func compareValues(itemValue, filterValue any) bool {
	switch fv := filterValue.(type) {
	case map[string]any:
		hasOp := false
		for k := range fv {
			if strings.HasPrefix(k, "$") {
				hasOp = true
				break
			}
		}
		if hasOp {
			for opKey, opVal := range fv {
				if !applyOperator(itemValue, opKey, opVal) {
					return false
				}
			}
			return true
		}
		iv, ok := itemValue.(map[string]any)
		if !ok {
			return false
		}
		for k, v := range fv {
			if !compareValues(iv[k], v) {
				return false
			}
		}
		return true
	case []any:
		iv, ok := itemValue.([]any)
		if !ok || len(iv) != len(fv) {
			return false
		}
		for i, v := range fv {
			if !compareValues(iv[i], v) {
				return false
			}
		}
		return true
	default:
		return reflect.DeepEqual(itemValue, filterValue)
	}
}

// applyOperator mirrors Python's `_apply_operator`. Comparison operators
// ($gt/$gte/$lt/$lte) coerce both operands to float64; a non-numeric operand
// makes the comparison false (documented divergence: Python raises ValueError
// via float(), failing the whole batch; Go treats it as no-match so a bad
// filter narrows to zero results rather than erroring the Search).
func applyOperator(value any, operator string, opValue any) bool {
	switch operator {
	case "$eq":
		return reflect.DeepEqual(value, opValue)
	case "$ne":
		return !reflect.DeepEqual(value, opValue)
	case "$gt":
		a, ok1 := toFloat64(value)
		b, ok2 := toFloat64(opValue)
		return ok1 && ok2 && a > b
	case "$gte":
		a, ok1 := toFloat64(value)
		b, ok2 := toFloat64(opValue)
		return ok1 && ok2 && a >= b
	case "$lt":
		a, ok1 := toFloat64(value)
		b, ok2 := toFloat64(opValue)
		return ok1 && ok2 && a < b
	case "$lte":
		a, ok1 := toFloat64(value)
		b, ok2 := toFloat64(opValue)
		return ok1 && ok2 && a <= b
	default:
		return false
	}
}

// toFloat64 coerces Go numeric kinds to float64, mirroring Python's float().
// Returns (0, false) for non-numeric values.
func toFloat64(v any) (float64, bool) {
	switch x := v.(type) {
	case int:
		return float64(x), true
	case int8:
		return float64(x), true
	case int16:
		return float64(x), true
	case int32:
		return float64(x), true
	case int64:
		return float64(x), true
	case uint:
		return float64(x), true
	case uint8:
		return float64(x), true
	case uint16:
		return float64(x), true
	case uint32:
		return float64(x), true
	case uint64:
		return float64(x), true
	case float32:
		return float64(x), true
	case float64:
		return x, true
	default:
		return 0, false
	}
}
