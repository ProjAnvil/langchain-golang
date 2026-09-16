package store

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// InMemoryStore is a dictionary-backed Store that keeps every item in process
// memory, mirroring Python's `langgraph.store.memory.InMemoryStore`. It is safe
// for concurrent use by any number of goroutines.
//
// Semantic search: NewInMemoryStore mirrors Python's InMemoryStore(index=None)
// — no vector index; SearchOptions.Query is accepted but silently ignored
// (Python's _batch_search drops the query the same way). Constructing with
// NewInMemoryStoreWithIndex mirrors InMemoryStore(index={"dims": ...,
// "embed": ..., "fields": [...]}): Put embeds each index field of the value
// and stores the vectors alongside the item, and Search with a non-empty
// Query embeds the query and ranks matching items by cosine similarity,
// descending. Vectors live only in memory; they are never persisted.
//
// Documented divergences from Python:
//   - Python returns items in insertion order; Go sorts deterministically by
//     (namespace, key), including on similarity ties (Python's stable sort
//     keeps insertion order).
//   - A Put replaces the item's vectors wholesale; Python merges per-path and
//     would keep stale vectors for indexed fields that disappeared from the
//     updated value.
//   - Index fields are plain names or "$" (the whole value as JSON); Python's
//     dotted paths, "[i]"/"[*]" indexing, "*", and "{a,b}" multi-field path
//     expressions are not ported.
//   - Non-string scalars are rendered approximately like Python's str() (e.g.
//     2.0 renders "2", not "2.0").
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
	// index is non-nil when the store was built with NewInMemoryStoreWithIndex
	// (Python's InMemoryStore(index=...)); it configures Put-time embedding
	// and Query-time cosine ranking. nil means Python's index=None default:
	// no vectors, Query ignored.
	index *IndexConfig
	// vectors mirrors data's shape — joinNS(namespace) → item key → index
	// field → embedding vector — and is only populated when index != nil.
	vectors map[string]map[string]map[string][]float64
}

// Compile-time assertion that InMemoryStore satisfies Store.
var _ Store = (*InMemoryStore)(nil)

// NewInMemoryStore returns an empty InMemoryStore without a semantic index
// (Python's InMemoryStore(index=None)): Put does not embed anything and Search
// ignores SearchOptions.Query.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		data:    map[string]map[string]*Item{},
		nsOf:    map[string][]string{},
		vectors: map[string]map[string]map[string][]float64{},
	}
}

// NewInMemoryStoreWithIndex returns an empty InMemoryStore with semantic search
// enabled, mirroring Python's InMemoryStore(index={"dims": ..., "embed": ...,
// "fields": [...]}): Put embeds each index field of the value (see IndexConfig
// for field semantics; Put's index argument can override them per call), and
// Search with a non-empty SearchOptions.Query ranks results by cosine
// similarity, descending. It panics when index.Embed is nil, mirroring the
// ValueError Python's ensure_embeddings raises for embed=None.
func NewInMemoryStoreWithIndex(index IndexConfig) *InMemoryStore {
	if index.Embed == nil {
		panic("store: NewInMemoryStoreWithIndex: IndexConfig.Embed must be provided")
	}
	// Copy the fields (Python copies its index_config) and apply the same
	// default Python does: fields or ["$"] → embed the whole value.
	fields := append([]string(nil), index.Fields...)
	if len(fields) == 0 {
		fields = []string{"$"}
	}
	return &InMemoryStore{
		data:    map[string]map[string]*Item{},
		nsOf:    map[string][]string{},
		index:   &IndexConfig{Dims: index.Dims, Embed: index.Embed, Fields: fields},
		vectors: map[string]map[string]map[string][]float64{},
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
// updates).
//
// When the store has an index (NewInMemoryStoreWithIndex), each index field of
// value is embedded before the item is stored, and the item's vectors are
// replaced wholesale — an update recomputes them (see the InMemoryStore doc
// comment for the Python comparison). The index argument overrides the
// configured fields for this call: nil uses them as configured; a non-nil
// slice embeds exactly those fields (an empty non-nil slice embeds nothing,
// standing in for Python's put(..., index=False)). An embedding error is
// returned and the item is NOT stored — mirroring Python's batch, which embeds
// (_extract_texts → embed_documents) before applying put ops. Returns
// ErrInvalidNamespace when namespace fails validation.
func (s *InMemoryStore) Put(ctx context.Context, namespace []string, key string, value map[string]any, index []string) error {
	if err := validateNamespace(namespace); err != nil {
		return err
	}
	nsCopy := append([]string(nil), namespace...)
	valCopy := cloneValue(value)

	// Embed before taking the lock: embedding may call out to a model. The
	// vectors are computed from valCopy, so the stored item and its vectors
	// always agree.
	var fieldVecs map[string][]float64
	if s.index != nil {
		fields := s.index.Fields
		if index != nil {
			fields = index
		}
		var err error
		fieldVecs, err = s.embedFields(ctx, valCopy, fields)
		if err != nil {
			return err
		}
	}

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
	if s.index != nil {
		if len(fieldVecs) == 0 {
			// Wholesale replace: nothing embedded → no vectors. (Python would
			// keep stale per-path vectors here; see the doc comment.)
			if vb := s.vectors[j]; vb != nil {
				delete(vb, key)
				if len(vb) == 0 {
					delete(s.vectors, j)
				}
			}
		} else {
			vb := s.vectors[j]
			if vb == nil {
				vb = map[string]map[string][]float64{}
				s.vectors[j] = vb
			}
			vb[key] = fieldVecs
		}
	}
	return nil
}

// embedFields embeds the given index fields of value, mirroring the
// _extract_texts → embed_documents → _insertinmem_store pipeline of Python's
// InMemoryStore.batch for a single Put. It returns a fresh field→vector map
// (nil when no field yields text) that the caller can store without aliasing.
func (s *InMemoryStore) embedFields(ctx context.Context, value map[string]any, fields []string) (map[string][]float64, error) {
	var texts, used []string
	for _, field := range fields {
		text, ok, err := textAtField(value, field)
		if err != nil {
			return nil, fmt.Errorf("store: render index field %q: %w", field, err)
		}
		if !ok {
			continue
		}
		texts = append(texts, text)
		used = append(used, field)
	}
	if len(texts) == 0 {
		return nil, nil
	}
	vectors, err := s.index.Embed.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("store: embed fields %v: %w", used, err)
	}
	if len(vectors) != len(texts) {
		// Mirrors Python's ValueError when the embeddings count does not match
		// the number of indices.
		return nil, fmt.Errorf("store: embedder returned %d vectors for %d texts", len(vectors), len(texts))
	}
	out := make(map[string][]float64, len(used))
	for i, field := range used {
		out[field] = append([]float64(nil), vectors[i]...)
	}
	return out, nil
}

// textAtField renders the text Python's get_text_at_path would embed for a
// simple index field: "$" encodes the whole value as sorted-key JSON
// (json.dumps(obj, sort_keys=True) — a nil value encodes as "null", matching
// json.dumps(None)); a named field embeds a string as-is, a map or slice as
// sorted-key JSON, and any other scalar via scalarText. A missing or nil field
// yields no text.
func textAtField(value map[string]any, field string) (string, bool, error) {
	var v any = value
	if field != "$" {
		raw, ok := value[field]
		if !ok || raw == nil {
			return "", false, nil
		}
		v = raw
	}
	switch x := v.(type) {
	case string:
		return x, true, nil
	case map[string]any, []any:
		encoded, err := json.Marshal(x)
		if err != nil {
			return "", false, err
		}
		return string(encoded), true, nil
	default:
		return scalarText(v), true, nil
	}
}

// scalarText approximates Python's str() for non-string scalars: bools render
// exactly ("True"/"False"); integers and floats render via strconv (float
// formatting differs from Python's repr, e.g. 2.0 renders "2" rather than
// "2.0"); anything else falls back to fmt.Sprint.
func scalarText(v any) string {
	switch x := v.(type) {
	case bool:
		if x {
			return "True"
		}
		return "False"
	case int:
		return strconv.FormatInt(int64(x), 10)
	case int8:
		return strconv.FormatInt(int64(x), 10)
	case int16:
		return strconv.FormatInt(int64(x), 10)
	case int32:
		return strconv.FormatInt(int64(x), 10)
	case int64:
		return strconv.FormatInt(x, 10)
	case uint:
		return strconv.FormatUint(uint64(x), 10)
	case uint8:
		return strconv.FormatUint(uint64(x), 10)
	case uint16:
		return strconv.FormatUint(uint64(x), 10)
	case uint32:
		return strconv.FormatUint(uint64(x), 10)
	case uint64:
		return strconv.FormatUint(x, 10)
	case float32:
		return strconv.FormatFloat(float64(x), 'g', -1, 32)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	default:
		return fmt.Sprint(v)
	}
}

// Delete removes the item stored under (namespace, key), together with its
// vectors (Python's PutOp(value=None) pops _vectors[namespace] too). It is NOT
// an error when no such item exists (mirroring Python, which pops a missing
// key silently). Empty namespace buckets are pruned so Search never iterates
// dead buckets.
func (s *InMemoryStore) Delete(_ context.Context, namespace []string, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	j := joinNS(namespace)
	bucket := s.data[j]
	if bucket == nil {
		return nil
	}
	delete(bucket, key)
	if vb := s.vectors[j]; vb != nil {
		delete(vb, key)
		if len(vb) == 0 {
			delete(s.vectors, j)
		}
	}
	if len(bucket) == 0 {
		delete(s.data, j)
		delete(s.nsOf, j)
	}
	return nil
}

// Search returns items whose Namespace has namespacePrefix as a prefix
// (element-wise; an empty prefix matches every namespace), filtered by
// opts.Filter, then paginated by opts.Offset/opts.Limit.
//
// Without a Query — or on a store without an index (Python's index=None, which
// drops the query silently) — results are ordered deterministically by
// (namespace, key). With an index and a non-empty Query, the query is embedded
// and candidates are ranked by cosine similarity against their stored field
// vectors, descending, max-pooled across each item's vectors, then paginated
// by Offset/Limit; unembedded items (score 0) pad the tail when fewer than
// Limit ranked results exist — mirroring Python's _batch_search. A
// zero/negative Limit defaults to 10 (Python parity).
func (s *InMemoryStore) Search(ctx context.Context, namespacePrefix []string, opts SearchOptions) ([]SearchItem, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	offset := max(opts.Offset, 0)

	// Python embeds every search query up front (_embed_search_queries),
	// before filtering and even when no item ends up matching, so an embedder
	// error always surfaces. Embedding runs outside the store lock.
	semantic := s.index != nil && opts.Query != ""
	var queryVec []float64
	if semantic {
		var err error
		queryVec, err = s.index.Embed.EmbedQuery(ctx, opts.Query)
		if err != nil {
			return nil, fmt.Errorf("store: embed query %q: %w", opts.Query, err)
		}
	}

	s.mu.RLock()
	var picks []searchCandidate
	for j, bucket := range s.data {
		ns := s.nsOf[j]
		if !namespaceHasPrefix(ns, namespacePrefix) {
			continue
		}
		var vecBucket map[string]map[string][]float64
		if semantic {
			vecBucket = s.vectors[j]
		}
		for _, it := range bucket {
			if opts.Filter != nil && !compareValues(it.Value, opts.Filter) {
				continue
			}
			c := searchCandidate{item: it}
			if semantic {
				for _, v := range vecBucket[it.Key] {
					c.vecs = append(c.vecs, v)
				}
			}
			picks = append(picks, c)
		}
	}
	s.mu.RUnlock()

	if !semantic {
		return searchUnranked(picks, offset, limit), nil
	}
	return searchRanked(picks, queryVec, offset, limit), nil
}

// searchCandidate pairs a filtered item with its field vectors (empty for the
// non-semantic path, or for items whose fields were never embedded).
type searchCandidate struct {
	item *Item
	vecs [][]float64
}

// searchUnranked implements the non-semantic Search path: deterministic
// (namespace, key) order (Python returns insertion order; Go map iteration is
// random, so this port pins a stable, testable ordering), then offset/limit
// pagination. All scores are zero.
func searchUnranked(picks []searchCandidate, offset, limit int) []SearchItem {
	slices.SortFunc(picks, func(a, b searchCandidate) int {
		return cmp.Or(
			strings.Compare(joinNS(a.item.Namespace), joinNS(b.item.Namespace)),
			cmp.Compare(a.item.Key, b.item.Key),
		)
	})
	if offset >= len(picks) {
		return []SearchItem{}
	}
	picks = picks[offset:]
	if limit < len(picks) {
		picks = picks[:limit]
	}
	out := make([]SearchItem, len(picks))
	for i, p := range picks {
		// Return defensive copies so callers cannot mutate the stored items.
		out[i] = SearchItem{Item: cloneItem(p.item)}
	}
	return out
}

// searchRanked implements the semantic Search path, mirroring Python's
// _batch_search: cosine-score every candidate that has vectors (max-pooling
// across its fields — Python sorts every (score, item) pair and dedupes by
// (namespace, key), keeping the max), sort descending with a deterministic
// (namespace, key) tie-break, slice [offset, offset+limit), then pad with
// unembedded candidates (score 0) whenever the window is shorter than limit —
// including Python's corner case of padding after a large offset. Python pads
// in insertion order; Go pads in the deterministic (namespace, key) order.
func searchRanked(picks []searchCandidate, queryVec []float64, offset, limit int) []SearchItem {
	type scored struct {
		score float64
		item  *Item
	}
	var ranked []scored
	var scoreless []*Item
	for _, p := range picks {
		if len(p.vecs) == 0 {
			scoreless = append(scoreless, p.item)
			continue
		}
		best := math.Inf(-1)
		for _, v := range p.vecs {
			if s := cosineSimilarity(queryVec, v); s > best {
				best = s
			}
		}
		ranked = append(ranked, scored{score: best, item: p.item})
	}
	slices.SortFunc(ranked, func(a, b scored) int {
		return cmp.Or(
			cmp.Compare(b.score, a.score),
			strings.Compare(joinNS(a.item.Namespace), joinNS(b.item.Namespace)),
			cmp.Compare(a.item.Key, b.item.Key),
		)
	})

	var kept []scored
	if offset < len(ranked) {
		window := ranked[offset:]
		if limit < len(window) {
			window = window[:limit]
		}
		kept = append(kept, window...)
	}
	if len(kept) < limit && len(scoreless) > 0 {
		slices.SortFunc(scoreless, func(a, b *Item) int {
			return cmp.Or(
				strings.Compare(joinNS(a.Namespace), joinNS(b.Namespace)),
				cmp.Compare(a.Key, b.Key),
			)
		})
		n := limit - len(kept)
		if n > len(scoreless) {
			n = len(scoreless)
		}
		for _, item := range scoreless[:n] {
			kept = append(kept, scored{score: 0, item: item})
		}
	}

	out := make([]SearchItem, len(kept))
	for i, p := range kept {
		out[i] = SearchItem{Item: cloneItem(p.item), Score: p.score}
	}
	return out
}

// cloneItem returns a defensive copy of item for Search results, so callers
// cannot mutate stored items through them.
func cloneItem(item *Item) Item {
	return Item{
		Namespace: append([]string(nil), item.Namespace...),
		Key:       item.Key,
		Value:     cloneValue(item.Value),
		CreatedAt: item.CreatedAt,
		UpdatedAt: item.UpdatedAt,
	}
}

// cosineSimilarity computes the cosine similarity of a and b, mirroring
// Python's pure-Python _cosine_similarity: the dot product truncates at the
// shorter vector (zip(strict=False)) while the norms use the full vectors, and
// a zero-norm vector scores 0 instead of dividing by zero.
func cosineSimilarity(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, normA, normB float64
	for i := range n {
		dot += a[i] * b[i]
	}
	for _, x := range a {
		normA += x * x
	}
	for _, x := range b {
		normB += x * x
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ListNamespaces returns the distinct namespaces present in the store, filtered
// by opts.Prefix (element-wise prefix) and opts.Suffix (element-wise suffix),
// truncated to opts.MaxDepth when set (truncated duplicates are deduped),
// sorted lexicographically by namespace segments, then paginated by
// opts.Offset/opts.Limit. A zero/negative Limit defaults to 100 (Python
// parity: Python's default limit is 100). Returned namespace slices are
// defensive copies.
func (s *InMemoryStore) ListNamespaces(_ context.Context, opts ListNamespacesOptions) ([][]string, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 100
	}
	offset := max(opts.Offset, 0)

	s.mu.RLock()
	// Collect distinct namespaces (one per non-empty data bucket) and apply
	// prefix/suffix filters. A map keyed by the (possibly truncated) joined
	// namespace dedupes, which is only ever needed after truncation — each
	// stored namespace is already distinct — but applying it unconditionally
	// keeps the code uniform and matches Python's set-based dedupe.
	unique := make(map[string][]string)
	for _, ns := range s.nsOf {
		if len(opts.Prefix) > 0 && !namespaceHasPrefix(ns, opts.Prefix) {
			continue
		}
		if len(opts.Suffix) > 0 && !namespaceHasSuffix(ns, opts.Suffix) {
			continue
		}
		key := ns
		if opts.MaxDepth != nil {
			d := max(*opts.MaxDepth, 0)
			if d < len(ns) {
				key = ns[:d]
			}
		}
		unique[joinNS(key)] = key
	}
	s.mu.RUnlock()

	namespaces := make([][]string, 0, len(unique))
	for _, ns := range unique {
		namespaces = append(namespaces, ns)
	}
	// Deterministic order: Python sorts the (truncated) namespace tuples
	// lexicographically; joinNS comparison reproduces that element-wise.
	slices.SortFunc(namespaces, func(a, b []string) int {
		return strings.Compare(joinNS(a), joinNS(b))
	})

	if offset >= len(namespaces) {
		return [][]string{}, nil
	}
	namespaces = namespaces[offset:]
	if limit < len(namespaces) {
		namespaces = namespaces[:limit]
	}
	out := make([][]string, len(namespaces))
	for i, ns := range namespaces {
		cp := make([]string, len(ns))
		copy(cp, ns)
		out[i] = cp
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
