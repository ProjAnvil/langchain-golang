package retrievers

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// TimeWeightedVectorStoreRetriever combines embedding similarity with recency
// and salience when retrieving documents. It keeps a memory stream of every
// document added, tags each with a buffer index and access timestamps, and on
// retrieval rescores the most recent documents plus the documents found
// salient to the query by:
//
//	score = (1 - decayRate)^hoursPassed + salience + similarity
//
// where hoursPassed is the time since the document was last accessed, salience
// is the sum of the configured other_score_keys metadata values, and
// similarity is the vector relevance score (or the configured default
// salience for recent documents that were not found salient). The top k
// documents by score are returned, and their last_accessed_at timestamps are
// refreshed so frequently accessed memories are not forgotten.
//
// Python parity: langchain_classic.retrievers.time_weighted_retriever.
// TimeWeightedVectorStoreRetriever (get_salient_docs, add_documents,
// _get_combined_score, _get_rescored_docs). The Go form exposes an injectable
// clock (WithTimeWeightedNow) so scoring is deterministic in tests.
type TimeWeightedVectorStoreRetriever struct {
	vectorstore  vectorstores.VectorStore
	decayRate    float64
	k            int
	searchKwargs map[string]any
	otherKeys    []string
	defaultSal   *float64
	now          func() time.Time

	mu           sync.Mutex
	memoryStream []documents.Document
}

// SalientDocument pairs a memory-stream document with the vector relevance
// score assigned to it by GetSalientDocs.
type SalientDocument struct {
	// Document is a defensive copy of the memory-stream document.
	Document documents.Document
	// Relevance is the vector relevance score for the document.
	Relevance float64
}

// TimeWeightedOption configures a TimeWeightedVectorStoreRetriever.
type TimeWeightedOption func(*TimeWeightedVectorStoreRetriever)

// WithTimeWeightedDecayRate sets the exponential decay factor used as
// (1 - decayRate)^hoursPassed (Python default 0.01). Values are clamped to
// [0, 1) during scoring.
func WithTimeWeightedDecayRate(rate float64) TimeWeightedOption {
	return func(r *TimeWeightedVectorStoreRetriever) {
		r.decayRate = rate
	}
}

// WithTimeWeightedK sets the maximum number of documents returned per call
// (Python default 4).
func WithTimeWeightedK(k int) TimeWeightedOption {
	return func(r *TimeWeightedVectorStoreRetriever) {
		if k > 0 {
			r.k = k
		}
	}
}

// WithTimeWeightedSearchKwargs sets the keyword arguments forwarded to the
// relevance-score search performed by GetSalientDocs (Python default
// {"k": 100}; the "k" entry is honored here).
func WithTimeWeightedSearchKwargs(kwargs map[string]any) TimeWeightedOption {
	return func(r *TimeWeightedVectorStoreRetriever) {
		r.searchKwargs = kwargs
	}
}

// WithTimeWeightedOtherScoreKeys sets additional metadata keys whose numeric
// values are added to each document's score (Python other_score_keys).
func WithTimeWeightedOtherScoreKeys(keys []string) TimeWeightedOption {
	return func(r *TimeWeightedVectorStoreRetriever) {
		r.otherKeys = slices.Clone(keys)
	}
}

// WithTimeWeightedDefaultSalience sets the salience assigned to recent
// documents that were not retrieved as salient. When unset (Python default
// None), such documents receive no salience/relevance boost.
func WithTimeWeightedDefaultSalience(value float64) TimeWeightedOption {
	return func(r *TimeWeightedVectorStoreRetriever) {
		v := value
		r.defaultSal = &v
	}
}

// WithTimeWeightedNow overrides the clock used for timestamps and scoring. It
// is primarily useful for deterministic tests; the default is time.Now.
func WithTimeWeightedNow(now func() time.Time) TimeWeightedOption {
	return func(r *TimeWeightedVectorStoreRetriever) {
		if now != nil {
			r.now = now
		}
	}
}

// NewTimeWeightedVectorStoreRetriever creates a time-weighted retriever over
// vectorstore. The store is required and must support relevance-score search
// (SimilaritySearchWithRelevanceScores) for GetSalientDocs.
func NewTimeWeightedVectorStoreRetriever(
	vectorstore vectorstores.VectorStore,
	opts ...TimeWeightedOption,
) (*TimeWeightedVectorStoreRetriever, error) {
	if vectorstore == nil {
		return nil, fmt.Errorf("vector store is required")
	}
	retriever := &TimeWeightedVectorStoreRetriever{
		vectorstore:  vectorstore,
		decayRate:    0.01,
		k:            4,
		searchKwargs: map[string]any{"k": 100},
		now:          time.Now,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(retriever)
		}
	}
	return retriever, nil
}

var _ Retriever = (*TimeWeightedVectorStoreRetriever)(nil)

// AddDocuments adds documents to the memory stream and the vector store. It
// deep-copies the inputs (the conversation "never mutate the caller's
// documents" rule), stamps last_accessed_at and created_at with the current
// clock when absent, and assigns each document a buffer_idx equal to its
// position in the memory stream (Python: add_documents). The stamped copies
// are what get indexed in the vector store.
func (r *TimeWeightedVectorStoreRetriever) AddDocuments(
	ctx context.Context,
	docs []documents.Document,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(docs) == 0 {
		return nil
	}
	now := r.now()

	r.mu.Lock()
	baseIdx := len(r.memoryStream)
	dupDocs := make([]documents.Document, len(docs))
	for i, doc := range docs {
		dup := doc.Clone()
		if dup.Metadata == nil {
			dup.Metadata = map[string]any{}
		}
		if _, ok := dup.Metadata["last_accessed_at"]; !ok {
			dup.Metadata["last_accessed_at"] = now
		}
		if _, ok := dup.Metadata["created_at"]; !ok {
			dup.Metadata["created_at"] = now
		}
		dup.Metadata["buffer_idx"] = baseIdx + i
		dupDocs[i] = dup
	}
	r.memoryStream = append(r.memoryStream, dupDocs...)
	r.mu.Unlock()

	if _, err := r.vectorstore.AddDocuments(ctx, dupDocs); err != nil {
		return fmt.Errorf("add documents to vector store: %w", err)
	}
	return nil
}

// GetSalientDocs returns the memory-stream documents that are salient to the
// query, keyed by their buffer index. It runs a relevance-score search over
// the vector store, and for each hit carrying a valid buffer_idx metadata
// entry returns the corresponding memory-stream document with its relevance
// score (Python: get_salient_docs).
func (r *TimeWeightedVectorStoreRetriever) GetSalientDocs(
	ctx context.Context,
	query string,
) (map[int]SalientDocument, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	searcher, ok := r.vectorstore.(relevanceScoreSearcher)
	if !ok {
		return nil, fmt.Errorf(
			"vector store %T does not support relevance-score search",
			r.vectorstore,
		)
	}
	k := searchKwargInt(r.searchKwargs, "k", 100)
	results, err := searcher.SimilaritySearchWithRelevanceScores(ctx, query, k, nil)
	if err != nil {
		return nil, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[int]SalientDocument, len(results))
	for _, result := range results {
		idx, ok := timeWeightedBufferIndex(result.Document)
		if !ok || idx < 0 || idx >= len(r.memoryStream) {
			continue
		}
		out[idx] = SalientDocument{
			Document:  r.memoryStream[idx].Clone(),
			Relevance: result.Score,
		}
	}
	return out, nil
}

// GetRelevantDocuments rescores the most recent memory-stream documents plus
// the query-salient documents and returns the top k by combined score,
// refreshing their last_accessed_at timestamps (Python:
// _get_relevant_documents + _get_rescored_docs).
func (r *TimeWeightedVectorStoreRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	salient, err := r.GetSalientDocs(ctx, query)
	if err != nil {
		return nil, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.now()

	// Candidates are keyed by buffer index. The most recent k documents enter
	// with the configured default salience as their relevance; salient hits
	// override them with their real vector relevance (Python: the
	// docs_and_scores dict updated by get_salient_docs).
	type candidate struct {
		doc       documents.Document
		relevance *float64
	}
	candidates := make(map[int]candidate)
	start := max(0, len(r.memoryStream)-r.k)
	for i := start; i < len(r.memoryStream); i++ {
		idx, ok := timeWeightedBufferIndex(r.memoryStream[i])
		if !ok {
			continue
		}
		candidates[idx] = candidate{doc: r.memoryStream[i].Clone(), relevance: r.defaultSal}
	}
	for idx, sd := range salient {
		relevance := sd.Relevance
		candidates[idx] = candidate{doc: sd.Document.Clone(), relevance: &relevance}
	}

	type scored struct {
		idx   int
		score float64
	}
	scoredDocs := make([]scored, 0, len(candidates))
	for idx, cand := range candidates {
		scoredDocs = append(scoredDocs, scored{
			idx:   idx,
			score: r.combinedScore(cand.doc, cand.relevance, now),
		})
	}
	// Stable descending sort; buffer index breaks ties so the result does not
	// depend on map iteration order.
	slices.SortStableFunc(scoredDocs, func(a, b scored) int {
		if a.score != b.score {
			if a.score > b.score {
				return -1
			}
			return 1
		}
		if a.idx < b.idx {
			return -1
		}
		if a.idx > b.idx {
			return 1
		}
		return 0
	})

	result := make([]documents.Document, 0, r.k)
	for i := 0; i < len(scoredDocs) && i < r.k; i++ {
		idx := scoredDocs[i].idx
		if idx < 0 || idx >= len(r.memoryStream) {
			continue
		}
		// Clone before mutating metadata, then refresh the access time.
		updated := r.memoryStream[idx].Clone()
		if updated.Metadata == nil {
			updated.Metadata = map[string]any{}
		}
		updated.Metadata["last_accessed_at"] = now
		r.memoryStream[idx] = updated
		result = append(result, updated.Clone())
	}
	return result, nil
}

// combinedScore mirrors Python's _get_combined_score: recency plus the sum of
// other_score_keys metadata values (salience) plus the optional vector
// relevance (similarity).
func (r *TimeWeightedVectorStoreRetriever) combinedScore(
	doc documents.Document,
	relevance *float64,
	now time.Time,
) float64 {
	hoursPassed := now.Sub(timeWeightedDocumentTime(doc, "last_accessed_at", now)).Hours()
	base := 1.0 - r.decayRate
	base = math.Max(0, math.Min(1, base))
	score := math.Pow(base, hoursPassed)
	for _, key := range r.otherKeys {
		if value, ok := timeWeightedNumericMetadata(doc, key); ok {
			score += value
		}
	}
	if relevance != nil {
		score += *relevance
	}
	return score
}

// timeWeightedBufferIndex reads the buffer_idx metadata value as an int.
func timeWeightedBufferIndex(doc documents.Document) (int, bool) {
	value, ok := doc.Metadata["buffer_idx"]
	if !ok || value == nil {
		return 0, false
	}
	switch n := value.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	case float32:
		return int(n), true
	default:
		return 0, false
	}
}

// timeWeightedNumericMetadata reads a numeric metadata value as a float64.
func timeWeightedNumericMetadata(doc documents.Document, key string) (float64, bool) {
	value, ok := doc.Metadata[key]
	if !ok || value == nil {
		return 0, false
	}
	switch n := value.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}

// timeWeightedDocumentTime reads a timestamp metadata value. It accepts a
// time.Time, or a Unix timestamp (int/int64/float64), defaulting to fallback
// when the field is absent or of an unsupported type (Python:
// _document_get_date falls back to datetime.now()).
func timeWeightedDocumentTime(doc documents.Document, key string, fallback time.Time) time.Time {
	value, ok := doc.Metadata[key]
	if !ok || value == nil {
		return fallback
	}
	switch t := value.(type) {
	case time.Time:
		return t
	case int64:
		return time.Unix(t, 0)
	case int:
		return time.Unix(int64(t), 0)
	case float64:
		return time.Unix(0, int64(t*float64(time.Second)))
	case float32:
		return time.Unix(0, int64(float64(t)*float64(time.Second)))
	default:
		return fallback
	}
}
