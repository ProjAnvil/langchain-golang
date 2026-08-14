package retrievers

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// Retriever returns documents relevant to a query.
type Retriever interface {
	GetRelevantDocuments(ctx context.Context, query string) ([]documents.Document, error)
}

// Static is a deterministic retriever useful for tests and examples.
type Static struct {
	Documents []documents.Document
}

// GetRelevantDocuments returns a defensive copy of configured documents.
func (r Static) GetRelevantDocuments(ctx context.Context, _ string) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([]documents.Document, len(r.Documents))
	for i, doc := range r.Documents {
		out[i] = doc.Clone()
	}
	return out, nil
}

// Search type values mirror Python's VectorStoreRetriever.allowed_search_types.
const (
	searchTypeSimilarity          = "similarity"
	searchTypeMMR                 = "mmr"
	searchTypeSimilarityThreshold = "similarity_score_threshold"
)

// RetrieverOption configures a VectorStoreRetriever.
type RetrieverOption func(*retrieverConfig)

type retrieverConfig struct {
	searchType   string
	searchKwargs map[string]any
}

// WithSearchType sets the search strategy performed by the retriever. Valid
// values are "similarity" (the default), "mmr", and
// "similarity_score_threshold".
func WithSearchType(searchType string) RetrieverOption {
	return func(c *retrieverConfig) {
		c.searchType = searchType
	}
}

// WithSearchKwargs sets the keyword arguments passed to the underlying search
// function. Recognized keys include "k", "fetch_k", "lambda_mult", and
// "score_threshold".
func WithSearchKwargs(kwargs map[string]any) RetrieverOption {
	return func(c *retrieverConfig) {
		c.searchKwargs = kwargs
	}
}

// mmrSearcher is implemented by vector stores that support maximal marginal
// relevance search.
type mmrSearcher interface {
	MaxMarginalRelevanceSearch(
		ctx context.Context,
		query string,
		k int,
		fetchK int,
		lambdaMult float64,
		filter vectorstores.Filter,
	) ([]documents.Document, error)
}

// relevanceScoreSearcher is implemented by vector stores that expose
// relevance-score search.
type relevanceScoreSearcher interface {
	SimilaritySearchWithRelevanceScores(
		ctx context.Context,
		query string,
		k int,
		scoreThreshold *float64,
	) ([]vectorstores.SearchResult, error)
}

// VectorStoreRetriever adapts a vector store to the retriever interface.
type VectorStoreRetriever struct {
	store        vectorstores.VectorStore
	k            int
	searchType   string
	searchKwargs map[string]any
}

// NewVectorStoreRetriever creates a retriever backed by a vector store using
// plain similarity search.
func NewVectorStoreRetriever(store vectorstores.VectorStore, k int) VectorStoreRetriever {
	if k <= 0 {
		k = 4
	}
	return VectorStoreRetriever{
		store:      store,
		k:          k,
		searchType: searchTypeSimilarity,
	}
}

// NewVectorStoreRetrieverWithOptions creates a retriever with additional
// configuration via functional options.
func NewVectorStoreRetrieverWithOptions(
	store vectorstores.VectorStore,
	k int,
	opts ...RetrieverOption,
) (VectorStoreRetriever, error) {
	r := NewVectorStoreRetriever(store, k)
	cfg := retrieverConfig{searchType: searchTypeSimilarity}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.searchType != "" {
		r.searchType = cfg.searchType
	}
	r.searchKwargs = cfg.searchKwargs
	if err := validateSearchType(r.searchType, r.searchKwargs); err != nil {
		return VectorStoreRetriever{}, err
	}
	return r, nil
}

// AsRetriever returns a VectorStoreRetriever initialized from the given vector
// store, mirroring VectorStore.as_retriever in langchain-core.
func AsRetriever(store vectorstores.VectorStore, opts ...RetrieverOption) (VectorStoreRetriever, error) {
	return NewVectorStoreRetrieverWithOptions(store, 0, opts...)
}

// GetRelevantDocuments returns the top documents for the query using the
// configured search type.
func (r VectorStoreRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r.store == nil {
		return nil, fmt.Errorf("vector store is required")
	}

	switch r.searchType {
	case searchTypeSimilarity:
		k := searchKwargInt(r.searchKwargs, "k", r.k)
		return r.store.SimilaritySearch(ctx, query, k)
	case searchTypeMMR:
		return r.mmr(ctx, query)
	case searchTypeSimilarityThreshold:
		return r.similarityScoreThreshold(ctx, query)
	default:
		return nil, fmt.Errorf(
			"search_type of %s not allowed. Expected search_type to be %q, %q or %q",
			r.searchType,
			searchTypeSimilarity,
			searchTypeSimilarityThreshold,
			searchTypeMMR,
		)
	}
}

func (r VectorStoreRetriever) mmr(ctx context.Context, query string) ([]documents.Document, error) {
	searcher, ok := r.store.(mmrSearcher)
	if !ok {
		return nil, fmt.Errorf("vector store %T does not support mmr search", r.store)
	}
	k := searchKwargInt(r.searchKwargs, "k", r.k)
	fetchK := searchKwargInt(r.searchKwargs, "fetch_k", 20)
	lambdaMult := searchKwargFloat(r.searchKwargs, "lambda_mult", 0.5)
	return searcher.MaxMarginalRelevanceSearch(ctx, query, k, fetchK, lambdaMult, nil)
}

func (r VectorStoreRetriever) similarityScoreThreshold(ctx context.Context, query string) ([]documents.Document, error) {
	searcher, ok := r.store.(relevanceScoreSearcher)
	if !ok {
		return nil, fmt.Errorf("vector store %T does not support similarity_score_threshold search", r.store)
	}
	k := searchKwargInt(r.searchKwargs, "k", r.k)
	threshold, hasThreshold := searchKwargOptionalFloat(r.searchKwargs, "score_threshold")
	var thresholdPtr *float64
	if hasThreshold {
		thresholdPtr = &threshold
	}
	results, err := searcher.SimilaritySearchWithRelevanceScores(ctx, query, k, thresholdPtr)
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, 0, len(results))
	for _, result := range results {
		docs = append(docs, result.Document)
	}
	return docs, nil
}

func validateSearchType(searchType string, kwargs map[string]any) error {
	switch searchType {
	case searchTypeSimilarity, searchTypeMMR:
		return nil
	case searchTypeSimilarityThreshold:
		if _, ok := searchKwargOptionalFloat(kwargs, "score_threshold"); !ok {
			return fmt.Errorf(
				"score_threshold must be provided in search_kwargs for search_type %q",
				searchTypeSimilarityThreshold,
			)
		}
		return nil
	default:
		return fmt.Errorf(
			"search_type of %s not allowed. Expected search_type to be %q, %q or %q",
			searchType,
			searchTypeSimilarity,
			searchTypeSimilarityThreshold,
			searchTypeMMR,
		)
	}
}

func searchKwargInt(kwargs map[string]any, key string, fallback int) int {
	if v, ok := kwargs[key]; ok {
		switch n := v.(type) {
		case int:
			return n
		case int64:
			return int(n)
		case float64:
			return int(n)
		case float32:
			return int(n)
		}
	}
	return fallback
}

func searchKwargFloat(kwargs map[string]any, key string, fallback float64) float64 {
	if v, ok := kwargs[key]; ok {
		switch f := v.(type) {
		case float64:
			return f
		case float32:
			return float64(f)
		case int:
			return float64(f)
		case int64:
			return float64(f)
		}
	}
	return fallback
}

func searchKwargOptionalFloat(kwargs map[string]any, key string) (float64, bool) {
	if v, ok := kwargs[key]; ok {
		switch f := v.(type) {
		case float64:
			return f, true
		case float32:
			return float64(f), true
		case int:
			return float64(f), true
		case int64:
			return float64(f), true
		}
	}
	return 0, false
}
