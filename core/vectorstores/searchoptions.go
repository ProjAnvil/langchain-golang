package vectorstores

import (
	"context"

	"github.com/projanvil/langchain-golang/core/documents"
)

// SearchOptions carries declarative search parameters for stores implementing
// OptionSearcher. It mirrors the field semantics of Python's SearchArgs
// (langchain_postgres/vectorstores.py: k, filter, score_threshold, fetch_k).
//
// Zero values mean "use the store default": K <= 0 becomes 4, FetchK <= 0
// becomes 20, and ScoreThreshold <= 0 disables thresholding, matching the
// Python SearchArgs defaults.
type SearchOptions struct {
	// K is the number of documents to return (Python SearchArgs.k, default 4).
	K int

	// Filter is a declarative metadata filter expressed in the shared DSL
	// understood by MatchFilter and ValidateFilter (Python SearchArgs.filter).
	// See filter.go for the operator set and semantics.
	Filter map[string]any

	// ScoreThreshold is the minimum relevance score (in [0, 1], 1 == most
	// similar) a document must reach to be returned (Python
	// SearchArgs.score_threshold). The zero value disables thresholding.
	ScoreThreshold float64

	// FetchK is the number of candidate documents to fetch before MMR
	// re-ranking (Python SearchArgs.fetch_k, default 20). It is ignored by
	// similarity search.
	FetchK int
}

// OptionSearcher is the optional capability interface for stores that support
// declarative search options, most notably the shared metadata filter DSL.
// The VectorStore interface stays closed (adding methods to it would break
// every implementer), so callers type-assert this interface the same way they
// assert TextAdder or the MMR/relevance-score searcher interfaces.
//
// Python parity: the option surface corresponds to VectorStore methods that
// accept **kwargs shaped by SearchArgs in langchain-postgres, plus the filter
// kwarg passed through by VectorStoreRetriever in langchain-core.
type OptionSearcher interface {
	// SimilaritySearchWithOptions returns the top K documents for the query,
	// restricted by the declarative filter and score threshold when set.
	SimilaritySearchWithOptions(
		ctx context.Context,
		query string,
		opts SearchOptions,
	) ([]documents.Document, error)

	// MMRSearchWithOptions returns up to K documents selected for relevance
	// and diversity from FetchK candidates, restricted by the declarative
	// filter when set. SearchOptions does not carry lambda_mult; stores apply
	// their default (0.5), matching Python SearchArgs.
	MMRSearchWithOptions(
		ctx context.Context,
		query string,
		opts SearchOptions,
	) ([]documents.Document, error)
}
