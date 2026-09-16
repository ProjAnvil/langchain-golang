package retrievers

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/documents"
)

// DocumentCompressor post-processes retrieved documents relative to the query
// that produced them. Implementations typically re-rank (e.g. hosted rerank
// APIs), filter, or shrink the retrieved documents before they are returned to
// the caller.
//
// Python parity: langchain_core.documents.compressor.BaseDocumentCompressor
// (compress_documents(documents, query, callbacks)). Callbacks are carried by
// the context in Go instead of a separate argument.
type DocumentCompressor interface {
	// CompressDocuments compresses retrieved documents given the query
	// context. It returns the compressed (re-ranked, filtered, or shrunk)
	// documents, usually fewer and better ordered than the input.
	CompressDocuments(
		ctx context.Context,
		docs []documents.Document,
		query string,
	) ([]documents.Document, error)
}

// ContextualCompressionRetriever wraps a base retriever and compresses its
// results through a DocumentCompressor: retrieval runs first, then the
// compressor re-ranks or filters the hits against the same query. When the
// base retriever returns nothing, the compressor is skipped and an empty
// result is returned immediately.
//
// Python parity: langchain_classic.retrievers.contextual_compression.
// ContextualCompressionRetriever (base_retriever + base_compressor; empty
// retrieval short-circuits: `if docs: compress else return []`).
type ContextualCompressionRetriever struct {
	baseRetriever Retriever
	compressor    DocumentCompressor
}

// NewContextualCompressionRetriever creates a retriever that compresses the
// output of base through compressor. Both are required.
func NewContextualCompressionRetriever(
	base Retriever,
	compressor DocumentCompressor,
) (ContextualCompressionRetriever, error) {
	if base == nil {
		return ContextualCompressionRetriever{}, fmt.Errorf("base retriever is required")
	}
	if compressor == nil {
		return ContextualCompressionRetriever{}, fmt.Errorf("document compressor is required")
	}
	return ContextualCompressionRetriever{
		baseRetriever: base,
		compressor:    compressor,
	}, nil
}

// GetRelevantDocuments retrieves documents from the base retriever and
// compresses them with the configured compressor.
func (r ContextualCompressionRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	docs, err := r.baseRetriever.GetRelevantDocuments(ctx, query)
	if err != nil {
		return nil, err
	}
	// Python parity: the compressor only runs when the base retriever
	// produced documents (contextual_compression.py `_get_relevant_documents`).
	if len(docs) == 0 {
		return []documents.Document{}, nil
	}
	return r.compressor.CompressDocuments(ctx, docs, query)
}

var _ Retriever = ContextualCompressionRetriever{}
