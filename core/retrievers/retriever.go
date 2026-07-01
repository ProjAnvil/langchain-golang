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

// VectorStoreRetriever adapts a vector store to the retriever interface.
type VectorStoreRetriever struct {
	store vectorstores.VectorStore
	k     int
}

// NewVectorStoreRetriever creates a retriever backed by a vector store.
func NewVectorStoreRetriever(store vectorstores.VectorStore, k int) VectorStoreRetriever {
	if k <= 0 {
		k = 4
	}
	return VectorStoreRetriever{
		store: store,
		k:     k,
	}
}

// GetRelevantDocuments returns the top documents for the query.
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
	return r.store.SimilaritySearch(ctx, query, r.k)
}
