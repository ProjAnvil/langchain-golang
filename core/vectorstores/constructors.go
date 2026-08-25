package vectorstores

import (
	"context"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

// FromOptions configures FromTexts and FromDocuments.
type FromOptions struct {
	// Metadatas attaches per-text metadata; extra entries are ignored and
	// missing entries default to empty metadata (Python from_texts semantics).
	Metadatas []map[string]any
	// IDs assigns explicit IDs. For FromDocuments, a non-nil IDs slice
	// overrides document IDs; leave nil to use document IDs.
	IDs []string
}

// FromOption configures FromTexts/FromDocuments.
type FromOption func(*FromOptions)

// WithMetadatas attaches per-text metadata.
func WithMetadatas(metadatas []map[string]any) FromOption {
	return func(o *FromOptions) { o.Metadatas = metadatas }
}

// WithIDs assigns explicit IDs, overriding any document IDs in FromDocuments.
func WithIDs(ids []string) FromOption {
	return func(o *FromOptions) { o.IDs = ids }
}

// FromTexts creates an InMemory store pre-populated with texts, mirroring
// Python's VectorStore.from_texts (vectorstores/base.py:848). It is a
// package-level constructor because Go interfaces cannot express Python's
// from_texts classmethod; InMemory is the store Python's base test-suite
// exercises for this path.
func FromTexts(
	ctx context.Context,
	embedder embeddings.Embeddings,
	texts []string,
	opts ...FromOption,
) (*InMemory, error) {
	options := FromOptions{}
	for _, opt := range opts {
		opt(&options)
	}
	store := NewInMemory(embedder)
	if _, err := store.AddTexts(ctx, texts, options.Metadatas, options.IDs); err != nil {
		return nil, err
	}
	return store, nil
}

// FromDocuments creates an InMemory store from documents, mirroring Python's
// VectorStore.from_documents (vectorstores/base.py:787): when WithIDs is not
// given and at least one document has a non-empty ID, the document IDs are
// used. Input documents are never mutated.
func FromDocuments(
	ctx context.Context,
	embedder embeddings.Embeddings,
	docs []documents.Document,
	opts ...FromOption,
) (*InMemory, error) {
	options := FromOptions{}
	for _, opt := range opts {
		opt(&options)
	}
	texts := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
		metadatas[i] = doc.Metadata
	}
	ids := options.IDs
	if ids == nil {
		docIDs := make([]string, len(docs))
		anyID := false
		for i, doc := range docs {
			docIDs[i] = doc.ID
			if doc.ID != "" {
				anyID = true
			}
		}
		if anyID {
			ids = docIDs
		}
	}
	return FromTexts(ctx, embedder, texts, WithMetadatas(metadatas), WithIDs(ids))
}
