package retrievers

import (
	"context"
	"fmt"
	"maps"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/stores"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// defaultMultiVectorIDKey mirrors Python's MultiVectorRetriever.id_key
// ("doc_id"), the child metadata key holding the parent document id.
const defaultMultiVectorIDKey = "doc_id"

// MultiVectorRetriever supports multiple embeddings per parent document.
// Small child chunks are embedded in a vector store and searched, but
// retrieval returns the whole parent documents those chunks came from: child
// hits are mapped back to parent ids found in their metadata (under id_key),
// deduplicated in hit order, and the corresponding parents are fetched from
// the docstore.
//
// Python parity: langchain_classic.retrievers.multi_vector.MultiVectorRetriever.
// Deviation: the Go form takes a byte-store-less stores.BaseStore[documents.Document]
// directly (Python optionally wraps a ByteStore via create_kv_docstore), and
// the async path is expressed through the context argument rather than a
// separate method. AddDocuments is intentionally not provided: callers index
// children through the vector store and parents through the docstore.
type MultiVectorRetriever struct {
	retriever VectorStoreRetriever
	docstore  stores.BaseStore[documents.Document]
	idKey     string
}

// MultiVectorOption configures a MultiVectorRetriever.
type MultiVectorOption func(*multiVectorConfig)

// multiVectorConfig holds the options applied before the child vector store
// retriever is constructed.
type multiVectorConfig struct {
	idKey        string
	searchType   string
	searchKwargs map[string]any
	k            int
}

// WithMultiVectorIDKey sets the child metadata key holding the parent
// document id (default "doc_id").
func WithMultiVectorIDKey(idKey string) MultiVectorOption {
	return func(c *multiVectorConfig) {
		c.idKey = idKey
	}
}

// WithMultiVectorSearchType sets the search strategy used over the child
// chunks. Valid values are "similarity" (the default), "mmr", and
// "similarity_score_threshold", mirroring Python's SearchType enum.
func WithMultiVectorSearchType(searchType string) MultiVectorOption {
	return func(c *multiVectorConfig) {
		c.searchType = searchType
	}
}

// WithMultiVectorSearchKwargs sets the keyword arguments forwarded to the
// underlying search (for example "k", "fetch_k", "lambda_mult", or
// "score_threshold"). Mirrors Python's MultiVectorRetriever.search_kwargs.
func WithMultiVectorSearchKwargs(kwargs map[string]any) MultiVectorOption {
	return func(c *multiVectorConfig) {
		c.searchKwargs = kwargs
	}
}

// WithMultiVectorK sets the default number of child hits to retrieve. A "k"
// entry in WithMultiVectorSearchKwargs takes precedence (default 4).
func WithMultiVectorK(k int) MultiVectorOption {
	return func(c *multiVectorConfig) {
		if k > 0 {
			c.k = k
		}
	}
}

// NewMultiVectorRetriever creates a retriever that searches child chunks in
// vectorstore and returns their whole parent documents from docstore. Both are
// required. The child search is delegated to a VectorStoreRetriever built with
// NewVectorStoreRetrieverWithOptions, so similarity, mmr, and
// similarity_score_threshold search types are all supported.
func NewMultiVectorRetriever(
	vectorstore vectorstores.VectorStore,
	docstore stores.BaseStore[documents.Document],
	opts ...MultiVectorOption,
) (MultiVectorRetriever, error) {
	if vectorstore == nil {
		return MultiVectorRetriever{}, fmt.Errorf("vector store is required")
	}
	if docstore == nil {
		return MultiVectorRetriever{}, fmt.Errorf("docstore is required")
	}
	cfg := multiVectorConfig{
		idKey:      defaultMultiVectorIDKey,
		searchType: searchTypeSimilarity,
		k:          4,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.idKey == "" {
		cfg.idKey = defaultMultiVectorIDKey
	}
	if cfg.searchType == "" {
		cfg.searchType = searchTypeSimilarity
	}
	childRetriever, err := NewVectorStoreRetrieverWithOptions(
		vectorstore,
		cfg.k,
		WithSearchType(cfg.searchType),
		WithSearchKwargs(maps.Clone(cfg.searchKwargs)),
	)
	if err != nil {
		return MultiVectorRetriever{}, err
	}
	return MultiVectorRetriever{
		retriever: childRetriever,
		docstore:  docstore,
		idKey:     cfg.idKey,
	}, nil
}

var _ Retriever = MultiVectorRetriever{}

// GetRelevantDocuments searches the child chunks, collects the unique parent
// ids from their metadata in hit order, and returns the corresponding parent
// documents from the docstore. Child hits without the id key are skipped and
// parents missing from the docstore are omitted, matching Python's
// `_get_relevant_documents` (`ids` list built in order, then `docstore.mget`).
func (r MultiVectorRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	subDocs, err := r.retriever.GetRelevantDocuments(ctx, query)
	if err != nil {
		return nil, err
	}

	ids := make([]string, 0, len(subDocs))
	seen := make(map[string]struct{}, len(subDocs))
	for _, doc := range subDocs {
		value, ok := doc.Metadata[r.idKey]
		if !ok || value == nil {
			continue
		}
		id, isString := value.(string)
		if !isString {
			id = fmt.Sprint(value)
		}
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return []documents.Document{}, nil
	}

	values, err := r.docstore.MGet(ctx, ids)
	if err != nil {
		return nil, fmt.Errorf("fetch parent documents from docstore: %w", err)
	}
	parents := make([]documents.Document, 0, len(values))
	for _, value := range values {
		if !value.Found {
			continue
		}
		parents = append(parents, value.Value.Clone())
	}
	return parents, nil
}
