package retrievers

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/stores"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// defaultParentIDKey mirrors Python's ParentDocumentRetriever.id_key
// ("doc_id"), the child metadata key holding the parent document id.
const defaultParentIDKey = "doc_id"

// ParentDocumentRetriever indexes small child chunks for precise similarity
// hits but returns the full parent documents those chunks came from: children
// are stored in the vector store tagged with their parent's id, parents are
// stored whole in the docstore, and retrieval maps child hits back to parents
// (deduplicated, first hit winning).
//
// Python parity: langchain_community.retrievers.ParentDocumentRetriever.
// Deviation: Python's optional parent_splitter is not carried (the task scope
// stores each parent whole in the docstore), and a nil child splitter indexes
// each parent as a single child instead of raising Python's ValueError, so
// the zero configuration is usable end to end.
type ParentDocumentRetriever struct {
	store          vectorstores.VectorStore
	childRetriever Retriever
	docstore       stores.BaseStore[documents.Document]
	childSplitter  documents.Transformer
	idKey          string
	searchK        int
}

// ParentDocumentOption configures a ParentDocumentRetriever.
type ParentDocumentOption func(*ParentDocumentRetriever)

// WithChildSplitter sets the transformer that splits parents into the child
// chunks indexed by the vector store. When unset, each parent is indexed as a
// single child.
func WithChildSplitter(splitter documents.Transformer) ParentDocumentOption {
	return func(r *ParentDocumentRetriever) {
		r.childSplitter = splitter
	}
}

// WithIDKey sets the child metadata key holding the parent document id
// (default "doc_id").
func WithIDKey(idKey string) ParentDocumentOption {
	return func(r *ParentDocumentRetriever) {
		r.idKey = idKey
	}
}

// WithSearchK sets how many child hits feed the parent lookup (default 4,
// mirroring Python's search_kwargs={"k": 4}).
func WithSearchK(k int) ParentDocumentOption {
	return func(r *ParentDocumentRetriever) {
		if k > 0 {
			r.searchK = k
		}
	}
}

// NewParentDocumentRetriever creates a retriever that returns whole parent
// documents stored in docstore for child hits retrieved from store. Both are
// required.
func NewParentDocumentRetriever(
	store vectorstores.VectorStore,
	docstore stores.BaseStore[documents.Document],
	opts ...ParentDocumentOption,
) (ParentDocumentRetriever, error) {
	if store == nil {
		return ParentDocumentRetriever{}, fmt.Errorf("vector store is required")
	}
	if docstore == nil {
		return ParentDocumentRetriever{}, fmt.Errorf("docstore is required")
	}
	retriever := ParentDocumentRetriever{
		store:    store,
		docstore: docstore,
		idKey:    defaultParentIDKey,
		searchK:  4,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&retriever)
		}
	}
	if retriever.idKey == "" {
		retriever.idKey = defaultParentIDKey
	}
	retriever.childRetriever = NewVectorStoreRetriever(store, retriever.searchK)
	return retriever, nil
}

var _ Retriever = ParentDocumentRetriever{}

// AddDocuments indexes documents for parent-document retrieval: each parent
// is stored whole in the docstore under its id (the document's own ID when
// set, otherwise a generated one), and its child chunks — produced by the
// configured splitter, or the parent itself when no splitter is set — are
// embedded into the vector store with the parent id recorded in each child's
// metadata under the configured id key (Python: `_doc.metadata[self.id_key]
// = _id`; docstore first, then vector store).
func (r ParentDocumentRetriever) AddDocuments(
	ctx context.Context,
	docs []documents.Document,
) error {
	if len(docs) == 0 {
		return nil
	}

	parents := make([]stores.KeyValue[documents.Document], 0, len(docs))
	children := make([]documents.Document, 0, len(docs))
	for _, doc := range docs {
		parentID := doc.ID
		if parentID == "" {
			parentID = newParentID()
		}
		parents = append(parents, stores.KeyValue[documents.Document]{
			Key:   parentID,
			Value: doc.Clone().WithID(parentID),
		})

		// Split the parent into children; without a splitter the parent is
		// its own single child (identity split).
		splits := []documents.Document{doc}
		if r.childSplitter != nil {
			var err error
			splits, err = r.childSplitter.TransformDocuments(ctx, []documents.Document{doc})
			if err != nil {
				return fmt.Errorf("split parent %q into children: %w", parentID, err)
			}
		}
		for _, split := range splits {
			child := split.Clone()
			if child.Metadata == nil {
				child.Metadata = map[string]any{}
			}
			child.Metadata[r.idKey] = parentID
			children = append(children, child)
		}
	}

	if err := r.docstore.MSet(ctx, parents); err != nil {
		return fmt.Errorf("store parents in docstore: %w", err)
	}
	if _, err := r.store.AddDocuments(ctx, children); err != nil {
		return fmt.Errorf("index children in vector store: %w", err)
	}
	return nil
}

// GetRelevantDocuments retrieves child chunks through the child retriever,
// collects their parent ids (deduplicated in hit order), and returns the
// corresponding full parent documents from the docstore. Parents missing from
// the docstore are skipped.
func (r ParentDocumentRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	children, err := r.childRetriever.GetRelevantDocuments(ctx, query)
	if err != nil {
		return nil, err
	}

	var parentIDs []string
	seen := make(map[string]struct{})
	for _, child := range children {
		value, ok := child.Metadata[r.idKey]
		if !ok {
			return nil, fmt.Errorf(
				"child document %q lacks parent id metadata key %q (was it indexed through AddDocuments?)",
				child.ID,
				r.idKey,
			)
		}
		parentID, ok := value.(string)
		if !ok {
			return nil, fmt.Errorf(
				"child document metadata %q must be a parent id string, got %T",
				r.idKey,
				value,
			)
		}
		if _, duplicate := seen[parentID]; duplicate {
			continue
		}
		seen[parentID] = struct{}{}
		parentIDs = append(parentIDs, parentID)
	}
	if len(parentIDs) == 0 {
		return []documents.Document{}, nil
	}

	values, err := r.docstore.MGet(ctx, parentIDs)
	if err != nil {
		return nil, fmt.Errorf("fetch parents from docstore: %w", err)
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

// newParentID generates a random parent document id (Python: uuid4().hex).
func newParentID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("parent-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}
