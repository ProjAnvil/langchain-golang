package indexing

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/utils"
)

// UpsertResponse reports which documents were successfully added or updated
// and which failed, mirroring langchain_core.indexing.UpsertResponse.
type UpsertResponse struct {
	// Succeeded holds the IDs that were successfully indexed, in input order.
	Succeeded []string
	// Failed holds the IDs that failed to index.
	Failed []string
}

// DeleteResponse reports which documents were successfully deleted and which
// failed, mirroring langchain_core.indexing.DeleteResponse.
type DeleteResponse struct {
	// Succeeded holds the IDs that were actually deleted. IDs that did not
	// exist are not included.
	Succeeded []string
	// Failed holds the IDs that failed to be deleted.
	Failed []string
	// NumDeleted counts the IDs that were actually deleted.
	NumDeleted int
	// NumFailed counts the IDs that failed to be deleted.
	NumFailed int
}

// DocumentIndex is a retriever that also supports storing, fetching, and
// deleting documents by ID, mirroring langchain_core.indexing.DocumentIndex.
type DocumentIndex interface {
	retrievers.Retriever
	Upsert(ctx context.Context, items []documents.Document) (UpsertResponse, error)
	Delete(ctx context.Context, ids []string) (DeleteResponse, error)
	Get(ctx context.Context, ids []string) ([]documents.Document, error)
}

// InMemoryDocumentIndex stores documents in an in-memory map and searches by
// counting how many times the query appears in each document's page content.
type InMemoryDocumentIndex struct {
	mu    sync.RWMutex
	store map[string]documents.Document
	topK  int
}

// NewInMemoryDocumentIndex creates an in-memory document index returning up to
// topK documents per query. A non-positive topK defaults to 4.
func NewInMemoryDocumentIndex(topK int) *InMemoryDocumentIndex {
	if topK <= 0 {
		topK = 4
	}
	return &InMemoryDocumentIndex{
		store: map[string]documents.Document{},
		topK:  topK,
	}
}

// Upsert adds or overwrites documents keyed by ID. Items with an empty ID get
// a generated UUID. The succeeded IDs are returned in input order.
func (m *InMemoryDocumentIndex) Upsert(_ context.Context, items []documents.Document) (UpsertResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	resp := UpsertResponse{
		Succeeded: make([]string, 0, len(items)),
		Failed:    []string{},
	}
	for _, item := range items {
		id := item.ID
		if id == "" {
			uuid, err := utils.UUID4()
			if err != nil {
				return UpsertResponse{}, err
			}
			id = uuid
			item = item.WithID(id)
		}
		m.store[id] = item
		resp.Succeeded = append(resp.Succeeded, id)
	}
	return resp, nil
}

// Delete removes the existing IDs. Missing IDs are silently skipped and are
// not counted as failures.
func (m *InMemoryDocumentIndex) Delete(_ context.Context, ids []string) (DeleteResponse, error) {
	if ids == nil {
		return DeleteResponse{}, fmt.Errorf("ids must be provided for deletion")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	resp := DeleteResponse{
		Succeeded: []string{},
		Failed:    []string{},
	}
	for _, id := range ids {
		if _, ok := m.store[id]; !ok {
			continue
		}
		delete(m.store, id)
		resp.Succeeded = append(resp.Succeeded, id)
		resp.NumDeleted++
	}
	return resp, nil
}

// Get returns the documents whose IDs exist, in the order of the input IDs.
func (m *InMemoryDocumentIndex) Get(_ context.Context, ids []string) ([]documents.Document, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]documents.Document, 0, len(ids))
	for _, id := range ids {
		if doc, ok := m.store[id]; ok {
			out = append(out, doc)
		}
	}
	return out, nil
}

// GetRelevantDocuments returns up to topK documents ranked by how many times
// query appears in their page content, in descending order.
func (m *InMemoryDocumentIndex) GetRelevantDocuments(ctx context.Context, query string) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	type scored struct {
		doc   documents.Document
		count int
	}
	counts := make([]scored, 0, len(m.store))
	for _, doc := range m.store {
		counts = append(counts, scored{doc: doc, count: strings.Count(doc.PageContent, query)})
	}
	sort.SliceStable(counts, func(i, j int) bool {
		return counts[i].count > counts[j].count
	})
	if len(counts) > m.topK {
		counts = counts[:m.topK]
	}

	out := make([]documents.Document, 0, len(counts))
	for _, entry := range counts {
		out = append(out, entry.doc.Clone())
	}
	return out, nil
}
