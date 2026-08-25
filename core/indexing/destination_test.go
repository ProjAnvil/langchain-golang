package indexing

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documentloaders"
	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// Mirrors test_indexing.py::test_index_into_document_index (line 2644):
// full lifecycle against a DocumentIndex destination.
func TestIndexDocumentsIntoDocumentIndex(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	documentIndex := NewInMemoryDocumentIndex(4)
	docs := []documents.Document{
		documents.New("This is a test document.", map[string]any{"source": "1"}),
		documents.New("This is another document.", map[string]any{"source": "2"}),
	}

	got, err := IndexDocuments(ctx, docs, manager, documentIndex, Options{Cleanup: CleanupFull})
	if err != nil {
		t.Fatalf("index first: %v", err)
	}
	if got.NumAdded != 2 || got.NumDeleted != 0 || got.NumSkipped != 0 || got.NumUpdated != 0 {
		t.Fatalf("first result: %+v", got)
	}

	got, err = IndexDocuments(ctx, docs, manager, documentIndex, Options{Cleanup: CleanupFull})
	if err != nil {
		t.Fatalf("index second: %v", err)
	}
	if got.NumAdded != 0 || got.NumDeleted != 0 || got.NumSkipped != 2 || got.NumUpdated != 0 {
		t.Fatalf("second result: %+v", got)
	}

	got, err = IndexDocuments(ctx, docs, manager, documentIndex, Options{Cleanup: CleanupFull, ForceUpdate: true})
	if err != nil {
		t.Fatalf("index force update: %v", err)
	}
	if got.NumAdded != 0 || got.NumDeleted != 0 || got.NumSkipped != 0 || got.NumUpdated != 2 {
		t.Fatalf("force update result: %+v", got)
	}

	got, err = IndexDocuments(ctx, []documents.Document{}, manager, documentIndex, Options{Cleanup: CleanupFull})
	if err != nil {
		t.Fatalf("index empty: %v", err)
	}
	if got.NumAdded != 0 || got.NumDeleted != 2 || got.NumSkipped != 0 || got.NumUpdated != 0 {
		t.Fatalf("empty result: %+v", got)
	}
}

// kwargsSpyStore mirrors test_index_with_upsert_kwargs's patched
// add_documents: it records the kwargs it receives.
type kwargsSpyStore struct {
	*vectorstores.InMemory
	calls      int
	lastKwargs map[string]any
}

func (s *kwargsSpyStore) AddDocumentsWithKwargs(
	ctx context.Context,
	docs []documents.Document,
	kwargs map[string]any,
) ([]string, error) {
	s.calls++
	s.lastKwargs = kwargs
	return s.InMemory.AddDocuments(ctx, docs)
}

// Mirrors test_indexing.py::test_index_with_upsert_kwargs (line 2782):
// upsert kwargs reach the vector store's add path.
func TestIndexDocumentsUpsertKwargsVectorStore(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	spy := &kwargsSpyStore{InMemory: vectorstores.NewInMemory(embeddings.NewFake(8))}
	docs := []documents.Document{
		documents.New("Test document 1", map[string]any{"source": "1"}),
		documents.New("Test document 2", map[string]any{"source": "2"}),
	}

	got, err := IndexDocuments(ctx, docs, manager, spy, Options{
		UpsertKwargs: map[string]any{"vector_field": "embedding"},
	})
	if err != nil {
		t.Fatalf("index: %v", err)
	}
	if got.NumAdded != 2 {
		t.Fatalf("result: %+v", got)
	}
	if spy.calls != 1 {
		t.Fatalf("AddDocumentsWithKwargs calls = %d, want 1", spy.calls)
	}
	if spy.lastKwargs["vector_field"] != "embedding" {
		t.Fatalf("kwargs: %v", spy.lastKwargs)
	}
}

// kwargsSpyIndex mirrors test_index_with_upsert_kwargs_for_document_indexer's
// upsert spy.
type kwargsSpyIndex struct {
	*InMemoryDocumentIndex
	calls      int
	lastKwargs map[string]any
}

func (s *kwargsSpyIndex) UpsertWithKwargs(
	ctx context.Context,
	items []documents.Document,
	kwargs map[string]any,
) (UpsertResponse, error) {
	s.calls++
	s.lastKwargs = kwargs
	return s.InMemoryDocumentIndex.Upsert(ctx, items)
}

// Mirrors test_indexing.py::test_index_with_upsert_kwargs_for_document_indexer
// (line 2835): upsert kwargs reach the document index's upsert.
func TestIndexDocumentsUpsertKwargsDocumentIndex(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	spy := &kwargsSpyIndex{InMemoryDocumentIndex: NewInMemoryDocumentIndex(4)}
	docs := []documents.Document{
		documents.New("This is a test document.", map[string]any{"source": "1"}),
		documents.New("This is another document.", map[string]any{"source": "2"}),
	}

	got, err := IndexDocuments(ctx, docs, manager, spy, Options{
		Cleanup:      CleanupFull,
		UpsertKwargs: map[string]any{"vector_field": "embedding"},
	})
	if err != nil {
		t.Fatalf("index: %v", err)
	}
	if got.NumAdded != 2 || got.NumDeleted != 0 || got.NumSkipped != 0 || got.NumUpdated != 0 {
		t.Fatalf("result: %+v", got)
	}
	if spy.calls != 1 {
		t.Fatalf("UpsertWithKwargs calls = %d, want 1", spy.calls)
	}
	if spy.lastKwargs["vector_field"] != "embedding" {
		t.Fatalf("kwargs: %v", spy.lastKwargs)
	}
}

// A vector store without KwargAdder rejects UpsertKwargs (Python would pass
// them to add_documents and raise TypeError on an unexpected kwarg).
func TestIndexDocumentsUpsertKwargsUnsupportedVectorStore(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	_, err := IndexDocuments(ctx, []documents.Document{documents.New("x", nil)}, manager, store, Options{
		UpsertKwargs: map[string]any{"vector_field": "embedding"},
	})
	if err == nil || !strings.Contains(err.Error(), "KwargAdder") {
		t.Fatalf("expected KwargAdder error, got %v", err)
	}
}

// Same for a DocumentIndex without KwargUpserter.
func TestIndexDocumentsUpsertKwargsUnsupportedDocumentIndex(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	documentIndex := NewInMemoryDocumentIndex(4)
	_, err := IndexDocuments(ctx, []documents.Document{documents.New("x", nil)}, manager, documentIndex, Options{
		UpsertKwargs: map[string]any{"vector_field": "embedding"},
	})
	if err == nil || !strings.Contains(err.Error(), "KwargUpserter") {
		t.Fatalf("expected KwargUpserter error, got %v", err)
	}
}

// Python raises TypeError when the destination is neither a VectorStore nor a
// DocumentIndex (indexing/api.py:445-450).
func TestIndexDocumentsRejectsUnknownDestination(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	_, err := IndexDocuments(ctx, []documents.Document{documents.New("x", nil)}, manager, "not-a-store", Options{})
	if err == nil || !strings.Contains(err.Error(), "VectorStore") || !strings.Contains(err.Error(), "DocumentIndex") {
		t.Fatalf("expected destination type error, got %v", err)
	}
}

// failingUpsertIndex reports every upsert as failed, mirroring Python's
// IndexingException path for DocumentIndex upsert/delete failures
// (indexing/api.py:267-271).
type failingUpsertIndex struct {
	*InMemoryDocumentIndex
}

func (f failingUpsertIndex) Upsert(_ context.Context, items []documents.Document) (UpsertResponse, error) {
	failed := make([]string, len(items))
	for i, item := range items {
		failed[i] = item.ID
	}
	return UpsertResponse{Succeeded: []string{}, Failed: failed}, nil
}

func TestIndexDocumentsDocumentIndexUpsertFailure(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	documentIndex := failingUpsertIndex{NewInMemoryDocumentIndex(4)}
	_, err := IndexDocuments(ctx, []documents.Document{documents.New("x", nil)}, manager, documentIndex, Options{})
	if err == nil || !strings.Contains(err.Error(), "DocumentIndex") {
		t.Fatalf("expected upsert failure error, got %v", err)
	}
}

// failingDeleteIndex reports delete failures; exercised via full cleanup.
type failingDeleteIndex struct {
	*InMemoryDocumentIndex
}

func (f failingDeleteIndex) Delete(_ context.Context, ids []string) (DeleteResponse, error) {
	return DeleteResponse{Succeeded: []string{}, Failed: append([]string(nil), ids...), NumFailed: len(ids)}, nil
}

func TestIndexDocumentsDocumentIndexDeleteFailure(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("stale", map[string]any{"source": "s"})}
	if _, err := IndexDocuments(ctx, docs, manager, store, Options{Cleanup: CleanupFull}); err != nil {
		t.Fatalf("seed index: %v", err)
	}
	// Second run against the same record manager but a failing-delete index:
	// full cleanup tries to delete the stale record and fails.
	_, err := IndexDocuments(ctx, []documents.Document{}, manager, failingDeleteIndex{NewInMemoryDocumentIndex(4)}, Options{Cleanup: CleanupFull})
	if err == nil || !strings.Contains(err.Error(), "delete operation to DocumentIndex failed") {
		t.Fatalf("expected delete failure error, got %v", err)
	}
}

// errorDeleteIndex returns a hard error from Delete.
type errorDeleteIndex struct {
	*InMemoryDocumentIndex
	err error
}

func (f errorDeleteIndex) Delete(context.Context, []string) (DeleteResponse, error) {
	return DeleteResponse{}, f.err
}

func TestIndexDocumentsDocumentIndexDeleteError(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("stale", map[string]any{"source": "s"})}
	if _, err := IndexDocuments(ctx, docs, manager, store, Options{Cleanup: CleanupFull}); err != nil {
		t.Fatalf("seed index: %v", err)
	}
	errTest := errors.New("delete blew up")
	_, err := IndexDocuments(ctx, []documents.Document{}, manager, errorDeleteIndex{NewInMemoryDocumentIndex(4), errTest}, Options{Cleanup: CleanupFull})
	if !errors.Is(err, errTest) {
		t.Fatalf("expected delete error, got %v", err)
	}
}

// IndexDocumentIterator accepts a DocumentIndex destination too.
func TestIndexDocumentIteratorIntoDocumentIndex(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	documentIndex := NewInMemoryDocumentIndex(4)
	iter := documentloaders.NewSliceIterator([]documents.Document{
		documents.New("iterated document", map[string]any{"source": "1"}),
	})
	got, err := IndexDocumentIterator(ctx, iter, manager, documentIndex, Options{})
	if err != nil {
		t.Fatalf("index iterator: %v", err)
	}
	if got.NumAdded != 1 {
		t.Fatalf("result: %+v", got)
	}
}

// errorUpsertIndex fails inside UpsertWithKwargs; the error propagates.
type errorUpsertIndex struct {
	*InMemoryDocumentIndex
	err error
}

func (s errorUpsertIndex) UpsertWithKwargs(
	context.Context,
	[]documents.Document,
	map[string]any,
) (UpsertResponse, error) {
	return UpsertResponse{}, s.err
}

func TestIndexDocumentsUpsertKwargsErrorPropagates(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	errTest := errors.New("upsert blew up")
	spy := errorUpsertIndex{InMemoryDocumentIndex: NewInMemoryDocumentIndex(4), err: errTest}
	_, err := IndexDocuments(ctx, []documents.Document{documents.New("x", nil)}, manager, spy, Options{
		UpsertKwargs: map[string]any{"vector_field": "embedding"},
	})
	if !errors.Is(err, errTest) {
		t.Fatalf("expected upsert error, got %v", err)
	}
}
