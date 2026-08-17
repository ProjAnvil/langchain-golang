package indexing

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

func TestIndexDocumentsAddsThenSkips(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{
		documents.New("alpha", map[string]any{"source": "a"}),
		documents.New("beta", map[string]any{"source": "b"}),
	}

	got, err := IndexDocuments(ctx, docs, manager, store, Options{SourceIDKey: "source"})
	if err != nil {
		t.Fatalf("index first: %v", err)
	}
	if got.NumAdded != 2 || got.NumSkipped != 0 || got.NumUpdated != 0 {
		t.Fatalf("first result: %+v", got)
	}

	got, err = IndexDocuments(ctx, docs, manager, store, Options{SourceIDKey: "source"})
	if err != nil {
		t.Fatalf("index second: %v", err)
	}
	if got.NumAdded != 0 || got.NumSkipped != 2 || got.NumUpdated != 0 {
		t.Fatalf("second result: %+v", got)
	}

	key, err := HashDocument(docs[0])
	if err != nil {
		t.Fatalf("hash: %v", err)
	}
	stored, err := store.GetByIDs(ctx, []string{key})
	if err != nil {
		t.Fatalf("get ids: %v", err)
	}
	if len(stored) != 1 || stored[0].PageContent != "alpha" {
		t.Fatalf("stored docs: %#v", stored)
	}
	keys, err := manager.ListKeys(ctx, []string{"a"}, time.Time{}, 0)
	if err != nil {
		t.Fatalf("list keys: %v", err)
	}
	if len(keys) != 1 || keys[0] != key {
		t.Fatalf("source keys: %#v want %q", keys, key)
	}
}

func TestIndexDocumentsForceUpdate(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("alpha", nil)}

	if _, err := IndexDocuments(ctx, docs, manager, store, Options{}); err != nil {
		t.Fatalf("index first: %v", err)
	}
	got, err := IndexDocuments(ctx, docs, manager, store, Options{ForceUpdate: true})
	if err != nil {
		t.Fatalf("index force: %v", err)
	}
	if got.NumUpdated != 1 || got.NumAdded != 0 || got.NumSkipped != 0 {
		t.Fatalf("force result: %+v", got)
	}
}

func TestIndexDocumentsFullCleanupDeletesAllStaleRecords(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	original := []documents.Document{
		documents.New("old a", map[string]any{"source": "a"}),
		documents.New("keep b", map[string]any{"source": "b"}),
	}
	if _, err := IndexDocuments(ctx, original, manager, store, Options{SourceIDKey: "source"}); err != nil {
		t.Fatalf("index original: %v", err)
	}
	oldKey, err := HashDocument(original[0])
	if err != nil {
		t.Fatalf("hash old: %v", err)
	}
	bKey, err := HashDocument(original[1])
	if err != nil {
		t.Fatalf("hash b: %v", err)
	}
	time.Sleep(time.Millisecond)

	replacement := []documents.Document{documents.New("new a", map[string]any{"source": "a"})}
	got, err := IndexDocuments(ctx, replacement, manager, store, Options{
		SourceIDKey: "source",
		Cleanup:     CleanupFull,
	})
	if err != nil {
		t.Fatalf("index replacement: %v", err)
	}
	if got.NumAdded != 1 || got.NumDeleted != 2 {
		t.Fatalf("cleanup result: %+v", got)
	}
	if docs, err := store.GetByIDs(ctx, []string{oldKey}); err != nil || len(docs) != 0 {
		t.Fatalf("old doc should be deleted: docs=%#v err=%v", docs, err)
	}
	if docs, err := store.GetByIDs(ctx, []string{bKey}); err != nil || len(docs) != 0 {
		t.Fatalf("other stale source should be deleted by full cleanup: docs=%#v err=%v", docs, err)
	}
}

func TestIndexDocumentsScopedFullCleanupDeletesOnlySeenSources(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	original := []documents.Document{
		documents.New("old a", map[string]any{"source": "a"}),
		documents.New("keep b", map[string]any{"source": "b"}),
	}
	if _, err := IndexDocuments(ctx, original, manager, store, Options{SourceIDKey: "source"}); err != nil {
		t.Fatalf("index original: %v", err)
	}
	oldKey, err := HashDocument(original[0])
	if err != nil {
		t.Fatalf("hash old: %v", err)
	}
	bKey, err := HashDocument(original[1])
	if err != nil {
		t.Fatalf("hash b: %v", err)
	}
	time.Sleep(time.Millisecond)

	replacement := []documents.Document{documents.New("new a", map[string]any{"source": "a"})}
	got, err := IndexDocuments(ctx, replacement, manager, store, Options{
		SourceIDKey: "source",
		Cleanup:     CleanupScopedFull,
	})
	if err != nil {
		t.Fatalf("index scoped replacement: %v", err)
	}
	if got.NumAdded != 1 || got.NumDeleted != 1 {
		t.Fatalf("cleanup result: %+v", got)
	}
	if docs, err := store.GetByIDs(ctx, []string{oldKey}); err != nil || len(docs) != 0 {
		t.Fatalf("old source doc should be deleted: docs=%#v err=%v", docs, err)
	}
	if docs, err := store.GetByIDs(ctx, []string{bKey}); err != nil || len(docs) != 1 {
		t.Fatalf("unseen source should remain: docs=%#v err=%v", docs, err)
	}
}

func TestIndexDocumentsFullCleanupRefreshesSkippedRecords(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("alpha", map[string]any{"source": "a"})}
	if _, err := IndexDocuments(ctx, docs, manager, store, Options{SourceIDKey: "source"}); err != nil {
		t.Fatalf("index original: %v", err)
	}
	key, err := HashDocument(docs[0])
	if err != nil {
		t.Fatalf("hash: %v", err)
	}
	time.Sleep(time.Millisecond)

	got, err := IndexDocuments(ctx, docs, manager, store, Options{
		SourceIDKey: "source",
		Cleanup:     CleanupFull,
	})
	if err != nil {
		t.Fatalf("index skipped cleanup: %v", err)
	}
	if got.NumSkipped != 1 || got.NumDeleted != 0 {
		t.Fatalf("cleanup result: %+v", got)
	}
	if docs, err := store.GetByIDs(ctx, []string{key}); err != nil || len(docs) != 1 {
		t.Fatalf("skipped doc should remain: docs=%#v err=%v", docs, err)
	}
}

func TestIndexDocumentsDeduplicatesWithinBatch(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	doc := documents.New("alpha", map[string]any{"source": "a"})
	docs := []documents.Document{doc, doc.Clone()}

	got, err := IndexDocuments(ctx, docs, manager, store, Options{SourceIDKey: "source"})
	if err != nil {
		t.Fatalf("index duplicates: %v", err)
	}
	if got.NumAdded != 1 || got.NumSkipped != 1 || got.NumUpdated != 0 {
		t.Fatalf("dedupe result: %+v", got)
	}
	key, err := HashDocument(doc)
	if err != nil {
		t.Fatalf("hash: %v", err)
	}
	stored, err := store.GetByIDs(ctx, []string{key})
	if err != nil {
		t.Fatalf("get ids: %v", err)
	}
	if len(stored) != 1 {
		t.Fatalf("expected one stored doc, got %#v", stored)
	}
}

func TestIndexDocumentIteratorStreamsAndCloses(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	original := []documents.Document{
		documents.New("old a", map[string]any{"source": "a"}),
		documents.New("old b", map[string]any{"source": "b"}),
	}
	if _, err := IndexDocuments(ctx, original, manager, store, Options{SourceIDKey: "source"}); err != nil {
		t.Fatalf("index original: %v", err)
	}
	oldAKey, err := HashDocument(original[0])
	if err != nil {
		t.Fatalf("hash old a: %v", err)
	}
	oldBKey, err := HashDocument(original[1])
	if err != nil {
		t.Fatalf("hash old b: %v", err)
	}
	time.Sleep(time.Millisecond)

	iter := &trackingIterator{docs: []documents.Document{
		documents.New("new a", map[string]any{"source": "a"}),
		documents.New("new a 2", map[string]any{"source": "a"}),
	}}
	got, err := IndexDocumentIterator(ctx, iter, manager, store, Options{
		BatchSize:   1,
		SourceIDKey: "source",
		Cleanup:     CleanupScopedFull,
	})
	if err != nil {
		t.Fatalf("index iterator: %v", err)
	}
	if !iter.closed {
		t.Fatal("iterator was not closed")
	}
	if got.NumAdded != 2 || got.NumDeleted != 1 || got.NumSkipped != 0 {
		t.Fatalf("iterator result: %+v", got)
	}
	if docs, err := store.GetByIDs(ctx, []string{oldAKey}); err != nil || len(docs) != 0 {
		t.Fatalf("old source doc should be deleted: docs=%#v err=%v", docs, err)
	}
	if docs, err := store.GetByIDs(ctx, []string{oldBKey}); err != nil || len(docs) != 1 {
		t.Fatalf("unseen source doc should remain: docs=%#v err=%v", docs, err)
	}
}

func TestIndexDocumentsIncrementalCleanupRequiresSourceIDKey(t *testing.T) {
	_, err := IndexDocuments(
		context.Background(),
		[]documents.Document{documents.New("alpha", nil)},
		NewInMemoryRecordManager("unit"),
		vectorstores.NewInMemory(embeddings.NewFake(8)),
		Options{Cleanup: CleanupIncremental},
	)
	if err == nil {
		t.Fatal("expected incremental cleanup source key error")
	}
}

func TestIndexDocumentsRejectsUnknownCleanupMode(t *testing.T) {
	_, err := IndexDocuments(
		context.Background(),
		[]documents.Document{documents.New("alpha", nil)},
		NewInMemoryRecordManager("unit"),
		vectorstores.NewInMemory(embeddings.NewFake(8)),
		Options{Cleanup: CleanupMode("invalid")},
	)
	if err == nil {
		t.Fatal("expected invalid cleanup mode error")
	}
}

func TestIndexDocumentsScopedFullCleanupRequiresSourceIDKey(t *testing.T) {
	_, err := IndexDocuments(
		context.Background(),
		[]documents.Document{documents.New("alpha", nil)},
		NewInMemoryRecordManager("unit"),
		vectorstores.NewInMemory(embeddings.NewFake(8)),
		Options{Cleanup: CleanupScopedFull},
	)
	if err == nil {
		t.Fatal("expected scoped full cleanup source key error")
	}
}

func TestIndexDocumentsSourceIDKeyRequiresStringMetadata(t *testing.T) {
	_, err := IndexDocuments(
		context.Background(),
		[]documents.Document{documents.New("alpha", nil)},
		NewInMemoryRecordManager("unit"),
		vectorstores.NewInMemory(embeddings.NewFake(8)),
		Options{SourceIDKey: "source", Cleanup: CleanupFull},
	)
	if err == nil {
		t.Fatal("expected missing source id error")
	}
}

func TestInMemoryRecordManagerUpdateValidation(t *testing.T) {
	manager := NewInMemoryRecordManager("unit")
	err := manager.Update(context.Background(), []string{"a", "b"}, []string{"one"}, time.Time{})
	if err == nil {
		t.Fatal("expected group length error")
	}
	err = manager.Update(context.Background(), []string{"a"}, nil, time.Now().Add(time.Hour))
	if err == nil {
		t.Fatal("expected future time error")
	}
}

func TestHashDocumentStable(t *testing.T) {
	first, err := HashDocument(documents.New("alpha", map[string]any{"source": "a"}))
	if err != nil {
		t.Fatalf("hash first: %v", err)
	}
	second, err := HashDocument(documents.New("alpha", map[string]any{"source": "a"}))
	if err != nil {
		t.Fatalf("hash second: %v", err)
	}
	third, err := HashDocument(documents.New("alpha", map[string]any{"source": "b"}))
	if err != nil {
		t.Fatalf("hash third: %v", err)
	}
	if first != second {
		t.Fatalf("hash not stable: %q %q", first, second)
	}
	if first == third {
		t.Fatal("hash should include metadata")
	}
}

type trackingIterator struct {
	docs   []documents.Document
	index  int
	closed bool
}

func (i *trackingIterator) Next(context.Context) (documents.Document, bool, error) {
	if i.index >= len(i.docs) {
		return documents.Document{}, false, nil
	}
	doc := i.docs[i.index].Clone()
	i.index++
	return doc, true, nil
}

func (i *trackingIterator) Close() error {
	i.closed = true
	return nil
}

var errTest = errors.New("test error")

// stubRecordManager delegates to an in-memory record manager but can be
// configured to fail individual methods.
type stubRecordManager struct {
	inner       *InMemoryRecordManager
	getTimeErr  error
	existsErr   error
	updateErr   error
	deleteErr   error
	listKeysErr error
}

func newStubRecordManager(namespace string) *stubRecordManager {
	return &stubRecordManager{inner: NewInMemoryRecordManager(namespace)}
}

func (m *stubRecordManager) GetTime(ctx context.Context) (time.Time, error) {
	if m.getTimeErr != nil {
		return time.Time{}, m.getTimeErr
	}
	return m.inner.GetTime(ctx)
}

func (m *stubRecordManager) Update(ctx context.Context, keys []string, groupIDs []string, timeAtLeast time.Time) error {
	if m.updateErr != nil {
		return m.updateErr
	}
	return m.inner.Update(ctx, keys, groupIDs, timeAtLeast)
}

func (m *stubRecordManager) Exists(ctx context.Context, keys []string) ([]bool, error) {
	if m.existsErr != nil {
		return nil, m.existsErr
	}
	return m.inner.Exists(ctx, keys)
}

func (m *stubRecordManager) DeleteKeys(ctx context.Context, keys []string) error {
	if m.deleteErr != nil {
		return m.deleteErr
	}
	return m.inner.DeleteKeys(ctx, keys)
}

func (m *stubRecordManager) ListKeys(ctx context.Context, groupIDs []string, before time.Time, limit int) ([]string, error) {
	if m.listKeysErr != nil {
		return nil, m.listKeysErr
	}
	return m.inner.ListKeys(ctx, groupIDs, before, limit)
}

// failingVectorStore wraps a vector store and fails AddDocuments/Delete on
// demand.
type failingVectorStore struct {
	vectorstores.VectorStore
	addErr    error
	deleteErr error
}

func (s failingVectorStore) AddDocuments(ctx context.Context, docs []documents.Document) ([]string, error) {
	if s.addErr != nil {
		return nil, s.addErr
	}
	return s.VectorStore.AddDocuments(ctx, docs)
}

func (s failingVectorStore) Delete(ctx context.Context, ids []string) error {
	if s.deleteErr != nil {
		return s.deleteErr
	}
	return s.VectorStore.Delete(ctx, ids)
}

type errorIterator struct{ err error }

func (i *errorIterator) Next(context.Context) (documents.Document, bool, error) {
	return documents.Document{}, false, i.err
}

func (i *errorIterator) Close() error { return nil }

func TestIndexDocumentsRequiresRecordManagerAndVectorStore(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("alpha", nil)}

	if _, err := IndexDocuments(ctx, docs, nil, store, Options{}); err == nil {
		t.Fatal("expected nil record manager error")
	}
	if _, err := IndexDocuments(ctx, docs, manager, nil, Options{}); err == nil {
		t.Fatal("expected nil vector store error")
	}
	if _, err := IndexDocumentIterator(ctx, &trackingIterator{}, nil, store, Options{}); err == nil {
		t.Fatal("expected nil record manager error from iterator indexing")
	}
}

func TestIndexDocumentIteratorNilIterator(t *testing.T) {
	_, err := IndexDocumentIterator(
		context.Background(),
		nil,
		NewInMemoryRecordManager("unit"),
		vectorstores.NewInMemory(embeddings.NewFake(8)),
		Options{},
	)
	if err == nil {
		t.Fatal("expected nil iterator error")
	}
}

func TestIndexDocumentsGetTimeError(t *testing.T) {
	manager := newStubRecordManager("unit")
	manager.getTimeErr = errTest
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("alpha", nil)}

	if _, err := IndexDocuments(context.Background(), docs, manager, store, Options{}); !errors.Is(err, errTest) {
		t.Fatalf("expected get time error, got %v", err)
	}
	if _, err := IndexDocumentIterator(context.Background(), &trackingIterator{docs: docs}, manager, store, Options{}); !errors.Is(err, errTest) {
		t.Fatalf("expected get time error from iterator indexing, got %v", err)
	}
}

func TestIndexDocumentsCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("alpha", nil)}

	if _, err := IndexDocuments(ctx, docs, manager, store, Options{}); err == nil {
		t.Fatal("expected context error")
	}
	if _, err := IndexDocumentIterator(ctx, &trackingIterator{docs: docs}, manager, store, Options{}); err == nil {
		t.Fatal("expected context error from iterator indexing")
	}
}

func TestIndexDocumentsUnhashableDocument(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("alpha", map[string]any{"bad": func() {}})}

	if _, err := IndexDocuments(ctx, docs, manager, store, Options{}); err == nil {
		t.Fatal("expected hash error for unmarshalable metadata")
	}
	if _, err := HashDocument(docs[0]); err == nil {
		t.Fatal("expected HashDocument error for unmarshalable metadata")
	}
}

func TestIndexDocumentsRecordManagerErrors(t *testing.T) {
	ctx := context.Background()
	docs := []documents.Document{documents.New("alpha", nil)}

	t.Run("exists", func(t *testing.T) {
		manager := newStubRecordManager("unit")
		manager.existsErr = errTest
		store := vectorstores.NewInMemory(embeddings.NewFake(8))
		if _, err := IndexDocuments(ctx, docs, manager, store, Options{}); !errors.Is(err, errTest) {
			t.Fatalf("expected exists error, got %v", err)
		}
	})

	t.Run("update", func(t *testing.T) {
		manager := newStubRecordManager("unit")
		manager.updateErr = errTest
		store := vectorstores.NewInMemory(embeddings.NewFake(8))
		if _, err := IndexDocuments(ctx, docs, manager, store, Options{}); !errors.Is(err, errTest) {
			t.Fatalf("expected update error, got %v", err)
		}
	})

	t.Run("add documents", func(t *testing.T) {
		manager := NewInMemoryRecordManager("unit")
		store := failingVectorStore{
			VectorStore: vectorstores.NewInMemory(embeddings.NewFake(8)),
			addErr:      errTest,
		}
		if _, err := IndexDocuments(ctx, docs, manager, store, Options{}); !errors.Is(err, errTest) {
			t.Fatalf("expected add documents error, got %v", err)
		}
	})
}

func TestIndexDocumentsIncrementalCleanupDeletesStaleRecords(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	original := []documents.Document{
		documents.New("old a1", map[string]any{"source": "a"}),
		documents.New("old a2", map[string]any{"source": "a"}),
	}
	if _, err := IndexDocuments(ctx, original, manager, store, Options{SourceIDKey: "source"}); err != nil {
		t.Fatalf("index original: %v", err)
	}
	oldKeys := make([]string, 0, len(original))
	for _, doc := range original {
		key, err := HashDocument(doc)
		if err != nil {
			t.Fatalf("hash: %v", err)
		}
		oldKeys = append(oldKeys, key)
	}
	time.Sleep(time.Millisecond)

	// Two docs from the same source exercise group ID deduplication.
	replacement := []documents.Document{
		documents.New("new a1", map[string]any{"source": "a"}),
		documents.New("new a2", map[string]any{"source": "a"}),
	}
	got, err := IndexDocuments(ctx, replacement, manager, store, Options{
		SourceIDKey: "source",
		Cleanup:     CleanupIncremental,
	})
	if err != nil {
		t.Fatalf("index replacement: %v", err)
	}
	if got.NumAdded != 2 || got.NumDeleted != 2 {
		t.Fatalf("incremental cleanup result: %+v", got)
	}
	stored, err := store.GetByIDs(ctx, oldKeys)
	if err != nil {
		t.Fatalf("get ids: %v", err)
	}
	if len(stored) != 0 {
		t.Fatalf("stale docs should be deleted, got %#v", stored)
	}
}

func TestIndexDocumentsIncrementalCleanupListKeysError(t *testing.T) {
	manager := newStubRecordManager("unit")
	manager.listKeysErr = errTest
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("alpha", map[string]any{"source": "a"})}

	_, err := IndexDocuments(context.Background(), docs, manager, store, Options{
		SourceIDKey: "source",
		Cleanup:     CleanupIncremental,
	})
	if !errors.Is(err, errTest) {
		t.Fatalf("expected list keys error, got %v", err)
	}
}

func TestIndexDocumentsFullCleanupErrors(t *testing.T) {
	ctx := context.Background()
	original := []documents.Document{documents.New("old a", map[string]any{"source": "a"})}
	replacement := []documents.Document{documents.New("new a", map[string]any{"source": "a"})}

	setup := func(t *testing.T) (*stubRecordManager, *vectorstores.InMemory) {
		t.Helper()
		manager := newStubRecordManager("unit")
		store := vectorstores.NewInMemory(embeddings.NewFake(8))
		if _, err := IndexDocuments(ctx, original, manager, store, Options{SourceIDKey: "source"}); err != nil {
			t.Fatalf("index original: %v", err)
		}
		time.Sleep(time.Millisecond)
		return manager, store
	}

	t.Run("list keys", func(t *testing.T) {
		manager, store := setup(t)
		manager.listKeysErr = errTest
		_, err := IndexDocuments(ctx, replacement, manager, store, Options{SourceIDKey: "source", Cleanup: CleanupFull})
		if !errors.Is(err, errTest) {
			t.Fatalf("expected list keys error, got %v", err)
		}
	})

	t.Run("vector store delete", func(t *testing.T) {
		manager, store := setup(t)
		failing := failingVectorStore{VectorStore: store, deleteErr: errTest}
		_, err := IndexDocuments(ctx, replacement, manager, failing, Options{SourceIDKey: "source", Cleanup: CleanupFull})
		if !errors.Is(err, errTest) {
			t.Fatalf("expected delete error, got %v", err)
		}
	})

	t.Run("delete keys", func(t *testing.T) {
		manager, store := setup(t)
		manager.deleteErr = errTest
		_, err := IndexDocuments(ctx, replacement, manager, store, Options{SourceIDKey: "source", Cleanup: CleanupFull})
		if !errors.Is(err, errTest) {
			t.Fatalf("expected delete keys error, got %v", err)
		}
	})

	t.Run("cancelled context", func(t *testing.T) {
		manager, store := setup(t)
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		// No docs to index, so the context error surfaces from cleanup.
		_, err := IndexDocuments(cancelled, nil, manager, store, Options{Cleanup: CleanupFull})
		if err == nil {
			t.Fatal("expected context error from cleanup")
		}
	})
}

func TestIndexDocumentsScopedFullCleanupWithoutDocs(t *testing.T) {
	got, err := IndexDocuments(
		context.Background(),
		nil,
		NewInMemoryRecordManager("unit"),
		vectorstores.NewInMemory(embeddings.NewFake(8)),
		Options{SourceIDKey: "source", Cleanup: CleanupScopedFull},
	)
	if err != nil {
		t.Fatalf("scoped full cleanup without docs: %v", err)
	}
	if got != (IndexingResult{}) {
		t.Fatalf("expected empty result, got %+v", got)
	}
}

func TestIndexDocumentIteratorNextError(t *testing.T) {
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))

	_, err := IndexDocumentIterator(context.Background(), &errorIterator{err: errTest}, manager, store, Options{})
	if !errors.Is(err, errTest) {
		t.Fatalf("expected iterator error, got %v", err)
	}
}

func TestIndexDocumentIteratorBatchErrors(t *testing.T) {
	ctx := context.Background()
	unhashable := documents.New("alpha", map[string]any{"bad": func() {}})

	t.Run("full batch", func(t *testing.T) {
		manager := NewInMemoryRecordManager("unit")
		store := vectorstores.NewInMemory(embeddings.NewFake(8))
		iter := &trackingIterator{docs: []documents.Document{unhashable}}
		if _, err := IndexDocumentIterator(ctx, iter, manager, store, Options{BatchSize: 1}); err == nil {
			t.Fatal("expected batch error")
		}
	})

	t.Run("trailing partial batch", func(t *testing.T) {
		manager := NewInMemoryRecordManager("unit")
		store := vectorstores.NewInMemory(embeddings.NewFake(8))
		iter := &trackingIterator{docs: []documents.Document{unhashable}}
		if _, err := IndexDocumentIterator(ctx, iter, manager, store, Options{BatchSize: 2}); err == nil {
			t.Fatal("expected trailing batch error")
		}
	})

	t.Run("cleanup", func(t *testing.T) {
		manager := newStubRecordManager("unit")
		manager.listKeysErr = errTest
		store := vectorstores.NewInMemory(embeddings.NewFake(8))
		iter := &trackingIterator{docs: []documents.Document{
			documents.New("alpha", map[string]any{"source": "a"}),
		}}
		_, err := IndexDocumentIterator(ctx, iter, manager, store, Options{
			SourceIDKey: "source",
			Cleanup:     CleanupFull,
		})
		if !errors.Is(err, errTest) {
			t.Fatalf("expected cleanup error, got %v", err)
		}
	})
}

func TestCleanupKeysDefaultsLimit(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	if err := manager.Update(ctx, []string{"stale"}, []string{"a"}, time.Time{}); err != nil {
		t.Fatalf("update: %v", err)
	}

	deleted, err := cleanupKeys(ctx, manager, store, nil, time.Now().Add(time.Hour), 0)
	if err != nil {
		t.Fatalf("cleanup keys: %v", err)
	}
	if deleted != 1 {
		t.Fatalf("deleted = %d, want 1", deleted)
	}
	exists, err := manager.Exists(ctx, []string{"stale"})
	if err != nil {
		t.Fatalf("exists: %v", err)
	}
	if exists[0] {
		t.Fatal("stale record should be deleted")
	}
}

func TestIndexDocumentIteratorTrailingPartialBatch(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	iter := &trackingIterator{docs: []documents.Document{
		documents.New("one", nil),
		documents.New("two", nil),
		documents.New("three", nil),
	}}

	got, err := IndexDocumentIterator(ctx, iter, manager, store, Options{BatchSize: 2})
	if err != nil {
		t.Fatalf("index iterator: %v", err)
	}
	if !iter.closed {
		t.Fatal("iterator was not closed")
	}
	if got.NumAdded != 3 || got.NumSkipped != 0 {
		t.Fatalf("trailing batch result: %+v", got)
	}
}

func TestInMemoryRecordManagerEmptyNamespace(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("")
	if err := manager.Update(ctx, []string{"a", "b"}, nil, time.Time{}); err != nil {
		t.Fatalf("update: %v", err)
	}
	exists, err := manager.Exists(ctx, []string{"a", "missing"})
	if err != nil {
		t.Fatalf("exists: %v", err)
	}
	if !exists[0] || exists[1] {
		t.Fatalf("exists: %#v", exists)
	}
	keys, err := manager.ListKeys(ctx, nil, time.Time{}, 0)
	if err != nil {
		t.Fatalf("list keys: %v", err)
	}
	for _, key := range keys {
		if key != "a" && key != "b" {
			t.Fatalf("unnamespaced key %q should be returned unchanged", key)
		}
	}
	if len(keys) != 2 {
		t.Fatalf("list keys: %#v", keys)
	}
	if err := manager.DeleteKeys(ctx, []string{"a"}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	keys, err = manager.ListKeys(ctx, nil, time.Time{}, 0)
	if err != nil {
		t.Fatalf("list keys after delete: %v", err)
	}
	if len(keys) != 1 || keys[0] != "b" {
		t.Fatalf("keys after delete: %#v", keys)
	}
}

func TestInMemoryRecordManagerListKeysFilters(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	if err := manager.Update(ctx, []string{"a", "b"}, []string{"g", "g"}, time.Time{}); err != nil {
		t.Fatalf("update: %v", err)
	}

	keys, err := manager.ListKeys(ctx, nil, time.Now().Add(-time.Hour), 0)
	if err != nil {
		t.Fatalf("list keys: %v", err)
	}
	if len(keys) != 0 {
		t.Fatalf("before filter should exclude all keys, got %#v", keys)
	}

	keys, err = manager.ListKeys(ctx, []string{"other"}, time.Time{}, 0)
	if err != nil {
		t.Fatalf("list keys: %v", err)
	}
	if len(keys) != 0 {
		t.Fatalf("group filter should exclude all keys, got %#v", keys)
	}

	keys, err = manager.ListKeys(ctx, nil, time.Now().Add(time.Hour), 1)
	if err != nil {
		t.Fatalf("list keys: %v", err)
	}
	if len(keys) != 1 {
		t.Fatalf("limit should cap results to 1, got %#v", keys)
	}
}

func TestInMemoryRecordManagerNamespaceRename(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("ab")
	if err := manager.Update(ctx, []string{"k"}, nil, time.Time{}); err != nil {
		t.Fatalf("update: %v", err)
	}
	// Renaming the namespace leaves stored keys that no longer carry the
	// current prefix; ListKeys must return them unchanged.
	manager.Namespace = "abcdefgh"
	keys, err := manager.ListKeys(ctx, nil, time.Time{}, 0)
	if err != nil {
		t.Fatalf("list keys: %v", err)
	}
	if len(keys) != 1 || keys[0] != "ab:k" {
		t.Fatalf("keys after namespace rename: %#v", keys)
	}
}
