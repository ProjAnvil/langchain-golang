package indexing

import (
	"context"
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
