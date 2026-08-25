package indexing

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// Mirrors test_hashed_document.py::test_hashing_custom_key_encoder: the
// callable encoder's return value becomes the document's dedup key.
func TestIndexDocumentsCustomKeyEncoderFunc(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{
		documents.New("Lorem ipsum dolor sit amet", map[string]any{"key": "like a duck"}),
	}
	options := Options{
		KeyEncoderFunc: func(doc documents.Document) (string, error) {
			return "quack-" + doc.Metadata["key"].(string), nil
		},
	}

	got, err := IndexDocuments(ctx, docs, manager, store, options)
	if err != nil {
		t.Fatalf("index: %v", err)
	}
	if got.NumAdded != 1 || got.NumSkipped != 0 {
		t.Fatalf("first result: %+v", got)
	}
	exists, err := manager.Exists(ctx, []string{"quack-like a duck"})
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if len(exists) != 1 || !exists[0] {
		t.Fatalf("record manager does not have the custom key: %v", exists)
	}
	stored, err := store.GetByIDs(ctx, []string{"quack-like a duck"})
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(stored) != 1 || stored[0].PageContent != "Lorem ipsum dolor sit amet" {
		t.Fatalf("stored doc: %+v", stored)
	}

	// Re-indexing with the same callable skips the document (dedup by custom key).
	got, err = IndexDocuments(ctx, docs, manager, store, options)
	if err != nil {
		t.Fatalf("re-index: %v", err)
	}
	if got.NumAdded != 0 || got.NumSkipped != 1 {
		t.Fatalf("second result: %+v", got)
	}
}

// KeyEncoderFunc errors propagate.
func TestIndexDocumentsKeyEncoderFuncError(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	errTest := errors.New("encoder blew up")
	_, err := IndexDocuments(ctx, []documents.Document{documents.New("x", nil)}, manager, store, Options{
		KeyEncoderFunc: func(documents.Document) (string, error) { return "", errTest },
	})
	if !errors.Is(err, errTest) {
		t.Fatalf("expected encoder error, got %v", err)
	}
}

// KeyEncoderFunc overrides KeyEncoder when both are set (Python's callable
// branch short-circuits the algorithm path, indexing/api.py:208-210).
func TestKeyEncoderFuncOverridesAlgorithm(t *testing.T) {
	ctx := context.Background()
	manager := NewInMemoryRecordManager("unit")
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docs := []documents.Document{documents.New("alpha", nil)}
	got, err := IndexDocuments(ctx, docs, manager, store, Options{
		KeyEncoder: KeyEncoderSHA1,
		KeyEncoderFunc: func(documents.Document) (string, error) {
			return "custom-key", nil
		},
	})
	if err != nil {
		t.Fatalf("index: %v", err)
	}
	if got.NumAdded != 1 {
		t.Fatalf("result: %+v", got)
	}
	exists, err := manager.Exists(ctx, []string{"custom-key"})
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if !exists[0] {
		t.Fatal("custom key not recorded")
	}
}
