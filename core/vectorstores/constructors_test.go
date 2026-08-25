package vectorstores

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

// Mirrors test_inmemory_similarity_search (test_in_memory.py:19): a store
// built from texts answers similarity search end to end.
func TestFromTextsEndToEnd(t *testing.T) {
	ctx := context.Background()
	store, err := FromTexts(ctx, embeddings.NewDeterministicFake(3), []string{"foo", "bar", "baz"})
	if err != nil {
		t.Fatalf("FromTexts: %v", err)
	}
	out, err := store.SimilaritySearch(ctx, "foo", 1)
	if err != nil {
		t.Fatalf("SimilaritySearch: %v", err)
	}
	if len(out) != 1 || out[0].PageContent != "foo" {
		t.Fatalf("search result: %+v", out)
	}
}

// Mirrors test_default_from_documents (test_vectorstore.py:242), first case:
// document IDs are used when WithIDs is absent.
func TestFromDocumentsUsesDocumentIDs(t *testing.T) {
	ctx := context.Background()
	store, err := FromDocuments(ctx, embeddings.NewFake(1), []documents.Document{
		documents.New("hello", map[string]any{"foo": "bar"}).WithID("1"),
	})
	if err != nil {
		t.Fatalf("FromDocuments: %v", err)
	}
	got, err := store.GetByIDs(ctx, []string{"1"})
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(got) != 1 || got[0].ID != "1" || got[0].PageContent != "hello" || got[0].Metadata["foo"] != "bar" {
		t.Fatalf("stored doc: %+v", got)
	}
}

// Second case: explicit ids are honored for documents without IDs.
func TestFromDocumentsWithIDsOption(t *testing.T) {
	ctx := context.Background()
	store, err := FromDocuments(ctx, embeddings.NewFake(1), []documents.Document{
		documents.New("hello", map[string]any{"foo": "bar"}),
	}, WithIDs([]string{"1"}))
	if err != nil {
		t.Fatalf("FromDocuments: %v", err)
	}
	got, err := store.GetByIDs(ctx, []string{"1"})
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(got) != 1 || got[0].ID != "1" || got[0].Metadata["foo"] != "bar" {
		t.Fatalf("stored doc: %+v", got)
	}
}

// Third case: ids win over document IDs, and the input document is not
// modified (Python asserts original_document.id == "7" afterwards).
func TestFromDocumentsIDsOverrideWithoutMutatingInput(t *testing.T) {
	ctx := context.Background()
	original := documents.New("baz", nil).WithID("7")
	store, err := FromDocuments(ctx, embeddings.NewFake(1), []documents.Document{original}, WithIDs([]string{"6"}))
	if err != nil {
		t.Fatalf("FromDocuments: %v", err)
	}
	if original.ID != "7" {
		t.Fatalf("original document mutated: %+v", original)
	}
	got, err := store.GetByIDs(ctx, []string{"6"})
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(got) != 1 || got[0].ID != "6" || got[0].PageContent != "baz" {
		t.Fatalf("stored doc: %+v", got)
	}
}

// Metadatas option flows through to stored documents.
func TestFromTextsWithMetadatas(t *testing.T) {
	ctx := context.Background()
	store, err := FromTexts(ctx, embeddings.NewFake(4), []string{"foo", "bar"},
		WithMetadatas([]map[string]any{{"id": 1}, {"id": 2}}))
	if err != nil {
		t.Fatalf("FromTexts: %v", err)
	}
	out, err := store.SimilaritySearchWithScoreFilter(ctx, "foo", 2, func(doc documents.Document) bool {
		return doc.Metadata["id"] == 2
	})
	if err != nil {
		t.Fatalf("SimilaritySearchWithScoreFilter: %v", err)
	}
	if len(out) != 1 || out[0].Document.PageContent != "bar" {
		t.Fatalf("filtered result: %+v", out)
	}
}

// A nil embedder surfaces the store's "embedder is required" error.
func TestFromTextsNilEmbedder(t *testing.T) {
	if _, err := FromTexts(context.Background(), nil, []string{"x"}); err == nil {
		t.Fatal("expected error for nil embedder")
	}
}
