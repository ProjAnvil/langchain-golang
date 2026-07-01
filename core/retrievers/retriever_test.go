package retrievers

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

func TestVectorStoreRetriever(t *testing.T) {
	store := vectorstores.NewInMemory(embeddings.NewFake(32))
	_, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", nil),
		documents.New("gamma delta", nil),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}

	retriever := NewVectorStoreRetriever(store, 1)
	docs, err := retriever.GetRelevantDocuments(context.Background(), "alpha")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs: got %d want 1", len(docs))
	}
	if docs[0].PageContent != "alpha beta" {
		t.Fatalf("top doc: got %q", docs[0].PageContent)
	}
}
