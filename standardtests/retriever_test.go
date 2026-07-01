package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

func TestRunRetrieverBasicsWithStaticRetriever(t *testing.T) {
	RunRetrieverBasics(t, func(t testing.TB) retrievers.Retriever {
		t.Helper()
		return retrievers.Static{Documents: []documents.Document{
			documents.New("alpha beta", map[string]any{"source": "unit"}),
		}}
	})
}

func TestRunRetrieverBasicsWithVectorStoreRetriever(t *testing.T) {
	RunRetrieverBasics(t, func(t testing.TB) retrievers.Retriever {
		t.Helper()
		store := vectorstores.NewInMemory(embeddings.NewFake(16))
		if _, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("alpha beta", map[string]any{"source": "unit"}),
			documents.New("gamma", map[string]any{"source": "unit"}),
		}); err != nil {
			t.Fatalf("add documents: %v", err)
		}
		return retrievers.NewVectorStoreRetriever(store, 1)
	})
}
