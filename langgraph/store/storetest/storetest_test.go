package storetest_test

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/store/storetest"
)

// TestInMemoryStoreConformance runs the shared Store conformance suite against
// the in-process InMemoryStore. Each subtest gets a freshly-constructed,
// empty store.
func TestInMemoryStoreConformance(t *testing.T) {
	storetest.Run(t, func(t *testing.T) store.Store {
		t.Helper()
		return store.NewInMemoryStore()
	})
}

// TestInMemoryStoreSemanticConformance runs the optional semantic-search
// conformance suite against an InMemoryStore built with a vector index
// (Python's InMemoryStore(index={...})), exercising put→search cosine ranking,
// filter+query combination, and vector recomputation on update.
func TestInMemoryStoreSemanticConformance(t *testing.T) {
	storetest.RunSemantic(t, func(t *testing.T, embed embeddings.Embeddings) store.Store {
		t.Helper()
		return store.NewInMemoryStoreWithIndex(store.IndexConfig{Embed: embed, Fields: []string{"text"}})
	})
}
