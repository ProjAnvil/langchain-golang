package storetest_test

import (
	"testing"

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
