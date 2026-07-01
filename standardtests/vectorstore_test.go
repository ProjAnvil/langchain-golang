package standardtests

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

func TestRunVectorStoreBasicsWithInMemoryStore(t *testing.T) {
	RunVectorStoreBasics(t, func(t testing.TB) vectorstores.VectorStore {
		t.Helper()
		return vectorstores.NewInMemory(embeddings.NewFake(32))
	})
}
