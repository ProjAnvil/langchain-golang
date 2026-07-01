package standardtests

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
)

func TestRunEmbeddingsBasicsWithFakeEmbeddings(t *testing.T) {
	RunEmbeddingsBasics(t, func(t testing.TB) embeddings.Embeddings {
		t.Helper()
		return embeddings.NewFake(8)
	})
}
