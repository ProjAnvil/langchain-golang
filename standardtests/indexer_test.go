package standardtests

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/indexing"
)

// TestRunDocumentIndexerConformanceWithInMemoryIndex wires the conformance
// suite to the in-memory DocumentIndex implementation, mirroring how Python
// implementers subclass DocumentIndexerTestSuite with an index fixture.
func TestRunDocumentIndexerConformanceWithInMemoryIndex(t *testing.T) {
	RunDocumentIndexerConformance(t, func(t testing.TB) indexing.DocumentIndex {
		t.Helper()
		return indexing.NewInMemoryDocumentIndex(4)
	})
}
