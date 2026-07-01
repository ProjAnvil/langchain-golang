package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documentloaders"
	"github.com/projanvil/langchain-golang/core/documents"
)

func TestRunDocumentLoaderBasicsWithSliceLoader(t *testing.T) {
	RunDocumentLoaderBasics(t, func(t testing.TB) documentloaders.LazyLoader {
		t.Helper()
		return sliceLoader{docs: []documents.Document{
			documents.New("alpha", map[string]any{"source": "unit"}),
			documents.New("beta", map[string]any{"source": "unit"}),
		}}
	})
}

type sliceLoader struct {
	docs []documents.Document
}

func (l sliceLoader) LazyLoad(context.Context) (documentloaders.DocumentIterator, error) {
	return documentloaders.NewSliceIterator(l.docs), nil
}
