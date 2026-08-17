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

// stubLoader is a lazy loader whose behavior is configured per scenario. Its
// iterator returns documents without cloning, so loads share metadata maps.
type stubLoader struct {
	docs    []documents.Document
	lazyErr error
	iterErr error
}

func (l stubLoader) LazyLoad(context.Context) (documentloaders.DocumentIterator, error) {
	if l.lazyErr != nil {
		return nil, l.lazyErr
	}
	return &stubIterator{docs: l.docs, err: l.iterErr}, nil
}

type stubIterator struct {
	docs  []documents.Document
	err   error
	index int
}

func (i *stubIterator) Next(context.Context) (documents.Document, bool, error) {
	if i.err != nil {
		return documents.Document{}, false, i.err
	}
	if i.index >= len(i.docs) {
		return documents.Document{}, false, nil
	}
	doc := i.docs[i.index]
	i.index++
	return doc, true, nil
}

func (i *stubIterator) Close() error { return nil }

// sequenceLoaderFactory returns the provided loaders in order, one per factory
// call, reusing the last loader once the sequence is exhausted.
func sequenceLoaderFactory(loaders ...documentloaders.LazyLoader) DocumentLoaderFactory {
	index := 0
	return func(t testing.TB) documentloaders.LazyLoader {
		t.Helper()
		if index >= len(loaders) {
			return loaders[len(loaders)-1]
		}
		loader := loaders[index]
		index++
		return loader
	}
}

func TestRunDocumentLoaderBasicsFailures(t *testing.T) {
	docs := []documents.Document{
		documents.New("alpha", map[string]any{"source": "unit"}),
		documents.New("beta", map[string]any{"source": "unit"}),
	}
	factory := func(loader stubLoader) DocumentLoaderFactory {
		return func(t testing.TB) documentloaders.LazyLoader {
			t.Helper()
			return loader
		}
	}

	expectConformanceFailure(t, "lazy load errors", func(t *testing.T) {
		RunDocumentLoaderBasics(t, factory(stubLoader{docs: docs, lazyErr: errConformanceStub}))
	})
	expectConformanceFailure(t, "second lazy load errors", func(t *testing.T) {
		RunDocumentLoaderBasics(t, sequenceLoaderFactory(
			stubLoader{docs: docs},
			stubLoader{lazyErr: errConformanceStub},
		))
	})
	expectConformanceFailure(t, "iterator errors", func(t *testing.T) {
		RunDocumentLoaderBasics(t, sequenceLoaderFactory(
			stubLoader{docs: docs},
			stubLoader{docs: docs, iterErr: errConformanceStub},
		))
	})
	expectConformanceFailure(t, "eager and lazy counts differ", func(t *testing.T) {
		RunDocumentLoaderBasics(t, sequenceLoaderFactory(
			stubLoader{docs: docs},
			stubLoader{docs: docs[:1]},
		))
	})
	expectConformanceFailure(t, "eager and lazy documents differ", func(t *testing.T) {
		RunDocumentLoaderBasics(t, sequenceLoaderFactory(
			stubLoader{docs: docs},
			stubLoader{docs: []documents.Document{
				documents.New("gamma", map[string]any{"source": "unit"}),
				documents.New("delta", map[string]any{"source": "unit"}),
			}},
		))
	})
	expectConformanceFailure(t, "loader returns no documents", func(t *testing.T) {
		RunDocumentLoaderBasics(t, factory(stubLoader{}))
	})
	expectConformanceFailure(t, "reload errors", func(t *testing.T) {
		RunDocumentLoaderBasics(t, sequenceLoaderFactory(
			stubLoader{docs: docs},
			stubLoader{docs: docs},
			stubLoader{docs: docs},
			stubLoader{lazyErr: errConformanceStub},
		))
	})
	expectConformanceFailure(t, "loader shares metadata", func(t *testing.T) {
		RunDocumentLoaderBasics(t, factory(stubLoader{docs: []documents.Document{
			documents.New("alpha", map[string]any{"source": "unit"}),
		}}))
	})
}
