package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
)

func TestRunEmbeddingsBasicsWithFakeEmbeddings(t *testing.T) {
	RunEmbeddingsBasics(t, func(t testing.TB) embeddings.Embeddings {
		t.Helper()
		return embeddings.NewFake(8)
	})
}

// stubEmbeddings is an embeddings implementation whose behavior is configured
// per scenario.
type stubEmbeddings struct {
	queryVector []float64
	queryErr    error
	docVectors  [][]float64
	docsErr     error
	ignoreCtx   bool
}

func (e stubEmbeddings) EmbedQuery(ctx context.Context, _ string) ([]float64, error) {
	if !e.ignoreCtx {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if e.queryErr != nil {
		return nil, e.queryErr
	}
	if e.queryVector == nil {
		return []float64{1, 1}, nil
	}
	return e.queryVector, nil
}

func (e stubEmbeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e.docsErr != nil {
		return nil, e.docsErr
	}
	if e.docVectors == nil {
		vectors := make([][]float64, len(texts))
		for i := range vectors {
			vectors[i] = []float64{1, 1}
		}
		return vectors, nil
	}
	return e.docVectors, nil
}

func TestRunEmbeddingsBasicsFailures(t *testing.T) {
	factory := func(model stubEmbeddings) EmbeddingsFactory {
		return func(t testing.TB) embeddings.Embeddings {
			t.Helper()
			return model
		}
	}

	expectConformanceFailure(t, "embed query errors", func(t *testing.T) {
		RunEmbeddingsBasics(t, factory(stubEmbeddings{queryErr: errConformanceStub}))
	})
	expectConformanceFailure(t, "embed query returns empty vector", func(t *testing.T) {
		RunEmbeddingsBasics(t, factory(stubEmbeddings{queryVector: []float64{}}))
	})
	expectConformanceFailure(t, "embed documents errors", func(t *testing.T) {
		RunEmbeddingsBasics(t, factory(stubEmbeddings{docsErr: errConformanceStub}))
	})
	expectConformanceFailure(t, "embed documents returns too few vectors", func(t *testing.T) {
		RunEmbeddingsBasics(t, factory(stubEmbeddings{docVectors: [][]float64{{1, 1}}}))
	})
	expectConformanceFailure(t, "embed documents returns empty vector", func(t *testing.T) {
		RunEmbeddingsBasics(t, factory(stubEmbeddings{docVectors: [][]float64{{}, {1, 1}}}))
	})
	expectConformanceFailure(t, "embed documents returns inconsistent dimensions", func(t *testing.T) {
		RunEmbeddingsBasics(t, factory(stubEmbeddings{docVectors: [][]float64{{1, 1}, {1}}}))
	})
	expectConformanceFailure(t, "context cancellation ignored", func(t *testing.T) {
		RunEmbeddingsBasics(t, factory(stubEmbeddings{ignoreCtx: true}))
	})
}
