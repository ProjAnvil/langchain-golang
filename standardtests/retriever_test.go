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

// stubRetriever is a retriever whose behavior is configured per scenario. The
// call counter is shared across instances so factories producing fresh
// retrievers can still script per-call behavior.
type stubRetriever struct {
	docs       []documents.Document
	err        error
	calls      *int
	failOnCall int
	ignoreCtx  bool
}

func (r stubRetriever) GetRelevantDocuments(ctx context.Context, _ string) ([]documents.Document, error) {
	if !r.ignoreCtx {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if r.calls != nil {
		*r.calls++
		if r.err != nil && *r.calls == r.failOnCall {
			return nil, r.err
		}
	} else if r.err != nil {
		return nil, r.err
	}
	return r.docs, nil
}

func TestRunRetrieverBasicsFailures(t *testing.T) {
	docs := []documents.Document{
		documents.New("alpha beta", map[string]any{"source": "unit"}),
	}
	factory := func(retriever stubRetriever) RetrieverFactory {
		return func(t testing.TB) retrievers.Retriever {
			t.Helper()
			return retriever
		}
	}

	expectConformanceFailure(t, "retrieve errors", func(t *testing.T) {
		RunRetrieverBasics(t, factory(stubRetriever{docs: docs, err: errConformanceStub}))
	})
	expectConformanceFailure(t, "retrieve returns no documents", func(t *testing.T) {
		RunRetrieverBasics(t, factory(stubRetriever{}))
	})
	expectConformanceFailure(t, "second retrieve errors", func(t *testing.T) {
		calls := 0
		RunRetrieverBasics(t, factory(stubRetriever{
			docs:       docs,
			err:        errConformanceStub,
			calls:      &calls,
			failOnCall: 2,
		}))
	})
	expectConformanceFailure(t, "retriever shares metadata", func(t *testing.T) {
		RunRetrieverBasics(t, factory(stubRetriever{docs: docs}))
	})
	expectConformanceFailure(t, "context cancellation ignored", func(t *testing.T) {
		RunRetrieverBasics(t, factory(stubRetriever{docs: docs, ignoreCtx: true}))
	})
}
