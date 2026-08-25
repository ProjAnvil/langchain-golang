package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
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

// stubIndex wraps a working in-memory index with configurable behavior
// overrides for failure-injection tests.
type stubIndex struct {
	inner          *indexing.InMemoryDocumentIndex
	upsertErr      error
	getErr         error
	deleteErr      error
	upsertResponse *indexing.UpsertResponse
	getResponse    []documents.Document
	deleteResponse *indexing.DeleteResponse
	deleteNoop     bool
	allowNilDelete bool
}

func newStubIndex() *stubIndex {
	return &stubIndex{inner: indexing.NewInMemoryDocumentIndex(4)}
}

func (s *stubIndex) Upsert(ctx context.Context, items []documents.Document) (indexing.UpsertResponse, error) {
	if s.upsertErr != nil {
		return indexing.UpsertResponse{}, s.upsertErr
	}
	if s.upsertResponse != nil {
		return *s.upsertResponse, nil
	}
	return s.inner.Upsert(ctx, items)
}

func (s *stubIndex) Delete(ctx context.Context, ids []string) (indexing.DeleteResponse, error) {
	if ids == nil && s.allowNilDelete {
		return indexing.DeleteResponse{}, nil
	}
	if s.deleteErr != nil {
		return indexing.DeleteResponse{}, s.deleteErr
	}
	if s.deleteNoop {
		return indexing.DeleteResponse{
			Succeeded:  append([]string(nil), ids...),
			Failed:     []string{},
			NumDeleted: len(ids),
		}, nil
	}
	if s.deleteResponse != nil {
		return *s.deleteResponse, nil
	}
	return s.inner.Delete(ctx, ids)
}

func (s *stubIndex) Get(ctx context.Context, ids []string) ([]documents.Document, error) {
	if s.getErr != nil {
		return nil, s.getErr
	}
	if s.getResponse != nil {
		return s.getResponse, nil
	}
	return s.inner.Get(ctx, ids)
}

func (s *stubIndex) GetRelevantDocuments(ctx context.Context, query string) ([]documents.Document, error) {
	return s.inner.GetRelevantDocuments(ctx, query)
}

func TestRunDocumentIndexerConformanceFailures(t *testing.T) {
	factory := func(configure func(*stubIndex)) IndexerFactory {
		return func(t testing.TB) indexing.DocumentIndex {
			t.Helper()
			index := newStubIndex()
			configure(index)
			return index
		}
	}

	expectConformanceFailure(t, "upsert errors", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.upsertErr = errConformanceStub
		}))
	})
	expectConformanceFailure(t, "get errors", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.getErr = errConformanceStub
		}))
	})
	expectConformanceFailure(t, "delete errors", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.deleteErr = errConformanceStub
		}))
	})
	expectConformanceFailure(t, "upsert reports ids it never stored", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.upsertResponse = &indexing.UpsertResponse{Succeeded: []string{"a", "b"}, Failed: []string{}}
		}))
	})
	expectConformanceFailure(t, "get fabricates documents with unknown ids", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.getResponse = []documents.Document{
				documents.New("foo", map[string]any{"id": 1}).WithID("zz"),
				documents.New("bar", map[string]any{"id": 2}).WithID("zz2"),
			}
		}))
	})
	expectConformanceFailure(t, "get duplicates the first document", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.getResponse = []documents.Document{
				documents.New("foo", map[string]any{"id": 1}).WithID("dup1"),
				documents.New("foo", map[string]any{"id": 1}).WithID("dup2"),
			}
		}))
	})
	expectConformanceFailure(t, "get drops document ids", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.getResponse = []documents.Document{
				documents.New("foo", map[string]any{"id": 1}),
				documents.New("bar", map[string]any{"id": 2}),
			}
		}))
	})
	expectConformanceFailure(t, "delete reports a missing id as deleted", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.deleteResponse = &indexing.DeleteResponse{Succeeded: []string{"1"}, Failed: []string{}, NumDeleted: 1}
		}))
	})
	expectConformanceFailure(t, "delete accepts nil ids", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.allowNilDelete = true
		}))
	})
	expectConformanceFailure(t, "delete reports success without deleting", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.deleteNoop = true
		}))
	})
	// The id-membership loop in "upsert no ids" Fatalfs before
	// requireIndexedDocument runs, so reaching its not-found and empty-ID
	// branches needs scenarios where upsert and get agree on IDs.
	expectConformanceFailure(t, "get serves the upserted ids with wrong content", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.upsertResponse = &indexing.UpsertResponse{Succeeded: []string{"x1", "x2"}, Failed: []string{}}
			s.getResponse = []documents.Document{
				documents.New("baz", map[string]any{"id": 3}).WithID("x1"),
				documents.New("qux", map[string]any{"id": 4}).WithID("x2"),
			}
		}))
	})
	expectConformanceFailure(t, "upsert and get agree on empty ids", func(t *testing.T) {
		RunDocumentIndexerConformance(t, factory(func(s *stubIndex) {
			s.upsertResponse = &indexing.UpsertResponse{Succeeded: []string{"", ""}, Failed: []string{}}
			s.getResponse = []documents.Document{
				documents.New("foo", map[string]any{"id": 1}),
				documents.New("bar", map[string]any{"id": 2}),
			}
		}))
	})
}
