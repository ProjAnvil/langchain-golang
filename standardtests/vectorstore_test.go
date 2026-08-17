package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

func TestRunVectorStoreBasicsWithInMemoryStore(t *testing.T) {
	RunVectorStoreBasics(t, func(t testing.TB) vectorstores.VectorStore {
		t.Helper()
		return vectorstores.NewInMemory(embeddings.NewFake(32))
	})
}

// minimalVectorStore hides the optional TextAdder, filtered-search, and MMR
// interfaces of the wrapped store so the optional standard tests skip.
type minimalVectorStore struct {
	vectorstores.VectorStore
}

func TestRunVectorStoreBasicsWithMinimalStore(t *testing.T) {
	RunVectorStoreBasics(t, func(t testing.TB) vectorstores.VectorStore {
		t.Helper()
		return minimalVectorStore{VectorStore: vectorstores.NewInMemory(embeddings.NewFake(16))}
	})
}

// stubVectorStore is a vector store with canned responses and error injection.
// It also implements the optional TextAdder, filtered-search, and MMR
// interfaces.
type stubVectorStore struct {
	addErr      error
	addIDs      []string
	getDocs     []documents.Document
	getErr      error
	searchRes   []vectorstores.SearchResult
	searchErr   error
	deleteErr   error
	addTextsIDs []string
	addTextsErr error
	filterRes   []vectorstores.SearchResult
	filterErr   error
	mmrDocs     []documents.Document
	mmrErr      error
}

func (s stubVectorStore) AddDocuments(context.Context, []documents.Document) ([]string, error) {
	if s.addErr != nil {
		return nil, s.addErr
	}
	if s.addIDs != nil {
		return s.addIDs, nil
	}
	return []string{"alpha", "gamma"}, nil
}

func (s stubVectorStore) Delete(context.Context, []string) error {
	return s.deleteErr
}

func (s stubVectorStore) GetByIDs(context.Context, []string) ([]documents.Document, error) {
	if s.getErr != nil {
		return nil, s.getErr
	}
	return s.getDocs, nil
}

func (s stubVectorStore) SimilaritySearch(
	ctx context.Context,
	query string,
	k int,
) ([]documents.Document, error) {
	results, err := s.SimilaritySearchWithScore(ctx, query, k)
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, len(results))
	for i, result := range results {
		docs[i] = result.Document
	}
	return docs, nil
}

func (s stubVectorStore) SimilaritySearchWithScore(
	context.Context,
	string,
	int,
) ([]vectorstores.SearchResult, error) {
	if s.searchErr != nil {
		return nil, s.searchErr
	}
	return s.searchRes, nil
}

func (s stubVectorStore) AddTexts(
	context.Context,
	[]string,
	[]map[string]any,
	[]string,
) ([]string, error) {
	if s.addTextsErr != nil {
		return nil, s.addTextsErr
	}
	return s.addTextsIDs, nil
}

func (s stubVectorStore) SimilaritySearchWithScoreFilter(
	context.Context,
	string,
	int,
	vectorstores.Filter,
) ([]vectorstores.SearchResult, error) {
	if s.filterErr != nil {
		return nil, s.filterErr
	}
	return s.filterRes, nil
}

func (s stubVectorStore) MaxMarginalRelevanceSearch(
	context.Context,
	string,
	int,
	int,
	float64,
	vectorstores.Filter,
) ([]documents.Document, error) {
	if s.mmrErr != nil {
		return nil, s.mmrErr
	}
	return s.mmrDocs, nil
}

func TestRunVectorStoreBasicsFailures(t *testing.T) {
	factory := func(store stubVectorStore) VectorStoreFactory {
		return func(t testing.TB) vectorstores.VectorStore {
			t.Helper()
			return store
		}
	}
	alphaResult := vectorstores.SearchResult{
		Document: documents.New("alpha beta", nil).WithID("alpha"),
		Score:    1,
	}
	textIDs := []string{"alpha-text", "gamma-text"}

	expectConformanceFailure(t, "add documents errors", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			addErr:      errConformanceStub,
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "add documents returns too few ids", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			addIDs:      []string{"alpha"},
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "get by ids errors", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			getErr:      errConformanceStub,
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "get by ids returns unexpected documents", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			getDocs: []documents.Document{
				documents.New("one", nil),
				documents.New("two", nil),
			},
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "get by ids returns wrong document id", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			getDocs:     []documents.Document{documents.New("other", nil).WithID("other")},
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "similarity search errors", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			searchErr:   errConformanceStub,
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "similarity search returns too many results", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			searchRes:   []vectorstores.SearchResult{alphaResult, alphaResult},
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "similarity search returns wrong top result", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			searchRes: []vectorstores.SearchResult{{
				Document: documents.New("gamma delta", nil),
				Score:    1,
			}},
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "delete errors", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			deleteErr:   errConformanceStub,
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "add texts errors", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{addTextsErr: errConformanceStub}))
	})
	expectConformanceFailure(t, "add texts returns wrong ids", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{addTextsIDs: []string{"wrong"}}))
	})
	expectConformanceFailure(t, "filtered search errors", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			filterErr:   errConformanceStub,
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "filtered search returns unexpected results", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			filterRes:   []vectorstores.SearchResult{alphaResult, alphaResult},
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "mmr search errors", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			mmrErr:      errConformanceStub,
			addTextsIDs: textIDs,
		}))
	})
	expectConformanceFailure(t, "mmr search returns too few documents", func(t *testing.T) {
		RunVectorStoreBasics(t, factory(stubVectorStore{
			mmrDocs:     []documents.Document{documents.New("alpha beta", nil)},
			addTextsIDs: textIDs,
		}))
	})
}
