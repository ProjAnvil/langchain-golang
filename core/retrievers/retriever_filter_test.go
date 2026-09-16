package retrievers

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// fakeOptionStore implements vectorstores.OptionSearcher (but neither the
// legacy MMR interface nor relevance-score search) and records its options.
type fakeOptionStore struct {
	fakeStore
	gotSimOpts vectorstores.SearchOptions
	gotMMROpts vectorstores.SearchOptions
	simDocs    []documents.Document
	mmrDocs    []documents.Document
	simCalled  bool
	mmrCalled  bool
}

func (f *fakeOptionStore) SimilaritySearchWithOptions(
	_ context.Context,
	_ string,
	opts vectorstores.SearchOptions,
) ([]documents.Document, error) {
	f.simCalled = true
	f.gotSimOpts = opts
	return f.simDocs, nil
}

func (f *fakeOptionStore) MMRSearchWithOptions(
	_ context.Context,
	_ string,
	opts vectorstores.SearchOptions,
) ([]documents.Document, error) {
	f.mmrCalled = true
	f.gotMMROpts = opts
	return f.mmrDocs, nil
}

// fakeFilterMMRStore additionally records the legacy MMR filter callback.
type fakeFilterMMRStore struct {
	fakeMMRStore
	mmrFilter vectorstores.Filter
}

func (f *fakeFilterMMRStore) MaxMarginalRelevanceSearch(
	ctx context.Context,
	query string,
	k int,
	fetchK int,
	lambdaMult float64,
	filter vectorstores.Filter,
) ([]documents.Document, error) {
	f.mmrFilter = filter
	return f.fakeMMRStore.MaxMarginalRelevanceSearch(ctx, query, k, fetchK, lambdaMult, filter)
}

func TestVectorStoreRetrieverSimilarityWithFilter(t *testing.T) {
	filter := map[string]any{"group": map[string]any{vectorstores.FilterEq: "a"}}

	t.Run("forwards filter via OptionSearcher", func(t *testing.T) {
		store := &fakeOptionStore{simDocs: []documents.Document{documents.New("doc", nil)}}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 3,
			WithSearchKwargs(map[string]any{"filter": filter}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		docs, err := r.GetRelevantDocuments(t.Context(), "q")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 1 {
			t.Fatalf("docs: %#v", docs)
		}
		if !store.simCalled {
			t.Fatal("expected SimilaritySearchWithOptions to be called")
		}
		if store.gotSimOpts.K != 3 {
			t.Fatalf("k: got %d want 3", store.gotSimOpts.K)
		}
		if store.gotSimOpts.Filter == nil {
			t.Fatal("expected filter to be forwarded")
		}
	})

	t.Run("kwarg k overrides retriever k", func(t *testing.T) {
		store := &fakeOptionStore{}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 3,
			WithSearchKwargs(map[string]any{"k": 7, "filter": filter}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.gotSimOpts.K != 7 {
			t.Fatalf("k: got %d want 7", store.gotSimOpts.K)
		}
	})

	t.Run("store without OptionSearcher errors when filter set", func(t *testing.T) {
		store := &fakeStore{}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 3,
			WithSearchKwargs(map[string]any{"filter": filter}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		_, err = r.GetRelevantDocuments(t.Context(), "q")
		if err == nil || !strings.Contains(err.Error(), "does not support declarative filters") {
			t.Fatalf("got %v", err)
		}
		if store.simCalled {
			t.Fatal("plain SimilaritySearch must not run when a filter is set")
		}
	})

	t.Run("no filter keeps plain similarity path", func(t *testing.T) {
		store := &fakeOptionStore{}
		r, err := NewVectorStoreRetrieverWithOptions(store, 3)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.simCalled {
			t.Fatal("expected plain SimilaritySearch, not the options path")
		}
	})

	t.Run("empty filter map keeps plain similarity path", func(t *testing.T) {
		store := &fakeOptionStore{}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 3,
			WithSearchKwargs(map[string]any{"filter": map[string]any{}}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.simCalled {
			t.Fatal("expected plain SimilaritySearch for empty filter")
		}
	})

	t.Run("non-map filter kwarg errors", func(t *testing.T) {
		store := &fakeOptionStore{}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 3,
			WithSearchKwargs(map[string]any{"filter": "group=a"}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		_, err = r.GetRelevantDocuments(t.Context(), "q")
		if err == nil || !strings.Contains(err.Error(), `"filter" must be a map[string]any`) {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("invalid filter operator errors", func(t *testing.T) {
		store := &fakeOptionStore{}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 3,
			WithSearchKwargs(map[string]any{"filter": map[string]any{"group": map[string]any{"$bogus": "a"}}}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		_, err = r.GetRelevantDocuments(t.Context(), "q")
		if err == nil || !strings.Contains(err.Error(), "invalid operator") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("in-memory store end to end", func(t *testing.T) {
		store := vectorstores.NewInMemory(embeddings.NewFake(16))
		_, err := store.AddDocuments(t.Context(), []documents.Document{
			documents.New("alpha one", map[string]any{"group": "a"}),
			documents.New("alpha two", map[string]any{"group": "a"}),
			documents.New("alpha three", map[string]any{"group": "b"}),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 3,
			WithSearchKwargs(map[string]any{"filter": map[string]any{"group": "a"}}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		docs, err := r.GetRelevantDocuments(t.Context(), "alpha")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
		for _, doc := range docs {
			if doc.Metadata["group"] != "a" {
				t.Fatalf("unexpected doc: %#v", doc)
			}
		}
	})
}

func TestVectorStoreRetrieverMMRWithFilter(t *testing.T) {
	filter := map[string]any{"group": map[string]any{vectorstores.FilterEq: "a"}}

	t.Run("legacy mmr store receives evaluated filter closure", func(t *testing.T) {
		store := &fakeFilterMMRStore{fakeMMRStore: fakeMMRStore{
			mmrDocs: []documents.Document{documents.New("doc", nil)},
		}}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("mmr"),
			WithSearchKwargs(map[string]any{"filter": filter, "lambda_mult": 0.7}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.mmrFilter == nil {
			t.Fatal("expected filter closure to be forwarded")
		}
		matching := documents.New("doc", map[string]any{"group": "a"})
		nonMatching := documents.New("doc", map[string]any{"group": "b"})
		if !store.mmrFilter(matching) {
			t.Fatal("closure should accept matching doc")
		}
		if store.mmrFilter(nonMatching) {
			t.Fatal("closure should reject non-matching doc")
		}
		if store.mmrLambdaMult != 0.7 {
			t.Fatalf("lambda_mult: got %v want 0.7", store.mmrLambdaMult)
		}
	})

	t.Run("option-only store gets MMRSearchWithOptions", func(t *testing.T) {
		store := &fakeOptionStore{mmrDocs: []documents.Document{documents.New("doc", nil)}}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("mmr"),
			WithSearchKwargs(map[string]any{"filter": filter, "fetch_k": 9}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		docs, err := r.GetRelevantDocuments(t.Context(), "q")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 1 {
			t.Fatalf("docs: %#v", docs)
		}
		if !store.mmrCalled {
			t.Fatal("expected MMRSearchWithOptions to be called")
		}
		if store.gotMMROpts.K != 2 || store.gotMMROpts.FetchK != 9 {
			t.Fatalf("opts: got %+v want K=2 FetchK=9", store.gotMMROpts)
		}
		if store.gotMMROpts.Filter == nil {
			t.Fatal("expected filter to be forwarded")
		}
	})

	t.Run("store supporting neither errors", func(t *testing.T) {
		store := &fakeStore{}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("mmr"),
			WithSearchKwargs(map[string]any{"filter": filter}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		_, err = r.GetRelevantDocuments(t.Context(), "q")
		if err == nil || !strings.Contains(err.Error(), "does not support mmr search") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("invalid filter errors before dispatch", func(t *testing.T) {
		store := &fakeOptionStore{}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("mmr"),
			WithSearchKwargs(map[string]any{"filter": map[string]any{"group": []any{"a"}}}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		_, err = r.GetRelevantDocuments(t.Context(), "q")
		if err == nil || !strings.Contains(err.Error(), "invalid filter condition") {
			t.Fatalf("got %v", err)
		}
		if store.mmrCalled {
			t.Fatal("search must not run with an invalid filter")
		}
	})

	t.Run("in-memory mmr end to end", func(t *testing.T) {
		store := vectorstores.NewInMemory(embeddings.NewFake(16))
		_, err := store.AddDocuments(t.Context(), []documents.Document{
			documents.New("alpha one", map[string]any{"group": "a"}),
			documents.New("alpha two", map[string]any{"group": "a"}),
			documents.New("alpha three", map[string]any{"group": "b"}),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("mmr"),
			WithSearchKwargs(map[string]any{"filter": map[string]any{"group": "a"}}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		docs, err := r.GetRelevantDocuments(t.Context(), "alpha")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
		for _, doc := range docs {
			if doc.Metadata["group"] != "a" {
				t.Fatalf("unexpected doc: %#v", doc)
			}
		}
	})
}
