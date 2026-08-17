package retrievers

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// fakeStore is a minimal VectorStore that records SimilaritySearch calls.
type fakeStore struct {
	simK      int
	simDocs   []documents.Document
	simErr    error
	simCalled bool
}

func (f *fakeStore) AddDocuments(context.Context, []documents.Document) ([]string, error) {
	return nil, nil
}

func (f *fakeStore) Delete(context.Context, []string) error { return nil }

func (f *fakeStore) GetByIDs(context.Context, []string) ([]documents.Document, error) {
	return nil, nil
}

func (f *fakeStore) SimilaritySearch(
	_ context.Context,
	_ string,
	k int,
) ([]documents.Document, error) {
	f.simCalled = true
	f.simK = k
	return f.simDocs, f.simErr
}

func (f *fakeStore) SimilaritySearchWithScore(
	context.Context,
	string,
	int,
) ([]vectorstores.SearchResult, error) {
	return nil, nil
}

// fakeMMRStore additionally implements mmrSearcher and records its arguments.
type fakeMMRStore struct {
	fakeStore
	mmrK          int
	mmrFetchK     int
	mmrLambdaMult float64
	mmrDocs       []documents.Document
	mmrErr        error
}

func (f *fakeMMRStore) MaxMarginalRelevanceSearch(
	_ context.Context,
	_ string,
	k int,
	fetchK int,
	lambdaMult float64,
	_ vectorstores.Filter,
) ([]documents.Document, error) {
	f.mmrK = k
	f.mmrFetchK = fetchK
	f.mmrLambdaMult = lambdaMult
	return f.mmrDocs, f.mmrErr
}

// fakeScoreStore additionally implements relevanceScoreSearcher.
type fakeScoreStore struct {
	fakeStore
	gotK         int
	gotThreshold *float64
	results      []vectorstores.SearchResult
	err          error
}

func (f *fakeScoreStore) SimilaritySearchWithRelevanceScores(
	_ context.Context,
	_ string,
	k int,
	scoreThreshold *float64,
) ([]vectorstores.SearchResult, error) {
	f.gotK = k
	f.gotThreshold = scoreThreshold
	return f.results, f.err
}

func TestStaticGetRelevantDocuments(t *testing.T) {
	r := Static{Documents: []documents.Document{
		documents.New("one", map[string]any{"n": 1}),
		documents.New("two", nil),
	}}

	docs, err := r.GetRelevantDocuments(context.Background(), "ignored")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2", len(docs))
	}

	// Returned documents must be defensive copies.
	docs[0].PageContent = "mutated"
	again, err := r.GetRelevantDocuments(context.Background(), "ignored")
	if err != nil {
		t.Fatalf("retrieve again: %v", err)
	}
	if again[0].PageContent != "one" {
		t.Fatalf("expected defensive copy, got %q", again[0].PageContent)
	}

	empty := Static{}
	docs, err = empty.GetRelevantDocuments(context.Background(), "q")
	if err != nil {
		t.Fatalf("empty retrieve: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("empty docs: got %d want 0", len(docs))
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.GetRelevantDocuments(ctx, "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled ctx: got %v want context.Canceled", err)
	}
}

func TestNewVectorStoreRetrieverDefaultK(t *testing.T) {
	store := &fakeStore{}
	r := NewVectorStoreRetriever(store, 0)
	if _, err := r.GetRelevantDocuments(context.Background(), "q"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if store.simK != 4 {
		t.Fatalf("default k: got %d want 4", store.simK)
	}

	r = NewVectorStoreRetriever(store, -3)
	if _, err := r.GetRelevantDocuments(context.Background(), "q"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if store.simK != 4 {
		t.Fatalf("negative k: got %d want 4", store.simK)
	}
}

func TestVectorStoreRetrieverGetRelevantDocumentsErrors(t *testing.T) {
	store := &fakeStore{}
	r := NewVectorStoreRetriever(store, 2)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.GetRelevantDocuments(ctx, "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled ctx: got %v want context.Canceled", err)
	}

	nilStore := NewVectorStoreRetriever(nil, 2)
	if _, err := nilStore.GetRelevantDocuments(context.Background(), "q"); err == nil ||
		!strings.Contains(err.Error(), "vector store is required") {
		t.Fatalf("nil store: got %v", err)
	}

	bogus := VectorStoreRetriever{store: store, searchType: "bogus"}
	_, err := bogus.GetRelevantDocuments(context.Background(), "q")
	if err == nil || !strings.Contains(err.Error(), `search_type of bogus not allowed`) {
		t.Fatalf("bogus search type: got %v", err)
	}

	wantErr := errors.New("boom")
	failStore := &fakeStore{simErr: wantErr}
	r = NewVectorStoreRetriever(failStore, 1)
	if _, err := r.GetRelevantDocuments(context.Background(), "q"); !errors.Is(err, wantErr) {
		t.Fatalf("store error: got %v want %v", err, wantErr)
	}
}

func TestNewVectorStoreRetrieverWithOptions(t *testing.T) {
	store := &fakeStore{}

	t.Run("nil option is skipped", func(t *testing.T) {
		r, err := NewVectorStoreRetrieverWithOptions(store, 2, nil)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(context.Background(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.simK != 2 {
			t.Fatalf("k: got %d want 2", store.simK)
		}
	})

	t.Run("empty search type keeps default", func(t *testing.T) {
		r, err := NewVectorStoreRetrieverWithOptions(store, 3, WithSearchType(""))
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(context.Background(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.simK != 3 {
			t.Fatalf("k: got %d want 3", store.simK)
		}
	})

	t.Run("search kwargs override k", func(t *testing.T) {
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchKwargs(map[string]any{"k": 5}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(context.Background(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.simK != 5 {
			t.Fatalf("k: got %d want 5", store.simK)
		}
	})

	t.Run("invalid search type", func(t *testing.T) {
		_, err := NewVectorStoreRetrieverWithOptions(store, 2, WithSearchType("nope"))
		if err == nil || !strings.Contains(err.Error(), `search_type of nope not allowed`) {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("threshold search without score_threshold", func(t *testing.T) {
		_, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("similarity_score_threshold"),
		)
		if err == nil || !strings.Contains(err.Error(), "score_threshold must be provided") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("threshold search with non-numeric score_threshold", func(t *testing.T) {
		_, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("similarity_score_threshold"),
			WithSearchKwargs(map[string]any{"score_threshold": "high"}),
		)
		if err == nil || !strings.Contains(err.Error(), "score_threshold must be provided") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("valid mmr", func(t *testing.T) {
		if _, err := NewVectorStoreRetrieverWithOptions(store, 2, WithSearchType("mmr")); err != nil {
			t.Fatalf("options: %v", err)
		}
	})

	t.Run("valid threshold", func(t *testing.T) {
		_, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("similarity_score_threshold"),
			WithSearchKwargs(map[string]any{"score_threshold": 0.5}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
	})
}

func TestAsRetriever(t *testing.T) {
	store := &fakeStore{}
	r, err := AsRetriever(store)
	if err != nil {
		t.Fatalf("as retriever: %v", err)
	}
	if _, err := r.GetRelevantDocuments(context.Background(), "q"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if store.simK != 4 {
		t.Fatalf("default k: got %d want 4", store.simK)
	}

	if _, err := AsRetriever(store, WithSearchType("nope")); err == nil {
		t.Fatal("expected error for invalid search type")
	}
}

func TestVectorStoreRetrieverMMR(t *testing.T) {
	t.Run("forwards parsed kwargs", func(t *testing.T) {
		store := &fakeMMRStore{mmrDocs: []documents.Document{documents.New("doc", nil)}}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("mmr"),
			WithSearchKwargs(map[string]any{
				"k":           int64(3),
				"fetch_k":     float64(10),
				"lambda_mult": float32(0.7),
			}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		docs, err := r.GetRelevantDocuments(context.Background(), "q")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 1 || docs[0].PageContent != "doc" {
			t.Fatalf("docs: got %v", docs)
		}
		if store.mmrK != 3 {
			t.Fatalf("mmr k: got %d want 3", store.mmrK)
		}
		if store.mmrFetchK != 10 {
			t.Fatalf("mmr fetch_k: got %d want 10", store.mmrFetchK)
		}
		if math.Abs(store.mmrLambdaMult-0.7) > 1e-6 {
			t.Fatalf("mmr lambda_mult: got %v want 0.7", store.mmrLambdaMult)
		}
	})

	t.Run("defaults without kwargs", func(t *testing.T) {
		store := &fakeMMRStore{}
		r, err := NewVectorStoreRetrieverWithOptions(store, 2, WithSearchType("mmr"))
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(context.Background(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.mmrK != 2 || store.mmrFetchK != 20 || store.mmrLambdaMult != 0.5 {
			t.Fatalf("defaults: got k=%d fetch_k=%d lambda=%v",
				store.mmrK, store.mmrFetchK, store.mmrLambdaMult)
		}
	})

	t.Run("unsupported store", func(t *testing.T) {
		store := &fakeStore{}
		r, err := NewVectorStoreRetrieverWithOptions(store, 2, WithSearchType("mmr"))
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		_, err = r.GetRelevantDocuments(context.Background(), "q")
		if err == nil || !strings.Contains(err.Error(), "does not support mmr search") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("store error propagates", func(t *testing.T) {
		wantErr := errors.New("mmr failed")
		store := &fakeMMRStore{mmrErr: wantErr}
		r, err := NewVectorStoreRetrieverWithOptions(store, 2, WithSearchType("mmr"))
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(context.Background(), "q"); !errors.Is(err, wantErr) {
			t.Fatalf("got %v want %v", err, wantErr)
		}
	})

	t.Run("in-memory store", func(t *testing.T) {
		store := vectorstores.NewInMemory(embeddings.NewFake(32))
		_, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("alpha beta", nil),
			documents.New("alpha gamma", nil),
			documents.New("delta epsilon", nil),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		r, err := NewVectorStoreRetrieverWithOptions(store, 2, WithSearchType("mmr"))
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		docs, err := r.GetRelevantDocuments(context.Background(), "alpha")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 2 {
			t.Fatalf("docs: got %d want 2", len(docs))
		}
	})
}

func TestVectorStoreRetrieverSimilarityScoreThreshold(t *testing.T) {
	t.Run("forwards parsed kwargs and unwraps results", func(t *testing.T) {
		store := &fakeScoreStore{results: []vectorstores.SearchResult{
			{Document: documents.New("hit", nil), Score: 0.9},
		}}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("similarity_score_threshold"),
			WithSearchKwargs(map[string]any{"score_threshold": 0.5, "k": 3}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		docs, err := r.GetRelevantDocuments(context.Background(), "q")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 1 || docs[0].PageContent != "hit" {
			t.Fatalf("docs: got %v", docs)
		}
		if store.gotK != 3 {
			t.Fatalf("k: got %d want 3", store.gotK)
		}
		if store.gotThreshold == nil || *store.gotThreshold != 0.5 {
			t.Fatalf("threshold: got %v want 0.5", store.gotThreshold)
		}
	})

	t.Run("nil threshold when kwarg absent", func(t *testing.T) {
		// Only reachable by constructing the retriever directly, since
		// NewVectorStoreRetrieverWithOptions requires score_threshold.
		store := &fakeScoreStore{}
		r := VectorStoreRetriever{
			store:      store,
			k:          2,
			searchType: searchTypeSimilarityThreshold,
		}
		if _, err := r.GetRelevantDocuments(context.Background(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if store.gotThreshold != nil {
			t.Fatalf("threshold: got %v want nil", *store.gotThreshold)
		}
	})

	t.Run("unsupported store", func(t *testing.T) {
		store := &fakeStore{}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("similarity_score_threshold"),
			WithSearchKwargs(map[string]any{"score_threshold": 0.5}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		_, err = r.GetRelevantDocuments(context.Background(), "q")
		if err == nil ||
			!strings.Contains(err.Error(), "does not support similarity_score_threshold search") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("store error propagates", func(t *testing.T) {
		wantErr := errors.New("search failed")
		store := &fakeScoreStore{err: wantErr}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("similarity_score_threshold"),
			WithSearchKwargs(map[string]any{"score_threshold": 0.5}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		if _, err := r.GetRelevantDocuments(context.Background(), "q"); !errors.Is(err, wantErr) {
			t.Fatalf("got %v want %v", err, wantErr)
		}
	})

	t.Run("in-memory store filters by threshold", func(t *testing.T) {
		store := vectorstores.NewInMemory(embeddings.NewFake(32))
		_, err := store.AddDocuments(context.Background(), []documents.Document{
			documents.New("alpha beta", nil),
			documents.New("gamma delta", nil),
		})
		if err != nil {
			t.Fatalf("add documents: %v", err)
		}
		r, err := NewVectorStoreRetrieverWithOptions(
			store, 2,
			WithSearchType("similarity_score_threshold"),
			WithSearchKwargs(map[string]any{"score_threshold": 0.0}),
		)
		if err != nil {
			t.Fatalf("options: %v", err)
		}
		docs, err := r.GetRelevantDocuments(context.Background(), "alpha")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) == 0 {
			t.Fatal("expected at least one document above threshold 0")
		}
	})
}

func TestSearchKwargInt(t *testing.T) {
	tests := []struct {
		name     string
		kwargs   map[string]any
		fallback int
		want     int
	}{
		{"missing key", map[string]any{}, 7, 7},
		{"nil map", nil, 7, 7},
		{"int", map[string]any{"k": 3}, 7, 3},
		{"int64", map[string]any{"k": int64(4)}, 7, 4},
		{"float64", map[string]any{"k": 5.9}, 7, 5},
		{"float32", map[string]any{"k": float32(6.5)}, 7, 6},
		{"wrong type", map[string]any{"k": "3"}, 7, 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := searchKwargInt(tt.kwargs, "k", tt.fallback); got != tt.want {
				t.Fatalf("got %d want %d", got, tt.want)
			}
		})
	}
}

func TestSearchKwargFloat(t *testing.T) {
	tests := []struct {
		name     string
		kwargs   map[string]any
		fallback float64
		want     float64
	}{
		{"missing key", map[string]any{}, 0.1, 0.1},
		{"nil map", nil, 0.1, 0.1},
		{"float64", map[string]any{"v": 0.25}, 0.1, 0.25},
		{"float32", map[string]any{"v": float32(0.5)}, 0.1, 0.5},
		{"int", map[string]any{"v": 1}, 0.1, 1.0},
		{"int64", map[string]any{"v": int64(2)}, 0.1, 2.0},
		{"wrong type", map[string]any{"v": "0.5"}, 0.1, 0.1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := searchKwargFloat(tt.kwargs, "v", tt.fallback); got != tt.want {
				t.Fatalf("got %v want %v", got, tt.want)
			}
		})
	}
}

func TestSearchKwargOptionalFloat(t *testing.T) {
	tests := []struct {
		name   string
		kwargs map[string]any
		want   float64
		wantOK bool
	}{
		{"missing key", map[string]any{}, 0, false},
		{"nil map", nil, 0, false},
		{"float64", map[string]any{"v": 0.25}, 0.25, true},
		{"float32", map[string]any{"v": float32(0.5)}, 0.5, true},
		{"int", map[string]any{"v": 1}, 1.0, true},
		{"int64", map[string]any{"v": int64(2)}, 2.0, true},
		{"wrong type", map[string]any{"v": true}, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := searchKwargOptionalFloat(tt.kwargs, "v")
			if got != tt.want || ok != tt.wantOK {
				t.Fatalf("got (%v, %v) want (%v, %v)", got, ok, tt.want, tt.wantOK)
			}
		})
	}
}
