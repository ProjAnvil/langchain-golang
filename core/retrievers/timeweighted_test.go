package retrievers_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// timeWeightedStoreStub implements vectorstores.VectorStore but deliberately
// does NOT implement SimilaritySearchWithRelevanceScores, so GetSalientDocs
// must reject it. addErr, when set, makes AddDocuments fail.
type timeWeightedStoreStub struct {
	addErr error
}

func (s timeWeightedStoreStub) AddDocuments(context.Context, []documents.Document) ([]string, error) {
	if s.addErr != nil {
		return nil, s.addErr
	}
	return nil, nil
}

func (s timeWeightedStoreStub) Delete(context.Context, []string) error { return nil }

func (s timeWeightedStoreStub) GetByIDs(context.Context, []string) ([]documents.Document, error) {
	return nil, nil
}

func (s timeWeightedStoreStub) SimilaritySearch(context.Context, string, int) ([]documents.Document, error) {
	return nil, nil
}

func (s timeWeightedStoreStub) SimilaritySearchWithScore(context.Context, string, int) ([]vectorstores.SearchResult, error) {
	return nil, nil
}

func newTimeWeightedStore() *vectorstores.InMemory {
	return vectorstores.NewInMemory(embeddings.NewDeterministicFake(8))
}

func TestTimeWeightedAddDocumentsStampsMetadataWithoutMutatingInput(t *testing.T) {
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	store := newTimeWeightedStore()
	retriever, err := retrievers.NewTimeWeightedVectorStoreRetriever(
		store,
		retrievers.WithTimeWeightedNow(func() time.Time { return now }),
	)
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	input := []documents.Document{
		documents.New("alpha", nil),
		documents.New("beta", map[string]any{"tag": "keep"}),
	}
	if err := retriever.AddDocuments(t.Context(), input); err != nil {
		t.Fatalf("add documents: %v", err)
	}

	// Inputs must not be mutated.
	for _, doc := range input {
		if _, ok := doc.Metadata["buffer_idx"]; ok {
			t.Fatalf("input document was mutated: %v", doc.Metadata)
		}
		if _, ok := doc.Metadata["last_accessed_at"]; ok {
			t.Fatalf("input document was mutated: %v", doc.Metadata)
		}
	}
	if got := input[1].Metadata["tag"]; got != "keep" {
		t.Fatalf("input metadata changed: %v", input[1].Metadata)
	}

	// The stamped copies are what the vector store received.
	stored, err := store.SimilaritySearch(t.Context(), "alpha beta", 10)
	if err != nil {
		t.Fatalf("search stored docs: %v", err)
	}
	if len(stored) != 2 {
		t.Fatalf("stored %d docs, want 2", len(stored))
	}
	indices := map[int]bool{}
	for _, doc := range stored {
		idx, ok := doc.Metadata["buffer_idx"].(int)
		if !ok {
			t.Fatalf("stored doc missing int buffer_idx: %v", doc.Metadata)
		}
		if idx < 0 || idx > 1 {
			t.Fatalf("unexpected buffer_idx %d", idx)
		}
		indices[idx] = true
		if _, ok := doc.Metadata["last_accessed_at"].(time.Time); !ok {
			t.Fatalf("stored doc missing last_accessed_at time: %v", doc.Metadata)
		}
		if _, ok := doc.Metadata["created_at"].(time.Time); !ok {
			t.Fatalf("stored doc missing created_at time: %v", doc.Metadata)
		}
	}
	if len(indices) != 2 {
		t.Fatalf("expected buffer indices 0 and 1, got %v", indices)
	}
}

func TestTimeWeightedAddDocumentsEmptyAndErrors(t *testing.T) {
	store := newTimeWeightedStore()
	retriever, err := retrievers.NewTimeWeightedVectorStoreRetriever(store)
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	if err := retriever.AddDocuments(t.Context(), nil); err != nil {
		t.Fatalf("add empty documents: %v", err)
	}

	wantErr := errors.New("store add failed")
	failing, err := retrievers.NewTimeWeightedVectorStoreRetriever(timeWeightedStoreStub{addErr: wantErr})
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	if err := failing.AddDocuments(t.Context(), []documents.Document{documents.New("x", nil)}); !errors.Is(err, wantErr) {
		t.Fatalf("AddDocuments error = %v, want %v", err, wantErr)
	}
}

func TestTimeWeightedGetSalientDocs(t *testing.T) {
	store := newTimeWeightedStore()
	retriever, err := retrievers.NewTimeWeightedVectorStoreRetriever(store)
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	if err := retriever.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha", nil),
		documents.New("beta", nil),
		documents.New("gamma", nil),
	}); err != nil {
		t.Fatalf("add documents: %v", err)
	}
	salient, err := retriever.GetSalientDocs(t.Context(), "query")
	if err != nil {
		t.Fatalf("get salient docs: %v", err)
	}
	if len(salient) != 3 {
		t.Fatalf("salient docs = %d, want 3", len(salient))
	}
	seen := map[string]bool{}
	for idx, sd := range salient {
		if idx < 0 || idx > 2 {
			t.Fatalf("unexpected salient index %d", idx)
		}
		seen[sd.Document.PageContent] = true
	}
	if !seen["alpha"] || !seen["beta"] || !seen["gamma"] {
		t.Fatalf("salient docs = %v, want alpha, beta, gamma", seen)
	}
}

func TestTimeWeightedGetSalientDocsEmptyAndErrors(t *testing.T) {
	store := newTimeWeightedStore()
	retriever, err := retrievers.NewTimeWeightedVectorStoreRetriever(store)
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	empty, err := retriever.GetSalientDocs(t.Context(), "query")
	if err != nil {
		t.Fatalf("get salient docs on empty stream: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("salient docs on empty stream = %d, want 0", len(empty))
	}

	unsupported, err := retrievers.NewTimeWeightedVectorStoreRetriever(timeWeightedStoreStub{})
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	if _, err := unsupported.GetSalientDocs(t.Context(), "query"); err == nil {
		t.Fatal("GetSalientDocs must error when the store lacks relevance-score search")
	}
}

func TestTimeWeightedGetRelevantDocumentsTopKAndRefresh(t *testing.T) {
	now := time.Date(2026, 2, 1, 9, 0, 0, 0, time.UTC)
	clock := func() time.Time { return now }
	store := newTimeWeightedStore()
	retriever, err := retrievers.NewTimeWeightedVectorStoreRetriever(
		store,
		retrievers.WithTimeWeightedNow(clock),
		retrievers.WithTimeWeightedK(2),
	)
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	if err := retriever.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha", nil),
		documents.New("beta", nil),
		documents.New("gamma", nil),
	}); err != nil {
		t.Fatalf("add documents: %v", err)
	}

	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs = %d, want top 2", len(docs))
	}
	for _, doc := range docs {
		if got, ok := doc.Metadata["last_accessed_at"].(time.Time); !ok || !got.Equal(now) {
			t.Fatalf("returned doc last_accessed_at = %v, want %v", doc.Metadata["last_accessed_at"], now)
		}
	}

	// Advancing the clock and retrieving again refreshes the timestamp.
	now = now.Add(24 * time.Hour)
	docs, err = retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve after clock advance: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs = %d, want top 2", len(docs))
	}
	for _, doc := range docs {
		if got, ok := doc.Metadata["last_accessed_at"].(time.Time); !ok || !got.Equal(now) {
			t.Fatalf("returned doc last_accessed_at = %v, want refreshed %v", doc.Metadata["last_accessed_at"], now)
		}
	}
}

func TestTimeWeightedGetRelevantDocumentsEmptyAndCanceled(t *testing.T) {
	store := newTimeWeightedStore()
	retriever, err := retrievers.NewTimeWeightedVectorStoreRetriever(store)
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve empty stream: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs on empty stream = %d, want 0", len(docs))
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := retriever.GetRelevantDocuments(ctx, "query"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled retrieval error = %v, want context.Canceled", err)
	}
}

func TestTimeWeightedScoreCombinesRecencyAndSimilarity(t *testing.T) {
	now := time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC)
	// Both documents are equally relevant (cosine 1); the older one has a
	// 10-hour penalty, so the recent one must win with k=1.
	store := vectorstores.NewInMemory(embeddings.Static{
		DocumentVectors: [][]float64{{1, 0}, {1, 0}},
		QueryVector:     []float64{1, 0},
	})
	retriever, err := retrievers.NewTimeWeightedVectorStoreRetriever(
		store,
		retrievers.WithTimeWeightedNow(func() time.Time { return now }),
		retrievers.WithTimeWeightedDecayRate(0.5),
		retrievers.WithTimeWeightedK(1),
	)
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	if err := retriever.AddDocuments(t.Context(), []documents.Document{
		documents.New("old", map[string]any{"last_accessed_at": now.Add(-10 * time.Hour)}),
		documents.New("recent", map[string]any{"last_accessed_at": now}),
	}); err != nil {
		t.Fatalf("add documents: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "recent" {
		t.Fatalf("docs = %v, want the recent document", docContents(docs))
	}
}

func TestTimeWeightedScoreAddsOtherScoreKeysAsSalience(t *testing.T) {
	now := time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC)
	// All documents have zero relevance and identical recency, so the
	// "importance" metadata key alone determines the ranking.
	store := vectorstores.NewInMemory(embeddings.Static{
		DocumentVectors: [][]float64{{0, 1}, {0, 1}, {0, 1}},
		QueryVector:     []float64{1, 0},
	})
	retriever, err := retrievers.NewTimeWeightedVectorStoreRetriever(
		store,
		retrievers.WithTimeWeightedNow(func() time.Time { return now }),
		retrievers.WithTimeWeightedK(2),
		retrievers.WithTimeWeightedOtherScoreKeys([]string{"importance"}),
	)
	if err != nil {
		t.Fatalf("new time-weighted retriever: %v", err)
	}
	if err := retriever.AddDocuments(t.Context(), []documents.Document{
		documents.New("zero", map[string]any{"importance": 0}),
		documents.New("five", map[string]any{"importance": 5}),
		documents.New("one", map[string]any{"importance": 1}),
	}); err != nil {
		t.Fatalf("add documents: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	got := docContents(docs)
	if len(got) != 2 || got[0] != "five" || got[1] != "one" {
		t.Fatalf("docs = %v, want [five one] ordered by salience", got)
	}
}

func TestTimeWeightedConstructorRequiresStore(t *testing.T) {
	if _, err := retrievers.NewTimeWeightedVectorStoreRetriever(nil); err == nil {
		t.Fatal("nil vector store must error")
	}
}
