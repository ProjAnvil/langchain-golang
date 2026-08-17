package chroma

import (
	"context"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

var errStubEmbed = errors.New("stub embed failure")

// stubEmbedder lets tests control embedding outputs and failures.
type stubEmbedder struct {
	docVectors  [][]float64
	queryVector []float64
	docErr      error
	queryErr    error
}

func (e stubEmbedder) EmbedDocuments(_ context.Context, _ []string) ([][]float64, error) {
	if e.docErr != nil {
		return nil, e.docErr
	}
	return e.docVectors, nil
}

func (e stubEmbedder) EmbedQuery(_ context.Context, _ string) ([]float64, error) {
	if e.queryErr != nil {
		return nil, e.queryErr
	}
	return e.queryVector, nil
}

func newTestStore(t *testing.T, baseURL string, embedder embeddings.Embeddings, opts ...Option) *Store {
	t.Helper()
	opts = append([]Option{WithBaseURL(baseURL), WithMaxRetries(0)}, opts...)
	store, err := New(context.Background(), "langchain", embedder, opts...)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	return store
}

// newFailAfterCreateServer serves collection creation and fails every other
// request with a non-retryable status.
func newFailAfterCreateServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/collections") {
			writeJSON(w, collectionResponse{ID: "collection-1", Name: "langchain"})
			return
		}
		http.Error(w, "boom", http.StatusBadRequest)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestNewValidationErrors(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	if _, err := New(context.Background(), "  ", embeddings.NewFake(4), WithBaseURL(server.URL)); err == nil {
		t.Fatal("expected error for blank collection name")
	}
	if _, err := New(context.Background(), "langchain", nil, WithBaseURL(server.URL)); err == nil {
		t.Fatal("expected error for nil embedder")
	}
}

func TestNewOptionFallbacks(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store, err := New(
		context.Background(),
		"langchain",
		embeddings.NewFake(4),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithHeader("X-One", "1"),
		WithHeader("X-Two", "2"),
		WithTenant("  "),
		WithDatabase(""),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	if store.tenant != defaultTenant || store.database != defaultDatabase {
		t.Fatalf("tenant/database fallbacks: got %q/%q", store.tenant, store.database)
	}
	if got, want := server.lastPath(), "/api/v2/tenants/default_tenant/databases/default_database/collections"; got != want {
		t.Fatalf("create path: got %q want %q", got, want)
	}
	if got := server.lastHeader("X-Two"); got != "2" {
		t.Fatalf("header: got %q", got)
	}
}

func TestNewCollectionIDFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, collectionResponse{ID: "", Name: "langchain"})
	}))
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(4))
	if store.collectionID != "langchain" {
		t.Fatalf("collection ID fallback: got %q", store.collectionID)
	}
}

func TestUpsertDocumentsEdgeCases(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(4))
	ids, err := store.UpsertDocuments(context.Background(), nil)
	if err != nil || ids != nil {
		t.Fatalf("empty upsert: ids=%v err=%v", ids, err)
	}

	failing := newTestStore(t, server.URL, stubEmbedder{docErr: errStubEmbed})
	if _, err := failing.UpsertDocuments(context.Background(), []documents.Document{
		documents.New("alpha", nil),
	}); !errors.Is(err, errStubEmbed) {
		t.Fatalf("embed error: got %v", err)
	}

	mismatch := newTestStore(t, server.URL, stubEmbedder{docVectors: [][]float64{{1, 0}}})
	_, err = mismatch.UpsertDocuments(context.Background(), []documents.Document{
		documents.New("alpha", nil),
		documents.New("beta", nil),
	})
	if err == nil || !strings.Contains(err.Error(), "embedding count mismatch") {
		t.Fatalf("count mismatch: got %v", err)
	}
}

func TestUpdateDocumentsEdgeCases(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(4))

	err := store.UpdateDocuments(context.Background(), []string{"a"}, nil)
	if err == nil || !strings.Contains(err.Error(), "length mismatch") {
		t.Fatalf("length mismatch: got %v", err)
	}
	if err := store.UpdateDocuments(context.Background(), nil, nil); err != nil {
		t.Fatalf("empty update: %v", err)
	}

	failing := newTestStore(t, server.URL, stubEmbedder{docErr: errStubEmbed})
	if err := failing.UpdateDocument(context.Background(), "a", documents.New("alpha", nil)); !errors.Is(err, errStubEmbed) {
		t.Fatalf("embed error: got %v", err)
	}

	mismatch := newTestStore(t, server.URL, stubEmbedder{docVectors: [][]float64{{1, 0}}})
	err = mismatch.UpdateDocuments(context.Background(), []string{"a", "b"}, []documents.Document{
		documents.New("alpha", nil),
		documents.New("beta", nil),
	})
	if err == nil || !strings.Contains(err.Error(), "embedding count mismatch") {
		t.Fatalf("count mismatch: got %v", err)
	}
}

func TestDeleteEdgeCases(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(4))
	if err := store.Delete(context.Background(), nil); err != nil {
		t.Fatalf("empty delete: %v", err)
	}

	failServer := newFailAfterCreateServer(t)
	failStore := newTestStore(t, failServer.URL, embeddings.NewFake(4))
	if err := failStore.Delete(context.Background(), []string{"a"}); err == nil {
		t.Fatal("expected delete error")
	}
}

func TestGetEdgeCases(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(4))
	docs, err := store.GetByIDs(context.Background(), nil)
	if err != nil || docs != nil {
		t.Fatalf("empty get: docs=%v err=%v", docs, err)
	}

	failServer := newFailAfterCreateServer(t)
	failStore := newTestStore(t, failServer.URL, embeddings.NewFake(4))
	if _, err := failStore.GetByIDs(context.Background(), []string{"a"}); err == nil {
		t.Fatal("expected get error")
	}

	// An unmarshalable filter value must surface the JSON marshal error.
	if _, err := store.Get(context.Background(), GetOptions{
		Where: map[string]any{"bad": make(chan int)},
	}); err == nil {
		t.Fatal("expected marshal error")
	}
}

func TestSimilaritySearchEmbedderErrors(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store := newTestStore(t, server.URL, stubEmbedder{queryErr: errStubEmbed})
	ctx := context.Background()

	if _, err := store.SimilaritySearch(ctx, "alpha", 2); !errors.Is(err, errStubEmbed) {
		t.Fatalf("similarity search: got %v", err)
	}
	if _, err := store.SimilaritySearchOptions(ctx, "alpha", QueryOptions{K: 2}); !errors.Is(err, errStubEmbed) {
		t.Fatalf("similarity search options: got %v", err)
	}
	if _, err := store.SimilaritySearchWithVectors(ctx, "alpha", QueryOptions{K: 2}); !errors.Is(err, errStubEmbed) {
		t.Fatalf("similarity search with vectors: got %v", err)
	}
	if _, err := store.MaxMarginalRelevanceSearch(ctx, "alpha", MMROptions{K: 2}); !errors.Is(err, errStubEmbed) {
		t.Fatalf("mmr: got %v", err)
	}
}

func TestSimilaritySearchServerErrors(t *testing.T) {
	failServer := newFailAfterCreateServer(t)
	store := newTestStore(t, failServer.URL, embeddings.NewFake(4))
	ctx := context.Background()
	vector := []float64{1, 0, 0, 0}

	if _, err := store.SimilaritySearchByVector(ctx, vector, QueryOptions{K: 2}); err == nil {
		t.Fatal("expected by-vector error")
	}
	if _, err := store.SimilaritySearchByVectorWithScore(ctx, vector, QueryOptions{K: 2}); err == nil {
		t.Fatal("expected by-vector-with-score error")
	}
	if _, err := store.SimilaritySearchByVectorWithVectors(ctx, vector, QueryOptions{K: 2}); err == nil {
		t.Fatal("expected by-vector-with-vectors error")
	}
	if _, err := store.MaxMarginalRelevanceSearchByVector(ctx, vector, MMROptions{K: 2}); err == nil {
		t.Fatal("expected mmr-by-vector error")
	}
}

func TestSimilaritySearchByVectorDefaults(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(8))
	ctx := context.Background()
	_, err := store.AddDocuments(ctx, []documents.Document{
		documents.New("alpha red", map[string]any{"color": "red"}).WithID("red"),
		documents.New("alpha blue", nil).WithID("blue"),
	})
	if err != nil {
		t.Fatalf("add: %v", err)
	}

	// K <= 0 falls back to the default of 4 results.
	docs, err := store.SimilaritySearchByVector(ctx, []float64{1, 0, 0, 0, 0, 0, 0, 0}, QueryOptions{})
	if err != nil {
		t.Fatalf("by-vector: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("by-vector docs: got %d", len(docs))
	}
	var payload queryRequest
	server.decodeLastBody(t, &payload)
	if payload.NResults != 4 {
		t.Fatalf("default n_results: got %d want 4", payload.NResults)
	}

	withVectors, err := store.SimilaritySearchWithVectors(ctx, "alpha", QueryOptions{K: 1})
	if err != nil {
		t.Fatalf("with vectors: %v", err)
	}
	if len(withVectors) != 1 || len(withVectors[0].Vector) != 8 {
		t.Fatalf("with vectors: %#v", withVectors)
	}
}

func TestSimilaritySearchEmptyResults(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(4))
	ctx := context.Background()

	// Querying an empty collection yields no scored documents.
	results, err := store.SimilaritySearchByVectorWithScore(ctx, []float64{1, 0, 0, 0}, QueryOptions{K: 2})
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("results: %#v", results)
	}

	// A response without an ids array yields nil results.
	emptyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/collections"):
			writeJSON(w, collectionResponse{ID: "collection-1", Name: "langchain"})
		case strings.HasSuffix(r.URL.Path, "/query"):
			writeJSON(w, map[string]any{})
		default:
			http.NotFound(w, r)
		}
	}))
	defer emptyServer.Close()

	emptyStore := newTestStore(t, emptyServer.URL, embeddings.NewFake(4))
	results, err = emptyStore.SimilaritySearchByVectorWithScore(ctx, []float64{1, 0, 0, 0}, QueryOptions{K: 2})
	if err != nil || results != nil {
		t.Fatalf("empty ids: results=%v err=%v", results, err)
	}
	vectors, err := emptyStore.SimilaritySearchByVectorWithVectors(ctx, []float64{1, 0, 0, 0}, QueryOptions{K: 2})
	if err != nil || vectors != nil {
		t.Fatalf("empty ids with vectors: vectors=%v err=%v", vectors, err)
	}
}

func TestMMRDefaultsAndDegenerateVectors(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	// Zero vectors exercise the zero-norm branch of cosineSimilarity.
	store := newTestStore(t, server.URL, stubEmbedder{
		docVectors:  [][]float64{{0, 0, 0, 0}, {0, 0, 0, 0}},
		queryVector: []float64{1, 0, 0, 0},
	})
	ctx := context.Background()
	_, err := store.AddDocuments(ctx, []documents.Document{
		documents.New("alpha", nil).WithID("a"),
		documents.New("beta", nil).WithID("b"),
	})
	if err != nil {
		t.Fatalf("add: %v", err)
	}

	// K/FetchK/LambdaMult defaults kick in for zero values and NaN.
	docs, err := store.MaxMarginalRelevanceSearch(ctx, "alpha", MMROptions{LambdaMult: math.NaN()})
	if err != nil {
		t.Fatalf("mmr: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("mmr docs: got %d", len(docs))
	}

	// A query vector of mismatched length exercises the length-mismatch
	// branch of cosineSimilarity.
	docs, err = store.MaxMarginalRelevanceSearchByVector(ctx, []float64{1, 0}, MMROptions{K: 1, FetchK: 2, LambdaMult: 0.5})
	if err != nil {
		t.Fatalf("mmr mismatch: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("mmr mismatch docs: got %d", len(docs))
	}
}

func TestMMREmptyCollection(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(4))
	docs, err := store.MaxMarginalRelevanceSearchByVector(
		context.Background(),
		[]float64{1, 0, 0, 0},
		MMROptions{K: 2, FetchK: 5, LambdaMult: 0.5},
	)
	if err != nil {
		t.Fatalf("mmr: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("mmr docs: %#v", docs)
	}
}

func TestCollectionLifecycleErrors(t *testing.T) {
	failServer := newFailAfterCreateServer(t)
	ctx := context.Background()

	store := newTestStore(t, failServer.URL, embeddings.NewFake(4))
	if err := store.DeleteCollection(ctx); err == nil {
		t.Fatal("expected delete collection error")
	}
	if err := store.ResetCollection(ctx); err == nil {
		t.Fatal("expected reset error from failed delete")
	}
	if _, err := store.Fork(ctx, "forked"); err == nil {
		t.Fatal("expected fork error")
	}

	// Reset fails when the collection recreation fails after a successful
	// delete.
	var mu sync.Mutex
	creates := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/collections") {
			mu.Lock()
			creates++
			n := creates
			mu.Unlock()
			if n > 1 {
				http.Error(w, "cannot create", http.StatusBadRequest)
				return
			}
			writeJSON(w, collectionResponse{ID: "collection-1", Name: "langchain"})
			return
		}
		writeJSON(w, map[string]any{})
	}))
	defer server.Close()

	resetStore := newTestStore(t, server.URL, embeddings.NewFake(4))
	if err := resetStore.ResetCollection(ctx); err == nil {
		t.Fatal("expected reset error from failed create")
	}
}

func TestForkIDFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/fork") {
			writeJSON(w, collectionResponse{ID: "", Name: "forked"})
			return
		}
		writeJSON(w, collectionResponse{ID: "collection-1", Name: "langchain"})
	}))
	defer server.Close()

	store := newTestStore(t, server.URL, embeddings.NewFake(4))
	fork, err := store.Fork(context.Background(), "forked")
	if err != nil {
		t.Fatalf("fork: %v", err)
	}
	if fork.collectionID != "forked" {
		t.Fatalf("fork ID fallback: got %q", fork.collectionID)
	}
}

func TestDoJSONRequestBuildError(t *testing.T) {
	// A base URL that fails to parse surfaces the request construction error.
	_, err := New(
		context.Background(),
		"langchain",
		embeddings.NewFake(4),
		WithBaseURL("http://invalid host"),
		WithMaxRetries(0),
	)
	if err == nil {
		t.Fatal("expected request build error")
	}
}

func TestDoJSONTransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := server.URL
	server.Close()

	_, err := New(
		context.Background(),
		"langchain",
		embeddings.NewFake(4),
		WithBaseURL(url),
		WithMaxRetries(0),
	)
	if err == nil {
		t.Fatal("expected transport error")
	}
}

func TestDoJSONDecodeAndEmptyBody(t *testing.T) {
	badJSON := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/collections") {
			writeJSON(w, collectionResponse{ID: "collection-1", Name: "langchain"})
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("not-json"))
	}))
	defer badJSON.Close()

	store := newTestStore(t, badJSON.URL, embeddings.NewFake(4))
	_, err := store.Get(context.Background(), GetOptions{})
	if err == nil || !strings.Contains(err.Error(), "decode chroma") {
		t.Fatalf("decode error: got %v", err)
	}

	emptyBody := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/collections") {
			writeJSON(w, collectionResponse{ID: "collection-1", Name: "langchain"})
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer emptyBody.Close()

	emptyStore := newTestStore(t, emptyBody.URL, embeddings.NewFake(4))
	docs, err := emptyStore.Get(context.Background(), GetOptions{})
	if err != nil {
		t.Fatalf("empty body get: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("empty body docs: %#v", docs)
	}
}

func TestDoJSONRetriesTransientFailures(t *testing.T) {
	var mu sync.Mutex
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attempts++
		n := attempts
		mu.Unlock()
		if n == 1 {
			http.Error(w, "slow down", http.StatusTooManyRequests)
			return
		}
		writeJSON(w, collectionResponse{ID: "collection-1", Name: "langchain"})
	}))
	defer server.Close()

	store, err := New(
		context.Background(),
		"langchain",
		embeddings.NewFake(4),
		WithBaseURL(server.URL),
		WithMaxRetries(1),
		WithRetryDelay(time.Millisecond),
	)
	if err != nil {
		t.Fatalf("new store after retry: %v", err)
	}
	if store.collectionID != "collection-1" {
		t.Fatalf("collection ID: got %q", store.collectionID)
	}
	mu.Lock()
	defer mu.Unlock()
	if attempts != 2 {
		t.Fatalf("attempts: got %d want 2", attempts)
	}
}
