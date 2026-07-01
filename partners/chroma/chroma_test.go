package chroma

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/vectorstores"
	"github.com/projanvil/langchain-golang/standardtests"
)

func TestStoreBasics(t *testing.T) {
	standardtests.RunVectorStoreBasics(t, func(t testing.TB) vectorstores.VectorStore {
		t.Helper()
		server := newChromaServer(t)
		t.Cleanup(server.Close)

		store, err := New(
			context.Background(),
			"langchain",
			embeddings.NewFake(8),
			WithBaseURL(server.URL),
			WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("new chroma store: %v", err)
		}
		return store
	})
}

func TestStoreRequestMapping(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store, err := New(
		context.Background(),
		"custom",
		embeddings.NewFake(4),
		WithBaseURL(server.URL),
		WithTenant("tenant-a"),
		WithDatabase("db-a"),
		WithHeader("X-Test", "present"),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	ids, err := store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha beta", map[string]any{"source": "unit"}).WithID("a"),
	})
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	if got, want := ids[0], "a"; got != want {
		t.Fatalf("id: got %q want %q", got, want)
	}

	if got, want := server.lastPath(), "/api/v2/tenants/tenant-a/databases/db-a/collections/collection-1/upsert"; got != want {
		t.Fatalf("last path: got %q want %q", got, want)
	}
	if got := server.lastHeader("X-Test"); got != "present" {
		t.Fatalf("header: got %q", got)
	}
	var payload writeRequest
	server.decodeLastBody(t, &payload)
	if len(payload.Embeddings) != 1 || len(payload.Embeddings[0]) != 4 {
		t.Fatalf("embedding dimensions: %#v", payload.Embeddings)
	}
	if got := payload.Metadatas[0]["source"]; got != "unit" {
		t.Fatalf("metadata source: got %#v", got)
	}
}

func TestStoreFiltersUpdateVectorsAndMMR(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store, err := New(
		context.Background(),
		"langchain",
		embeddings.NewFake(16),
		WithBaseURL(server.URL),
		WithCollectionMetadata(map[string]any{"owner": "tests"}),
		WithCollectionConfiguration(map[string]any{"hnsw": map[string]any{"space": "cosine"}}),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	var createPayload createCollectionRequest
	server.decodeLastBody(t, &createPayload)
	if got := createPayload.Metadata["owner"]; got != "tests" {
		t.Fatalf("collection metadata: got %#v", got)
	}

	_, err = store.AddDocuments(context.Background(), []documents.Document{
		documents.New("alpha red", map[string]any{"color": "red"}).WithID("red"),
		documents.New("alpha blue", map[string]any{"color": "blue"}).WithID("blue"),
		documents.New("gamma red", map[string]any{"color": "red"}).WithID("gamma"),
	})
	if err != nil {
		t.Fatalf("add: %v", err)
	}

	filtered, err := store.SimilaritySearchOptions(context.Background(), "alpha", QueryOptions{
		K:             2,
		Where:         map[string]any{"color": "red"},
		WhereDocument: map[string]any{"$contains": "alpha"},
	})
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if len(filtered) != 1 || filtered[0].ID != "red" {
		t.Fatalf("filtered docs: %#v", filtered)
	}

	err = store.UpdateDocument(context.Background(), "blue", documents.New("alpha green", map[string]any{"color": "green"}))
	if err != nil {
		t.Fatalf("update: %v", err)
	}
	got, err := store.Get(context.Background(), GetOptions{Where: map[string]any{"color": "green"}})
	if err != nil {
		t.Fatalf("get with filter: %v", err)
	}
	if len(got) != 1 || got[0].ID != "blue" || got[0].PageContent != "alpha green" {
		t.Fatalf("updated doc: %#v", got)
	}

	queryVector, err := embeddings.NewFake(16).EmbedQuery(context.Background(), "alpha")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	withVectors, err := store.SimilaritySearchByVectorWithVectors(context.Background(), queryVector, QueryOptions{K: 2})
	if err != nil {
		t.Fatalf("search with vectors: %v", err)
	}
	if len(withVectors) != 2 || len(withVectors[0].Vector) != 16 {
		t.Fatalf("with vectors: %#v", withVectors)
	}

	mmr, err := store.MaxMarginalRelevanceSearch(context.Background(), "alpha", MMROptions{
		K:          2,
		FetchK:     3,
		LambdaMult: 0.5,
	})
	if err != nil {
		t.Fatalf("mmr: %v", err)
	}
	if len(mmr) != 2 {
		t.Fatalf("mmr docs: got %d", len(mmr))
	}
}

func TestStoreResetAndFork(t *testing.T) {
	server := newChromaServer(t)
	defer server.Close()

	store, err := New(
		context.Background(),
		"langchain",
		embeddings.NewFake(8),
		WithBaseURL(server.URL),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	fork, err := store.Fork(context.Background(), "forked")
	if err != nil {
		t.Fatalf("fork: %v", err)
	}
	if fork.collectionName != "forked" || fork.collectionID != "forked-id" {
		t.Fatalf("fork store: name=%q id=%q", fork.collectionName, fork.collectionID)
	}

	if err := store.ResetCollection(context.Background()); err != nil {
		t.Fatalf("reset: %v", err)
	}
	if got, want := server.deletedCollections(), []string{"collection-1"}; !equalStrings(got, want) {
		t.Fatalf("deleted collections: got %#v want %#v", got, want)
	}
}

func TestStoreErrorTranslation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "slow down", http.StatusTooManyRequests)
	}))
	defer server.Close()

	_, err := New(
		context.Background(),
		"langchain",
		embeddings.NewFake(4),
		WithBaseURL(server.URL),
		WithMaxRetries(0),
	)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, lcerrors.ErrRateLimited) {
		t.Fatalf("error type: got %v want rate limited", err)
	}
}

type chromaServer struct {
	*httptest.Server
	t           testing.TB
	mu          sync.Mutex
	lastRequest recordedRequest
	docs        map[string]documents.Document
	vectors     map[string][]float64
	deleted     []string
}

type recordedRequest struct {
	path    string
	headers http.Header
	body    []byte
}

func newChromaServer(t testing.TB) *chromaServer {
	s := &chromaServer{
		t:       t,
		docs:    map[string]documents.Document{},
		vectors: map[string][]float64{},
	}
	s.Server = httptest.NewServer(http.HandlerFunc(s.handle))
	return s
}

func (s *chromaServer) handle(w http.ResponseWriter, r *http.Request) {
	var body json.RawMessage
	if r.Body != nil {
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil && err.Error() != "EOF" {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}

	s.mu.Lock()
	s.lastRequest = recordedRequest{
		path:    r.URL.Path,
		headers: r.Header.Clone(),
		body:    append([]byte(nil), body...),
	}
	s.mu.Unlock()

	switch {
	case strings.HasSuffix(r.URL.Path, "/collections"):
		writeJSON(w, collectionResponse{ID: "collection-1", Name: "langchain"})
	case strings.HasSuffix(r.URL.Path, "/upsert"):
		var req writeRequest
		decodeBody(w, body, &req)
		s.add(req)
		writeJSON(w, map[string]any{})
	case strings.HasSuffix(r.URL.Path, "/update"):
		var req writeRequest
		decodeBody(w, body, &req)
		s.add(req)
		writeJSON(w, map[string]any{})
	case strings.HasSuffix(r.URL.Path, "/get"):
		var req getRequest
		decodeBody(w, body, &req)
		writeJSON(w, s.get(req))
	case strings.HasSuffix(r.URL.Path, "/query"):
		var req queryRequest
		decodeBody(w, body, &req)
		writeJSON(w, s.query(req))
	case strings.HasSuffix(r.URL.Path, "/delete"):
		var req deleteRequest
		decodeBody(w, body, &req)
		s.delete(req.IDs)
		writeJSON(w, map[string]any{})
	case strings.HasSuffix(r.URL.Path, "/fork"):
		writeJSON(w, collectionResponse{ID: "forked-id", Name: "forked"})
	case r.Method == http.MethodDelete && strings.Contains(r.URL.Path, "/collections/"):
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		s.mu.Lock()
		s.deleted = append(s.deleted, parts[len(parts)-1])
		s.docs = map[string]documents.Document{}
		s.vectors = map[string][]float64{}
		s.mu.Unlock()
		writeJSON(w, map[string]any{})
	default:
		http.NotFound(w, r)
	}
}

func (s *chromaServer) add(req writeRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i, id := range req.IDs {
		doc := documents.Document{ID: id}
		if i < len(req.Documents) {
			doc.PageContent = req.Documents[i]
		}
		if i < len(req.Metadatas) {
			doc.Metadata = req.Metadatas[i]
		}
		s.docs[id] = doc
		if i < len(req.Embeddings) {
			s.vectors[id] = req.Embeddings[i]
		}
	}
}

func (s *chromaServer) get(req getRequest) getResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	resp := getResponse{}
	ids := req.IDs
	if len(ids) == 0 {
		for id := range s.docs {
			ids = append(ids, id)
		}
		sort.Strings(ids)
	}
	if req.Offset != nil && *req.Offset < len(ids) {
		ids = ids[*req.Offset:]
	}
	if req.Limit != nil && *req.Limit < len(ids) {
		ids = ids[:*req.Limit]
	}
	for _, id := range ids {
		doc, ok := s.docs[id]
		if !ok {
			continue
		}
		if !matchesWhere(doc, req.Where) || !matchesWhereDocument(doc, req.WhereDocument) {
			continue
		}
		resp.IDs = append(resp.IDs, doc.ID)
		resp.Documents = append(resp.Documents, doc.PageContent)
		resp.Metadatas = append(resp.Metadatas, doc.Metadata)
	}
	return resp
}

func (s *chromaServer) query(req queryRequest) queryResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	query := []float64(nil)
	if len(req.QueryEmbeddings) > 0 {
		query = req.QueryEmbeddings[0]
	}
	type scored struct {
		id       string
		distance float64
	}
	scoredDocs := make([]scored, 0, len(s.docs))
	for id, vector := range s.vectors {
		doc := s.docs[id]
		if !matchesWhere(doc, req.Where) || !matchesWhereDocument(doc, req.WhereDocument) {
			continue
		}
		scoredDocs = append(scoredDocs, scored{
			id:       id,
			distance: 1 - cosine(query, vector),
		})
	}
	sort.SliceStable(scoredDocs, func(i, j int) bool {
		return scoredDocs[i].distance < scoredDocs[j].distance
	})
	if req.NResults > 0 && len(scoredDocs) > req.NResults {
		scoredDocs = scoredDocs[:req.NResults]
	}

	resp := queryResponse{IDs: [][]string{{}}, Documents: [][]string{{}}, Metadatas: [][]map[string]any{{}}, Distances: [][]float64{{}}, Embeddings: [][][]float64{{}}}
	for _, item := range scoredDocs {
		doc := s.docs[item.id]
		resp.IDs[0] = append(resp.IDs[0], doc.ID)
		resp.Documents[0] = append(resp.Documents[0], doc.PageContent)
		resp.Metadatas[0] = append(resp.Metadatas[0], doc.Metadata)
		resp.Distances[0] = append(resp.Distances[0], item.distance)
		resp.Embeddings[0] = append(resp.Embeddings[0], append([]float64(nil), s.vectors[item.id]...))
	}
	return resp
}

func (s *chromaServer) delete(ids []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range ids {
		delete(s.docs, id)
		delete(s.vectors, id)
	}
}

func (s *chromaServer) lastPath() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastRequest.path
}

func (s *chromaServer) lastHeader(name string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastRequest.headers.Get(name)
}

func (s *chromaServer) decodeLastBody(t *testing.T, out any) {
	t.Helper()
	s.mu.Lock()
	body := append([]byte(nil), s.lastRequest.body...)
	s.mu.Unlock()

	if err := json.Unmarshal(body, out); err != nil {
		t.Fatalf("decode last body: %v", err)
	}
}

func (s *chromaServer) deletedCollections() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.deleted...)
}

func decodeBody(w http.ResponseWriter, body json.RawMessage, out any) {
	if err := json.Unmarshal(body, out); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func cosine(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot float64
	var normA float64
	var normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func matchesWhere(doc documents.Document, where map[string]any) bool {
	for key, want := range where {
		if doc.Metadata[key] != want {
			return false
		}
	}
	return true
}

func matchesWhereDocument(doc documents.Document, where map[string]any) bool {
	contains, ok := where["$contains"].(string)
	if !ok || contains == "" {
		return true
	}
	return strings.Contains(doc.PageContent, contains)
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
