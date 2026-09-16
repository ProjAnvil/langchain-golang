package cohere

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

// rerankServer is an httptest fake of the Cohere /v2/rerank endpoint. It
// records the last request and replies with configured results.
type rerankServer struct {
	*httptest.Server

	mu       sync.Mutex
	requests int
	lastPath string
	lastAuth string
	lastBody []byte
	results  []rerankResult
	status   int
}

func newRerankServer(t *testing.T, results ...rerankResult) *rerankServer {
	t.Helper()
	server := &rerankServer{results: results}
	server.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body json.RawMessage
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		server.mu.Lock()
		server.requests++
		server.lastPath = r.URL.Path
		server.lastAuth = r.Header.Get("Authorization")
		server.lastBody = append([]byte(nil), body...)
		results := server.results
		status := server.status
		server.mu.Unlock()

		if status != 0 {
			http.Error(w, "rerank unavailable", status)
			return
		}
		writeJSON(w, rerankResponse{ID: "rerank-1", Results: results})
	}))
	t.Cleanup(server.Close)
	return server
}

func (s *rerankServer) requestCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests
}

func (s *rerankServer) decodeLastBody(t *testing.T) rerankRequest {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	var req rerankRequest
	if err := json.Unmarshal(s.lastBody, &req); err != nil {
		t.Fatalf("decode request body %s: %v", s.lastBody, err)
	}
	return req
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

// Python parity: langchain_cohere.CohereRerank.compress_documents sends
// model/query/documents/top_n to the rerank API and rebuilds the output from
// results[].index, annotating relevance_score metadata.
func TestRerankerReranksByResultIndex(t *testing.T) {
	server := newRerankServer(t,
		rerankResult{Index: 2, RelevanceScore: 0.98},
		rerankResult{Index: 0, RelevanceScore: 0.42},
	)

	reranker := New(WithBaseURL(server.URL), WithAPIKey("secret-key"))
	docs, err := reranker.CompressDocuments(t.Context(), []documents.Document{
		documents.New("first", map[string]any{"source": "a"}),
		documents.New("second", nil),
		documents.New("third", nil),
	}, "what is the capital of france")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}

	if len(docs) != 2 || docs[0].PageContent != "third" || docs[1].PageContent != "first" {
		t.Fatalf("docs: got %v", docs)
	}
	if got, ok := docs[0].Metadata["relevance_score"]; !ok || got != 0.98 {
		t.Fatalf("relevance score metadata: got %v", docs[0].Metadata)
	}
	if docs[1].Metadata["source"] != "a" {
		t.Fatalf("input metadata must be preserved: %v", docs[1].Metadata)
	}
	if got, ok := docs[1].Metadata["relevance_score"]; !ok || got != 0.42 {
		t.Fatalf("relevance score metadata: got %v", docs[1].Metadata)
	}

	if server.lastPath != "/v2/rerank" {
		t.Fatalf("path: got %q want /v2/rerank", server.lastPath)
	}
	if server.lastAuth != "Bearer secret-key" {
		t.Fatalf("auth header: got %q", server.lastAuth)
	}
	payload := server.decodeLastBody(t)
	if payload.Model != "rerank-v3.5" {
		t.Fatalf("model: got %q want rerank-v3.5", payload.Model)
	}
	if payload.Query != "what is the capital of france" {
		t.Fatalf("query: got %q", payload.Query)
	}
	if len(payload.Documents) != 3 || payload.Documents[0] != "first" ||
		payload.Documents[1] != "second" || payload.Documents[2] != "third" {
		t.Fatalf("documents: got %v", payload.Documents)
	}
	if payload.TopN != 3 {
		t.Fatalf("top_n: got %d want 3", payload.TopN)
	}
}

func TestRerankerOptionsOverrideDefaults(t *testing.T) {
	server := newRerankServer(t, rerankResult{Index: 0, RelevanceScore: 0.5})

	reranker := New(
		WithBaseURL(server.URL),
		WithAPIKey("key"),
		WithModel("rerank-v3.0"),
		WithTopN(2),
	)
	if _, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	); err != nil {
		t.Fatalf("compress: %v", err)
	}
	payload := server.decodeLastBody(t)
	if payload.Model != "rerank-v3.0" {
		t.Fatalf("model: got %q want rerank-v3.0", payload.Model)
	}
	if payload.TopN != 2 {
		t.Fatalf("top_n: got %d want 2", payload.TopN)
	}
}

// Python parity: compress_documents returns [] without calling the API when
// the input is empty.
func TestRerankerEmptyInputSkipsAPI(t *testing.T) {
	server := newRerankServer(t, rerankResult{Index: 0, RelevanceScore: 1})

	reranker := New(WithBaseURL(server.URL), WithAPIKey("key"))
	docs, err := reranker.CompressDocuments(t.Context(), nil, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs: got %d want 0", len(docs))
	}
	if server.requestCount() != 0 {
		t.Fatalf("API must not be called for empty input, called %d times", server.requestCount())
	}
}

func TestRerankerEmptyResults(t *testing.T) {
	server := newRerankServer(t)

	reranker := New(WithBaseURL(server.URL), WithAPIKey("key"))
	docs, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	)
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs: got %d want 0", len(docs))
	}
}

func TestRerankerAPIKeyFromEnv(t *testing.T) {
	server := newRerankServer(t, rerankResult{Index: 0, RelevanceScore: 0.5})
	t.Setenv("COHERE_API_KEY", "env-key")

	reranker := New(WithBaseURL(server.URL))
	if _, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	); err != nil {
		t.Fatalf("compress: %v", err)
	}
	if server.lastAuth != "Bearer env-key" {
		t.Fatalf("auth header: got %q want Bearer env-key", server.lastAuth)
	}
}

func TestRerankerProviderErrorPropagates(t *testing.T) {
	server := newRerankServer(t)
	server.mu.Lock()
	server.status = http.StatusServiceUnavailable
	server.mu.Unlock()

	reranker := New(WithBaseURL(server.URL), WithAPIKey("key"), WithMaxRetries(0))
	_, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	)
	if providerErr, ok := errors.AsType[*lcerrors.ProviderError](err); !ok {
		t.Fatalf("want provider error, got %v", err)
	} else if providerErr.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status: got %d want 503", providerErr.StatusCode)
	}
}

func TestRerankerRejectsOutOfRangeIndex(t *testing.T) {
	server := newRerankServer(t, rerankResult{Index: 7, RelevanceScore: 0.5})

	reranker := New(WithBaseURL(server.URL), WithAPIKey("key"))
	_, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	)
	if err == nil {
		t.Fatal("out-of-range result index must error")
	}
}

// Compile-time guard: Reranker implements the core retriever compressor
// interface so it can drive a ContextualCompressionRetriever.
var _ retrievers.DocumentCompressor = (*Reranker)(nil)
