package jina

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

// rerankServer is an httptest fake of the Jina /v1/rerank endpoint. It records
// the last request and replies with configured results.
type rerankServer struct {
	*httptest.Server

	mu       sync.Mutex
	requests int
	lastPath string
	lastAuth string
	lastBody []byte
	header   http.Header
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
		server.header = r.Header.Clone()
		results := server.results
		status := server.status
		server.mu.Unlock()

		if status != 0 {
			http.Error(w, "rerank unavailable", status)
			return
		}
		writeJSON(w, rerankResponse{
			Model:   "jina-reranker-v2-base-multilingual",
			Results: results,
		})
	}))
	t.Cleanup(server.Close)
	return server
}

func (s *rerankServer) requestCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests
}

func (s *rerankServer) lastHeader(name string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.header.Get(name)
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

// Python parity: langchain_community JinaRerank.compress_documents sends
// model/query/documents/top_n to the rerank API and rebuilds the output from
// results[].index, annotating relevance_score metadata.
func TestRerankerReranksByResultIndex(t *testing.T) {
	server := newRerankServer(t,
		rerankResult{Index: 1, RelevanceScore: 0.99},
		rerankResult{Index: 2, RelevanceScore: 0.31},
	)

	reranker := New(WithBaseURL(server.URL), WithAPIKey("jina-key"))
	docs, err := reranker.CompressDocuments(t.Context(), []documents.Document{
		documents.New("first", nil),
		documents.New("second", map[string]any{"source": "b"}),
		documents.New("third", nil),
	}, "distance to the moon")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}

	if len(docs) != 2 || docs[0].PageContent != "second" || docs[1].PageContent != "third" {
		t.Fatalf("docs: got %v", docs)
	}
	if docs[0].Metadata["source"] != "b" {
		t.Fatalf("input metadata must be preserved: %v", docs[0].Metadata)
	}
	if got, ok := docs[0].Metadata["relevance_score"]; !ok || got != 0.99 {
		t.Fatalf("relevance score metadata: got %v", docs[0].Metadata)
	}
	if got, ok := docs[1].Metadata["relevance_score"]; !ok || got != 0.31 {
		t.Fatalf("relevance score metadata: got %v", docs[1].Metadata)
	}

	if server.lastPath != "/v1/rerank" {
		t.Fatalf("path: got %q want /v1/rerank", server.lastPath)
	}
	if server.lastAuth != "Bearer jina-key" {
		t.Fatalf("auth header: got %q", server.lastAuth)
	}
	payload := server.decodeLastBody(t)
	if payload.Model != "jina-reranker-v2-base-multilingual" {
		t.Fatalf(
			"model: got %q want jina-reranker-v2-base-multilingual",
			payload.Model,
		)
	}
	if payload.Query != "distance to the moon" {
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
		WithModel("jina-reranker-m0"),
		WithTopN(1),
	)
	if _, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	); err != nil {
		t.Fatalf("compress: %v", err)
	}
	payload := server.decodeLastBody(t)
	if payload.Model != "jina-reranker-m0" {
		t.Fatalf("model: got %q want jina-reranker-m0", payload.Model)
	}
	if payload.TopN != 1 {
		t.Fatalf("top_n: got %d want 1", payload.TopN)
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
	t.Setenv("JINA_API_KEY", "env-key")

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
	server.status = http.StatusUnauthorized
	server.mu.Unlock()

	reranker := New(WithBaseURL(server.URL), WithAPIKey("bad"), WithMaxRetries(0))
	_, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	)
	if providerErr, ok := errors.AsType[*lcerrors.ProviderError](err); !ok {
		t.Fatalf("want provider error, got %v", err)
	} else if providerErr.StatusCode != http.StatusUnauthorized {
		t.Fatalf("status: got %d want 401", providerErr.StatusCode)
	}
}

func TestRerankerRejectsOutOfRangeIndex(t *testing.T) {
	server := newRerankServer(t, rerankResult{Index: -1, RelevanceScore: 0.5})

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

// recordingTransport counts RoundTrips and replays a canned rerank response so
// tests can prove a WithHTTPClient-injected client is the one making requests.
type recordingTransport struct {
	roundTrips int
}

func (t *recordingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	t.roundTrips++
	body := strings.NewReader(`{"model":"jina-reranker-v2-base-multilingual","results":[{"index":0,"relevance_score":0.5}]}`)
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(body),
	}, nil
}

func TestRerankerWithHTTPClientUsesInjectedClient(t *testing.T) {
	transport := &recordingTransport{}

	reranker := New(
		WithBaseURL("http://rerank.invalid"),
		WithAPIKey("key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	docs, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	)
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "doc" {
		t.Fatalf("docs: got %v", docs)
	}
	if transport.roundTrips != 1 {
		t.Fatalf("injected client must issue the request, got %d round trips", transport.roundTrips)
	}
}

func TestRerankerWithHeaderSendsCustomHeaders(t *testing.T) {
	server := newRerankServer(t, rerankResult{Index: 0, RelevanceScore: 0.5})

	reranker := New(
		WithBaseURL(server.URL),
		WithAPIKey("key"),
		WithHeader("X-Tenant", "acme"),
		WithHeader("X-Trace-Id", "trace-7"),
	)
	if _, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("doc", nil)},
		"q",
	); err != nil {
		t.Fatalf("compress: %v", err)
	}
	if got := server.lastHeader("X-Tenant"); got != "acme" {
		t.Fatalf("X-Tenant header: got %q want acme", got)
	}
	if got := server.lastHeader("X-Trace-Id"); got != "trace-7" {
		t.Fatalf("X-Trace-Id header: got %q want trace-7", got)
	}
	if server.lastHeader("Authorization") != "Bearer key" {
		t.Fatalf("custom headers must not replace authorization: %q", server.lastHeader("Authorization"))
	}
}

func TestRerankerBlankBaseURLFallsBackToDefault(t *testing.T) {
	reranker := New(WithBaseURL("  "))
	if reranker.cfg.BaseURL != defaultBaseURL {
		t.Fatalf("base URL: got %q want %q", reranker.cfg.BaseURL, defaultBaseURL)
	}
}

func TestRerankerBlankModelFallsBackToDefault(t *testing.T) {
	reranker := New(WithModel("  "))
	if reranker.cfg.Model != defaultModel {
		t.Fatalf("model: got %q want %q", reranker.cfg.Model, defaultModel)
	}
}

// WithHeader must lazily create the header map when the config carries none.
func TestRerankerWithHeaderInitializesNilHeaderMap(t *testing.T) {
	reranker := &Reranker{}
	WithHeader("X-Tenant", "acme")(reranker)
	if reranker.cfg.Headers["X-Tenant"] != "acme" {
		t.Fatalf("headers: got %v", reranker.cfg.Headers)
	}
}

// A document built without metadata must still gain a relevance_score map, the
// annotation must not panic on a nil Metadata field.
func TestRerankerAnnotatesDocumentWithoutMetadata(t *testing.T) {
	server := newRerankServer(t, rerankResult{Index: 0, RelevanceScore: 0.77})

	reranker := New(WithBaseURL(server.URL), WithAPIKey("key"))
	docs, err := reranker.CompressDocuments(t.Context(), []documents.Document{
		{PageContent: "bare doc"},
	}, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "bare doc" {
		t.Fatalf("docs: got %v", docs)
	}
	if got, ok := docs[0].Metadata["relevance_score"]; !ok || got != 0.77 {
		t.Fatalf("relevance score metadata: got %v", docs[0].Metadata)
	}
}

// Compile-time guard: Reranker implements the core retriever compressor
// interface so it can drive a ContextualCompressionRetriever.
var _ retrievers.DocumentCompressor = (*Reranker)(nil)
