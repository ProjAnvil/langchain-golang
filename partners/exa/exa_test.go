package exa

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
)

// exaServer is an httptest fake of the Exa REST API. It records the last
// request and replies with configured results (or a configured error).
type exaServer struct {
	*httptest.Server

	mu         sync.Mutex
	requests   int
	lastPath   string
	lastHeader http.Header
	lastBody   []byte
	results    []Result
	autoprompt string
	status     int
	errorBody  string
}

func newExaServer(t *testing.T) *exaServer {
	t.Helper()
	server := &exaServer{}
	server.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		server.mu.Lock()
		server.requests++
		server.lastPath = r.URL.Path
		server.lastHeader = r.Header.Clone()
		server.lastBody = raw
		results := server.results
		autoprompt := server.autoprompt
		status := server.status
		errorBody := server.errorBody
		server.mu.Unlock()

		if status != 0 {
			http.Error(w, errorBody, status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(searchResponse{Results: results, AutopromptString: autoprompt})
	}))
	t.Cleanup(server.Close)
	return server
}

func (s *exaServer) requestCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests
}

func (s *exaServer) header(name string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastHeader.Get(name)
}

func (s *exaServer) lastBodyMap(t *testing.T) map[string]any {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	var body map[string]any
	if err := json.Unmarshal(s.lastBody, &body); err != nil {
		t.Fatalf("decode request body %s: %v", s.lastBody, err)
	}
	return body
}

// setResults configures the search results the fake replies with.
func (s *exaServer) setResults(results ...Result) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.results = results
}

// setError configures a non-2xx reply.
func (s *exaServer) setError(status int, body string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status = status
	s.errorBody = body
}

// assertExactKeys fails unless body has exactly the wanted top-level keys.
func assertExactKeys(t *testing.T, body map[string]any, want ...string) {
	t.Helper()
	if len(body) != len(want) {
		t.Fatalf("request body keys: got %v, want exactly %v (body: %v)", keysOf(body), want, body)
	}
	for _, key := range want {
		if _, ok := body[key]; !ok {
			t.Fatalf("request body is missing key %q (body: %v)", key, body)
		}
	}
}

func keysOf(body map[string]any) []string {
	out := make([]string, 0, len(body))
	for key := range body {
		out = append(out, key)
	}
	return out
}

// subObject returns body[key] as a decoded JSON object.
func subObject(t *testing.T, body map[string]any, key string) map[string]any {
	t.Helper()
	value, ok := body[key].(map[string]any)
	if !ok {
		t.Fatalf("request body key %q is not an object: %v", key, body[key])
	}
	return value
}

// Python parity: langchain_exa 1.1.0 talks to the Exa REST API with an
// x-api-key header and camelCase wire fields; Go implements the endpoints
// directly (design decision #6 in the parity wave 3 spec).
func TestClientSearchSendsQueryAndAuth(t *testing.T) {
	server := newExaServer(t)

	client := &Client{APIKey: "secret-key", BaseURL: server.URL}
	results, err := client.Search(t.Context(), searchRequest{Query: "what is the weather in SF", NumResults: 10})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("results: got %v", results)
	}

	if server.lastPath != "/search" {
		t.Fatalf("path: got %q want /search", server.lastPath)
	}
	if got := server.header("x-api-key"); got != "secret-key" {
		t.Fatalf("x-api-key header: got %q want secret-key", got)
	}
	if got := server.header("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type header: got %q", got)
	}
	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "query", "numResults")
	if body["query"] != "what is the weather in SF" {
		t.Fatalf("query: got %v", body["query"])
	}
	if body["numResults"] != float64(10) {
		t.Fatalf("numResults: got %v want 10", body["numResults"])
	}
}

func TestClientSearchFullRequestBody(t *testing.T) {
	server := newExaServer(t)
	livecrawl := "fallback"

	client := &Client{APIKey: "key", BaseURL: server.URL}
	_, err := client.Search(t.Context(), searchRequest{
		Query:              "exa optimized query",
		NumResults:         3,
		LiveCrawl:          &livecrawl,
		StartPublishedDate: "2024-01-01",
		EndPublishedDate:   "2024-12-31",
		IncludeDomains:     []string{"example.com", "docs.example.com"},
		ExcludeDomains:     []string{"spam.example"},
		Contents: &contentsRequest{
			Text:       &TextOptions{MaxCharacters: 1000},
			Highlights: &HighlightOptions{Query: "exa", NumSentences: 3},
		},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	body := server.lastBodyMap(t)
	assertExactKeys(t, body,
		"query", "numResults", "livecrawl", "startPublishedDate", "endPublishedDate",
		"includeDomains", "excludeDomains", "contents",
	)
	if body["livecrawl"] != "fallback" {
		t.Fatalf("livecrawl: got %v", body["livecrawl"])
	}
	if body["startPublishedDate"] != "2024-01-01" || body["endPublishedDate"] != "2024-12-31" {
		t.Fatalf("published dates: got %v / %v", body["startPublishedDate"], body["endPublishedDate"])
	}
	if domains, ok := body["includeDomains"].([]any); !ok ||
		len(domains) != 2 || domains[0] != "example.com" || domains[1] != "docs.example.com" {
		t.Fatalf("includeDomains: got %v", body["includeDomains"])
	}
	if domains, ok := body["excludeDomains"].([]any); !ok || len(domains) != 1 || domains[0] != "spam.example" {
		t.Fatalf("excludeDomains: got %v", body["excludeDomains"])
	}
	contents := subObject(t, body, "contents")
	assertExactKeys(t, contents, "text", "highlights")
	text := subObject(t, contents, "text")
	if text["maxCharacters"] != float64(1000) {
		t.Fatalf("maxCharacters: got %v", text["maxCharacters"])
	}
	highlights := subObject(t, contents, "highlights")
	if highlights["query"] != "exa" || highlights["numSentences"] != float64(3) {
		t.Fatalf("highlights: got %v", highlights)
	}
}

func TestClientFindSimilarRequest(t *testing.T) {
	server := newExaServer(t)

	client := &Client{APIKey: "key", BaseURL: server.URL}
	_, err := client.FindSimilar(t.Context(), searchRequest{
		URL:        "https://example.com/post",
		NumResults: 5,
		Category:   "news",
	})
	if err != nil {
		t.Fatalf("findSimilar: %v", err)
	}

	if server.lastPath != "/findSimilar" {
		t.Fatalf("path: got %q want /findSimilar", server.lastPath)
	}
	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "url", "numResults", "category")
	if body["url"] != "https://example.com/post" {
		t.Fatalf("url: got %v", body["url"])
	}
	if body["category"] != "news" {
		t.Fatalf("category: got %v", body["category"])
	}
}

func TestClientDecodesResults(t *testing.T) {
	server := newExaServer(t)
	server.setResults(Result{
		Title:           "title",
		URL:             "https://example.com",
		PublishedDate:   "2023-02-23T01:17:06.594Z",
		Author:          "author",
		Score:           0.1843988299369812,
		Text:            "text",
		Highlights:      []string{"one", "two"},
		ID:              "id",
		Summary:         "summary",
		HighlightScores: []float64{0.9, 0.8},
	})

	client := &Client{APIKey: "key", BaseURL: server.URL}
	results, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("results: got %v", results)
	}
	got := results[0]
	if got.Title != "title" || got.URL != "https://example.com" ||
		got.PublishedDate != "2023-02-23T01:17:06.594Z" || got.Author != "author" ||
		got.Text != "text" || got.ID != "id" || got.Summary != "summary" {
		t.Fatalf("result fields: got %+v", got)
	}
	if got.Score != 0.1843988299369812 {
		t.Fatalf("score: got %v", got.Score)
	}
	if len(got.Highlights) != 2 || got.Highlights[0] != "one" || got.Highlights[1] != "two" {
		t.Fatalf("highlights: got %v", got.Highlights)
	}
	if len(got.HighlightScores) != 2 || got.HighlightScores[0] != 0.9 || got.HighlightScores[1] != 0.8 {
		t.Fatalf("highlightScores: got %v", got.HighlightScores)
	}
}

// Python parity: initialize_client resolves the API key from the EXA_API_KEY
// environment variable when it is not passed explicitly.
func TestClientAPIKeyFromEnv(t *testing.T) {
	server := newExaServer(t)
	t.Setenv("EXA_API_KEY", "env-key")

	client := &Client{BaseURL: server.URL}
	if _, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1}); err != nil {
		t.Fatalf("search: %v", err)
	}
	if got := server.header("x-api-key"); got != "env-key" {
		t.Fatalf("x-api-key header: got %q want env-key", got)
	}
}

func TestClientMissingAPIKey(t *testing.T) {
	server := newExaServer(t)
	t.Setenv("EXA_API_KEY", "")

	client := &Client{BaseURL: server.URL}
	_, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1})
	if err == nil {
		t.Fatal("missing API key must error")
	}
	if !strings.Contains(err.Error(), "EXA_API_KEY") {
		t.Fatalf("error must mention EXA_API_KEY: %v", err)
	}
	if server.requestCount() != 0 {
		t.Fatalf("no request may be sent without a key, sent %d", server.requestCount())
	}
}

// Python parity: HTTP failures surface as typed provider errors (HTTP 429 ->
// rate limit sentinel, 5xx -> server sentinel, 401 -> auth sentinel), matching
// langchain-core's standard model error classification.
func TestClientErrorMapping(t *testing.T) {
	cases := []struct {
		status   int
		sentinel error
	}{
		{http.StatusUnauthorized, lcerrors.ErrModelAuth},
		{http.StatusTooManyRequests, lcerrors.ErrModelRateLimit},
		{http.StatusInternalServerError, lcerrors.ErrModelServer},
	}
	for _, testCase := range cases {
		t.Run(fmt.Sprintf("status_%d", testCase.status), func(t *testing.T) {
			server := newExaServer(t)
			server.setError(testCase.status, "exa exploded")

			client := &Client{APIKey: "key", BaseURL: server.URL}
			_, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1})
			if !errors.Is(err, testCase.sentinel) {
				t.Fatalf("status %d: want %v, got %v", testCase.status, testCase.sentinel, err)
			}
			providerErr, ok := errors.AsType[*lcerrors.ProviderError](err)
			if !ok {
				t.Fatalf("status %d: want provider error, got %v", testCase.status, err)
			}
			if providerErr.Provider != "exa" || providerErr.Endpoint != "/search" ||
				providerErr.StatusCode != testCase.status {
				t.Fatalf("provider error fields: got %+v", providerErr)
			}
			if !strings.Contains(providerErr.Body, "exa exploded") {
				t.Fatalf("provider error body: got %q", providerErr.Body)
			}
		})
	}
}

func TestClientDefaultBaseURL(t *testing.T) {
	var requestURL string
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		requestURL = r.URL.String()
		return jsonResponse(`{"results":[]}`), nil
	})

	client := &Client{APIKey: "key", HTTPClient: &http.Client{Transport: transport}}
	if _, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1}); err != nil {
		t.Fatalf("search: %v", err)
	}
	if requestURL != "https://api.exa.ai/search" {
		t.Fatalf("url: got %q want https://api.exa.ai/search", requestURL)
	}
}

// A trailing-slash base URL must not produce a double slash in the path.
func TestClientTrailingSlashBaseURL(t *testing.T) {
	var requestPath string
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		requestPath = r.URL.Path
		return jsonResponse(`{"results":[]}`), nil
	})

	client := &Client{APIKey: "key", BaseURL: "http://exa.invalid/", HTTPClient: &http.Client{Transport: transport}}
	if _, err := client.FindSimilar(t.Context(), searchRequest{URL: "https://x", NumResults: 1}); err != nil {
		t.Fatalf("findSimilar: %v", err)
	}
	if requestPath != "/findSimilar" {
		t.Fatalf("path: got %q want /findSimilar", requestPath)
	}
}

func TestClientTransportErrorWrapsConnection(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	server.Close()

	client := &Client{APIKey: "key", BaseURL: server.URL}
	_, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1})
	if !errors.Is(err, lcerrors.ErrModelConnection) {
		t.Fatalf("want connection sentinel, got %v", err)
	}
}

// An unparseable base URL fails at request construction before any I/O.
func TestClientInvalidBaseURL(t *testing.T) {
	client := &Client{APIKey: "key", BaseURL: "http://exa.invalid/\x7f"}

	_, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1})
	if err == nil || !strings.Contains(err.Error(), "build") {
		t.Fatalf("invalid base URL must fail at request build: %v", err)
	}
}

// A response body that fails mid-read surfaces as a read error.
func TestClientReadBodyError(t *testing.T) {
	client := &Client{
		APIKey:     "key",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return failingBody(), nil })},
	}

	_, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1})
	if err == nil || !strings.Contains(err.Error(), "read") {
		t.Fatalf("failing body must surface a read error: %v", err)
	}
}

// A 2xx reply whose body is not JSON fails response decoding.
func TestClientInvalidJSONResponse(t *testing.T) {
	client := &Client{
		APIKey:     "key",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return jsonResponse("<html>oops</html>"), nil })},
	}

	_, err := client.Search(t.Context(), searchRequest{Query: "q", NumResults: 1})
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Fatalf("non-JSON body must fail decoding: %v", err)
	}
}

// errReader simulates a response body that errors while being read.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("body exploded") }

func failingBody() *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(errReader{}),
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func jsonResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
