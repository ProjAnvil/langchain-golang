package nomic

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	coreembeddings "github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/lcerrors"
)

// Compile-time guard: Embeddings satisfies the core embeddings interface.
var _ coreembeddings.Embeddings = (*Embeddings)(nil)

// embedServer is an httptest fake of the Nomic Atlas /embedding/text endpoint.
// It records the last request and replies with a canned response body (or a
// configured error status).
type embedServer struct {
	*httptest.Server

	mu       sync.Mutex
	requests int
	lastPath string
	lastAuth string
	lastBody []byte
	status   int
	respBody string
}

func newEmbedServer(t *testing.T, responseBody string) *embedServer {
	t.Helper()
	server := &embedServer{respBody: responseBody}
	server.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		server.mu.Lock()
		server.requests++
		server.lastPath = r.URL.Path
		server.lastAuth = r.Header.Get("Authorization")
		server.lastBody = body
		status := server.status
		respBody := server.respBody
		server.mu.Unlock()

		if status != 0 {
			http.Error(w, "nomic unavailable", status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, respBody)
	}))
	t.Cleanup(server.Close)
	return server
}

func (s *embedServer) setStatus(status int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status = status
}

func (s *embedServer) requestCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests
}

// lastBodyMap returns the last request body decoded as a raw JSON object so
// tests can assert the exact wire key set (e.g. dimensionality omitted).
func (s *embedServer) lastBodyMap(t *testing.T) map[string]any {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	var body map[string]any
	if err := json.Unmarshal(s.lastBody, &body); err != nil {
		t.Fatalf("decode request body %s: %v", s.lastBody, err)
	}
	return body
}

// wireRequest mirrors the Atlas /embedding/text request body.
type wireRequest struct {
	Model          string   `json:"model"`
	Texts          []string `json:"texts"`
	TaskType       string   `json:"task_type"`
	Dimensionality *int     `json:"dimensionality"`
}

func (s *embedServer) lastWireRequest(t *testing.T) wireRequest {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	var req wireRequest
	if err := json.Unmarshal(s.lastBody, &req); err != nil {
		t.Fatalf("decode request body %s: %v", s.lastBody, err)
	}
	return req
}

// Python parity: langchain_nomic.NomicEmbeddings.embed_documents (embeddings.py:116-126)
// sends task_type="search_document" and returns output["embeddings"] in input
// order; auth is a Bearer token against the Atlas API.
func TestEmbedDocumentsGoldenRequest(t *testing.T) {
	server := newEmbedServer(t,
		`{"embeddings":[[0.1,0.2],[-0.3,0.4]],"usage":{"prompt_tokens":9,"total_tokens":9}}`)

	model := &Embeddings{APIKey: "test-key", BaseURL: server.URL}
	vectors, err := model.EmbedDocuments(t.Context(), []string{"first doc", "second doc"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	if len(vectors) != 2 ||
		vectors[0][0] != 0.1 || vectors[0][1] != 0.2 ||
		vectors[1][0] != -0.3 || vectors[1][1] != 0.4 {
		t.Fatalf("vectors: got %v", vectors)
	}

	if server.lastPath != "/embedding/text" {
		t.Fatalf("path: got %q want /embedding/text", server.lastPath)
	}
	if server.lastAuth != "Bearer test-key" {
		t.Fatalf("auth header: got %q want Bearer test-key", server.lastAuth)
	}
	if keys := server.lastBodyMap(t); len(keys) != 3 {
		t.Fatalf("request keys: got %v want exactly model/texts/task_type", keys)
	}
	wire := server.lastWireRequest(t)
	if wire.Model != "nomic-embed-text-v1.5" {
		t.Fatalf("model: got %q want nomic-embed-text-v1.5", wire.Model)
	}
	if len(wire.Texts) != 2 || wire.Texts[0] != "first doc" || wire.Texts[1] != "second doc" {
		t.Fatalf("texts: got %v", wire.Texts)
	}
	if wire.TaskType != "search_document" {
		t.Fatalf("task_type: got %q want search_document", wire.TaskType)
	}
}

// Python parity: langchain_nomic.NomicEmbeddings.embed_query (embeddings.py:128-138)
// sends task_type="search_query" with a single text and unwraps the first vector.
func TestEmbedQueryGoldenRequest(t *testing.T) {
	server := newEmbedServer(t,
		`{"embeddings":[[0.5,-0.25]],"usage":{"prompt_tokens":3,"total_tokens":3}}`)

	model := &Embeddings{APIKey: "test-key", BaseURL: server.URL}
	vector, err := model.EmbedQuery(t.Context(), "what is the capital of france")
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if len(vector) != 2 || vector[0] != 0.5 || vector[1] != -0.25 {
		t.Fatalf("vector: got %v", vector)
	}

	if server.lastAuth != "Bearer test-key" {
		t.Fatalf("auth header: got %q want Bearer test-key", server.lastAuth)
	}
	wire := server.lastWireRequest(t)
	if wire.Model != "nomic-embed-text-v1.5" {
		t.Fatalf("model: got %q want nomic-embed-text-v1.5", wire.Model)
	}
	if len(wire.Texts) != 1 || wire.Texts[0] != "what is the capital of france" {
		t.Fatalf("texts: got %v", wire.Texts)
	}
	if wire.TaskType != "search_query" {
		t.Fatalf("task_type: got %q want search_query", wire.TaskType)
	}
}

// An explicit TaskType overrides the per-method default for both entry points,
// covering all four values documented by langchain_nomic.NomicEmbeddings.embed
// (embeddings.py:97-114): search_query, search_document, classification,
// clustering.
func TestExplicitTaskTypeOverridesMethodDefault(t *testing.T) {
	for _, taskType := range []string{
		"search_query", "search_document", "classification", "clustering",
	} {
		for _, useQuery := range []bool{true, false} {
			method := "embed_documents"
			if useQuery {
				method = "embed_query"
			}
			t.Run(taskType+"/"+method, func(t *testing.T) {
				server := newEmbedServer(t, `{"embeddings":[[1.0]],"usage":{}}`)

				model := &Embeddings{APIKey: "key", BaseURL: server.URL, TaskType: taskType}
				if useQuery {
					if _, err := model.EmbedQuery(t.Context(), "q"); err != nil {
						t.Fatalf("embed query: %v", err)
					}
				} else {
					if _, err := model.EmbedDocuments(t.Context(), []string{"doc"}); err != nil {
						t.Fatalf("embed documents: %v", err)
					}
				}
				if wire := server.lastWireRequest(t); wire.TaskType != taskType {
					t.Fatalf("task_type: got %q want %q", wire.TaskType, taskType)
				}
			})
		}
	}
}

// Divergence (Go addition, Python has no such validation): an explicit
// TaskType outside the four documented values fails fast without a request.
func TestInvalidTaskTypeErrors(t *testing.T) {
	server := newEmbedServer(t, `{"embeddings":[[1.0]],"usage":{}}`)

	for _, useQuery := range []bool{true, false} {
		model := &Embeddings{APIKey: "key", BaseURL: server.URL, TaskType: "summarize"}
		var err error
		if useQuery {
			_, err = model.EmbedQuery(t.Context(), "q")
		} else {
			_, err = model.EmbedDocuments(t.Context(), []string{"doc"})
		}
		if err == nil {
			t.Fatal("invalid task type must error")
		}
		if !errors.Is(err, lcerrors.ErrInvalidInput) {
			t.Fatalf("error: got %v want lcerrors.ErrInvalidInput", err)
		}
		if !strings.Contains(err.Error(), "summarize") {
			t.Fatalf("error must name the offending task type: %v", err)
		}
	}
	if server.requestCount() != 0 {
		t.Fatalf("invalid task type must not reach the API, got %d requests", server.requestCount())
	}
}

// Divergence: an empty Model defaults to nomic-embed-text-v1.5 (Python
// requires the model argument); an explicit Model is passed through.
func TestModelOverride(t *testing.T) {
	server := newEmbedServer(t, `{"embeddings":[[1.0]],"usage":{}}`)

	model := &Embeddings{APIKey: "key", BaseURL: server.URL, Model: "nomic-embed-text-v1"}
	if _, err := model.EmbedQuery(t.Context(), "q"); err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if wire := server.lastWireRequest(t); wire.Model != "nomic-embed-text-v1" {
		t.Fatalf("model: got %q want nomic-embed-text-v1", wire.Model)
	}
}

// dimensionality (Matryoshka truncation, embeddings.py:72-73) is omitted from
// the wire body when nil and sent verbatim when set.
func TestDimensionalityOmittedAndPresent(t *testing.T) {
	server := newEmbedServer(t, `{"embeddings":[[1.0]],"usage":{}}`)

	model := &Embeddings{APIKey: "key", BaseURL: server.URL}
	if _, err := model.EmbedQuery(t.Context(), "q"); err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if _, ok := server.lastBodyMap(t)["dimensionality"]; ok {
		t.Fatal("dimensionality must be omitted when nil")
	}

	dimensionality := 64
	model.Dimensionality = &dimensionality
	if _, err := model.EmbedQuery(t.Context(), "q"); err != nil {
		t.Fatalf("embed query: %v", err)
	}
	wire := server.lastWireRequest(t)
	if wire.Dimensionality == nil || *wire.Dimensionality != 64 {
		t.Fatalf("dimensionality: got %v want 64", wire.Dimensionality)
	}
}

// Python parity: the API key falls back to the NOMIC_API_KEY environment
// variable (embeddings.py:88). Go additionally fails fast when neither is set.
func TestAPIKeyResolution(t *testing.T) {
	t.Run("env fallback", func(t *testing.T) {
		server := newEmbedServer(t, `{"embeddings":[[1.0]],"usage":{}}`)
		t.Setenv("NOMIC_API_KEY", "env-key")

		model := &Embeddings{BaseURL: server.URL}
		if _, err := model.EmbedQuery(t.Context(), "q"); err != nil {
			t.Fatalf("embed query: %v", err)
		}
		if server.lastAuth != "Bearer env-key" {
			t.Fatalf("auth header: got %q want Bearer env-key", server.lastAuth)
		}
	})

	t.Run("explicit field wins over env", func(t *testing.T) {
		server := newEmbedServer(t, `{"embeddings":[[1.0]],"usage":{}}`)
		t.Setenv("NOMIC_API_KEY", "env-key")

		model := &Embeddings{APIKey: "field-key", BaseURL: server.URL}
		if _, err := model.EmbedQuery(t.Context(), "q"); err != nil {
			t.Fatalf("embed query: %v", err)
		}
		if server.lastAuth != "Bearer field-key" {
			t.Fatalf("auth header: got %q want Bearer field-key", server.lastAuth)
		}
	})

	t.Run("missing key errors without a request", func(t *testing.T) {
		server := newEmbedServer(t, `{"embeddings":[[1.0]],"usage":{}}`)
		t.Setenv("NOMIC_API_KEY", "")

		model := &Embeddings{BaseURL: server.URL}
		_, err := model.EmbedDocuments(t.Context(), []string{"doc"})
		if err == nil {
			t.Fatal("missing API key must error")
		}
		if !errors.Is(err, lcerrors.ErrInvalidInput) {
			t.Fatalf("error: got %v want lcerrors.ErrInvalidInput", err)
		}
		if !strings.Contains(err.Error(), "NOMIC_API_KEY") {
			t.Fatalf("error must mention NOMIC_API_KEY: %v", err)
		}
		if server.requestCount() != 0 {
			t.Fatalf("missing API key must not reach the API, got %d requests", server.requestCount())
		}
	})
}

// Non-2xx responses surface as typed provider errors: 429 classifies as a
// rate limit, 5xx as a server failure (core/lcerrors #39538 mapping).
func TestProviderErrorMapping(t *testing.T) {
	cases := []struct {
		name   string
		status int
		want   error
	}{
		{"rate limited", http.StatusTooManyRequests, lcerrors.ErrModelRateLimit},
		{"server error", http.StatusInternalServerError, lcerrors.ErrModelServer},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := newEmbedServer(t, "")
			server.setStatus(tc.status)

			model := &Embeddings{APIKey: "key", BaseURL: server.URL}
			_, err := model.EmbedDocuments(t.Context(), []string{"doc"})
			if !errors.Is(err, tc.want) {
				t.Fatalf("error: got %v want %v", err, tc.want)
			}
			providerErr, ok := errors.AsType[*lcerrors.ProviderError](err)
			if !ok {
				t.Fatalf("want provider error, got %v", err)
			}
			if providerErr.Provider != "nomic" {
				t.Fatalf("provider: got %q want nomic", providerErr.Provider)
			}
			if providerErr.Endpoint != "/embedding/text" {
				t.Fatalf("endpoint: got %q want /embedding/text", providerErr.Endpoint)
			}
			if providerErr.StatusCode != tc.status {
				t.Fatalf("status: got %d want %d", providerErr.StatusCode, tc.status)
			}
		})
	}
}

// A 200 response with a malformed body must fail instead of returning zero
// vectors.
func TestMalformedResponseErrors(t *testing.T) {
	server := newEmbedServer(t, "not-json")

	model := &Embeddings{APIKey: "key", BaseURL: server.URL}
	if _, err := model.EmbedDocuments(t.Context(), []string{"doc"}); err == nil {
		t.Fatal("malformed response must error")
	}
}

// EmbedQuery unwraps the first vector, so an empty embeddings array must fail
// rather than panic.
func TestEmbedQueryEmptyResponseErrors(t *testing.T) {
	server := newEmbedServer(t, `{"embeddings":[],"usage":{}}`)

	model := &Embeddings{APIKey: "key", BaseURL: server.URL}
	_, err := model.EmbedQuery(t.Context(), "q")
	if err == nil {
		t.Fatal("empty embeddings response must error for a query")
	}
	if !errors.Is(err, lcerrors.ErrSchemaValidation) {
		t.Fatalf("error: got %v want lcerrors.ErrSchemaValidation", err)
	}
}

// Divergence: the zero-value struct resolves model/base URL defaults without
// touching the network.
func TestEmbeddingsDefaults(t *testing.T) {
	model := &Embeddings{}
	if got := model.resolvedModel(); got != defaultModel {
		t.Fatalf("model: got %q want %q", got, defaultModel)
	}
	if got := model.resolvedBaseURL(); got != defaultBaseURL {
		t.Fatalf("base URL: got %q want %q", got, defaultBaseURL)
	}

	model = &Embeddings{Model: "  ", BaseURL: "  "}
	if got := model.resolvedModel(); got != defaultModel {
		t.Fatalf("blank model: got %q want %q", got, defaultModel)
	}
	if got := model.resolvedBaseURL(); got != defaultBaseURL {
		t.Fatalf("blank base URL: got %q want %q", got, defaultBaseURL)
	}

	model = &Embeddings{Model: "nomic-embed-text-v1", BaseURL: "http://example.test"}
	if got := model.resolvedModel(); got != "nomic-embed-text-v1" {
		t.Fatalf("model: got %q", got)
	}
	if got := model.resolvedBaseURL(); got != "http://example.test" {
		t.Fatalf("base URL: got %q", got)
	}
}
