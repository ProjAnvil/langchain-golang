package typesafe

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

// Client tests mirror langchain_typesafe classifier.py + client.py (#40542,
// #40570): POST {base}/v1/systemone with state/model/questions, Bearer
// auth, and the Go User-Agent divergence
// "langchain-typesafe-go/0.0.1" (DIVERGENCES #5).

func testClassifier(t *testing.T, baseURL string, httpClient *http.Client) *Classifier {
	t.Helper()
	return &Classifier{
		Model:      "jev-latest",
		APIKey:     "test-key",
		BaseURL:    baseURL,
		HTTPClient: httpClient,
		Timeout:    10 * time.Second,
	}
}

func TestClassifierInvokeGolden(t *testing.T) {
	var gotMethod, gotPath string
	var gotAuth, gotContentType, gotUserAgent string
	var gotBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		gotUserAgent = r.Header.Get("User-Agent")
		buf := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(buf)
		gotBody = string(buf)
		w.Header().Set("x-typesafe-request-id", "req-42")
		_, _ = w.Write([]byte(`{
			"model": "jev-latest",
			"answers": {
				"urgent": {"type": "noul", "noul": 0.75},
				"route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "technical": 0.2}, "confidence": 0.8},
				"frustration": {"type": "score", "score": 1.25, "legend": {"0": "calm", "1": "angry"}, "probabilities": {"0": 0.75, "1": 0.25}, "confidence": 0.6}
			},
			"usage": {"input_tokens": 3, "output_tokens": 5}
		}`))
	}))
	defer server.Close()

	classifier := testClassifier(t, server.URL, server.Client())
	response, err := classifier.Invoke(t.Context(), Request{
		State: []messages.Message{messages.Human("Stripe fails.")},
		Questions: map[string]Question{
			"urgent": Noul{Instructions: "Urgent?", Criteria: &NoulCriteria{True: "t", False: "f"}},
			"route": Choice{
				Instructions: "Which?",
				Criteria:     []ChoiceCriterion{{Label: "billing", Description: "B."}, {Label: "technical", Description: "T."}},
			},
			"frustration": Score{Instructions: "How mad?", Criteria: []any{"calm", "angry"}},
		},
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if gotMethod != http.MethodPost {
		t.Errorf("method = %s, want POST", gotMethod)
	}
	if gotPath != "/v1/systemone" {
		t.Errorf("path = %s, want /v1/systemone", gotPath)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("authorization = %q", gotAuth)
	}
	if gotContentType != "application/json" {
		t.Errorf("content-type = %q", gotContentType)
	}
	if gotUserAgent != "langchain-typesafe-go/0.0.1" {
		t.Errorf("user-agent = %q", gotUserAgent)
	}

	// Questions are keyed by name in stable sorted order; each question's
	// wire object follows the pydantic field declaration order.
	wantBody := `{"state":[{"content":"Stripe fails.","role":"user"}],"model":"jev-latest",` +
		`"questions":{` +
		`"frustration":{"type":"score","criteria":["calm","angry"],"instructions":"How mad?"},` +
		`"route":{"type":"choice","criteria":{"billing":"B.","technical":"T."},"instructions":"Which?"},` +
		`"urgent":{"type":"noul","instructions":"Urgent?","criteria":{"true":"t","false":"f"}}}}`
	if strings.TrimSpace(gotBody) != wantBody {
		t.Errorf("request body mismatch:\n got %s\nwant %s", gotBody, wantBody)
	}

	if response.Model != "jev-latest" {
		t.Errorf("model = %s", response.Model)
	}
	if response.RequestID != "req-42" {
		t.Errorf("request id = %q", response.RequestID)
	}
	if response.Usage.InputTokens == nil || *response.Usage.InputTokens != 3 {
		t.Errorf("usage input tokens = %v", response.Usage.InputTokens)
	}
	if response.Usage.OutputTokens == nil || *response.Usage.OutputTokens != 5 {
		t.Errorf("usage output tokens = %v", response.Usage.OutputTokens)
	}
	if nouls := response.Nouls(); nouls["urgent"].Noul != 0.75 {
		t.Errorf("noul answer = %#v", nouls)
	}
	choice := response.Choices()["route"]
	if choice.Choice != "billing" || choice.Confidence != 0.8 || choice.Probabilities["technical"] != 0.2 {
		t.Errorf("choice answer = %#v", choice)
	}
	score := response.Scores()["frustration"]
	if score.Score != 1.25 || score.Legend[0] != "calm" || score.Probabilities[1] != 0.25 || score.Confidence != 0.6 {
		t.Errorf("score answer = %#v", score)
	}
}

func TestClassifierInvokeStateError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("server should not be called for invalid state")
	}))
	defer server.Close()

	classifier := testClassifier(t, server.URL, server.Client())
	if _, err := classifier.Invoke(t.Context(), Request{State: 42}); err == nil {
		t.Fatal("expected state validation error")
	}
}

func TestClassifierErrorTable(t *testing.T) {
	cases := []struct {
		name         string
		status       int
		body         string
		wantSentinel error
		wantKind     lcerrors.ModelErrorKind
	}{
		{name: "400", status: 400, body: `{"error":"bad"}`, wantSentinel: lcerrors.ErrModelInvalidRequest, wantKind: lcerrors.ModelKindInvalidRequest},
		{name: "400 overflow marker", status: 400, body: `{"error":"context_length_exceeded"}`, wantSentinel: lcerrors.ErrModelContextOverflow, wantKind: lcerrors.ModelKindContextOverflow},
		{name: "401", status: 401, body: `{}`, wantSentinel: lcerrors.ErrModelAuth, wantKind: lcerrors.ModelKindAuth},
		{name: "403", status: 403, body: `{}`, wantSentinel: lcerrors.ErrModelPermissionDenied, wantKind: lcerrors.ModelKindPermissionDenied},
		{name: "404", status: 404, body: `{}`, wantSentinel: lcerrors.ErrModelNotFound, wantKind: lcerrors.ModelKindNotFound},
		{name: "422", status: 422, body: `{}`, wantSentinel: lcerrors.ErrModelInvalidRequest, wantKind: lcerrors.ModelKindInvalidRequest},
		{name: "429", status: 429, body: `{}`, wantSentinel: lcerrors.ErrModelRateLimit, wantKind: lcerrors.ModelKindRateLimit},
		{name: "500", status: 500, body: `{}`, wantSentinel: lcerrors.ErrModelServer, wantKind: lcerrors.ModelKindServer},
		{name: "503", status: 503, body: `{}`, wantSentinel: lcerrors.ErrModelServer, wantKind: lcerrors.ModelKindServer},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("x-typesafe-request-id", "req-err")
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()

			classifier := testClassifier(t, server.URL, server.Client())
			_, err := classifier.Invoke(t.Context(), Request{
				State:     "hello",
				Questions: map[string]Question{"q": Noul{Instructions: "i"}},
			})
			if err == nil {
				t.Fatal("expected APIError")
			}
			var apiErr *APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("expected *APIError, got %T: %v", err, err)
			}
			if !errors.Is(err, tc.wantSentinel) {
				t.Errorf("sentinel mismatch: %v", err)
			}
			var pe *lcerrors.ProviderError
			if !errors.As(err, &pe) {
				t.Fatal("errors.As(*ProviderError) failed")
			}
			if pe.ModelKind != tc.wantKind {
				t.Errorf("kind = %s, want %s", pe.ModelKind, tc.wantKind)
			}
			if apiErr.RequestID != "req-err" {
				t.Errorf("request id = %q", apiErr.RequestID)
			}
			// Endpoint is the sanitized method + URL of the request.
			wantEndpoint := fmt.Sprintf("POST %s/v1/systemone", server.URL)
			if pe.Endpoint != wantEndpoint {
				t.Errorf("endpoint = %q, want %q", pe.Endpoint, wantEndpoint)
			}
			// The body stays inspectable via ProviderError but never leaks
			// through Error().
			if !strings.Contains(pe.Body, tc.body) {
				t.Errorf("provider body = %q", pe.Body)
			}
		})
	}
}

func TestClassifierRetryAfter(t *testing.T) {
	future := time.Now().Add(2 * time.Hour).UTC().Format(http.TimeFormat)
	past := time.Now().Add(-2 * time.Hour).UTC().Format(http.TimeFormat)
	cases := []struct {
		name        string
		headers     map[string]string
		want        time.Duration
		wantNonZero bool
	}{
		{name: "ms header wins", headers: map[string]string{"retry-after-ms": "500", "retry-after": "9"}, want: 500 * time.Millisecond},
		{name: "seconds header", headers: map[string]string{"retry-after": "2"}, want: 2 * time.Second},
		{name: "fractional seconds", headers: map[string]string{"retry-after": "1.5"}, want: 1500 * time.Millisecond},
		{name: "ms garbage falls through to seconds", headers: map[string]string{"retry-after-ms": "soon", "retry-after": "3"}, want: 3 * time.Second},
		{name: "empty string is zero", headers: map[string]string{"retry-after": ""}, want: 0},
		{name: "empty ms string is zero", headers: map[string]string{"retry-after-ms": ""}, want: 0},
		{name: "negative ignored", headers: map[string]string{"retry-after": "-5"}, want: 0},
		{name: "negative ms falls through then none", headers: map[string]string{"retry-after-ms": "-5"}, want: 0},
		{name: "invalid header ignored", headers: map[string]string{"retry-after": "garbage"}, want: 0},
		{name: "http date future", headers: map[string]string{"retry-after": future}, wantNonZero: true},
		{name: "http date past clamped to zero", headers: map[string]string{"retry-after": past}, want: 0},
		{name: "no headers", headers: map[string]string{}, want: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				for k, v := range tc.headers {
					w.Header().Set(k, v)
				}
				w.WriteHeader(http.StatusTooManyRequests)
			}))
			defer server.Close()

			classifier := testClassifier(t, server.URL, server.Client())
			_, err := classifier.Invoke(t.Context(), Request{
				State:     "s",
				Questions: map[string]Question{"q": Noul{Instructions: "i"}},
			})
			var apiErr *APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("expected APIError, got %v", err)
			}
			got := apiErr.RetryAfter
			if tc.wantNonZero {
				if got <= 0 || got > 3*time.Hour {
					t.Errorf("retry-after = %v, want positive future delta", got)
				}
				return
			}
			if got != tc.want {
				t.Errorf("retry-after = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestClassifierRetryAfterIgnoredForNon429(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("retry-after-ms", "500")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	classifier := testClassifier(t, server.URL, server.Client())
	_, err := classifier.Invoke(t.Context(), Request{
		State:     "s",
		Questions: map[string]Question{"q": Noul{Instructions: "i"}},
	})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %v", err)
	}
	if apiErr.RetryAfter != 0 {
		t.Errorf("retry-after parsed for non-429 status: %v", apiErr.RetryAfter)
	}
}

func TestClassifierTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(300 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	classifier := testClassifier(t, server.URL, &http.Client{Timeout: 40 * time.Millisecond})
	_, err := classifier.Invoke(t.Context(), Request{
		State:     "s",
		Questions: map[string]Question{"q": Noul{Instructions: "i"}},
	})
	if !errors.Is(err, lcerrors.ErrModelTimeout) {
		t.Fatalf("expected ErrModelTimeout, got %v", err)
	}
	if got, want := err.Error(), "Request timed out (timeout=40ms)."; got != want {
		t.Errorf("timeout message mismatch:\n got %q\nwant %q", got, want)
	}
}

func TestClassifierConnectionError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	server.Close() // immediately close so the port refuses connections

	classifier := testClassifier(t, server.URL, &http.Client{Timeout: 2 * time.Second})
	_, err := classifier.Invoke(t.Context(), Request{
		State:     "s",
		Questions: map[string]Question{"q": Noul{Instructions: "i"}},
	})
	if !errors.Is(err, lcerrors.ErrModelConnection) {
		t.Fatalf("expected ErrModelConnection, got %v", err)
	}
	if errors.Is(err, lcerrors.ErrModelTimeout) {
		t.Errorf("connection failure misclassified as timeout: %v", err)
	}
}

func TestClassifierResponseValidation(t *testing.T) {
	cases := []struct {
		name      string
		body      string
		wantField string
	}{
		{name: "not json", body: `oops`, wantField: "response"},
		{name: "json array", body: `[]`, wantField: "response"},
		{name: "missing model", body: `{"answers":{}}`, wantField: "model"},
		{name: "model wrong type", body: `{"model":5,"answers":{}}`, wantField: "model"},
		{name: "missing answers", body: `{"model":"m"}`, wantField: "answers"},
		{name: "answers wrong type", body: `{"model":"m","answers":[]}`, wantField: "answers"},
		{name: "unknown answer type", body: `{"model":"m","answers":{"q":{"type":"weird"}}}`, wantField: "answers.q.type"},
		{name: "answer not object", body: `{"model":"m","answers":{"q":"x"}}`, wantField: "answers.q"},
		{name: "noul missing value", body: `{"model":"m","answers":{"q":{"type":"noul"}}}`, wantField: "answers.q.noul"},
		{name: "noul out of range", body: `{"model":"m","answers":{"q":{"type":"noul","noul":1.5}}}`, wantField: "answers.q.noul"},
		{name: "choice missing confidence", body: `{"model":"m","answers":{"q":{"type":"choice","choice":"a","probabilities":{"a":1}}}}`, wantField: "answers.q.confidence"},
		{name: "choice probabilities wrong type", body: `{"model":"m","answers":{"q":{"type":"choice","choice":"a","probabilities":5,"confidence":0.5}}}`, wantField: "answers.q.probabilities"},
		{name: "score legend non numeric key", body: `{"model":"m","answers":{"q":{"type":"score","score":1,"legend":{"x":"d"},"probabilities":{"0":1},"confidence":0.5}}}`, wantField: "answers.q.legend"},
		{name: "score missing score", body: `{"model":"m","answers":{"q":{"type":"score","legend":{"0":"d"},"probabilities":{"0":1},"confidence":0.5}}}`, wantField: "answers.q.score"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()

			classifier := testClassifier(t, server.URL, server.Client())
			_, err := classifier.Invoke(t.Context(), Request{
				State:     "s",
				Questions: map[string]Question{"q": Noul{Instructions: "i"}},
			})
			var validationErr *ResponseValidationError
			if !errors.As(err, &validationErr) {
				t.Fatalf("expected ResponseValidationError, got %T: %v", err, err)
			}
			if validationErr.FieldPath != tc.wantField {
				t.Errorf("field path = %q, want %q", validationErr.FieldPath, tc.wantField)
			}
			if !errors.Is(err, lcerrors.ErrSchemaValidation) {
				t.Error("response validation error does not wrap ErrSchemaValidation")
			}
		})
	}
}

func TestClassifierUsageDefaults(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"jev-latest","answers":{"q":{"type":"noul","noul":0.1}}}`))
	}))
	defer server.Close()

	classifier := testClassifier(t, server.URL, server.Client())
	response, err := classifier.Invoke(t.Context(), Request{
		State:     "s",
		Questions: map[string]Question{"q": Noul{Instructions: "i"}},
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if response.Usage.InputTokens != nil || response.Usage.OutputTokens != nil {
		t.Errorf("usage should default to nil, got %#v", response.Usage)
	}
	if response.RequestID != "" {
		t.Errorf("request id should be empty, got %q", response.RequestID)
	}
}

func TestNewClassifierDefaults(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "env-key")
	t.Setenv("TYPESAFE_BASE_URL", "")

	classifier, err := NewClassifier()
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	if classifier.Model != "jev-latest" {
		t.Errorf("model = %s", classifier.Model)
	}
	if classifier.APIKey != "env-key" {
		t.Errorf("api key = %q", classifier.APIKey)
	}
	if classifier.BaseURL != "https://api.typesafe.ai" {
		t.Errorf("base url = %s", classifier.BaseURL)
	}
	if classifier.Timeout != 30*time.Second {
		t.Errorf("timeout = %s", classifier.Timeout)
	}
	if classifier.HTTPClient != nil {
		t.Error("http client should default to nil (built per request)")
	}
}

func TestNewClassifierEnvBaseURL(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "env-key")
	t.Setenv("TYPESAFE_BASE_URL", "https://proxy.example.com/")

	classifier, err := NewClassifier()
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	if classifier.BaseURL != "https://proxy.example.com/" {
		t.Errorf("base url = %s", classifier.BaseURL)
	}
}

func TestNewClassifierMissingAPIKey(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	_, err := NewClassifier()
	if err == nil {
		t.Fatal("expected missing key error")
	}
	want := "TypeSafe API key is required. Pass `api_key` or set `TYPESAFE_API_KEY`."
	if err.Error() != want {
		t.Errorf("error mismatch:\n got %q\nwant %q", err.Error(), want)
	}
}

func TestNewClassifierWhitespaceAPIKey(t *testing.T) {
	if _, err := NewClassifier(WithAPIKey("   ")); err == nil {
		t.Fatal("expected whitespace key error")
	}
}

func TestNewClassifierModelValidation(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "env-key")
	if _, err := NewClassifier(WithModel("   ")); err == nil {
		t.Fatal("expected empty model error")
	} else if err.Error() != "TypeSafe model must not be empty." {
		t.Errorf("error mismatch: %q", err.Error())
	}

	// Model values are stripped (classifier.py:246-253).
	classifier, err := NewClassifier(WithModel("  jev-mini  "))
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	if classifier.Model != "jev-mini" {
		t.Errorf("model = %q, want jev-mini", classifier.Model)
	}
}

func TestNewClassifierDefaultsWithoutEnv(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	t.Setenv("TYPESAFE_BASE_URL", "https://from-env.example.com")
	classifier := &Classifier{Model: "m", APIKey: "k"}
	if err := applyDefaults(classifier); err != nil {
		t.Fatalf("applyDefaults: %v", err)
	}
	if classifier.BaseURL != "https://from-env.example.com" {
		t.Errorf("base url = %s", classifier.BaseURL)
	}
	if classifier.Timeout != 30*time.Second {
		t.Errorf("timeout = %s", classifier.Timeout)
	}
}

func TestClassifierOptions(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	httpClient := &http.Client{Timeout: 5 * time.Second}
	classifier, err := NewClassifier(
		WithModel("jev-mini"),
		WithAPIKey("explicit"),
		WithBaseURL("https://elsewhere.example.com"),
		WithHTTPClient(httpClient),
		WithTimeout(15*time.Second),
	)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	if classifier.Model != "jev-mini" || classifier.APIKey != "explicit" ||
		classifier.BaseURL != "https://elsewhere.example.com" ||
		classifier.HTTPClient != httpClient || classifier.Timeout != 15*time.Second {
		t.Errorf("options not applied: %#v", classifier)
	}
}

func TestClassifierEffectiveTimeout(t *testing.T) {
	c := &Classifier{Timeout: 5 * time.Second}
	if got := c.effectiveTimeout(&http.Client{}); got != 5*time.Second {
		t.Errorf("injected client without timeout: %s", got)
	}
	if got := c.effectiveTimeout(&http.Client{Timeout: 2 * time.Second}); got != 2*time.Second {
		t.Errorf("injected client with timeout: %s", got)
	}
	c.HTTPClient = nil
	if got := c.effectiveTimeout(nil); got != 5*time.Second {
		t.Errorf("classifier-created client: %s", got)
	}
}

func TestClassifierBadURL(t *testing.T) {
	classifier := testClassifier(t, "://not-a-url", nil)
	_, err := classifier.Invoke(t.Context(), Request{
		State:     "s",
		Questions: map[string]Question{"q": Noul{Instructions: "i"}},
	})
	if !errors.Is(err, lcerrors.ErrModelConnection) {
		t.Fatalf("expected connection error for invalid URL, got %v", err)
	}
}

func TestClassifierContextDeadlineTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(300 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(t.Context(), 40*time.Millisecond)
	defer cancel()

	classifier := testClassifier(t, server.URL, server.Client())
	_, err := classifier.Invoke(ctx, Request{
		State:     "s",
		Questions: map[string]Question{"q": Noul{Instructions: "i"}},
	})
	if !errors.Is(err, lcerrors.ErrModelTimeout) {
		t.Fatalf("expected ErrModelTimeout, got %v", err)
	}
}

func TestSanitizedEndpoint(t *testing.T) {
	resp := &http.Response{Request: &http.Request{
		Method: http.MethodPost,
		URL:    mustParseURL("https://api.typesafe.ai/v1/systemone?foo=1#frag"),
	}}
	if got, want := sanitizedEndpoint(resp), "POST https://api.typesafe.ai/v1/systemone"; got != want {
		t.Errorf("plain endpoint = %q, want %q", got, want)
	}

	resp = &http.Response{Request: &http.Request{
		Method: http.MethodPost,
		// Userinfo must be stripped, IPv6 host bracketed, port kept.
		URL: mustParseURL("http://user:secret@[::1]:8443/v1/systemone?x=1"),
	}}
	if got, want := sanitizedEndpoint(resp), "POST http://[::1]:8443/v1/systemone"; got != want {
		t.Errorf("ipv6 endpoint = %q, want %q", got, want)
	}

	resp = &http.Response{Request: nil}
	if got := sanitizedEndpoint(resp); got != "" {
		t.Errorf("nil request endpoint = %q, want empty", got)
	}
}

func mustParseURL(raw string) *url.URL {
	parsed, err := url.Parse(raw)
	if err != nil {
		panic(err)
	}
	return parsed
}

func TestParseResponseScoreProbabilityKeyAndUsageNull(t *testing.T) {
	// Non-numeric score probability keys fail at answers.<name>.probabilities.
	bad := `{"model":"m","answers":{"q":{"type":"score","score":1,"legend":{"0":"d"},"probabilities":{"x":1},"confidence":0.5}}}`
	if _, err := parseResponse(&http.Response{}, []byte(bad)); err == nil {
		t.Error("expected error for non-numeric score probability key")
	} else {
		var validationErr *ResponseValidationError
		if !errors.As(err, &validationErr) || validationErr.FieldPath != "answers.q.probabilities" {
			t.Errorf("field path mismatch: %v", err)
		}
	}

	good := `{"model":"m","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":null,"output_tokens":null}}`
	response, err := parseResponse(&http.Response{}, []byte(good))
	if err != nil {
		t.Fatalf("parseResponse: %v", err)
	}
	if response.Usage.InputTokens != nil || response.Usage.OutputTokens != nil {
		t.Errorf("null usage tokens should stay nil: %#v", response.Usage)
	}
}

func TestIsTimeoutErrorPlainError(t *testing.T) {
	if isTimeoutError(errors.New("plain")) {
		t.Error("plain error misclassified as timeout")
	}
	if isTimeoutError(nil) {
		t.Error("nil error misclassified as timeout")
	}
}

func TestSanitizedEndpointNoHost(t *testing.T) {
	resp := &http.Response{Request: &http.Request{
		Method: http.MethodPost,
		URL:    mustParseURL("mailto:support@typesafe.ai"),
	}}
	if got := sanitizedEndpoint(resp); got != "" {
		t.Errorf("hostless endpoint = %q, want empty", got)
	}
}

func TestParseAnswerEdgeBranches(t *testing.T) {
	// Choice probability value that is not a number.
	bad := `{"model":"m","answers":{"q":{"type":"choice","choice":"a","probabilities":{"a":"high"},"confidence":0.5}}}`
	_, err := parseResponse(&http.Response{}, []byte(bad))
	var validationErr *ResponseValidationError
	if !errors.As(err, &validationErr) || validationErr.FieldPath != "answers.q.probabilities" {
		t.Errorf("field path mismatch: %v", err)
	}

	// Choice missing the selected label.
	bad = `{"model":"m","answers":{"q":{"type":"choice","probabilities":{"a":1},"confidence":0.5}}}`
	if _, err = parseResponse(&http.Response{}, []byte(bad)); err == nil {
		t.Error("expected error for missing choice label")
	}
}
