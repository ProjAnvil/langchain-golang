package httpclient

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

type widget struct {
	Name string `json:"name"`
}

func cfg(serverURL string, retries int, opts ...func(*modelconfig.Config)) modelconfig.Config {
	c := modelconfig.New(
		modelconfig.WithBaseURL(serverURL),
		modelconfig.WithMaxRetries(retries),
	)
	for _, opt := range opts {
		opt(&c)
	}
	return c
}

func TestPostJSONDecodesSuccess(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q", ct)
		}
		if r.Header.Get("X-Provider-Token") != "secret" {
			t.Errorf("configure hook did not set provider header")
		}
		_ = json.NewEncoder(w).Encode(widget{Name: "ok"})
	}))
	defer server.Close()

	got, err := PostJSON[widget](context.Background(), "openai", cfg(server.URL, 0), "/x", map[string]any{"q": 1},
		func(req *http.Request) { req.Header.Set("X-Provider-Token", "secret") })
	if err != nil {
		t.Fatalf("PostJSON: %v", err)
	}
	if got.Name != "ok" {
		t.Fatalf("decoded = %+v", got)
	}
}

func TestPostJSONRetriesThenSucceeds(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			http.Error(w, "slow down", http.StatusTooManyRequests)
			return
		}
		_ = json.NewEncoder(w).Encode(widget{Name: "ok"})
	}))
	defer server.Close()

	got, err := PostJSON[widget](context.Background(), "openai", cfg(server.URL, 2), "/x", nil, nil)
	if err != nil {
		t.Fatalf("PostJSON: %v", err)
	}
	if got.Name != "ok" || attempts != 2 {
		t.Fatalf("attempts=%d got=%+v", attempts, got)
	}
}

func TestPostJSONRateLimitedExhaustsRetries(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "2")
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer server.Close()

	_, err := PostJSON[widget](context.Background(), "openai", cfg(server.URL, 1), "/x", nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, lcerrors.ErrRateLimited) {
		t.Fatalf("err not ErrRateLimited: %v", err)
	}
	var pe *lcerrors.ProviderError
	if !errors.As(err, &pe) || pe.RetryAfter != 2*time.Second {
		t.Fatalf("RetryAfter = %v", pe.RetryAfter)
	}
	if !IsRetryable(err) {
		t.Fatal("429 should be retryable")
	}
}

func TestPostJSONServerErrorIsProviderAndRetryable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadGateway)
	}))
	defer server.Close()

	_, err := PostJSON[widget](context.Background(), "anthropic", cfg(server.URL, 0), "/x", nil, nil)
	if !errors.Is(err, lcerrors.ErrProvider) {
		t.Fatalf("err not ErrProvider: %v", err)
	}
	if errors.Is(err, lcerrors.ErrRateLimited) {
		t.Fatal("502 must not be ErrRateLimited")
	}
	if !IsRetryable(err) {
		t.Fatal("502 should be retryable")
	}
}

func TestPostJSONBadRequestIsProviderNotRetryable(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		http.Error(w, "bad", http.StatusBadRequest)
	}))
	defer server.Close()

	_, err := PostJSON[widget](context.Background(), "ollama", cfg(server.URL, 3), "/x", nil, nil)
	if !errors.Is(err, lcerrors.ErrProvider) {
		t.Fatalf("err not ErrProvider: %v", err)
	}
	if attempts != 1 {
		t.Fatalf("400 must not retry, attempts=%d", attempts)
	}
	if IsRetryable(err) {
		t.Fatal("400 should not be retryable")
	}
}

func TestPostJSONTimeoutIsErrTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
	}))
	defer server.Close()

	clientCfg := cfg(server.URL, 0, func(c *modelconfig.Config) {
		c.HTTPClient = &http.Client{Timeout: time.Millisecond}
	})
	_, err := PostJSON[widget](context.Background(), "openai", clientCfg, "/x", nil, nil)
	if !errors.Is(err, lcerrors.ErrTimeout) {
		t.Fatalf("err not ErrTimeout: %v", err)
	}
	if !IsRetryable(err) {
		t.Fatal("timeout should be retryable")
	}
}

func TestResponseErrorClassifiesNon2xx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"nope"}`, http.StatusUnauthorized)
	}))
	defer server.Close()

	resp, err := http.Post(server.URL+"/x", "application/json", nil)
	if err != nil {
		t.Fatalf("http.Post: %v", err)
	}
	got := ResponseError("openai", "/x", resp)
	if !errors.Is(got, lcerrors.ErrProvider) {
		t.Fatalf("err not ErrProvider: %v", got)
	}
	var pe *lcerrors.ProviderError
	if !errors.As(got, &pe) || pe.StatusCode != http.StatusUnauthorized || pe.Body == "" {
		t.Fatalf("ProviderError = %+v", pe)
	}
}

func TestRetryAfterHeader(t *testing.T) {
	resp := &http.Response{Header: http.Header{}}
	resp.Header.Set("Retry-After", "3")
	if d := RetryAfter(resp); d != 3*time.Second {
		t.Fatalf("delta-seconds = %v", d)
	}
	resp.Header.Set("Retry-After", "Wed, 21 Oct 2015 07:28:00 GMT")
	_ = RetryAfter(resp) // must not panic; value depends on clock
	resp.Header.Del("Retry-After")
	if d := RetryAfter(resp); d != 0 {
		t.Fatalf("absent = %v", d)
	}
}

func TestRetryAfterEdgeCases(t *testing.T) {
	resp := &http.Response{Header: http.Header{}}

	resp.Header.Set("Retry-After", "-5")
	if d := RetryAfter(resp); d != 0 {
		t.Fatalf("negative delta-seconds = %v, want 0", d)
	}

	resp.Header.Set("Retry-After", "not-a-date")
	if d := RetryAfter(resp); d != 0 {
		t.Fatalf("unparseable = %v, want 0", d)
	}

	future := time.Now().Add(time.Minute).UTC().Format(http.TimeFormat)
	resp.Header.Set("Retry-After", future)
	if d := RetryAfter(resp); d <= 0 || d > time.Minute {
		t.Fatalf("future HTTP-date = %v, want within (0, 1m]", d)
	}

	past := time.Now().Add(-time.Minute).UTC().Format(http.TimeFormat)
	resp.Header.Set("Retry-After", past)
	if d := RetryAfter(resp); d != 0 {
		t.Fatalf("past HTTP-date = %v, want 0", d)
	}
}

func TestPostJSONMarshalError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server must not be reached when the payload cannot be marshaled")
	}))
	defer server.Close()

	_, err := PostJSON[widget](context.Background(), "openai", cfg(server.URL, 3), "/x",
		make(chan int), nil)
	if err == nil {
		t.Fatal("expected marshal error")
	}
}

func TestPostJSONInvalidEndpointURL(t *testing.T) {
	// A '%' in BaseURL produces a URL the request constructor rejects.
	c := modelconfig.New(modelconfig.WithBaseURL("%"), modelconfig.WithMaxRetries(3))
	_, err := PostJSON[widget](context.Background(), "openai", c, "/x", nil, nil)
	if err == nil {
		t.Fatal("expected URL construction error")
	}
	if IsRetryable(err) {
		t.Fatal("URL construction error must not be retried")
	}
}

func TestPostJSONDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("{not json"))
	}))
	defer server.Close()

	_, err := PostJSON[widget](context.Background(), "openai", cfg(server.URL, 0), "/x", nil, nil)
	if err == nil {
		t.Fatal("expected decode error")
	}
	if errors.Is(err, lcerrors.ErrProvider) {
		t.Fatalf("decode failure must not be a provider error: %v", err)
	}
}

// errReader fails on every Read to exercise the response-body error paths.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("read failure") }
func (errReader) Close() error             { return nil }

type errBodyTransport struct{}

func (errBodyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{},
		Body:       errReader{},
		Request:    req,
	}, nil
}

func TestPostJSONBodyReadError(t *testing.T) {
	c := modelconfig.New(
		modelconfig.WithBaseURL("http://example.invalid"),
		modelconfig.WithMaxRetries(0),
	)
	c.HTTPClient = &http.Client{Transport: errBodyTransport{}}

	_, err := PostJSON[widget](context.Background(), "openai", c, "/x", nil, nil)
	if err == nil {
		t.Fatal("expected body read error")
	}
}

func TestResponseErrorBodyReadError(t *testing.T) {
	resp := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Header:     http.Header{},
		Body:       errReader{},
	}
	err := ResponseError("openai", "/x", resp)
	if err == nil {
		t.Fatal("expected error from unreadable body")
	}
	var pe *lcerrors.ProviderError
	if errors.As(err, &pe) {
		t.Fatalf("read failure must not become a ProviderError: %v", err)
	}
}
