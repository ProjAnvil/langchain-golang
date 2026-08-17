package modelconfig

import (
	"net/http"
	"testing"
	"time"
)

func TestNewDefaults(t *testing.T) {
	cfg := New()

	if cfg.MaxRetries != 2 {
		t.Fatalf("default max retries: got %d", cfg.MaxRetries)
	}
	if cfg.Headers == nil {
		t.Fatal("default headers map must be non-nil")
	}
	if cfg.Extra == nil {
		t.Fatal("default extra map must be non-nil")
	}
	if cfg.Temperature != nil {
		t.Fatalf("default temperature: got %v", *cfg.Temperature)
	}
	if cfg.MaxTokens != nil {
		t.Fatalf("default max tokens: got %v", *cfg.MaxTokens)
	}
}

func TestWithBaseURL(t *testing.T) {
	cfg := New(WithBaseURL("https://example.test/v1"))
	if cfg.BaseURL != "https://example.test/v1" {
		t.Fatalf("base url: got %q", cfg.BaseURL)
	}
}

func TestWithHTTPClient(t *testing.T) {
	client := &http.Client{Timeout: 42 * time.Second}
	cfg := New(WithHTTPClient(client))
	if cfg.HTTPClient != client {
		t.Fatalf("http client: got %p, want %p", cfg.HTTPClient, client)
	}
}

func TestWithHeaderOnNilMap(t *testing.T) {
	var cfg Config
	WithHeader("X-Key", "value")(&cfg)
	if cfg.Headers["X-Key"] != "value" {
		t.Fatalf("header: got %q", cfg.Headers["X-Key"])
	}
}

func TestWithExtraOnNilMap(t *testing.T) {
	var cfg Config
	WithExtra("seed", 42)(&cfg)
	if cfg.Extra["seed"] != 42 {
		t.Fatalf("extra: got %v", cfg.Extra["seed"])
	}
}

func TestOptionsApplyInOrder(t *testing.T) {
	cfg := New(
		WithModel("first"),
		WithModel("second"),
		WithHeader("X-Dup", "a"),
		WithHeader("X-Dup", "b"),
	)
	if cfg.Model != "second" {
		t.Fatalf("model: got %q", cfg.Model)
	}
	if cfg.Headers["X-Dup"] != "b" {
		t.Fatalf("header: got %q", cfg.Headers["X-Dup"])
	}
}

func TestCloneOfZeroConfig(t *testing.T) {
	var cfg Config
	clone := cfg.Clone()

	if clone.Headers == nil {
		t.Fatal("cloned headers map must be non-nil")
	}
	if clone.Extra == nil {
		t.Fatal("cloned extra map must be non-nil")
	}
	if clone.Temperature != nil {
		t.Fatalf("cloned temperature: got %v", *clone.Temperature)
	}
	if clone.MaxTokens != nil {
		t.Fatalf("cloned max tokens: got %v", *clone.MaxTokens)
	}
}

func TestCloneCopiesMaxTokensAndScalars(t *testing.T) {
	client := &http.Client{}
	cfg := New(
		WithModel("m"),
		WithBaseURL("u"),
		WithMaxTokens(64),
		WithHTTPClient(client),
	)
	clone := cfg.Clone()

	if clone.MaxTokens == nil || *clone.MaxTokens != 64 {
		t.Fatalf("max tokens: got %v", clone.MaxTokens)
	}
	if clone.MaxTokens == cfg.MaxTokens {
		t.Fatal("max tokens pointer was shared with the original")
	}
	*clone.MaxTokens = 1
	if *cfg.MaxTokens != 64 {
		t.Fatalf("max tokens mutated: %d", *cfg.MaxTokens)
	}
	if clone.Model != "m" || clone.BaseURL != "u" {
		t.Fatalf("scalars not copied: %+v", clone)
	}
	if clone.HTTPClient != client {
		t.Fatal("http client should be shared, not deep-copied")
	}
}
