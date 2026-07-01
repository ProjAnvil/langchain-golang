package modelconfig

import (
	"testing"
	"time"
)

func TestNewConfig(t *testing.T) {
	cfg := New(
		WithModel("fake-model"),
		WithAPIKey("secret"),
		WithHeader("X-Test", "true"),
		WithTemperature(0.2),
		WithMaxTokens(128),
		WithTimeout(3*time.Second),
		WithMaxRetries(4),
		WithRetryDelay(100*time.Millisecond),
		WithRetryBackoffMultiplier(2),
		WithRetryMaxDelay(time.Second),
		WithExtra("reasoning_effort", "low"),
	)

	if cfg.Model != "fake-model" {
		t.Fatalf("model: got %q", cfg.Model)
	}
	if cfg.Headers["X-Test"] != "true" {
		t.Fatalf("header: got %q", cfg.Headers["X-Test"])
	}
	if cfg.Temperature == nil || *cfg.Temperature != 0.2 {
		t.Fatalf("temperature: got %v", cfg.Temperature)
	}
	if cfg.MaxTokens == nil || *cfg.MaxTokens != 128 {
		t.Fatalf("max tokens: got %v", cfg.MaxTokens)
	}
	if cfg.Timeout != 3*time.Second {
		t.Fatalf("timeout: got %v", cfg.Timeout)
	}
	if cfg.MaxRetries != 4 {
		t.Fatalf("max retries: got %d", cfg.MaxRetries)
	}
	if cfg.RetryDelay != 100*time.Millisecond {
		t.Fatalf("retry delay: got %v", cfg.RetryDelay)
	}
	if cfg.RetryBackoffMultiplier != 2 {
		t.Fatalf("retry backoff multiplier: got %v", cfg.RetryBackoffMultiplier)
	}
	if cfg.RetryMaxDelay != time.Second {
		t.Fatalf("retry max delay: got %v", cfg.RetryMaxDelay)
	}
	if cfg.Extra["reasoning_effort"] != "low" {
		t.Fatalf("extra: got %v", cfg.Extra["reasoning_effort"])
	}
}

func TestConfigCloneCopiesMapsAndPointers(t *testing.T) {
	cfg := New(
		WithHeader("X-Test", "true"),
		WithTemperature(0.3),
		WithExtra("foo", "bar"),
	)
	clone := cfg.Clone()
	clone.Headers["X-Test"] = "false"
	clone.Extra["foo"] = "baz"
	*clone.Temperature = 0.9

	if cfg.Headers["X-Test"] != "true" {
		t.Fatalf("header mutated: %q", cfg.Headers["X-Test"])
	}
	if cfg.Extra["foo"] != "bar" {
		t.Fatalf("extra mutated: %v", cfg.Extra["foo"])
	}
	if *cfg.Temperature != 0.3 {
		t.Fatalf("temperature mutated: %v", *cfg.Temperature)
	}
}
