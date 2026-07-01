package lcerrors

import (
	"errors"
	"testing"
	"time"
)

type fakeNetTimeout struct{}

func (fakeNetTimeout) Error() string   { return "i/o timeout" }
func (fakeNetTimeout) Timeout() bool   { return true }
func (fakeNetTimeout) Temporary() bool { return false }

type fakeNetConnRefused struct{}

func (fakeNetConnRefused) Error() string   { return "connection refused" }
func (fakeNetConnRefused) Timeout() bool   { return false }
func (fakeNetConnRefused) Temporary() bool { return false }

func TestNewProviderErrorClassification(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		wantSent   error
		retryable  bool
	}{
		{"rate limited", 429, ErrRateLimited, true},
		{"request timeout", 408, ErrTimeout, true},
		{"server error", 500, ErrProvider, true},
		{"bad gateway", 502, ErrProvider, true},
		{"bad request", 400, ErrProvider, false},
		{"unauthorized", 401, ErrProvider, false},
		{"not found", 404, ErrProvider, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := NewProviderError("openai", "/chat", tc.status, "body", 0)
			if !errors.Is(err, tc.wantSent) {
				t.Fatalf("errors.Is(%d, %v) = false, want true", tc.status, tc.wantSent)
			}
			var pe *ProviderError
			if !errors.As(err, &pe) {
				t.Fatalf("errors.As into *ProviderError = false")
			}
			if pe.StatusCode != tc.status {
				t.Fatalf("StatusCode = %d, want %d", pe.StatusCode, tc.status)
			}
			if pe.Provider != "openai" {
				t.Fatalf("Provider = %q", pe.Provider)
			}
			if pe.Body != "body" {
				t.Fatalf("Body = %q", pe.Body)
			}
			if IsRetryableStatus(tc.status) != tc.retryable {
				t.Fatalf("IsRetryableStatus(%d) = %v, want %v", tc.status, IsRetryableStatus(tc.status), tc.retryable)
			}
		})
	}
}

func TestProviderErrorUnwrapChainsToExactlyOneSentinel(t *testing.T) {
	err := NewProviderError("anthropic", "/messages", 429, "slow down", 5*time.Second)
	if !errors.Is(err, ErrRateLimited) {
		t.Fatal("expected ErrRateLimited")
	}
	if errors.Is(err, ErrProvider) {
		t.Fatal("429 must not also match ErrProvider")
	}
	if errors.Is(err, ErrTimeout) {
		t.Fatal("429 must not also match ErrTimeout")
	}
	var pe *ProviderError
	if !errors.As(err, &pe) || pe.RetryAfter != 5*time.Second {
		t.Fatalf("RetryAfter = %v, want 5s", pe.RetryAfter)
	}
}

func TestProviderErrorMessageIncludesDetails(t *testing.T) {
	err := NewProviderError("ollama", "/api/chat", 500, "boom", 0)
	want := "ollama /api/chat returned 500: boom"
	if err.Error() != want {
		t.Fatalf("Error() = %q, want %q", err.Error(), want)
	}
	empty := NewProviderError("ollama", "/api/chat", 500, "", 0)
	if got := empty.Error(); got != "ollama /api/chat returned 500" {
		t.Fatalf("Error() with empty body = %q", got)
	}
}

func TestWrapTransportClassifiesTimeout(t *testing.T) {
	if err := WrapTransport(fakeNetTimeout{}); !errors.Is(err, ErrTimeout) {
		t.Fatalf("timeout not classified as ErrTimeout: %v", err)
	}
	if err := WrapTransport(fakeNetConnRefused{}); errors.Is(err, ErrTimeout) {
		t.Fatalf("non-timeout transport error must not be ErrTimeout: %v", err)
	}
	if err := WrapTransport(nil); err != nil {
		t.Fatalf("WrapTransport(nil) = %v, want nil", err)
	}
}
