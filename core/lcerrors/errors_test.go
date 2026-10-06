package lcerrors

import (
	"errors"
	"fmt"
	"net"
	"os"
	"syscall"
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
		name      string
		status    int
		wantSent  error
		retryable bool
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
			pe, ok := errors.AsType[*ProviderError](err)
			if !ok {
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
	pe, ok := errors.AsType[*ProviderError](err)
	if !ok || pe.RetryAfter != 5*time.Second {
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

func TestModelErrorKindForStatus(t *testing.T) {
	cases := []struct {
		status int
		body   string
		want   ModelErrorKind
	}{
		{401, "", ModelKindAuth},
		{403, "", ModelKindPermissionDenied},
		{400, "", ModelKindInvalidRequest},
		{422, "", ModelKindInvalidRequest},
		{404, "", ModelKindNotFound},
		{429, "", ModelKindRateLimit},
		{500, "", ModelKindServer},
		{503, "", ModelKindServer},
		{408, "", ModelKindTimeout},
		{200, "", ModelKindNone},
		{400, `{"error":{"message":"This model's maximum context length is 4096 tokens, however you requested 5000 tokens."}}`, ModelKindContextOverflow},
		{400, `{"error":{"code":"context_length_exceeded","message":"too long"}}`, ModelKindContextOverflow},
		{400, `{"error":{"message":"your prompt is too long (19543 tokens), max is 8192"}}`, ModelKindContextOverflow},
		{400, `{"error":{"message":"Request too large for model context window"}}`, ModelKindContextOverflow},
		{413, `{"error":{"message":"request entity too large"}}`, ModelKindNone},
	}
	for _, tc := range cases {
		if got := ModelErrorKindForStatus(tc.status, tc.body); got != tc.want {
			t.Errorf("status=%d body=%q: got %q, want %q", tc.status, tc.body, got, tc.want)
		}
	}
}

func TestProviderErrorModelKindAndRetryable(t *testing.T) {
	pe := NewProviderError("openai", "/responses", 429, "", 0)
	if pe.ModelKind != ModelKindRateLimit {
		t.Fatalf("429 kind = %q, want %q", pe.ModelKind, ModelKindRateLimit)
	}
	if !pe.IsModelRetryable() {
		t.Fatal("429 must be model-retryable")
	}
	if !errors.Is(pe, ErrModelRateLimit) || !errors.Is(pe, ErrRateLimited) {
		t.Fatal("429 must satisfy both ErrModelRateLimit and ErrRateLimited")
	}
	for _, status := range []int{401, 403, 400, 404} {
		pe := NewProviderError("openai", "/responses", status, "", 0)
		if pe.IsModelRetryable() {
			t.Errorf("status %d must not be model-retryable", status)
		}
	}
	for _, status := range []int{500, 503} {
		pe := NewProviderError("openai", "/responses", status, "", 0)
		if !pe.IsModelRetryable() {
			t.Errorf("status %d must be model-retryable", status)
		}
	}
	pe = NewProviderError("openai", "/responses", 400, "maximum context length is 4096 tokens", 0)
	if pe.ModelKind != ModelKindContextOverflow || pe.IsModelRetryable() {
		t.Fatalf("context overflow: kind=%q retryable=%v", pe.ModelKind, pe.IsModelRetryable())
	}
}

func TestProviderErrorModelSentinelsAndLegacyCompat(t *testing.T) {
	if !errors.Is(NewProviderError("x", "", 401, "", 0), ErrModelAuth) {
		t.Fatal("401 must match ErrModelAuth")
	}
	if !errors.Is(NewProviderError("x", "", 403, "", 0), ErrModelPermissionDenied) {
		t.Fatal("403 must match ErrModelPermissionDenied")
	}
	if !errors.Is(NewProviderError("x", "", 404, "", 0), ErrModelNotFound) {
		t.Fatal("404 must match ErrModelNotFound")
	}
	if !errors.Is(NewProviderError("x", "", 400, "", 0), ErrModelInvalidRequest) {
		t.Fatal("400 must match ErrModelInvalidRequest")
	}
	if !errors.Is(NewProviderError("x", "", 500, "", 0), ErrModelServer) {
		t.Fatal("500 must match ErrModelServer")
	}
	if !errors.Is(NewProviderError("x", "", 408, "", 0), ErrModelTimeout) {
		t.Fatal("408 must match ErrModelTimeout")
	}
	// Legacy compatibility: the fine-grained 4xx/5xx kinds remain provider
	// errors, while 429/408 keep their exclusive sentinels.
	for _, status := range []int{400, 401, 403, 404, 500, 502} {
		if !errors.Is(NewProviderError("x", "", status, "", 0), ErrProvider) {
			t.Errorf("status %d must still match ErrProvider", status)
		}
	}
	if errors.Is(NewProviderError("x", "", 429, "", 0), ErrProvider) {
		t.Fatal("429 must not match ErrProvider")
	}
	// Connection kind is set by WrapTransport classification, not by status.
	conn := fmt.Errorf("%w: %v", ErrModelConnection, fakeNetConnRefused{})
	if !errors.Is(conn, ErrModelConnection) || !errors.Is(conn, ErrModelConnection) {
		t.Fatal("connection sentinel")
	}
}

func TestWrapTransportClassifiesConnectionRefused(t *testing.T) {
	wrapped := WrapTransport(&net.OpError{Op: "dial", Err: &os.SyscallError{Syscall: "connect", Err: syscall.ECONNREFUSED}})
	if !errors.Is(wrapped, ErrModelConnection) {
		t.Fatalf("connection refused must classify as ErrModelConnection: %v", wrapped)
	}
	if errors.Is(wrapped, ErrTimeout) {
		t.Fatal("connection refused must not be ErrTimeout")
	}
	reset := WrapTransport(&net.OpError{Op: "read", Err: &os.SyscallError{Syscall: "read", Err: syscall.ECONNRESET}})
	if !errors.Is(reset, ErrModelConnection) {
		t.Fatalf("connection reset must classify as ErrModelConnection: %v", reset)
	}
	// Unrelated errors pass through unchanged.
	plain := errors.New("boom")
	if got := WrapTransport(plain); got != plain {
		t.Fatalf("plain error must pass through, got %v", got)
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
