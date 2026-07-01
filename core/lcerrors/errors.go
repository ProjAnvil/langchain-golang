// Package lcerrors defines the typed error vocabulary used across
// langchain-golang to classify provider and client failures, as specified in
// MIGRATION_PLAN.md (Core API Design: Context and Errors).
//
// Sentinel values (ErrProvider, ErrRateLimited, ErrTimeout, ...) are compared
// with errors.Is. ProviderError carries the HTTP response details and is read
// with errors.As.
package lcerrors

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"
)

// Typed error kinds. Use errors.Is to branch on the category of a wrapped error.
var (
	// ErrInvalidInput indicates the caller supplied invalid input.
	ErrInvalidInput = errors.New("invalid input")
	// ErrProvider indicates a generic provider-side failure (e.g. HTTP 4xx/5xx).
	ErrProvider = errors.New("provider error")
	// ErrRateLimited indicates the provider returned a rate-limit response (HTTP 429).
	ErrRateLimited = errors.New("rate limited")
	// ErrTimeout indicates a request timed out (connect, read, or deadline).
	ErrTimeout = errors.New("timeout")
	// ErrSchemaValidation indicates a request or response failed schema validation.
	ErrSchemaValidation = errors.New("schema validation error")
	// ErrToolExecution indicates a tool invocation failed.
	ErrToolExecution = errors.New("tool execution error")
	// ErrUnsupportedFeature indicates the provider does not support a requested feature.
	ErrUnsupportedFeature = errors.New("unsupported feature")
)

// ProviderError describes a non-2xx provider HTTP response. It wraps exactly one
// of the sentinels above so callers can branch with errors.Is while still
// reading the status code, body, and retry-after hint with errors.As.
type ProviderError struct {
	Provider   string        // e.g. "openai", "anthropic", "ollama"
	StatusCode int           // HTTP status code
	Endpoint   string        // request path, e.g. "/messages"
	Body       string        // raw response body
	RetryAfter time.Duration // parsed Retry-After header, 0 when absent
	Err        error         // sentinel this error compares as
}

// Error implements the error interface.
func (e *ProviderError) Error() string {
	msg := fmt.Sprintf("%s %s returned %d", e.Provider, e.Endpoint, e.StatusCode)
	if e.Body != "" {
		msg += ": " + e.Body
	}
	return msg
}

// Unwrap allows errors.Is(err, ErrRateLimited) and similar to match.
func (e *ProviderError) Unwrap() error { return e.Err }

// NewProviderError classifies an HTTP status code into a typed ProviderError.
// retryAfter carries the parsed Retry-After hint (0 when absent).
func NewProviderError(provider, endpoint string, statusCode int, body string, retryAfter time.Duration) *ProviderError {
	return &ProviderError{
		Provider:   provider,
		StatusCode: statusCode,
		Endpoint:   endpoint,
		Body:       body,
		RetryAfter: retryAfter,
		Err:        classifyStatus(statusCode),
	}
}

// classifyStatus maps an HTTP status code to a sentinel error kind.
func classifyStatus(statusCode int) error {
	switch statusCode {
	case http.StatusTooManyRequests:
		return ErrRateLimited
	case http.StatusRequestTimeout:
		return ErrTimeout
	default:
		return ErrProvider
	}
}

// IsRetryableStatus reports whether an HTTP status code should trigger a retry
// (HTTP 429 and any 5xx response).
func IsRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests ||
		statusCode == http.StatusRequestTimeout ||
		statusCode >= 500
}

// WrapTransport inspects a network/transport error and returns an error that
// compares as ErrTimeout when the failure was a timeout (connect, read, or
// request deadline); otherwise it returns the original error unchanged.
func WrapTransport(err error) error {
	if err == nil {
		return nil
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return fmt.Errorf("%w: %v", ErrTimeout, err)
	}
	return err
}
