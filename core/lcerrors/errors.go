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
	"strings"
	"syscall"
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

// Standard model error sentinels, mirroring Python langchain-core 1.6's
// standard model exception types (#39538). ErrModelRateLimit and
// ErrModelTimeout alias the existing sentinels so both spellings share one
// identity. The fine-grained 4xx/5xx kinds are subsumed by ErrProvider (see
// ProviderError.Is); 429/408 keep their exclusive sentinels.
var (
	// ErrModelAuth indicates provider authentication failed (HTTP 401,
	// Python ModelAuthenticationError).
	ErrModelAuth = errors.New("model authentication failed")
	// ErrModelPermissionDenied indicates credentials lack permission (HTTP 403,
	// Python ModelPermissionDeniedError).
	ErrModelPermissionDenied = errors.New("model permission denied")
	// ErrModelInvalidRequest indicates the provider rejected the request as
	// invalid (HTTP 400/422, Python ModelInvalidRequestError).
	ErrModelInvalidRequest = errors.New("model invalid request")
	// ErrModelNotFound indicates the requested model was not found (HTTP 404,
	// Python ModelNotFoundError).
	ErrModelNotFound = errors.New("model not found")
	// ErrModelServer indicates a provider server failure (HTTP 5xx, Python
	// ModelAPIError).
	ErrModelServer = errors.New("model server error")
	// ErrModelConnection indicates the provider could not be reached (Python
	// ModelConnectionError); set by WrapTransport, never by status code.
	ErrModelConnection = errors.New("model connection error")
	// ErrModelRateLimit indicates a rate limit was exceeded (HTTP 429).
	ErrModelRateLimit = ErrRateLimited
	// ErrModelTimeout indicates a model request timed out (HTTP 408/transport).
	ErrModelTimeout = ErrTimeout
	// ErrModelContextOverflow indicates the input exceeded the model's context
	// limit (HTTP 400 + body marker, Python ContextOverflowError).
	ErrModelContextOverflow = errors.New("model context overflow")
)

// ModelErrorKind classifies a provider failure, mirroring Python
// langchain-core 1.6's standard model exception types (#39538).
type ModelErrorKind string

const (
	// ModelKindNone means the failure carries no model classification.
	ModelKindNone ModelErrorKind = ""
	// ModelKindAuth maps to ModelAuthenticationError (HTTP 401).
	ModelKindAuth ModelErrorKind = "authentication"
	// ModelKindPermissionDenied maps to ModelPermissionDeniedError (HTTP 403).
	ModelKindPermissionDenied ModelErrorKind = "permission_denied"
	// ModelKindInvalidRequest maps to ModelInvalidRequestError (HTTP 400/422).
	ModelKindInvalidRequest ModelErrorKind = "invalid_request"
	// ModelKindNotFound maps to ModelNotFoundError (HTTP 404).
	ModelKindNotFound ModelErrorKind = "not_found"
	// ModelKindRateLimit maps to ModelRateLimitError (HTTP 429).
	ModelKindRateLimit ModelErrorKind = "rate_limit"
	// ModelKindServer maps to ModelAPIError (HTTP 5xx).
	ModelKindServer ModelErrorKind = "server_error"
	// ModelKindConnection maps to ModelConnectionError (transport).
	ModelKindConnection ModelErrorKind = "connection"
	// ModelKindTimeout maps to ModelTimeoutError (HTTP 408 / net timeout).
	ModelKindTimeout ModelErrorKind = "timeout"
	// ModelKindContextOverflow maps to ContextOverflowError (HTTP 400 + body marker).
	ModelKindContextOverflow ModelErrorKind = "context_overflow"
)

// contextOverflowMarkers are substrings (lowercased) that mark an HTTP 400
// body as a context-window overflow, aligned with the markers Python's
// openai partner sniffs for ContextWindowExceededError (#39300).
var contextOverflowMarkers = []string{
	"context_length_exceeded",
	"maximum context length",
	"context window",
	"prompt is too long",
}

// ModelErrorKindForStatus classifies an HTTP status (plus response body for
// the 400 context-overflow sniff) into a ModelErrorKind. It is pure so
// partners and tests can reuse the mapping.
func ModelErrorKindForStatus(statusCode int, body string) ModelErrorKind {
	switch {
	case statusCode == 401:
		return ModelKindAuth
	case statusCode == 403:
		return ModelKindPermissionDenied
	case statusCode == 404:
		return ModelKindNotFound
	case statusCode == 429:
		return ModelKindRateLimit
	case statusCode == 408:
		return ModelKindTimeout
	case statusCode >= 500:
		return ModelKindServer
	case statusCode == 400 || statusCode == 422:
		lowered := strings.ToLower(body)
		for _, marker := range contextOverflowMarkers {
			if strings.Contains(lowered, marker) {
				return ModelKindContextOverflow
			}
		}
		return ModelKindInvalidRequest
	default:
		return ModelKindNone
	}
}

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
	ModelKind  ModelErrorKind
}

// providerSubsumedSentinels are the fine-grained model sentinels that remain
// subsumed by ErrProvider for backwards compatibility (429/408 deliberately
// keep their exclusive sentinels).
var providerSubsumedSentinels = map[error]bool{
	ErrModelAuth:             true,
	ErrModelPermissionDenied: true,
	ErrModelInvalidRequest:   true,
	ErrModelNotFound:         true,
	ErrModelServer:           true,
	ErrModelContextOverflow:  true,
}

// Is reports whether the error matches target. The fine-grained model
// sentinels (auth, permission, invalid request, not found, server, context
// overflow) also match ErrProvider, preserving the pre-1.6 classification
// surface while adding the standard kinds on top (#39538).
func (e *ProviderError) Is(target error) bool {
	if target == error(e.Err) {
		return true
	}
	return target == ErrProvider && providerSubsumedSentinels[e.Err]
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

// IsModelRetryable reports whether retrying the same model request may
// succeed, mirroring Python's ModelError.is_retryable defaults (#39538):
// rate limits, server failures, connection errors, and timeouts are
// retryable; everything else is not.
func (e *ProviderError) IsModelRetryable() bool {
	switch e.ModelKind {
	case ModelKindRateLimit, ModelKindServer, ModelKindConnection, ModelKindTimeout:
		return true
	default:
		return false
	}
}

// NewProviderError classifies an HTTP status code into a typed ProviderError.
// retryAfter carries the parsed Retry-After hint (0 when absent).
func NewProviderError(provider, endpoint string, statusCode int, body string, retryAfter time.Duration) *ProviderError {
	return &ProviderError{
		Provider:   provider,
		StatusCode: statusCode,
		Endpoint:   endpoint,
		Body:       body,
		RetryAfter: retryAfter,
		Err:        classifyStatus(statusCode, body),
		ModelKind:  ModelErrorKindForStatus(statusCode, body),
	}
}

// classifyStatus maps an HTTP status code (plus body, for the 400
// context-overflow sniff) to a sentinel error kind.
func classifyStatus(statusCode int, body string) error {
	switch ModelErrorKindForStatus(statusCode, body) {
	case ModelKindRateLimit:
		return ErrRateLimited
	case ModelKindTimeout:
		return ErrTimeout
	case ModelKindAuth:
		return ErrModelAuth
	case ModelKindPermissionDenied:
		return ErrModelPermissionDenied
	case ModelKindNotFound:
		return ErrModelNotFound
	case ModelKindInvalidRequest:
		return ErrModelInvalidRequest
	case ModelKindContextOverflow:
		return ErrModelContextOverflow
	case ModelKindServer:
		return ErrModelServer
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

// WrapTransport inspects a network/transport error and classifies it: a
// timeout (connect, read, or request deadline) compares as ErrTimeout; a
// connection refusal/reset compares as ErrModelConnection (Python
// ModelConnectionError, #39538); anything else is returned unchanged.
func WrapTransport(err error) error {
	if err == nil {
		return nil
	}
	if netErr, ok := errors.AsType[net.Error](err); ok && netErr.Timeout() {
		return fmt.Errorf("%w: %v", ErrTimeout, err)
	}
	if isConnectionFailure(err) {
		return fmt.Errorf("%w: %v", ErrModelConnection, err)
	}
	return err
}

// isConnectionFailure reports whether err is (or wraps) a connection
// refusal or reset.
func isConnectionFailure(err error) bool {
	for err != nil {
		if errno, ok := errors.AsType[syscall.Errno](err); ok &&
			(errno == syscall.ECONNREFUSED || errno == syscall.ECONNRESET) {
			return true
		}
		err = errors.Unwrap(err)
	}
	return false
}
