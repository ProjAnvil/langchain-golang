package typesafe

import (
	"fmt"
	"net/http"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
)

// This file mirrors langchain_typesafe/client.py's error vocabulary
// (#40542/#40570) without copying Python's dual-inheritance class tree: the
// Go APIError embeds lcerrors.ProviderError so errors.As keeps reading
// status/body/retry-after and errors.Is keeps matching the model sentinels,
// while Error() is overridden to the safe summary format (spec decision #5).

// safeStatusMessage covers statuses with no standard http.StatusText phrase
// (client.py:31: 529 -> "Overloaded", anything unknown -> "API request
// failed").
func safeStatusMessage(status int) string {
	if status == 529 {
		return "Overloaded"
	}
	if phrase := http.StatusText(status); phrase != "" {
		return phrase
	}
	return "API request failed"
}

// APIError describes an unsuccessful TypeSafe HTTP response. The embedded
// ProviderError carries the status code, sanitized endpoint, body, retry
// hint, and sentinel kind; RequestID mirrors the x-typesafe-request-id
// response header.
//
// Error() deliberately excludes the body and headers (Python keeps them off
// __str__/__repr__ so classified state and credentials never reach logs —
// client.py:37-51).
type APIError struct {
	lcerrors.ProviderError
	// RequestID is the x-typesafe-request-id header value, empty when absent.
	RequestID string
}

// Error returns "{endpoint}: {status} {phrase} (request_id=...)", omitting
// the endpoint prefix when empty and the request-id suffix when absent
// (client.py:88-100). It never includes the response body or headers.
func (e *APIError) Error() string {
	message := fmt.Sprintf("%d %s", e.StatusCode, safeStatusMessage(e.StatusCode))
	if e.Endpoint != "" {
		message = e.Endpoint + ": " + message
	}
	if e.RequestID != "" {
		message = fmt.Sprintf("%s (request_id=%s)", message, e.RequestID)
	}
	return message
}

// Unwrap returns the embedded ProviderError so errors.As(err, *ProviderError)
// and errors.Is(err, sentinel) both hold: the embedded error's Is/Unwrap
// semantics are not swallowed by the outer type.
func (e *APIError) Unwrap() error { return &e.ProviderError }

// ResponseValidationError reports a successful response that did not match
// the expected schema (client.py:196-228). FieldPath is the dotted path to
// the first field that failed validation, "response" when the body is not a
// JSON object.
type ResponseValidationError struct {
	// FieldPath is the dotted location of the first validation failure.
	FieldPath string
}

// Error returns the verbatim Python message (client.py:225).
func (e *ResponseValidationError) Error() string {
	return fmt.Sprintf("Invalid response data at '%s'.", e.FieldPath) //nolint:staticcheck // ST1005: verbatim Python text (client.py:225)
}

// Unwrap ties response-validation failures to the schema-validation
// sentinel so callers can branch with errors.Is.
func (e *ResponseValidationError) Unwrap() error { return lcerrors.ErrSchemaValidation }

// timeoutError mirrors TypeSafeAPITimeoutError (client.py:165-193): the
// message reports only the configured timeout, and the error compares as
// lcerrors.ErrModelTimeout.
type timeoutError struct {
	timeout time.Duration
}

func (e *timeoutError) Error() string {
	return fmt.Sprintf("Request timed out (timeout=%s).", e.timeout) //nolint:staticcheck // ST1005: verbatim Python text (client.py:188)
}

func (e *timeoutError) Unwrap() error { return lcerrors.ErrModelTimeout }

// connectionError mirrors TypeSafeAPIConnectionError (client.py:161-163):
// a request failed without receiving an HTTP response. It compares as
// lcerrors.ErrModelConnection and preserves the transport cause for
// errors.Is.
type connectionError struct {
	cause error
}

func (e *connectionError) Error() string {
	return "Unable to connect to the TypeSafe API." //nolint:staticcheck // ST1005: verbatim Python text (classifier.py:369)
}

func (e *connectionError) Unwrap() []error {
	return []error{lcerrors.ErrModelConnection, e.cause}
}
