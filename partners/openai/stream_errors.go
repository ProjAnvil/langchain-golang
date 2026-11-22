package openai

import "fmt"

// In-band stream error classification, porting Python langchain-openai's
// #41952: OpenAI can accept a streaming request with HTTP 200 and then fail
// partway through by sending an error event. A transient server-error code
// carries the same retryable classification an HTTP 5xx gets, without
// fabricating a status code.

// transientStreamErrorCodes are in-band stream error codes that indicate a
// transient backend failure (base.py _TRANSIENT_STREAM_ERROR_CODES).
var transientStreamErrorCodes = map[string]bool{
	"server_error": true,
}

// StreamServerError is a server error reported inside an already-accepted
// stream, classified as retryable (#41952). It implements the middleware
// retry duck type (IsModelRetryable) like a ModelAPIError from HTTP 5xx.
type StreamServerError struct {
	// Message is the provider's error message.
	Message string
	// Code is the error event's code (e.g. "server_error").
	Code string
	// Type is the error event's type spelling, when present.
	Type string
	// Event is the stream event type that carried the error.
	Event string
}

// Error implements the error interface.
func (e *StreamServerError) Error() string {
	return fmt.Sprintf("openai stream error (%s event, code %q): %s", e.Event, e.Code, e.Message)
}

// IsModelRetryable reports true: transient server errors may succeed on a
// retried request.
func (e *StreamServerError) IsModelRetryable() bool { return true }

// streamServerError builds the error for one in-band error event: a
// transient code yields a retryable StreamServerError, anything else the
// plain stream error as before (#41952).
func streamServerError(eventType string, payload streamError) error {
	if transientStreamErrorCodes[payload.Code] || transientStreamErrorCodes[payload.Type] {
		return &StreamServerError{
			Message: payload.Message,
			Code:    payload.Code,
			Type:    payload.Type,
			Event:   eventType,
		}
	}
	if payload.Message == "" {
		return fmt.Errorf("openai stream error")
	}
	return fmt.Errorf("openai stream error: %s", payload.Message)
}
