package openai

import "fmt"

// In-band stream error classification, porting Python langchain-openai's
// #41952: OpenAI can accept a streaming request with HTTP 200 and then fail
// partway through by sending an error event. A transient server-error code
// carries the same retryable classification an HTTP 5xx gets, without
// fabricating a status code.

// transientStreamErrorCodes are the in-band stream error codes that
// indicate a transient backend failure (base.py _TRANSIENT_STREAM_ERROR_CODES).
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
	// Type is the Chat Completions APIError type spelling, when present.
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

// streamServerError builds the error for one Responses API in-band error:
// the classification reads only the error code (base.py:5746 checks
// chunk.code alone); a code prefixes the message (base.py:5745
// f"{chunk.code}: {chunk.message}"). Anything else is the plain stream
// error as before (#41952).
func streamServerError(eventType, code, message string) error {
	if transientStreamErrorCodes[code] {
		return &StreamServerError{Message: message, Code: code, Event: eventType}
	}
	if code != "" {
		if message == "" {
			return fmt.Errorf("openai stream error: %s", code)
		}
		return fmt.Errorf("openai stream error: %s: %s", code, message)
	}
	if message == "" {
		return fmt.Errorf("openai stream error")
	}
	return fmt.Errorf("openai stream error: %s", message)
}

// chatCompletionsStreamError classifies an in-band error event on a Chat
// Completions stream. The SDK-level check matches (e.code, e.type) against
// the transient set (base.py:777-783), so both spellings count here; the
// payload is nested ({"error": {"message", "code", "type"}}).
func chatCompletionsStreamError(payload streamError) error {
	if transientStreamErrorCodes[payload.Code] || transientStreamErrorCodes[payload.Type] {
		return &StreamServerError{Message: payload.Message, Code: payload.Code, Type: payload.Type, Event: "chat.completions"}
	}
	return streamServerError("chat.completions", payload.Code, payload.Message)
}
