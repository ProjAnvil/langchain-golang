package openai

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// TestStreamServerErrorClassification pins the #41952 in-band stream error
// classification: a server_error code is retryable, other codes are not,
// and the code prefixes the plain message (base.py:5745).
func TestStreamServerErrorClassification(t *testing.T) {
	retryable := streamServerError("error", "server_error", "backend exploded")
	if _, ok := retryable.(*StreamServerError); !ok {
		t.Fatalf("server_error code: %#v", retryable)
	}
	if !retryable.(*StreamServerError).IsModelRetryable() {
		t.Fatal("StreamServerError must be retryable")
	}

	plain := streamServerError("error", "invalid_request", "bad input")
	if _, ok := plain.(*StreamServerError); ok {
		t.Fatalf("non-transient code must stay plain: %#v", plain)
	}
	if plain.Error() != "openai stream error: invalid_request: bad input" {
		t.Fatalf("code-prefixed message: %q", plain.Error())
	}

	codeOnly := streamServerError("error", "weird", "")
	if codeOnly.Error() != "openai stream error: weird" {
		t.Fatalf("code-only message: %q", codeOnly.Error())
	}

	fallback := streamServerError("response.failed", "", "")
	if fallback == nil || fallback.Error() == "" {
		t.Fatalf("empty payload: %#v", fallback)
	}
}

// TestResponsesStreamServerErrorEvent drives a real-shape top-level error
// event through Stream and checks the surfaced error is the retryable
// classification (the wire carries code/message at the top level, not
// nested — base.py:5744-5748).
func TestResponsesStreamServerErrorEvent(t *testing.T) {
	server := sseServer(t,
		"data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"upstream failed\"}\n\n")
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(t.Context(), nil)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, _, err = stream.Next(t.Context())
	serverErr, ok := err.(*StreamServerError)
	if !ok {
		t.Fatalf("error type: %#v", err)
	}
	if !serverErr.IsModelRetryable() || serverErr.Code != "server_error" || serverErr.Message != "upstream failed" {
		t.Fatalf("classification: %#v", serverErr)
	}
}

// TestResponsesStreamFailedEvent pins the response.failed path: the error
// rides the response payload, and a server_error code is retryable
// (base.py:5732-5743).
func TestResponsesStreamFailedEvent(t *testing.T) {
	server := sseServer(t,
		"data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_9\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"backend died\"}}}\n\n")
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(t.Context(), nil)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, _, err = stream.Next(t.Context())
	serverErr, ok := err.(*StreamServerError)
	if !ok || serverErr.Code != "server_error" || serverErr.Message != "backend died" {
		t.Fatalf("response.failed classification: %#v", err)
	}
}

// TestResponsesStreamFailedWithoutErrorObject pins the bare failure text:
// no error object means "Response {id} failed." (base.py:5736-5739).
func TestResponsesStreamFailedWithoutErrorObject(t *testing.T) {
	server := sseServer(t,
		"data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_9\",\"status\":\"failed\"}}\n\n")
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(t.Context(), nil)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, _, err = stream.Next(t.Context())
	if err == nil || err.Error() != "openai stream error: Response resp_9 failed." {
		t.Fatalf("bare failure text: %#v", err)
	}
}

// TestChatCompletionsStreamErrorEvent pins the Chat Completions in-band
// error path: the nested shape (correct for this API) with a transient
// code is retryable; the stream terminates with the classified error
// instead of dropping it (#41952, base.py:777-783).
func TestChatCompletionsStreamErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"error\":{\"message\":\"mid-stream failure\",\"code\":\"server_error\",\"type\":\"server_error\"}}\n\n")
	}))
	defer server.Close()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(t.Context(), nil)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, _, err = stream.Next(t.Context())
	serverErr, ok := err.(*StreamServerError)
	if !ok || !serverErr.IsModelRetryable() || serverErr.Message != "mid-stream failure" {
		t.Fatalf("chat completions classification: %#v", err)
	}
}
