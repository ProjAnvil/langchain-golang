package openai

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// TestStreamServerErrorClassification pins the #41952 in-band stream error
// classification: a server_error code is retryable, other codes are not.
func TestStreamServerErrorClassification(t *testing.T) {
	retryable := streamServerError("error", streamError{Message: "backend exploded", Code: "server_error"})
	if _, ok := retryable.(*StreamServerError); !ok {
		t.Fatalf("server_error code: %#v", retryable)
	}
	if !retryable.(*StreamServerError).IsModelRetryable() {
		t.Fatal("StreamServerError must be retryable")
	}

	plain := streamServerError("error", streamError{Message: "bad input", Code: "invalid_request"})
	if _, ok := plain.(*StreamServerError); ok {
		t.Fatalf("non-transient code must stay plain: %#v", plain)
	}

	fallback := streamServerError("response.failed", streamError{})
	if fallback == nil || fallback.Error() == "" {
		t.Fatalf("empty payload: %#v", fallback)
	}
}

// TestResponsesStreamServerErrorEvent drives an in-band error event through
// Stream and checks the surfaced error is the retryable classification.
func TestResponsesStreamServerErrorEvent(t *testing.T) {
	server := sseServer(t,
		"data: {\"type\":\"error\",\"error\":{\"message\":\"upstream failed\",\"code\":\"server_error\"}}\n\n")
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	_, _, err = stream.Next(t.Context())
	if err == nil {
		t.Fatal("expected stream error")
	}
	serverErr, ok := err.(*StreamServerError)
	if !ok {
		t.Fatalf("error type: %#v", err)
	}
	if !serverErr.IsModelRetryable() || serverErr.Code != "server_error" {
		t.Fatalf("classification: %#v", serverErr)
	}
}
