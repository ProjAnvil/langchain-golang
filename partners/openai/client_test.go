package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestChatModelRetriesRateLimit(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"resp_123",
			"model":"gpt-test",
			"output":[{
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"ok"}]
			}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
		modelconfig.WithMaxRetries(1),
	)
	response, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if response.Content != "ok" {
		t.Fatalf("content: got %q", response.Content)
	}
	if attempts != 2 {
		t.Fatalf("attempts: got %d want 2", attempts)
	}
}

func TestChatModelDoesNotRetryBadRequest(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
		modelconfig.WithMaxRetries(2),
	)
	_, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("hello"),
	})
	if err == nil {
		t.Fatal("expected bad request error")
	}
	if attempts != 1 {
		t.Fatalf("attempts: got %d want 1", attempts)
	}
}
