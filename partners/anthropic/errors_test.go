package anthropic

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestInvokeRateLimitedIsTyped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"type":"error","error":{"type":"rate_limit_error"}}`, http.StatusTooManyRequests)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("claude-test"))
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if !errors.Is(err, lcerrors.ErrRateLimited) {
		t.Fatalf("invoke err not ErrRateLimited: %v", err)
	}
}

func TestInvokeServerErrorIsTypedProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
		modelconfig.WithMaxRetries(0),
	)
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if !errors.Is(err, lcerrors.ErrProvider) {
		t.Fatalf("invoke err not ErrProvider: %v", err)
	}
}

func TestStreamRateLimitedIsTyped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("claude-test"))
	_, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if !errors.Is(err, lcerrors.ErrRateLimited) {
		t.Fatalf("stream err not ErrRateLimited: %v", err)
	}
}
