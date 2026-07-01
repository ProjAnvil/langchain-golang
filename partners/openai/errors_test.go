package openai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestInvokeRateLimitedIsTyped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "5")
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if !errors.Is(err, lcerrors.ErrRateLimited) {
		t.Fatalf("invoke err not ErrRateLimited: %v", err)
	}
	if errors.Is(err, lcerrors.ErrProvider) {
		t.Fatal("429 must not match ErrProvider")
	}
	var pe *lcerrors.ProviderError
	if !errors.As(err, &pe) || pe.RetryAfter != 5*time.Second {
		t.Fatalf("ProviderError RetryAfter = %v", pe)
	}
}

func TestInvokeServerErrorIsTypedProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadGateway)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
		modelconfig.WithMaxRetries(0),
	)
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if !errors.Is(err, lcerrors.ErrProvider) {
		t.Fatalf("invoke err not ErrProvider: %v", err)
	}
	if errors.Is(err, lcerrors.ErrRateLimited) {
		t.Fatal("502 must not match ErrRateLimited")
	}
}

func TestInvokeTimeoutIsTyped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(50 * time.Millisecond)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
		func(c *modelconfig.Config) { c.HTTPClient = &http.Client{Timeout: time.Millisecond} },
	)
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if !errors.Is(err, lcerrors.ErrTimeout) {
		t.Fatalf("invoke err not ErrTimeout: %v", err)
	}
}

func TestStreamRateLimitedIsTyped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil && stream != nil {
		_, _, err = stream.Next(context.Background())
	}
	if !errors.Is(err, lcerrors.ErrRateLimited) {
		t.Fatalf("stream err not ErrRateLimited: %v", err)
	}
}
