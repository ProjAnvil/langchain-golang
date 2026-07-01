package middleware

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestModelFallbackMiddlewareTriesPrimaryThenFallbacks(t *testing.T) {
	fallback := NewModelFallbackMiddleware("fallback-a", "fallback-b")
	request, err := NewModelRequest(ModelRequest{Model: "primary"})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	var seen []any
	response, err := fallback.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		seen = append(seen, request.Model)
		if request.Model != "fallback-b" {
			return ModelResponse{}, errors.New("failed")
		}
		return ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if response.Result[0].Content != "ok" {
		t.Fatalf("response mismatch: %#v", response.Result)
	}
	wantSeen := []any{"primary", "fallback-a", "fallback-b"}
	if !reflect.DeepEqual(seen, wantSeen) {
		t.Fatalf("model order mismatch: got %#v want %#v", seen, wantSeen)
	}
	if request.Model != "primary" {
		t.Fatalf("original request mutated: %#v", request.Model)
	}
}

func TestModelFallbackMiddlewareReturnsPrimarySuccess(t *testing.T) {
	fallback := NewModelFallbackMiddleware("fallback")
	request, err := NewModelRequest(ModelRequest{Model: "primary"})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	calls := 0
	response, err := fallback.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{Result: []messages.Message{messages.AI(request.Model.(string))}}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if calls != 1 {
		t.Fatalf("call count mismatch: got %d", calls)
	}
	if response.Result[0].Content != "primary" {
		t.Fatalf("response mismatch: %q", response.Result[0].Content)
	}
}

func TestModelFallbackMiddlewareReturnsLastError(t *testing.T) {
	lastErr := errors.New("fallback failed")
	fallback := NewModelFallbackMiddleware("fallback")
	request, err := NewModelRequest(ModelRequest{Model: "primary"})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	_, err = fallback.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		if request.Model == "fallback" {
			return ModelResponse{}, lastErr
		}
		return ModelResponse{}, errors.New("primary failed")
	})
	if !errors.Is(err, lastErr) {
		t.Fatalf("expected last fallback error, got %v", err)
	}
}
