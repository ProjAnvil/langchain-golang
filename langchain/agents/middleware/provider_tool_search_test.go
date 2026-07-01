package middleware

import (
	"context"
	"strings"
	"testing"
)

func TestProviderToolSearchMiddlewareDefersSearchableTools(t *testing.T) {
	search := mustTool(t, "search")
	calc := mustTool(t, "calc")
	middleware := NewProviderToolSearchMiddleware("search")
	request, err := NewModelRequest(ModelRequest{
		Model: "openai:gpt-5",
		Tools: []any{search, calc},
	})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}

	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		if len(request.Tools) != 3 {
			t.Fatalf("tool count mismatch: %#v", request.Tools)
		}
		deferred, ok := request.Tools[0].(DeferredTool)
		if !ok || deferred.Name() != "search" || !deferred.DeferLoading() {
			t.Fatalf("deferred tool mismatch: %#v", request.Tools[0])
		}
		if request.Tools[1].(interface{ Name() string }).Name() != "calc" {
			t.Fatalf("calc tool mismatch: %#v", request.Tools[1])
		}
		spec, ok := request.Tools[2].(map[string]any)
		if !ok || spec["type"] != "tool_search" {
			t.Fatalf("provider search spec mismatch: %#v", request.Tools[2])
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestProviderToolSearchMiddlewarePassesThroughWhenNothingDeferred(t *testing.T) {
	search := mustTool(t, "search")
	middleware := NewProviderToolSearchMiddleware()
	request, err := NewModelRequest(ModelRequest{
		Model: "unsupported:model",
		Tools: []any{search},
	})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}

	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		if len(request.Tools) != 1 {
			t.Fatalf("tool count mismatch: %#v", request.Tools)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestProviderToolSearchMiddlewareRejectsUnknownSearchableTool(t *testing.T) {
	middleware := NewProviderToolSearchMiddleware("missing")
	request, err := NewModelRequest(ModelRequest{
		Model: "openai:gpt-5",
		Tools: []any{mustTool(t, "search")},
	})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}

	_, err = middleware.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "not bound") {
		t.Fatalf("expected unknown tool error, got %v", err)
	}
}

func TestProviderToolSearchMiddlewareRejectsUnsupportedProvider(t *testing.T) {
	middleware := NewProviderToolSearchMiddleware("search")
	request, err := NewModelRequest(ModelRequest{
		Model: "ollama:llama3",
		Tools: []any{mustTool(t, "search")},
	})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}

	_, err = middleware.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "requires a provider") {
		t.Fatalf("expected unsupported provider error, got %v", err)
	}
}

func TestProviderToolSearchMiddlewareAnthropicSpec(t *testing.T) {
	middleware := NewProviderToolSearchMiddleware("search")
	request, err := NewModelRequest(ModelRequest{
		Model: "anthropic:claude",
		Tools: []any{mustTool(t, "search")},
	})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}

	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		spec := request.Tools[1].(map[string]any)
		if spec["type"] != "tool_search_tool_bm25_20251119" || spec["name"] != "tool_search_tool_bm25" {
			t.Fatalf("anthropic spec mismatch: %#v", spec)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}
