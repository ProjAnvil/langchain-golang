package middleware

import (
	"context"
	"strings"
	"testing"
)

// FakeAnthropicChatModel exercises provider inference by class name: its
// reflect type name contains "Anthropic".
type FakeAnthropicChatModel struct{}

func TestDeferredToolWrapper(t *testing.T) {
	tool := mustTool(t, "search")
	deferred := NewDeferredTool(tool)
	if !deferred.DeferLoading() {
		t.Fatal("expected defer_loading extra")
	}
	if deferred.Name() != "search" || deferred.Description() != tool.Description() {
		t.Fatalf("deferred metadata mismatch: %q %q", deferred.Name(), deferred.Description())
	}
	if deferred.ArgsSchema() == nil {
		t.Fatal("expected args schema to be forwarded")
	}
	result, err := deferred.Invoke(context.Background(), map[string]any{"x": 1})
	if err != nil || result.Content != "ok" {
		t.Fatalf("invoke mismatch: %#v %v", result, err)
	}

	// Extras without defer_loading report false.
	plain := DeferredTool{Tool: tool, Extras: map[string]any{}}
	if plain.DeferLoading() {
		t.Fatal("expected DeferLoading false without the extra")
	}
}

func TestProviderToolSearchUnknownSearchableTool(t *testing.T) {
	middleware := NewProviderToolSearchMiddleware("ghost")
	request, err := NewModelRequest(ModelRequest{
		Model:    "anthropic:claude-3",
		Messages: nil,
		Tools:    []any{mustTool(t, "search")},
	})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	_, err = middleware.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "ghost") {
		t.Fatalf("expected unknown searchable tool error, got %v", err)
	}
}

func TestProviderToolSearchNoDeferredToolsPassThrough(t *testing.T) {
	middleware := NewProviderToolSearchMiddleware()
	tool := mustTool(t, "search")
	request, err := NewModelRequest(ModelRequest{Model: "anthropic:claude-3", Tools: []any{tool}})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if len(req.Tools) != 1 {
			t.Fatalf("request should pass through unchanged: %#v", req.Tools)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestProviderToolSearchAddsServerToolSpec(t *testing.T) {
	cases := []struct {
		name     string
		model    any
		specType string
	}{
		{"anthropic", "anthropic:claude-3", "tool_search_tool_bm25_20251119"},
		{"openai", "openai:gpt-4o", "tool_search"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			middleware := NewProviderToolSearchMiddleware()
			deferred := NewDeferredTool(mustTool(t, "search"))
			other := map[string]any{"type": "custom_tool"}
			request, err := NewModelRequest(ModelRequest{Model: tt.model, Tools: []any{deferred, other}})
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
				if len(req.Tools) != 3 {
					t.Fatalf("expected deferred + custom + spec tools: %#v", req.Tools)
				}
				wrapped, ok := req.Tools[0].(DeferredTool)
				if !ok || !wrapped.DeferLoading() {
					t.Fatalf("deferred tool should stay deferred: %#v", req.Tools[0])
				}
				spec, ok := req.Tools[2].(map[string]any)
				if !ok || spec["type"] != tt.specType {
					t.Fatalf("server tool spec mismatch: %#v", req.Tools[2])
				}
				return ModelResponse{}, nil
			})
			if err != nil {
				t.Fatalf("wrap model call: %v", err)
			}
		})
	}
}

func TestProviderToolSearchUndeterminedProvider(t *testing.T) {
	middleware := NewProviderToolSearchMiddleware()
	request, err := NewModelRequest(ModelRequest{Model: "llama3", Tools: []any{NewDeferredTool(mustTool(t, "search"))}})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	_, err = middleware.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "could not determine the provider") {
		t.Fatalf("expected provider inference error, got %v", err)
	}
}

func TestProviderToolSearchUnsupportedProvider(t *testing.T) {
	middleware := NewProviderToolSearchMiddleware()
	// "foo:bar" infers provider "foo", which has no server-side search spec.
	request, err := NewModelRequest(ModelRequest{Model: "foo:bar", Tools: []any{NewDeferredTool(mustTool(t, "search"))}})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	_, err = middleware.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "requires a provider with server-side tool search") {
		t.Fatalf("expected unsupported provider error, got %v", err)
	}
}

func TestInferProvider(t *testing.T) {
	cases := []struct {
		name    string
		model   any
		runtime any
		want    string
	}{
		{"nil", nil, nil, ""},
		{"colon spec", "anthropic:claude-3", nil, "anthropic"},
		{"claude name", "claude-3-opus", nil, "anthropic"},
		{"gpt name", "gpt-4o", nil, "openai"},
		{"unknown name", "llama3", nil, ""},
		{"config model_provider", map[string]any{"model_provider": "Anthropic"}, nil, "anthropic"},
		{"config model", map[string]any{"model": "openai:gpt-4"}, nil, "openai"},
		{"config ls_provider", map[string]any{"ls_provider": "OPENAI"}, nil, "openai"},
		{"class name", FakeAnthropicChatModel{}, nil, "anthropic"},
		{"runtime fallback", nil, map[string]any{"model_provider": "openai"}, "openai"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := InferProvider(tt.model, tt.runtime); got != tt.want {
				t.Fatalf("InferProvider(%v, %v) = %q, want %q", tt.model, tt.runtime, got, tt.want)
			}
		})
	}
}
