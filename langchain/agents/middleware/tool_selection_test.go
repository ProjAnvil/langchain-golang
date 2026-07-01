package middleware

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestLLMToolSelectorMiddlewareFiltersTools(t *testing.T) {
	search := mustTool(t, "search")
	calc := mustTool(t, "calc")
	weather := mustTool(t, "weather")
	maxTools := 1
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorMaxTools(maxTools),
		WithToolSelectorAlwaysInclude("weather"),
		WithToolSelectorFunc(func(request ToolSelectionRequest) ([]string, error) {
			if request.LastUserMessage.Content != "find things" {
				t.Fatalf("last user mismatch: %#v", request.LastUserMessage)
			}
			if len(request.AvailableTools) != 2 {
				t.Fatalf("available tools mismatch: %#v", request.AvailableTools)
			}
			if !strings.Contains(request.SystemMessage, "only the first 1 will be used") {
				t.Fatalf("system message missing max instruction: %q", request.SystemMessage)
			}
			return []string{"calc", "search"}, nil
		}),
	)

	request, err := NewModelRequest(ModelRequest{
		Model:    "main-model",
		Messages: []messages.Message{messages.Human("find things")},
		Tools:    []any{search, calc, weather, map[string]any{"type": "provider_tool"}},
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	var seenTools []any
	_, err = selector.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		seenTools = request.Tools
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if len(seenTools) != 3 {
		t.Fatalf("tool count mismatch: %#v", seenTools)
	}
	if seenTools[0].(interface{ Name() string }).Name() != "calc" {
		t.Fatalf("selected tool mismatch: %#v", seenTools[0])
	}
	if seenTools[1].(interface{ Name() string }).Name() != "weather" {
		t.Fatalf("always included tool mismatch: %#v", seenTools[1])
	}
	if _, ok := seenTools[2].(map[string]any); !ok {
		t.Fatalf("provider tool was not preserved: %#v", seenTools[2])
	}
}

func TestLLMToolSelectorMiddlewareNoToolsCallsHandler(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware()
	request, err := NewModelRequest(ModelRequest{
		Model:    "main-model",
		Messages: []messages.Message{messages.Human("hi")},
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	called := false
	_, err = selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		called = true
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if !called {
		t.Fatal("expected handler call")
	}
}

func TestLLMToolSelectorMiddlewareMissingAlwaysInclude(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorAlwaysInclude("missing"),
		WithToolSelectorFunc(func(ToolSelectionRequest) ([]string, error) {
			return []string{}, nil
		}),
	)
	request, err := NewModelRequest(ModelRequest{
		Model:    "main-model",
		Messages: []messages.Message{messages.Human("hi")},
		Tools:    []any{mustTool(t, "search")},
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	_, err = selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "always_include") {
		t.Fatalf("expected always include error, got %v", err)
	}
}

func TestLLMToolSelectorMiddlewareInvalidSelection(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorFunc(func(ToolSelectionRequest) ([]string, error) {
			return []string{"missing"}, nil
		}),
	)
	request, err := NewModelRequest(ModelRequest{
		Model:    "main-model",
		Messages: []messages.Message{messages.Human("hi")},
		Tools:    []any{mustTool(t, "search")},
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	_, err = selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "invalid tools") {
		t.Fatalf("expected invalid selection error, got %v", err)
	}
}

func TestLLMToolSelectorMiddlewareRequiresHumanMessage(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorFunc(func(ToolSelectionRequest) ([]string, error) {
			return []string{"search"}, nil
		}),
	)
	request, err := NewModelRequest(ModelRequest{
		Model:    "main-model",
		Messages: []messages.Message{messages.AI("hello")},
		Tools:    []any{mustTool(t, "search")},
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	_, err = selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "no user message") {
		t.Fatalf("expected no user message error, got %v", err)
	}
}
