package middleware

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// TestLLMToolSelectorMiddlewareUsesStructuredOutput exercises the spec-mandated
// primary path: when a real language.ChatModel is configured (via
// WithToolSelectorModel) the middleware routes through language.InvokeStructured
// instead of the ToolSelectionFunc callback. It also asserts that
// processSelectionResponse still filters/preserves tools (selected + always
// include) on the structured path exactly as it does on the callback path.
func TestLLMToolSelectorMiddlewareUsesStructuredOutput(t *testing.T) {
	search := mustTool(t, "search")
	calc := mustTool(t, "calc")
	weather := mustTool(t, "weather")

	fake := newStructuredFakeChatModel(messages.AI(`{"tools":["calc","search"]}`))
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorModel(fake),
		WithToolSelectorAlwaysInclude("weather"),
		// Deliberately NO WithToolSelectorFunc — the structured path must win
		// and the callback (when set) is ignored.
	)

	request, err := NewModelRequest(ModelRequest{
		Model:    "ignored-when-middleware-model-set",
		Messages: []messages.Message{messages.Human("find things")},
		Tools:    []any{search, calc, weather},
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

	if fake.structuredCalls != 1 {
		t.Fatalf("expected InvokeStructured called once, got %d", fake.structuredCalls)
	}

	// Schema must constrain the array to valid tool names (an enum).
	if len(fake.capturedSchemas) != 1 {
		t.Fatalf("captured schemas: %#v", fake.capturedSchemas)
	}
	props, _ := fake.capturedSchemas[0]["properties"].(map[string]any)
	toolsProp, _ := props["tools"].(schema.Schema)
	items, _ := toolsProp["items"].(schema.Schema)
	enum, _ := items["enum"].([]string)
	if len(enum) != 2 || enum[0] != "search" || enum[1] != "calc" {
		t.Fatalf("schema enum should be valid tool names: %#v", enum)
	}

	if len(seenTools) != 3 {
		t.Fatalf("tool count mismatch: %#v", seenTools)
	}
	// processSelectionResponse preserves availableTools order (search, calc)
	// then appends always-include tools.
	if seenTools[0].(interface{ Name() string }).Name() != "search" {
		t.Fatalf("first selected tool mismatch: %#v", seenTools[0])
	}
	if seenTools[1].(interface{ Name() string }).Name() != "calc" {
		t.Fatalf("second selected tool mismatch: %#v", seenTools[1])
	}
	if seenTools[2].(interface{ Name() string }).Name() != "weather" {
		t.Fatalf("always-included tool mismatch: %#v", seenTools[2])
	}
}

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
