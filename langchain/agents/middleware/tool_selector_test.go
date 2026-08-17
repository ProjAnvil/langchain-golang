package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// failingStructuredChatModel satisfies language.ChatModel (via the embedded
// fake) but always fails InvokeStructured, exercising the structured-path
// error branch.
type failingStructuredChatModel struct {
	*language.FakeChatModel
	err error
}

func (m *failingStructuredChatModel) InvokeStructured(context.Context, []messages.Message, schema.Schema) (messages.Message, error) {
	return messages.Message{}, m.err
}

func selectorRequest(t *testing.T, model any, tools ...any) ModelRequest {
	t.Helper()
	request, err := NewModelRequest(ModelRequest{
		Model:    model,
		Messages: []messages.Message{messages.Human("find things")},
		Tools:    tools,
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}
	return request
}

func TestToolSelectorSystemPromptOption(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorSystemPrompt("custom prompt"),
		WithToolSelectorFunc(func(request ToolSelectionRequest) ([]string, error) {
			if request.SystemMessage != "custom prompt" {
				t.Fatalf("system prompt mismatch: %q", request.SystemMessage)
			}
			return []string{"search"}, nil
		}),
	)
	request := selectorRequest(t, "model", mustTool(t, "search"))
	_, err := selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestToolSelectorCallbackErrorPropagates(t *testing.T) {
	wantErr := errors.New("selection failed")
	selector := NewLLMToolSelectorMiddleware(WithToolSelectorFunc(func(ToolSelectionRequest) ([]string, error) {
		return nil, wantErr
	}))
	request := selectorRequest(t, "model", mustTool(t, "search"))
	_, err := selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected selection error, got %v", err)
	}
}

func TestToolSelectorRequiresSelectionFunc(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware()
	request := selectorRequest(t, "model", mustTool(t, "search"))
	_, err := selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "requires a selection function") {
		t.Fatalf("expected selection function error, got %v", err)
	}
}

func TestToolSelectorStructuredErrorPropagates(t *testing.T) {
	fake := &failingStructuredChatModel{FakeChatModel: language.NewFakeChatModel(), err: errors.New("structured down")}
	selector := NewLLMToolSelectorMiddleware(WithToolSelectorModel(fake))
	request := selectorRequest(t, "model", mustTool(t, "search"))
	_, err := selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "structured output") {
		t.Fatalf("expected structured output error, got %v", err)
	}
}

func TestToolSelectorStructuredParseFailures(t *testing.T) {
	cases := []struct {
		name     string
		response messages.Message
		wantErr  string
	}{
		{"invalid json", messages.AI("not json"), "parse structured response"},
		{"missing tools key", messages.AI(`{"other":["search"]}`), "missing"},
		{"tools not array", messages.AI(`{"tools":"search"}`), "is not an array"},
		{"item not string", messages.AI(`{"tools":[42]}`), "is not a string"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			fake := newStructuredFakeChatModel(tt.response)
			selector := NewLLMToolSelectorMiddleware(WithToolSelectorModel(fake))
			request := selectorRequest(t, "model", mustTool(t, "search"))
			_, err := selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
				return ModelResponse{}, nil
			})
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected %q error, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestToolSelectorAllToolsAlwaysIncludedPassesThrough(t *testing.T) {
	// When every tool is always-included there is nothing to select, so the
	// request passes through without invoking any selection mechanism.
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorAlwaysInclude("search"),
		WithToolSelectorFunc(func(ToolSelectionRequest) ([]string, error) {
			t.Fatal("selection should not run when nothing is selectable")
			return nil, nil
		}),
	)
	request := selectorRequest(t, "model", mustTool(t, "search"))
	called := false
	_, err := selector.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		called = true
		if len(req.Tools) != 1 {
			t.Fatalf("request should pass through unchanged: %#v", req.Tools)
		}
		return ModelResponse{}, nil
	})
	if err != nil || !called {
		t.Fatalf("wrap model call: %v called=%v", err, called)
	}
}

func TestToolSelectorDeduplicatesSelection(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware(WithToolSelectorFunc(func(ToolSelectionRequest) ([]string, error) {
		return []string{"search", "search"}, nil
	}))
	request := selectorRequest(t, "model", mustTool(t, "search"), mustTool(t, "calc"))
	_, err := selector.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if len(req.Tools) != 1 {
			t.Fatalf("duplicate selection should yield one tool: %#v", req.Tools)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestToolSelectorMaxToolsTruncates(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorMaxTools(1),
		WithToolSelectorFunc(func(request ToolSelectionRequest) ([]string, error) {
			if !strings.Contains(request.SystemMessage, "only the first 1 will be used") {
				t.Fatalf("max tools hint missing from system prompt: %q", request.SystemMessage)
			}
			return []string{"search", "calc"}, nil
		}),
	)
	request := selectorRequest(t, "model", mustTool(t, "search"), mustTool(t, "calc"))
	_, err := selector.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if len(req.Tools) != 1 || req.Tools[0].(interface{ Name() string }).Name() != "search" {
			t.Fatalf("max tools truncation mismatch: %#v", req.Tools)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestToolSelectorAlwaysIncludeMissingTool(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware(
		WithToolSelectorAlwaysInclude("ghost"),
		WithToolSelectorFunc(func(ToolSelectionRequest) ([]string, error) { return nil, nil }),
	)
	request := selectorRequest(t, "model", mustTool(t, "search"))
	_, err := selector.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "always_include") {
		t.Fatalf("expected always_include error, got %v", err)
	}
}

func TestToolSelectorNoUserMessage(t *testing.T) {
	selector := NewLLMToolSelectorMiddleware(WithToolSelectorFunc(func(ToolSelectionRequest) ([]string, error) {
		return []string{"search"}, nil
	}))
	request, err := NewModelRequest(ModelRequest{
		Model:    "model",
		Messages: []messages.Message{messages.AI("no human here")},
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
