package middleware

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestTodoListMiddlewareToolInvocation(t *testing.T) {
	middleware, err := NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}
	if len(middleware.Tools) != 1 || middleware.Tools[0].Name() != WriteTodosToolName {
		t.Fatalf("tools mismatch: %#v", middleware.Tools)
	}
	result, err := middleware.Tools[0].Invoke(context.Background(), map[string]any{
		"todos": []any{map[string]any{"content": "task", "status": "pending"}},
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !strings.Contains(result.Content, "Updated todo list") {
		t.Fatalf("result content mismatch: %q", result.Content)
	}
}

func TestTodoListMiddlewareWrapModelCallWithoutSystemMessage(t *testing.T) {
	middleware, err := NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}
	request, err := NewModelRequest(ModelRequest{Model: "model"})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if req.SystemMessage == nil {
			t.Fatal("expected system message to be created")
		}
		if got := req.SystemPromptText(); !strings.Contains(got, "write_todos") {
			t.Fatalf("system prompt missing todo instructions: %q", got)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestTodoListMiddlewareWrapModelCallAppendsToExistingSystem(t *testing.T) {
	middleware, err := NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}
	request, err := NewModelRequest(ModelRequest{Model: "model", SystemPrompt: "base instructions"})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if req.SystemMessage == nil || req.SystemMessage.Content != "base instructions" {
			t.Fatalf("base system content should be preserved: %#v", req.SystemMessage)
		}
		found := false
		for _, block := range req.SystemMessage.ContentBlocks {
			if text, ok := messages.BlockToMap(block)["text"].(string); ok && strings.Contains(text, "write_todos") {
				found = true
			}
		}
		if !found {
			t.Fatalf("todo instructions should be appended as a content block: %#v", req.SystemMessage.ContentBlocks)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestTodoListMiddlewareWrapModelCallDefaultPrompt(t *testing.T) {
	// A zero-value middleware falls back to the default system prompt.
	middleware := &TodoListMiddleware{}
	request, err := NewModelRequest(ModelRequest{Model: "model"})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if got := req.SystemPromptText(); !strings.Contains(got, "write_todos") {
			t.Fatalf("default prompt mismatch: %q", got)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestTodoListMiddlewareAfterModel(t *testing.T) {
	middleware, err := NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}

	// No messages: nil update.
	update, err := middleware.AfterModel(context.Background(), map[string]any{})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without messages: %#v %v", update, err)
	}

	// AI message without tool calls: nil update.
	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{messages.AI("hi")}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without tool calls: %#v %v", update, err)
	}

	// A single write_todos call is fine.
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: WriteTodosToolName}}
	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update for a single call: %#v %v", update, err)
	}

	// Parallel write_todos calls produce error tool messages.
	ai.ToolCalls = []messages.ToolCall{
		{ID: "1", Name: WriteTodosToolName},
		{ID: "2", Name: WriteTodosToolName},
		{ID: "3", Name: "search"},
	}
	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 2 {
		t.Fatalf("expected two error messages, got %#v", msgs)
	}
	for _, msg := range msgs {
		if msg.ResponseMetadata["status"] != "error" || !strings.Contains(msg.Content, "never be called multiple times") {
			t.Fatalf("error message mismatch: %#v", msg)
		}
	}
}
