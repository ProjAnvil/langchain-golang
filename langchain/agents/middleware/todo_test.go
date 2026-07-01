package middleware

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestTodoListMiddlewareWrapModelCallAddsSystemPrompt(t *testing.T) {
	middleware, err := NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}
	request, err := NewModelRequest(ModelRequest{Model: "model"})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		if request.SystemMessage == nil || len(request.SystemMessage.ContentBlocks) != 1 {
			t.Fatalf("system message mismatch: %#v", request.SystemMessage)
		}
		if !strings.Contains(request.SystemMessage.ContentBlocks[0]["text"].(string), "write_todos") {
			t.Fatalf("system prompt missing todo text: %#v", request.SystemMessage.ContentBlocks)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}

func TestTodoListMiddlewareRejectsParallelWriteTodos(t *testing.T) {
	middleware, err := NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: WriteTodosToolName}, {ID: "2", Name: WriteTodosToolName}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 2 || msgs[0].ResponseMetadata["status"] != "error" || msgs[1].ToolCallID != "2" {
		t.Fatalf("error messages mismatch: %#v", msgs)
	}
}
