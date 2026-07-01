package middleware

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestClearToolUsesEditClearsOldToolResults(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{
		{ID: "1", Name: "search", Args: map[string]any{"query": "old"}},
		{ID: "2", Name: "calc", Args: map[string]any{"expr": "1+1"}},
	}
	msgs := []messages.Message{
		messages.Human("hi"),
		ai,
		messages.Tool("1", "very large old result"),
		messages.Tool("2", "recent result"),
	}
	msgs[2].Name = "search"
	msgs[3].Name = "calc"

	edit := ClearToolUsesEdit{
		Trigger:         1,
		Keep:            1,
		ClearToolInputs: true,
		Placeholder:     "[cleared]",
	}
	edited := edit.Apply(msgs, ApproximateTokenCount)

	if edited[2].Content != "[cleared]" {
		t.Fatalf("old tool result not cleared: %#v", edited[2])
	}
	if edited[3].Content != "recent result" {
		t.Fatalf("recent tool result should be kept: %#v", edited[3])
	}
	if edited[1].ToolCalls[0].Args == nil || len(edited[1].ToolCalls[0].Args) != 0 {
		t.Fatalf("tool inputs not cleared: %#v", edited[1].ToolCalls[0].Args)
	}
	if edited[2].ResponseMetadata["context_editing"] == nil {
		t.Fatalf("context editing metadata missing: %#v", edited[2].ResponseMetadata)
	}
	if msgs[2].Content == "[cleared]" {
		t.Fatal("original messages mutated")
	}
}

func TestContextEditingMiddlewareWrapModelCall(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	request, err := NewModelRequest(ModelRequest{
		Model: "model",
		Messages: []messages.Message{
			ai,
			messages.Tool("1", "old result"),
		},
	})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	middleware := NewContextEditingMiddleware(ClearToolUsesEdit{Trigger: 0, Keep: 0, Placeholder: "[x]"})
	middleware.CountTokens = func([]messages.Message) int { return 10 }

	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, request ModelRequest) (ModelResponse, error) {
		if request.Messages[1].Content != "[x]" {
			t.Fatalf("edited content mismatch: %#v", request.Messages)
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}
