package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestLLMToolEmulatorEmulatesSelectedTool(t *testing.T) {
	emulator := NewLLMToolEmulator([]string{"search"}, func(request ToolCallRequest, prompt string) (string, error) {
		if !strings.Contains(prompt, "Tool: search") || !strings.Contains(prompt, "Arguments: map[q:test]") {
			t.Fatalf("prompt mismatch: %q", prompt)
		}
		return "emulated result", nil
	})
	response, err := emulator.WrapToolCall(context.Background(), ToolCallRequest{
		ToolCall: ToolCall{Name: "search", ID: "1", Args: map[string]any{"q": "test"}},
		Tool:     mustTool(t, "search"),
	}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("should not call handler")
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}
	if response.Content != "emulated result" || response.ToolCallID != "1" || response.Name != "search" {
		t.Fatalf("response mismatch: %#v", response)
	}
}

func TestLLMToolEmulatorPassesThroughUnselectedTool(t *testing.T) {
	emulator := NewLLMToolEmulator([]string{"search"}, nil)
	response, err := emulator.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "calc", ID: "2"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Tool("2", "real result"), nil
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}
	if response.Content != "real result" {
		t.Fatalf("response mismatch: %#v", response)
	}
}
