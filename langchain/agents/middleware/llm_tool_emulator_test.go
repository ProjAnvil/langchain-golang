package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestLLMToolEmulatorEmulateAllTools(t *testing.T) {
	// nil toolNames emulates every tool.
	emulator := NewLLMToolEmulator(nil, WithToolEmulatorFunc(func(ToolCallRequest, string) (string, error) {
		return "fake", nil
	}))
	if !emulator.EmulateAll {
		t.Fatal("expected EmulateAll for nil tool names")
	}
	response, err := emulator.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "anything", ID: "1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("should not call handler")
	})
	if err != nil || response.Content != "fake" {
		t.Fatalf("emulate-all mismatch: %#v %v", response, err)
	}
}

func TestLLMToolEmulatorEmulateFuncError(t *testing.T) {
	wantErr := errors.New("emulation failed")
	emulator := NewLLMToolEmulator([]string{"search"}, WithToolEmulatorFunc(func(ToolCallRequest, string) (string, error) {
		return "", wantErr
	}))
	_, err := emulator.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("should not call handler")
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected emulation error, got %v", err)
	}
}

func TestLLMToolEmulatorRequiresEmulationFunc(t *testing.T) {
	emulator := NewLLMToolEmulator([]string{"search"})
	_, err := emulator.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("should not call handler")
	})
	if err == nil || !strings.Contains(err.Error(), "requires an emulation function") {
		t.Fatalf("expected emulation function error, got %v", err)
	}
}

func TestLLMToolEmulatorStructuredParseFailures(t *testing.T) {
	cases := []struct {
		name     string
		response messages.Message
		wantErr  string
	}{
		{"invalid json", messages.AI("not json"), "parse structured response"},
		{"missing output key", messages.AI(`{"other":"x"}`), "missing"},
		{"output not a string", messages.AI(`{"output":42}`), "is not a string"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			fake := newStructuredFakeChatModel(tt.response)
			emulator := NewLLMToolEmulator([]string{"search"}, WithToolEmulatorModel(fake))
			_, err := emulator.WrapToolCall(context.Background(), ToolCallRequest{ToolCall: ToolCall{Name: "search", ID: "1"}}, func(context.Context, ToolCallRequest) (messages.Message, error) {
				return messages.Message{}, errors.New("should not call handler")
			})
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected %q error, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestBuildToolEmulationPromptWithoutTool(t *testing.T) {
	prompt := BuildToolEmulationPrompt(ToolCallRequest{ToolCall: ToolCall{Name: "search", Args: map[string]any{"q": "x"}}})
	if !strings.Contains(prompt, "No description available") || !strings.Contains(prompt, "Tool: search") {
		t.Fatalf("prompt mismatch: %q", prompt)
	}
}
