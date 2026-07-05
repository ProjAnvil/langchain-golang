package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestLLMToolEmulatorEmulatesSelectedTool(t *testing.T) {
	emulator := NewLLMToolEmulator([]string{"search"}, WithToolEmulatorFunc(func(request ToolCallRequest, prompt string) (string, error) {
		if !strings.Contains(prompt, "Tool: search") || !strings.Contains(prompt, "Arguments: map[q:test]") {
			t.Fatalf("prompt mismatch: %q", prompt)
		}
		return "emulated result", nil
	}))
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
	emulator := NewLLMToolEmulator([]string{"search"})
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

// TestLLMToolEmulatorUsesStructuredOutput exercises the spec-mandated primary
// path: when a real language.ChatModel is configured (via WithToolEmulatorModel)
// the middleware routes through language.InvokeStructured instead of the
// ToolEmulationFunc callback. ToolCallRequest has no model field, so the model
// must be supplied explicitly via the option.
func TestLLMToolEmulatorUsesStructuredOutput(t *testing.T) {
	fake := newStructuredFakeChatModel(messages.AI(`{"output":"emulated"}`))
	emulator := NewLLMToolEmulator(
		[]string{"search"},
		WithToolEmulatorModel(fake),
		// Deliberately NO WithToolEmulatorFunc — the structured path must win
		// and the callback (when set) is ignored.
	)

	response, err := emulator.WrapToolCall(context.Background(), ToolCallRequest{
		ToolCall: ToolCall{Name: "search", ID: "1", Args: map[string]any{"q": "test"}},
		Tool:     mustTool(t, "search"),
	}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("should not call handler")
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}

	if fake.structuredCalls != 1 {
		t.Fatalf("expected InvokeStructured called once, got %d", fake.structuredCalls)
	}
	if response.Content != "emulated" {
		t.Fatalf("content mismatch: %#v", response)
	}
	if response.ToolCallID != "1" {
		t.Fatalf("tool call id mismatch: %#v", response)
	}
	if response.Name != "search" {
		t.Fatalf("name mismatch: %#v", response)
	}
}

// TestLLMToolEmulatorStructuredBeatsCallback locks the
// structured-precedence-over-callback behavior: when BOTH a real ChatModel and
// a ToolEmulationFunc are configured, the structured (InvokeStructured) path
// runs and the callback is NEVER invoked.
func TestLLMToolEmulatorStructuredBeatsCallback(t *testing.T) {
	fake := newStructuredFakeChatModel(messages.AI(`{"output":"emulated"}`))
	emulator := NewLLMToolEmulator(
		[]string{"search"},
		WithToolEmulatorModel(fake),
		WithToolEmulatorFunc(func(ToolCallRequest, string) (string, error) {
			t.Fatalf("ToolEmulationFunc callback must not be invoked when structured path is available")
			return "", nil
		}),
	)

	response, err := emulator.WrapToolCall(context.Background(), ToolCallRequest{
		ToolCall: ToolCall{Name: "search", ID: "1", Args: map[string]any{"q": "test"}},
		Tool:     mustTool(t, "search"),
	}, func(context.Context, ToolCallRequest) (messages.Message, error) {
		return messages.Message{}, errors.New("should not call handler")
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}

	if fake.structuredCalls != 1 {
		t.Fatalf("expected InvokeStructured called once, got %d", fake.structuredCalls)
	}
	if response.Content != "emulated" {
		t.Fatalf("content mismatch: %#v", response)
	}
}
