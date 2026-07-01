package middleware

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestModelRequestCreatesWithVariousSystemInputs(t *testing.T) {
	tests := []struct {
		name           string
		systemMessage  *messages.Message
		systemPrompt   string
		expectMessage  bool
		expectedPrompt string
	}{
		{
			name:           "system message",
			systemMessage:  ptr(messages.System("You are helpful")),
			expectMessage:  true,
			expectedPrompt: "You are helpful",
		},
		{
			name: "none",
		},
		{
			name:           "system prompt string",
			systemPrompt:   "You are helpful",
			expectMessage:  true,
			expectedPrompt: "You are helpful",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request, err := NewModelRequest(ModelRequest{
				Model:         "fake",
				SystemMessage: tt.systemMessage,
				SystemPrompt:  tt.systemPrompt,
				Messages:      []messages.Message{messages.Human("Hi")},
			})
			if err != nil {
				t.Fatalf("new model request: %v", err)
			}

			if !tt.expectMessage {
				if request.SystemMessage != nil {
					t.Fatalf("expected nil system message, got %#v", *request.SystemMessage)
				}
				if request.SystemPromptText() != "" {
					t.Fatalf("expected empty system prompt, got %q", request.SystemPromptText())
				}
				return
			}

			if request.SystemMessage == nil {
				t.Fatal("expected system message")
			}
			if request.SystemMessage.Content != tt.expectedPrompt {
				t.Fatalf("system message mismatch: got %q", request.SystemMessage.Content)
			}
			if request.SystemPromptText() != tt.expectedPrompt {
				t.Fatalf("system prompt mismatch: got %q", request.SystemPromptText())
			}
		})
	}
}

func TestModelRequestSystemPromptWithContentBlocks(t *testing.T) {
	systemMessage := messages.System("")
	systemMessage.ContentBlocks = []messages.ContentBlock{
		{"type": "text", "text": "Part 1"},
		{"type": "text", "text": "Part 2"},
	}
	request, err := NewModelRequest(ModelRequest{
		Model:         "fake",
		SystemMessage: &systemMessage,
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	prompt := request.SystemPromptText()
	if !strings.Contains(prompt, "Part 1") || !strings.Contains(prompt, "Part 2") {
		t.Fatalf("system prompt did not include content block text: %q", prompt)
	}
}

func TestModelRequestOverrideSystemMessageAndPrompt(t *testing.T) {
	original, err := NewModelRequest(ModelRequest{
		Model:         "fake",
		SystemMessage: ptr(messages.System("Original")),
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	withMessage, err := original.Override(WithSystemMessage(ptr(messages.System("New"))))
	if err != nil {
		t.Fatalf("override system message: %v", err)
	}
	if withMessage.SystemPromptText() != "New" {
		t.Fatalf("override message mismatch: got %q", withMessage.SystemPromptText())
	}
	if original.SystemPromptText() != "Original" {
		t.Fatalf("original was mutated: got %q", original.SystemPromptText())
	}

	withPrompt, err := original.Override(WithSystemPrompt("New prompt"))
	if err != nil {
		t.Fatalf("override system prompt: %v", err)
	}
	if withPrompt.SystemPromptText() != "New prompt" {
		t.Fatalf("override prompt mismatch: got %q", withPrompt.SystemPromptText())
	}
	if original.SystemPromptText() != "Original" {
		t.Fatalf("original was mutated after prompt override: got %q", original.SystemPromptText())
	}
}

func TestModelRequestOverrideSystemPromptToNone(t *testing.T) {
	original, err := NewModelRequest(ModelRequest{
		Model:         "fake",
		SystemMessage: ptr(messages.System("Original")),
	})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	overridden, err := original.Override(WithSystemPromptNone())
	if err != nil {
		t.Fatalf("override system prompt none: %v", err)
	}
	if overridden.SystemMessage != nil {
		t.Fatalf("expected nil system message, got %#v", *overridden.SystemMessage)
	}
	if overridden.SystemPromptText() != "" {
		t.Fatalf("expected empty system prompt, got %q", overridden.SystemPromptText())
	}
}

func TestModelRequestCannotSetBothSystemPromptAndSystemMessage(t *testing.T) {
	_, err := NewModelRequest(ModelRequest{
		Model:         "fake",
		SystemPrompt:  "String prompt",
		SystemMessage: ptr(messages.System("Message prompt")),
	})
	if err == nil {
		t.Fatal("expected constructor error")
	}
	if !strings.Contains(err.Error(), "Cannot specify both") {
		t.Fatalf("unexpected constructor error: %v", err)
	}

	request, err := NewModelRequest(ModelRequest{Model: "fake"})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}
	_, err = request.Override(
		WithSystemPrompt("String prompt"),
		WithSystemMessage(ptr(messages.System("Message prompt"))),
	)
	if err == nil {
		t.Fatal("expected override error")
	}
	if !strings.Contains(err.Error(), "Cannot specify both") {
		t.Fatalf("unexpected override error: %v", err)
	}
}

func ptr[T any](value T) *T {
	return &value
}
