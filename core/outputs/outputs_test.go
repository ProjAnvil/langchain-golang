package outputs

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestMergeGenerationChunks(t *testing.T) {
	got := MergeGenerationChunks(
		NewGeneration("hel", map[string]any{"a": map[string]any{"x": 1}}),
		NewGeneration("lo", map[string]any{"a": map[string]any{"y": 2}}),
	)
	if got.Text != "hello" {
		t.Fatalf("Text = %q, want hello", got.Text)
	}
	info := got.GenerationInfo["a"].(map[string]any)
	if info["x"] != 1 || info["y"] != 2 {
		t.Fatalf("nested info not merged: %#v", got.GenerationInfo)
	}
}

func TestChatGenerationTextFromBlocks(t *testing.T) {
	gen := NewChatGeneration(messages.Message{
		Role: messages.RoleAI,
		ContentBlocks: []messages.ContentBlock{
			messages.TextBlock{Text: "a"},
			messages.NonStandardContentBlock{Type: "", Value: map[string]any{"text": "b"}},
			messages.NonStandardContentBlock{Type: "image", Value: map[string]any{"text": "ignored"}},
		},
	}, nil)
	if gen.Text != "ab" {
		t.Fatalf("Text = %q, want ab", gen.Text)
	}
}
