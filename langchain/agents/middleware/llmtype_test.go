package middleware

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// fakeLLMTypeModel is a minimal stand-in implementing LLMTypeProvider so the
// duck-type assertion in summarization (Task 7) can be exercised without a
// real partner import.
type fakeLLMTypeModel struct {
	llmType string
}

func (f fakeLLMTypeModel) LLMType() string { return f.llmType }

// Compile-time guard: fakeLLMTypeModel satisfies LLMTypeProvider.
var _ LLMTypeProvider = fakeLLMTypeModel{}

func TestLLMTypeProviderDuckTyping(t *testing.T) {
	cases := []struct {
		model   any
		want    string
		present bool
	}{
		{fakeLLMTypeModel{llmType: "anthropic-chat"}, "anthropic-chat", true},
		{fakeLLMTypeModel{llmType: "ollama-chat"}, "ollama-chat", true},
		{nil, "", false},
	}
	for _, tt := range cases {
		got, ok := "", false
		if lt, asserts := tt.model.(LLMTypeProvider); asserts {
			ok = true
			got = lt.LLMType()
		}
		if ok != tt.present {
			t.Fatalf("model=%v: LLMTypeProvider present=%v, want %v", tt.model, ok, tt.present)
		}
		if ok && got != tt.want {
			t.Fatalf("LLMType()=%q, want %q", got, tt.want)
		}
	}
	// Reference a messages.Message so the import stays used in this file.
	_ = messages.Message{}
}
