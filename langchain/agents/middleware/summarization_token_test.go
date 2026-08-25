package middleware

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// These tests reuse fakeLLMTypeModel from llmtype_test.go (Task 1, same package).

// TestResolveTokenCounter_AnthropicUsesCharsPerToken mirrors Python's
// _get_approximate_token_counter (summarization.py:208-216): an anthropic-chat
// model selects the 3.3 chars/token counter; a non-anthropic model selects the
// default 4.0 chars/token counter; a model without LLMType also defaults.
func TestResolveTokenCounter_AnthropicUsesCharsPerToken(t *testing.T) {
	anthropicCounter := resolveTokenCounter(fakeLLMTypeModel{"anthropic-chat"}, nil)
	otherCounter := resolveTokenCounter(fakeLLMTypeModel{"openai-chat"}, nil)
	noLLMType := resolveTokenCounter(nil, nil)

	// A single 97-char content: the default counter sees 3+ceil(97/4)=28; the
	// anthropic counter sees 3+ceil(97/3.3)=33. They must disagree, proving
	// provider-specific tuning kicked in.
	longWord := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" // 97 'a's
	if len(longWord) < 90 {
		t.Fatalf("test setup: longWord too short")
	}
	msg := []messages.Message{{Role: messages.RoleAI, Content: longWord}}
	if anthropicCounter(msg) == otherCounter(msg) {
		t.Fatalf("anthropic counter should differ from default: got %d for both", anthropicCounter(msg))
	}
	if anthropicCounter(msg) < 25 {
		t.Fatalf("anthropic counter should be ~chars/3.3 >= 25, got %d", anthropicCounter(msg))
	}
	if otherCounter(msg) != noLLMType(msg) {
		t.Fatalf("non-anthropic and no-LLMType should both use the default counter")
	}
}

// TestShouldSummarizeBasedOnReportedTokens mirrors Python's
// _should_summarize_based_on_reported_tokens (summarization.py:561-581):
// when the last AI message's reported total_tokens crosses the threshold and
// the message's model_provider matches the model's LLMType prefix, trigger.
func TestShouldSummarizeBasedOnReportedTokens(t *testing.T) {
	mw := &SummarizationMiddleware{Model: fakeLLMTypeModel{"anthropic-chat"}}

	msgs := []messages.Message{
		messages.Human("hi"),
		{Role: messages.RoleAI, Content: "hello",
			UsageMetadata:    messages.UsageMetadata{TotalTokens: 5000},
			ResponseMetadata: map[string]any{"model_provider": "anthropic"}},
	}
	if !mw.shouldSummarizeBasedOnReportedTokens(msgs, 4000) {
		t.Fatalf("expected reported-tokens signal to fire (5000 >= 4000, provider matches)")
	}
	if mw.shouldSummarizeBasedOnReportedTokens(msgs, 6000) {
		t.Fatalf("expected no fire when reported tokens (5000) below threshold (6000)")
	}

	// Provider mismatch (message says ollama, model is anthropic) → no fire.
	mismatch := []messages.Message{
		{Role: messages.RoleAI, Content: "x",
			UsageMetadata:    messages.UsageMetadata{TotalTokens: 5000},
			ResponseMetadata: map[string]any{"model_provider": "ollama"}},
	}
	if mw.shouldSummarizeBasedOnReportedTokens(mismatch, 4000) {
		t.Fatalf("expected no fire on provider mismatch")
	}
}

// TestResolveTokenCounter_UsageMetadataScalingWired verifies that the counters
// returned by resolveTokenCounter enable usage-metadata scaling, mirroring
// Python's partial(count_tokens_approximately, use_usage_metadata_scaling=True).
func TestResolveTokenCounter_UsageMetadataScalingWired(t *testing.T) {
	ai := messages.AI("hi")
	ai.UsageMetadata = messages.UsageMetadata{TotalTokens: 20}
	ai.ResponseMetadata = map[string]any{"model_provider": "anthropic"}
	msgs := []messages.Message{
		messages.Human("hello world"),
		ai,
		messages.Human("more text here"),
	}

	// Anthropic (3.3 chars/token): unscaled = 8+7+9 = 24, approx at AI = 15,
	// factor 20/15 → clamped 1.25 → ceil(24*1.25) = 30.
	if got := resolveTokenCounter(fakeLLMTypeModel{"anthropic-chat"}, nil)(msgs); got != 30 {
		t.Fatalf("anthropic scaled counter = %d, want 30", got)
	}
	// Default (4.0 chars/token): unscaled = 7+6+8 = 21, factor 20/13 → 1.25
	// → ceil(21*1.25) = 27.
	if got := resolveTokenCounter(fakeLLMTypeModel{"openai-chat"}, nil)(msgs); got != 27 {
		t.Fatalf("default scaled counter = %d, want 27", got)
	}
}
