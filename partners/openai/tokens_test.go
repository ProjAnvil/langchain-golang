package openai

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// Mirrors libs/partners/openai/tests/unit_tests/test_token_counts.py in
// structure. Python asserts exact tiktoken counts; the Go port approximates
// (see GetTokenIDs doc), so we assert determinism, positivity, and the
// get_num_tokens == len(get_token_ids) invariant instead.
func TestChatModelGetNumTokensApproximate(t *testing.T) {
	model := NewChatModel(modelconfig.WithModel("gpt-4.1"))
	text := "表情符号是\n🦜🔗"
	first := model.GetNumTokens(text)
	if first <= 0 {
		t.Fatalf("GetNumTokens = %d, want > 0", first)
	}
	if again := model.GetNumTokens(text); again != first {
		t.Fatalf("GetNumTokens not deterministic: %d vs %d", first, again)
	}
	if got := model.GetNumTokens(""); got != 0 {
		t.Fatalf("GetNumTokens(empty) = %d, want 0", got)
	}
	if ids := model.GetTokenIDs(text); len(ids) != first {
		t.Fatalf("len(GetTokenIDs) = %d, want GetNumTokens %d", len(ids), first)
	}
	// language-package dispatch hits the model's own counter.
	if got := language.GetNumTokens(model, text); got != first {
		t.Fatalf("language.GetNumTokens = %d, want %d", got, first)
	}
}

// Hand-computed overhead cases mirroring test_base.py::test_get_num_tokens_from_messages
// (tokens_per_message=3, tokens_per_name=1, +3 reply primer; counts use the
// 4-runes-per-token approximation, so "user"=1, "hi"=1, "assistant"=3).
func TestChatModelGetNumTokensFromMessagesOverhead(t *testing.T) {
	model := NewChatModel(modelconfig.WithModel("gpt-4.1"))

	// 3 (per message) + 1 (role "user") + 1 ("hi") + 3 (primer) = 8.
	got, err := model.GetNumTokensFromMessages([]messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 8 {
		t.Fatalf("single human message = %d, want 8", got)
	}

	// Adding name "bob" adds count("bob")=1 plus tokens_per_name=1: 8 + 2 = 10.
	named := messages.Human("hi")
	named.Name = "bob"
	got, err = model.GetNumTokensFromMessages([]messages.Message{named})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 10 {
		t.Fatalf("named human message = %d, want 10", got)
	}

	// gpt-3.5-turbo-0301: tokens_per_message=4, tokens_per_name=-1:
	// 4 + 1 + 1 + (1 - 1) + 3 = 9.
	legacy := NewChatModel(modelconfig.WithModel("gpt-3.5-turbo-0301"))
	got, err = legacy.GetNumTokensFromMessages([]messages.Message{named})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 9 {
		t.Fatalf("gpt-3.5-turbo-0301 named message = %d, want 9", got)
	}

	// Tool message: tool_call_id contributes a flat 3
	// (3 + 1 role "tool" + 1 "ok" + 3 tool_call_id + 3 primer = 11).
	got, err = model.GetNumTokensFromMessages([]messages.Message{messages.Tool("call_1", "ok")})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 11 {
		t.Fatalf("tool message = %d, want 11", got)
	}

	// AI message with a tool call: 3 + 3 (role "assistant") + 1 ("bar") +
	// 4 (`{"arg1":"arg1"}` is 16 runes) + 3 primer = 14.
	ai := messages.Message{
		Role:      messages.RoleAI,
		ToolCalls: []messages.ToolCall{{ID: "foo", Name: "bar", Args: map[string]any{"arg1": "arg1"}}},
	}
	got, err = model.GetNumTokensFromMessages([]messages.Message{ai})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 14 {
		t.Fatalf("tool-call message = %d, want 14", got)
	}

	// System message: exercises the RoleSystem arm of openAIWireRole
	// (3 + 2 for role "system" (6 runes) + 1 "hi" + 3 primer = 9).
	got, err = model.GetNumTokensFromMessages([]messages.Message{messages.System("hi")})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 9 {
		t.Fatalf("system message = %d, want 9", got)
	}

	// Unknown role: exercises the default arm of openAIWireRole
	// ("function" is 8 runes = 2 tokens; 3 + 2 + 1 "hi" + 3 primer = 9).
	got, err = model.GetNumTokensFromMessages([]messages.Message{{Role: "function", Content: "hi"}})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 9 {
		t.Fatalf("custom-role message = %d, want 9", got)
	}

	// TextBlock content: exercises the TextBlock arm of the ContentBlocks
	// switch (3 + 1 role "user" + 1 "hi" + 3 primer = 8).
	texty := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.TextBlock{Text: "hi"},
	})
	got, err = model.GetNumTokensFromMessages([]messages.Message{texty})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 8 {
		t.Fatalf("text-block message = %d, want 8", got)
	}
}

// Low-detail images cost a flat 85 tokens; other images are ignored (the Go
// port never fetches image URLs — documented divergence from Python, which
// sizes them via PIL/httpx).
func TestChatModelGetNumTokensFromMessagesImages(t *testing.T) {
	model := NewChatModel(modelconfig.WithModel("gpt-4.1"))
	low := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{URL: "https://example.com/x.png", Extras: map[string]any{"detail": "low"}},
	})
	got, err := model.GetNumTokensFromMessages([]messages.Message{low})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 92 { // 3 + 1 (role) + 85 + 3 primer
		t.Fatalf("low-detail image message = %d, want 92", got)
	}

	high := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{URL: "https://example.com/x.png"},
	})
	got, err = model.GetNumTokensFromMessages([]messages.Message{high})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 7 { // 3 + 1 (role) + 3 primer; unfetched image ignored
		t.Fatalf("default-detail image message = %d, want 7", got)
	}
}

// Python raises NotImplementedError for models outside the
// gpt-3.5-turbo/gpt-4/gpt-5 families (chat_models/base.py:2142-2149).
func TestChatModelGetNumTokensFromMessagesUnsupportedModel(t *testing.T) {
	model := NewChatModel(modelconfig.WithModel("o3"))
	_, err := model.GetNumTokensFromMessages([]messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected error for o3")
	}
	if !strings.Contains(err.Error(), "not presently implemented") || !strings.Contains(err.Error(), "o3") {
		t.Fatalf("error = %v, want 'not presently implemented' mentioning o3", err)
	}
}
