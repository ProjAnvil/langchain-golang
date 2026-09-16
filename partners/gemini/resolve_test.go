package gemini

import (
	"testing"

	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// TestGeminiSelfRegisters proves init() registered the "gemini" factory into
// chatmodels.Resolve, mirroring partners/anthropic/resolve_test.go.
func TestGeminiSelfRegisters(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	t.Setenv("GOOGLE_GEMINI_BASE_URL", "https://example.test")

	model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{Provider: "gemini", Model: "gemini-2.5-pro"})
	if err != nil {
		t.Fatalf("Resolve gemini: %v", err)
	}
	cm, ok := model.(ChatModel)
	if !ok {
		t.Fatalf("resolved gemini model is %T, not ChatModel", model)
	}
	if cm.config.APIKey != "gemini-key" {
		t.Fatalf("APIKey: %q", cm.config.APIKey)
	}
	if cm.config.BaseURL != "https://example.test" {
		t.Fatalf("BaseURL: %q", cm.config.BaseURL)
	}
	if cm.config.Model != "gemini-2.5-pro" {
		t.Fatalf("Model: %q", cm.config.Model)
	}
	if cm.LLMType() != "chat-google-generative-ai" {
		t.Fatalf("LLMType()=%q", cm.LLMType())
	}
}

// TestGeminiFactoryGoogleKeyPrecedence pins the key precedence: GOOGLE_API_KEY
// overrides GEMINI_API_KEY when both are set (the genai SDK's documented
// rule, mirrored by the factory reading GOOGLE last).
func TestGeminiFactoryGoogleKeyPrecedence(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	t.Setenv("GOOGLE_API_KEY", "google-key")

	model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{Provider: "gemini"})
	if err != nil {
		t.Fatalf("Resolve gemini: %v", err)
	}
	cm, ok := model.(ChatModel)
	if !ok {
		t.Fatalf("resolved gemini model is %T, not ChatModel", model)
	}
	if cm.config.APIKey != "google-key" {
		t.Fatalf("APIKey: %q want google-key", cm.config.APIKey)
	}
	// Empty model spec falls back to the adapter default.
	if cm.config.Model != defaultModel {
		t.Fatalf("default model: %q", cm.config.Model)
	}
}
