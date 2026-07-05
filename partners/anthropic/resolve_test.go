package anthropic

import (
	"testing"

	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// TestAnthropicSelfRegisters proves init() registered the "anthropic" factory
// into chatmodels.Resolve, mirroring partners/openai/resolve_test.go. It also
// pins LLMType(); the StructuredCaller satisfaction is locked by the package-
// level compile guard added in Task 2 (var _ language.StructuredCaller).
func TestAnthropicSelfRegisters(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("ANTHROPIC_BASE_URL", "https://example.test/v1")

	model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{Provider: "anthropic", Model: "claude-sonnet-4-5"})
	if err != nil {
		t.Fatalf("Resolve anthropic: %v", err)
	}
	cm, ok := model.(ChatModel)
	if !ok {
		t.Fatalf("resolved anthropic model is %T, not ChatModel", model)
	}
	if cm.LLMType() != "anthropic-chat" {
		t.Fatalf("LLMType()=%q, want anthropic-chat", cm.LLMType())
	}
}
