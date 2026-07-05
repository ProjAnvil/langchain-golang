package ollama

import (
	"testing"

	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// TestOllamaSelfRegisters proves init() registered the "ollama" factory,
// mirroring partners/openai/resolve_test.go.
func TestOllamaSelfRegisters(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://example.test:11434")
	model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{Provider: "ollama", Model: "llama3"})
	if err != nil {
		t.Fatalf("Resolve ollama: %v", err)
	}
	cm, ok := model.(ChatModel)
	if !ok {
		t.Fatalf("resolved ollama model is %T, not ChatModel", model)
	}
	if cm.LLMType() != "ollama-chat" {
		t.Fatalf("LLMType()=%q, want ollama-chat", cm.LLMType())
	}
}
