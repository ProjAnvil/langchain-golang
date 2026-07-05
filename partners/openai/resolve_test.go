package openai

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// TestResolveOpenAI verifies the partners/openai package self-registers its
// ProviderFactory (via init()) so chatmodels.Resolve can construct it without
// any manual registration in the test. This is the wiring Task 5.6 adds; it
// makes WithAgentModel("openai:...") work end-to-end (Task 5.7).
func TestResolveOpenAI(t *testing.T) {
	// No manual RegisterProvider here: the test binary imports partners/openai
	// (this file is package openai), so its init() runs and registers the
	// factory under "openai".
	model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
		Provider: "openai",
		Model:    "gpt-test",
	})
	if err != nil {
		t.Fatalf("Resolve: unexpected error: %v", err)
	}

	// The product must be the concrete openai.ChatModel (not a wrapper) so the
	// agent runtime gets the real adapter. The test is in package openai, so
	// it can read the unexported config field.
	openaiModel, ok := model.(ChatModel)
	if !ok {
		t.Fatalf("Resolve returned %T, want openai.ChatModel", model)
	}
	if got := openaiModel.config.Model; got != "gpt-test" {
		t.Fatalf("config.Model: got %q want %q", got, "gpt-test")
	}
	// NewChatModel defaults BaseURL when unset; assert the default is applied
	// so the resolved model is callable out-of-the-box against the real API.
	if openaiModel.config.BaseURL == "" {
		t.Fatal("config.BaseURL: empty, expected NewChatModel default")
	}

	// The registered product must also satisfy language.StructuredCaller so
	// the agent's ProviderStrategy native structured-output path
	// (agents.invokeModel -> language.InvokeStructured) can use the resolved
	// model without a separate cast or adapter.
	if _, ok := model.(language.StructuredCaller); !ok {
		t.Fatalf("Resolve returned %T which does not satisfy language.StructuredCaller", model)
	}
}

// TestResolveOpenAINormalizesProvider ensures Resolve's provider-name
// normalization (NormalizeProvider lowercases the name) hits the openai
// registration, so "OpenAI" and "OPENAI" resolve to the same factory. (Note
// NormalizeProvider maps "-" to "_", so "open-ai" -> "open_ai", which is a
// distinct provider name and intentionally not covered here.)
func TestResolveOpenAINormalizesProvider(t *testing.T) {
	for _, provider := range []string{"OpenAI", "OPENAI"} {
		model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
			Provider: provider,
			Model:    "gpt-test",
		})
		if err != nil {
			t.Fatalf("Resolve(%q): unexpected error: %v", provider, err)
		}
		if _, ok := model.(ChatModel); !ok {
			t.Fatalf("Resolve(%q): returned %T, want openai.ChatModel", provider, model)
		}
	}
}
