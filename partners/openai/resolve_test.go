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

// TestFactoryReadsEnvForResolve covers Task 5.7 Part A: openaiFactory reads
// OPENAI_API_KEY and OPENAI_BASE_URL from the environment (via os.LookupEnv)
// so chatmodels.Resolve produces a model that works against the real API
// without callers having to plumb an API key through modelconfig separately.
// modelconfig itself does NOT read env (verified: no os.Getenv in
// core/modelconfig), so the factory is the natural home for env->config
// mapping, matching the OpenAI SDK convention.
//
// t.Setenv is used so each subtest restores the prior value on completion
// (no parallel-test flakiness, no leakage into TestResolveOpenAI above or
// other packages in the same binary).
func TestFactoryReadsEnvForResolve(t *testing.T) {
	t.Run("env vars applied to resolved model", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "env-key")
		t.Setenv("OPENAI_BASE_URL", "https://env.example/v1")

		model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
			Provider: "openai",
			Model:    "gpt-test",
		})
		if err != nil {
			t.Fatalf("Resolve: unexpected error: %v", err)
		}
		openaiModel, ok := model.(ChatModel)
		if !ok {
			t.Fatalf("Resolve returned %T, want openai.ChatModel", model)
		}
		if got, want := openaiModel.config.APIKey, "env-key"; got != want {
			t.Fatalf("config.APIKey: got %q want %q", got, want)
		}
		if got, want := openaiModel.config.BaseURL, "https://env.example/v1"; got != want {
			t.Fatalf("config.BaseURL: got %q want %q", got, want)
		}
		if got, want := openaiModel.config.Model, "gpt-test"; got != want {
			t.Fatalf("config.Model: got %q want %q", got, want)
		}
	})

	t.Run("unset env yields default base URL and empty API key", func(t *testing.T) {
		// t.Setenv to empty: os.LookupEnv returns ("", true), but the factory
		// treats empty as "not set" (matching the OpenAI SDK convention), so
		// NewChatModel's built-in defaults apply (defaultBaseURL, empty APIKey).
		// This also covers the fully-unset case since this test process did not
		// have OPENAI_*_KEY set at startup (the factory's "not set" and "empty"
		// branches collapse to the same behavior).
		t.Setenv("OPENAI_API_KEY", "")
		t.Setenv("OPENAI_BASE_URL", "")

		model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
			Provider: "openai",
			Model:    "gpt-test",
		})
		if err != nil {
			t.Fatalf("Resolve: unexpected error: %v", err)
		}
		openaiModel, ok := model.(ChatModel)
		if !ok {
			t.Fatalf("Resolve returned %T, want openai.ChatModel", model)
		}
		if openaiModel.config.APIKey != "" {
			t.Fatalf("config.APIKey: got %q want empty", openaiModel.config.APIKey)
		}
		if openaiModel.config.BaseURL != defaultBaseURL {
			t.Fatalf("config.BaseURL: got %q want default %q",
				openaiModel.config.BaseURL, defaultBaseURL)
		}
		if openaiModel.config.Model != "gpt-test" {
			t.Fatalf("config.Model: got %q want %q", openaiModel.config.Model, "gpt-test")
		}
	})

	t.Run("only API key set leaves base URL at default", func(t *testing.T) {
		// Mixed case: API key from env, base URL unset (so defaulted).
		// Confirms the two env vars are read independently rather than as a
		// pair (a partial config doesn't fall through to "all defaults").
		t.Setenv("OPENAI_API_KEY", "only-key")
		t.Setenv("OPENAI_BASE_URL", "")

		model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
			Provider: "openai",
			Model:    "gpt-test",
		})
		if err != nil {
			t.Fatalf("Resolve: unexpected error: %v", err)
		}
		openaiModel, ok := model.(ChatModel)
		if !ok {
			t.Fatalf("Resolve returned %T, want openai.ChatModel", model)
		}
		if openaiModel.config.APIKey != "only-key" {
			t.Fatalf("config.APIKey: got %q want %q", openaiModel.config.APIKey, "only-key")
		}
		if openaiModel.config.BaseURL != defaultBaseURL {
			t.Fatalf("config.BaseURL: got %q want default %q",
				openaiModel.config.BaseURL, defaultBaseURL)
		}
	})
}
