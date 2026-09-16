package openaicompat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
	openaipartner "github.com/projanvil/langchain-golang/partners/openai"
)

// TestProvidersTable pins the static descriptor for every provider this
// package registers: registry name, default base URL (the Python SDK default
// each langchain_* package relies on), API key env var, base URL env var, and
// default model. These values are the contract Python parity is built on, so
// they are asserted verbatim rather than derived.
func TestProvidersTable(t *testing.T) {
	want := map[string]CompatProvider{
		"deepseek": {
			Name:           "deepseek",
			DefaultBaseURL: "https://api.deepseek.com/v1",
			DefaultModel:   "deepseek-chat",
			APIKeyEnv:      "DEEPSEEK_API_KEY",
			BaseURLEnv:     "DEEPSEEK_API_BASE",
		},
		"fireworks": {
			Name:           "fireworks",
			DefaultBaseURL: "https://api.fireworks.ai/inference/v1",
			DefaultModel:   "accounts/fireworks/models/llama-v3p1-8b-instruct",
			APIKeyEnv:      "FIREWORKS_API_KEY",
			BaseURLEnv:     "FIREWORKS_API_BASE",
		},
		"groq": {
			Name:           "groq",
			DefaultBaseURL: "https://api.groq.com/openai/v1",
			DefaultModel:   "openai/gpt-oss-20b",
			APIKeyEnv:      "GROQ_API_KEY",
			BaseURLEnv:     "GROQ_API_BASE",
		},
		"mistralai": {
			Name:           "mistralai",
			DefaultBaseURL: "https://api.mistral.ai/v1",
			DefaultModel:   "mistral-small",
			APIKeyEnv:      "MISTRAL_API_KEY",
			BaseURLEnv:     "MISTRAL_BASE_URL",
		},
		"openrouter": {
			Name:           "openrouter",
			DefaultBaseURL: "https://openrouter.ai/api/v1",
			DefaultModel:   "openrouter/auto",
			APIKeyEnv:      "OPENROUTER_API_KEY",
			BaseURLEnv:     "OPENROUTER_API_BASE",
		},
		"perplexity": {
			Name:           "perplexity",
			DefaultBaseURL: "https://api.perplexity.ai",
			DefaultModel:   "sonar",
			APIKeyEnv:      "PPLX_API_KEY",
			// Python's ChatPerplexity exposes no base URL override at all
			// (the SDK default is used), so BaseURLEnv is empty here.
			BaseURLEnv: "",
		},
		"xai": {
			Name:           "xai",
			DefaultBaseURL: "https://api.x.ai/v1",
			DefaultModel:   "grok-4",
			APIKeyEnv:      "XAI_API_KEY",
			BaseURLEnv:     "XAI_API_BASE",
		},
	}

	registered := Providers()
	if len(registered) != len(want) {
		t.Fatalf("Providers(): got %d providers, want %d", len(registered), len(want))
	}
	for _, got := range registered {
		exp, ok := want[got.Name]
		if !ok {
			t.Fatalf("Providers(): unexpected provider %q", got.Name)
		}
		if got.DefaultBaseURL != exp.DefaultBaseURL {
			t.Errorf("%s DefaultBaseURL: got %q want %q", got.Name, got.DefaultBaseURL, exp.DefaultBaseURL)
		}
		if got.DefaultModel != exp.DefaultModel {
			t.Errorf("%s DefaultModel: got %q want %q", got.Name, got.DefaultModel, exp.DefaultModel)
		}
		if got.APIKeyEnv != exp.APIKeyEnv {
			t.Errorf("%s APIKeyEnv: got %q want %q", got.Name, got.APIKeyEnv, exp.APIKeyEnv)
		}
		if got.BaseURLEnv != exp.BaseURLEnv {
			t.Errorf("%s BaseURLEnv: got %q want %q", got.Name, got.BaseURLEnv, exp.BaseURLEnv)
		}
	}

	for name := range want {
		if _, ok := LookupProvider(name); !ok {
			t.Errorf("LookupProvider(%q): not found", name)
		}
	}
	if _, ok := LookupProvider("no-such-provider"); ok {
		t.Error(`LookupProvider("no-such-provider"): found, want not found`)
	}
}

// TestResolveAllProvidersOverWire resolves every provider that exposes a base
// URL env var through the chatmodels registry with fake env (t.Setenv) and
// asserts the request lands on the fake server's /chat/completions with the
// right model and bearer token. Perplexity is excluded here because Python
// exposes no base URL override for it (its SDK default is used verbatim); its
// construction path is covered by TestResolveAllProvidersConstructs below and
// its factory branch logic by TestCompatFactoryBranches.
func TestResolveAllProvidersOverWire(t *testing.T) {
	cases := []struct {
		provider   string
		model      string
		apiKeyEnv  string
		baseURLEnv string
	}{
		{"deepseek", "deepseek-reasoner", "DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE"},
		{"fireworks", "accounts/fireworks/models/gpt-oss-120b", "FIREWORKS_API_KEY", "FIREWORKS_API_BASE"},
		{"groq", "llama-3.3-70b-versatile", "GROQ_API_KEY", "GROQ_API_BASE"},
		{"mistralai", "mistral-large-latest", "MISTRAL_API_KEY", "MISTRAL_BASE_URL"},
		{"openrouter", "anthropic/claude-sonnet-4-5", "OPENROUTER_API_KEY", "OPENROUTER_API_BASE"},
		{"xai", "grok-3-mini", "XAI_API_KEY", "XAI_API_BASE"},
	}
	for _, tc := range cases {
		t.Run(tc.provider, func(t *testing.T) {
			var gotPath string
			var gotAuth string
			var gotBody map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				gotAuth = r.Header.Get("Authorization")
				_ = json.NewDecoder(r.Body).Decode(&gotBody)
				_, _ = w.Write([]byte(`{
					"id":"cc-1",
					"model":"` + tc.model + `",
					"choices":[{"message":{"role":"assistant","content":"ok"}}],
					"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
				}`))
			}))
			defer server.Close()

			t.Setenv(tc.apiKeyEnv, "test-key")
			t.Setenv(tc.baseURLEnv, server.URL)

			model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
				Provider: tc.provider,
				Model:    tc.model,
			})
			if err != nil {
				t.Fatalf("Resolve(%s): %v", tc.provider, err)
			}
			resp, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
			if err != nil {
				t.Fatalf("Invoke(%s): %v", tc.provider, err)
			}
			if resp.Content != "ok" {
				t.Fatalf("%s content = %q want %q", tc.provider, resp.Content, "ok")
			}
			if gotPath != "/chat/completions" {
				t.Fatalf("%s path = %q want /chat/completions (Chat Completions shape)", tc.provider, gotPath)
			}
			if gotAuth != "Bearer test-key" {
				t.Fatalf("%s Authorization = %q want Bearer test-key", tc.provider, gotAuth)
			}
			if gotBody["model"] != tc.model {
				t.Fatalf("%s body model = %v want %v", tc.provider, gotBody["model"], tc.model)
			}
			if msgs, ok := gotBody["messages"].([]any); !ok || len(msgs) != 1 {
				t.Fatalf("%s body messages = %v want 1 entry", tc.provider, gotBody["messages"])
			}
		})
	}
}

// TestResolveAllProvidersConstructs covers all 7 — including perplexity —
// resolving through the registry to the concrete partners/openai ChatModel
// with the agent runtime's required capabilities. Construction performs no
// I/O, so no env or fake server is needed.
func TestResolveAllProvidersConstructs(t *testing.T) {
	for _, name := range []string{
		"deepseek", "fireworks", "groq", "mistralai", "openrouter", "perplexity", "xai",
	} {
		t.Run(name, func(t *testing.T) {
			model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
				Provider: name,
				Model:    "test-model",
			})
			if err != nil {
				t.Fatalf("Resolve(%s): %v", name, err)
			}
			if _, ok := model.(openaipartner.ChatModel); !ok {
				t.Fatalf("Resolve(%s) returned %T, want openai.ChatModel", name, model)
			}
			if _, ok := model.(language.StructuredCaller); !ok {
				t.Fatalf("Resolve(%s) returned %T lacking language.StructuredCaller", name, model)
			}
			if _, ok := model.(language.ToolBinder); !ok {
				t.Fatalf("Resolve(%s) returned %T lacking language.ToolBinder", name, model)
			}
		})
	}
}

// TestParseModelStringResolves exercises the 'provider:model' parsing path the
// agent runtime uses (chatmodels.ParseModelString + Resolve) end to end for a
// provider that has no inference prefix (groq).
func TestParseModelStringResolves(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"cc-2",
			"model":"llama-3.3-70b-versatile",
			"choices":[{"message":{"role":"assistant","content":"parsed"}}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()
	t.Setenv("GROQ_API_KEY", "test-key")
	t.Setenv("GROQ_API_BASE", server.URL)

	spec, err := chatmodels.ParseModelString("groq:llama-3.3-70b-versatile")
	if err != nil {
		t.Fatalf("ParseModelString: %v", err)
	}
	if spec.Provider != "groq" || spec.Model != "llama-3.3-70b-versatile" {
		t.Fatalf("spec = %+v want {groq llama-3.3-70b-versatile}", spec)
	}
	model, err := chatmodels.Resolve(spec)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotBody["model"] != "llama-3.3-70b-versatile" {
		t.Fatalf("body model = %v", gotBody["model"])
	}
}

// TestInferredProvidersResolve covers chatmodels.ParseModel's provider
// inference (no 'provider:' prefix): the model-name prefixes registered in
// chatmodels (mistral*/deepseek*/grok*/sonar*/accounts/fireworks*) must
// resolve to this package's factories.
func TestInferredProvidersResolve(t *testing.T) {
	for model, provider := range map[string]string{
		"mistral-small":                          "mistralai",
		"deepseek-chat":                          "deepseek",
		"grok-4":                                 "xai",
		"sonar":                                  "perplexity",
		"accounts/fireworks/models/gpt-oss-120b": "fireworks",
	} {
		spec, err := chatmodels.ParseModel(model)
		if err != nil {
			t.Fatalf("ParseModel(%q): %v", model, err)
		}
		if spec.Provider != provider {
			t.Fatalf("ParseModel(%q).Provider = %q want %q", model, spec.Provider, provider)
		}
		resolved, err := chatmodels.Resolve(spec)
		if err != nil {
			t.Fatalf("Resolve(ParseModel(%q)): %v", model, err)
		}
		if _, ok := resolved.(openaipartner.ChatModel); !ok {
			t.Fatalf("Resolve(ParseModel(%q)) returned %T, want openai.ChatModel", model, resolved)
		}
	}
}

// TestDefaultModelAppliedWhenEmpty verifies Resolve with an empty Model half
// falls back to the provider's DefaultModel rather than partners/openai's
// built-in "gpt-4.1" (which would be invalid for the provider).
func TestDefaultModelAppliedWhenEmpty(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"cc-3",
			"model":"openai/gpt-oss-20b",
			"choices":[{"message":{"role":"assistant","content":"ok"}}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()
	t.Setenv("GROQ_API_KEY", "test-key")
	t.Setenv("GROQ_API_BASE", server.URL)

	model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{Provider: "groq"})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotBody["model"] != "openai/gpt-oss-20b" {
		t.Fatalf("body model = %v want default %q", gotBody["model"], "openai/gpt-oss-20b")
	}
}

// TestCompatFactoryBranches drives the shared factory against a synthetic
// descriptor so every branch is covered without depending on any one real
// provider's env surface (this is also the only way to observe perplexity's
// no-env default-base behavior, since Python exposes no override for it).
func TestCompatFactoryBranches(t *testing.T) {
	newTestDescriptor := func() CompatProvider {
		return CompatProvider{
			Name:           "fakecompat",
			DefaultBaseURL: "https://default.example/v1",
			DefaultModel:   "default-model",
			APIKeyEnv:      "FAKECOMPAT_API_KEY",
			BaseURLEnv:     "FAKECOMPAT_BASE_URL",
		}
	}

	t.Run("no env applies descriptor default base URL", func(t *testing.T) {
		t.Setenv("FAKECOMPAT_API_KEY", "")
		t.Setenv("FAKECOMPAT_BASE_URL", "")
		// The descriptor default base URL must be applied even with no env
		// override (this is the branch perplexity always takes, since Python
		// exposes no override for it): point DefaultBaseURL at a fake server
		// and observe the request arrive without setting any env var.
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"x"}}]}`))
		}))
		defer server.Close()
		desc := newTestDescriptor()
		desc.DefaultBaseURL = server.URL
		model, err := compatFactory(desc)("", nil)
		if err != nil {
			t.Fatalf("factory: %v", err)
		}
		if _, ok := model.(openaipartner.ChatModel); !ok {
			t.Fatalf("factory returned %T, want openai.ChatModel", model)
		}
		if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
			t.Fatalf("Invoke against descriptor default base URL: %v", err)
		}
	})

	t.Run("empty model constructs with DefaultModel", func(t *testing.T) {
		// The DefaultModel substitution itself is asserted over the wire in
		// TestDefaultModelAppliedWhenEmpty; here we verify construction
		// succeeds with an empty model string (no panic, right concrete type).
		t.Setenv("FAKECOMPAT_API_KEY", "")
		t.Setenv("FAKECOMPAT_BASE_URL", "")
		model, err := compatFactory(newTestDescriptor())("", nil)
		if err != nil {
			t.Fatalf("factory: %v", err)
		}
		openaiModel, ok := model.(openaipartner.ChatModel)
		if !ok {
			t.Fatalf("factory returned %T, want openai.ChatModel", model)
		}
		if got := openaiModel.LLMType(); got != "openai-chat" {
			t.Fatalf("LLMType = %q want openai-chat", got)
		}
	})

	t.Run("env override wins over default base", func(t *testing.T) {
		var gotAuth string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotAuth = r.Header.Get("Authorization")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"x"}}]}`))
		}))
		defer server.Close()
		t.Setenv("FAKECOMPAT_API_KEY", "branch-key")
		t.Setenv("FAKECOMPAT_BASE_URL", server.URL)
		factory := compatFactory(newTestDescriptor())
		model, err := factory("m1", nil)
		if err != nil {
			t.Fatalf("factory: %v", err)
		}
		if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
			t.Fatalf("Invoke: %v", err)
		}
		if gotAuth != "Bearer branch-key" {
			t.Fatalf("Authorization = %q want Bearer branch-key", gotAuth)
		}
	})
}
