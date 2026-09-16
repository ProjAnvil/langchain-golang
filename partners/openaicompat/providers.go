// Package openaicompat registers the OpenAI-compatible partner chat
// providers (groq, mistralai, deepseek, xai, openrouter, fireworks,
// perplexity) into the chatmodels provider registry.
//
// In Python each of these ships as its own langchain-* package, but every one
// of them talks to an OpenAI-shaped Chat Completions endpoint — the packages
// differ only in base URL, environment variable names, default model, and a
// few provider quirks. The Go port therefore does NOT create seven partner
// packages: this one package holds a descriptor table and registers a factory
// per provider that constructs a partners/openai ChatModel switched to the
// Chat Completions API (WithChatCompletions) with the provider's base URL and
// env-derived credentials applied. Importing this package (e.g. via a blank
// import in the application entry point) activates all seven names for
// chatmodels.Resolve and the 'provider:model' forms such as
// "groq:llama-3.3-70b-versatile".
//
// The descriptor values mirror the Python packages (base URLs are the SDK
// defaults each langchain_* wrapper relies on; env var names match
// secret_from_env/from_env defaults). Provider-specific behavior beyond
// request shaping (e.g. deepseek reasoning_content surfacing, xAI stop-parameter
// rejection) is documented per provider below and, where it only affects
// request construction the shared CC path already handles, intentionally not
// re-implemented.
package openaicompat

import (
	"cmp"
	"os"
	"slices"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
	openaipartner "github.com/projanvil/langchain-golang/partners/openai"
)

// CompatProvider is the static description of one OpenAI-compatible provider
// registered by this package: everything the shared factory needs to shape a
// partners/openai ChatModel for that provider. Use Providers/LookupProvider
// to introspect the table.
type CompatProvider struct {
	// Name is the chatmodels registry key ("groq", "deepseek", ...). It
	// matches the BuiltinProviders spelling, so specs produced by
	// ParseModel/ParseModelString/InitChatModel resolve to this factory.
	Name string
	// DefaultBaseURL is applied whenever BaseURLEnv is unset or empty. It is
	// the SDK default the corresponding Python package relies on.
	DefaultBaseURL string
	// DefaultModel is substituted when the factory is invoked with an empty
	// model string (Resolve(ChatModelSpec{Provider: name})); see the
	// per-provider entries below for each value's provenance.
	DefaultModel string
	// APIKeyEnv is the environment variable read for the API key (applied
	// when set and non-empty; no error when missing — the request fails
	// server-side with 401 like the openai factory).
	APIKeyEnv string
	// BaseURLEnv is the environment variable read to override
	// DefaultBaseURL. Empty when the Python package exposes no override
	// (perplexity).
	BaseURLEnv string
	// defaultHeaders are extra request headers applied after auth, each
	// env-overridable with a fallback default (OpenRouter attribution).
	defaultHeaders []headerDefault
}

// headerDefault is one default request header with an env override.
type headerDefault struct {
	Header   string
	Env      string
	Fallback string
}

// providers is the registration table. Kept sorted by name; init registers
// one factory per entry. Every entry's fields are pinned by
// TestProvidersTable.
var providers = []CompatProvider{
	{
		// langchain_deepseek.ChatDeepSeek subclasses BaseChatOpenAI against
		// the OpenAI SDK with api_base default https://api.deepseek.com/v1
		// (chat_models.py DEFAULT_API_BASE) and env DEEPSEEK_API_KEY /
		// DEEPSEEK_API_BASE.
		//
		// DefaultModel "deepseek-chat" was the package's historic default
		// and remains the documented starter model; Python now requires the
		// model explicitly.
		//
		// Known limitations vs Python:
		//   - reasoning models (deepseek-reasoner) return a reasoning_content
		//     field Python surfaces in additional_kwargs; the shared CC
		//     response parser does not surface it yet.
		//   - Python routes strict structured output through a beta base URL
		//     (https://api.deepseek.com/beta); the Go factory always uses
		//     the main base.
		Name:           "deepseek",
		DefaultBaseURL: "https://api.deepseek.com/v1",
		DefaultModel:   "deepseek-chat",
		APIKeyEnv:      "DEEPSEEK_API_KEY",
		BaseURLEnv:     "DEEPSEEK_API_BASE",
	},
	{
		// langchain_fireworks.ChatFireworks uses the Fireworks SDK whose
		// default base is https://api.fireworks.ai/inference/v1, with env
		// FIREWORKS_API_KEY (required in Python) / FIREWORKS_API_BASE.
		//
		// DefaultModel is the package's historic default, removed from
		// Python in April 2025 (the model is now required there); model IDs
		// are "accounts/fireworks/models/..." namespaced.
		Name:           "fireworks",
		DefaultBaseURL: "https://api.fireworks.ai/inference/v1",
		DefaultModel:   "accounts/fireworks/models/llama-v3p1-8b-instruct",
		APIKeyEnv:      "FIREWORKS_API_KEY",
		BaseURLEnv:     "FIREWORKS_API_BASE",
	},
	{
		// langchain_groq.ChatGroq uses the groq SDK (default base
		// https://api.groq.com/openai/v1) with env GROQ_API_KEY /
		// GROQ_API_BASE on the Chat Completions API.
		//
		// DefaultModel "openai/gpt-oss-20b" is the current package docstring
		// example. The long-standing default "llama-3.3-70b-versatile" was
		// deprecated by Groq and shut down 2026-08-16, so it is deliberately
		// not used here.
		//
		// Known limitations vs Python:
		//   - strict tool-calling is unsupported (Python drops strict=);
		//     the Go adapter has no strict tool flag to drop, so nothing to
		//     do.
		//   - Groq-specific request knobs reasoning_format /
		//     reasoning_effort / service_tier are not modeled; callers can
		//     add them via modelconfig.WithExtra if a future adapter honors
		//     it.
		//   - Groq returns tool_call arguments as JSON null for empty args;
		//     Python rewrites to {}. The shared CC parser tolerates both.
		Name:           "groq",
		DefaultBaseURL: "https://api.groq.com/openai/v1",
		DefaultModel:   "openai/gpt-oss-20b",
		APIKeyEnv:      "GROQ_API_KEY",
		BaseURLEnv:     "GROQ_API_BASE",
	},
	{
		// langchain_mistralai.ChatMistralAI posts to {base}/chat/completions
		// with base default https://api.mistral.ai/v1; env MISTRAL_API_KEY,
		// plus MISTRAL_BASE_URL as the fallback when no explicit endpoint is
		// configured (validate_environment).
		//
		// DefaultModel "mistral-small" matches the Python field default.
		Name:           "mistralai",
		DefaultBaseURL: "https://api.mistral.ai/v1",
		DefaultModel:   "mistral-small",
		APIKeyEnv:      "MISTRAL_API_KEY",
		BaseURLEnv:     "MISTRAL_BASE_URL",
	},
	{
		// langchain_openrouter.ChatOpenRouter targets the OpenRouter API
		// (default https://openrouter.ai/api/v1) with env OPENROUTER_API_KEY
		// / OPENROUTER_API_BASE. Model IDs are "vendor/model" strings.
		//
		// App-attribution headers mirror _build_client: HTTP-Referer from
		// OPENROUTER_APP_URL (default https://docs.langchain.com) and X-Title
		// from OPENROUTER_APP_TITLE (default "LangChain").
		//
		// DefaultModel "openrouter/auto" is OpenRouter's auto-routing model;
		// Python requires an explicit model, so this default is Go-side
		// sugar for Resolve calls without a model.
		//
		// Known limitations vs Python: provider routing prefs (route,
		// provider order, plugins) and OpenRouter-only fallback metadata are
		// not modeled.
		Name:           "openrouter",
		DefaultBaseURL: "https://openrouter.ai/api/v1",
		DefaultModel:   "openrouter/auto",
		APIKeyEnv:      "OPENROUTER_API_KEY",
		BaseURLEnv:     "OPENROUTER_API_BASE",
		defaultHeaders: []headerDefault{
			{Header: "HTTP-Referer", Env: "OPENROUTER_APP_URL", Fallback: "https://docs.langchain.com"},
			{Header: "X-Title", Env: "OPENROUTER_APP_TITLE", Fallback: "LangChain"},
		},
	},
	{
		// langchain_perplexity.ChatPerplexity uses the Perplexity SDK
		// (default base https://api.perplexity.ai) with env PPLX_API_KEY —
		// note the PPLX spelling, not PERPLEXITY. Python exposes no base URL
		// override at all, so BaseURLEnv is empty and DefaultBaseURL always
		// applies. DefaultModel "sonar" matches the Python field default.
		//
		// Known limitations vs Python: search_results / citations metadata
		// and the optional Responses ("Agent") API routing are not modeled —
		// the Go factory always uses Chat Completions.
		Name:           "perplexity",
		DefaultBaseURL: "https://api.perplexity.ai",
		DefaultModel:   "sonar",
		APIKeyEnv:      "PPLX_API_KEY",
	},
	{
		// langchain_xai.ChatXAI subclasses BaseChatOpenAI against the OpenAI
		// SDK with base default https://api.x.ai/v1 and env XAI_API_KEY /
		// XAI_API_BASE. DefaultModel "grok-4" matches the Python field
		// default.
		//
		// Known limitation vs Python: xAI rejects the `stop` parameter for
		// reasoning models (grok-3*/grok-4*/grok-code-fast...); Python drops
		// stop for those via model profiles. The Go adapter has no
		// per-provider stop stripping — avoid WithStop on those models.
		Name:           "xai",
		DefaultBaseURL: "https://api.x.ai/v1",
		DefaultModel:   "grok-4",
		APIKeyEnv:      "XAI_API_KEY",
		BaseURLEnv:     "XAI_API_BASE",
	},
}

// init self-registers every provider in the table into the chatmodels
// registry, following the same init()-self-registration pattern as
// partners/openai, partners/anthropic, and partners/ollama. No import cycle:
// langchain/chatmodels does not import this package.
func init() {
	for _, p := range providers {
		chatmodels.RegisterProvider(p.Name, compatFactory(p))
	}
}

// compatFactory adapts a CompatProvider to the chatmodels.ProviderFactory
// signature. It builds a partners/openai ChatModel switched to the Chat
// Completions API (WithChatCompletions — all seven providers speak the CC
// shape, not OpenAI's Responses API) with:
//
//   - the model from the spec, or DefaultModel when the spec's model is
//     empty (partners/openai's own "gpt-4.1" fallback would be an invalid
//     model ID for these providers, so it must be overridden);
//   - DefaultBaseURL, overridden by BaseURLEnv when set and non-empty
//     (options apply in order, so the later WithBaseURL wins);
//   - the API key from APIKeyEnv when set and non-empty (no error when
//     missing, matching the openai/anthropic factories);
//   - each defaultHeader, env-overridable with the fallback default.
//
// opts is reserved for future expansion (not parsed today), matching the
// other partner factories.
func compatFactory(p CompatProvider) chatmodels.ProviderFactory {
	return func(model string, opts map[string]any) (language.ChatModel, error) {
		model = cmp.Or(model, p.DefaultModel)
		configOpts := []modelconfig.Option{
			modelconfig.WithModel(model),
			modelconfig.WithBaseURL(p.DefaultBaseURL),
		}
		if p.BaseURLEnv != "" {
			if v, ok := os.LookupEnv(p.BaseURLEnv); ok && v != "" {
				configOpts = append(configOpts, modelconfig.WithBaseURL(v))
			}
		}
		if v, ok := os.LookupEnv(p.APIKeyEnv); ok && v != "" {
			configOpts = append(configOpts, modelconfig.WithAPIKey(v))
		}
		for _, h := range p.defaultHeaders {
			value, ok := os.LookupEnv(h.Env)
			if !ok || value == "" {
				value = h.Fallback
			}
			configOpts = append(configOpts, modelconfig.WithHeader(h.Header, value))
		}
		return openaipartner.NewChatModel(configOpts...).WithChatCompletions(), nil
	}
}

// Providers returns the descriptors this package registers, sorted by name.
// The returned slice is a copy; callers may mutate it freely.
func Providers() []CompatProvider {
	out := make([]CompatProvider, len(providers))
	copy(out, providers)
	slices.SortFunc(out, func(a, b CompatProvider) int { return cmp.Compare(a.Name, b.Name) })
	return out
}

// LookupProvider returns the descriptor registered under name. The name is
// normalized exactly like the registry (chatmodels.NormalizeProvider), so
// "Groq" and "groq" find the same descriptor.
func LookupProvider(name string) (CompatProvider, bool) {
	normalized := chatmodels.NormalizeProvider(name)
	for _, p := range providers {
		if chatmodels.NormalizeProvider(p.Name) == normalized {
			return p, true
		}
	}
	return CompatProvider{}, false
}
