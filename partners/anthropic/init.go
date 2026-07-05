package anthropic

import (
	"os"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// init self-registers the anthropic ProviderFactory into the chatmodels
// registry so chatmodels.Resolve(ChatModelSpec{Provider: "anthropic", ...})
// produces an anthropic.ChatModel. Importing this package (blank import in the
// application entry point) activates "anthropic" as a resolvable provider — the
// same pattern Python langchain uses where importing langchain_anthropic
// registers the integration. No import cycle: langchain/chatmodels does not
// import partners/anthropic.
func init() {
	chatmodels.RegisterProvider("anthropic", anthropicFactory)
}

// anthropicFactory adapts the chatmodels.ProviderFactory signature to
// NewChatModel. It reads ANTHROPIC_API_KEY and the base URL from the
// environment itself (via os.LookupEnv, applying only when set and non-empty),
// matching the openai factory pattern. Base URL resolution mirrors Python
// (langchain_anthropic/chat_models.py:949): ANTHROPIC_API_URL first, then
// ANTHROPIC_BASE_URL. NewChatModel's built-in default BaseURL
// (https://api.anthropic.com/v1) still applies when neither env var is set.
// opts is reserved for future expansion (not parsed today).
func anthropicFactory(model string, opts map[string]any) (language.ChatModel, error) {
	configOpts := []modelconfig.Option{modelconfig.WithModel(model)}
	if v, ok := os.LookupEnv("ANTHROPIC_API_KEY"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithAPIKey(v))
	}
	if v, ok := os.LookupEnv("ANTHROPIC_API_URL"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithBaseURL(v))
	} else if v, ok := os.LookupEnv("ANTHROPIC_BASE_URL"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithBaseURL(v))
	}
	return NewChatModel(configOpts...), nil
}
