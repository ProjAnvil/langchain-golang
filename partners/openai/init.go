package openai

import (
	"os"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// init self-registers the openai ProviderFactory into the chatmodels registry
// so chatmodels.Resolve(ChatModelSpec{Provider: "openai", Model: ...}) produces
// an openai.ChatModel without callers needing to wire it up. Importing this
// package (e.g. via a blank import in the application entry point) is what
// activates "openai" as a resolvable provider — the same pattern Python
// langchain uses where importing langchain_openai registers the integration.
//
// There is no import cycle: langchain/chatmodels does not import
// partners/openai (verified), so this partners/openai -> langchain/chatmodels
// edge is one-directional.
func init() {
	chatmodels.RegisterProvider("openai", openaiFactory)
}

// openaiFactory adapts the chatmodels.ProviderFactory signature to NewChatModel.
//
// It reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment itself
// (via os.LookupEnv, only applying when set and non-empty) and passes them as
// modelconfig.WithAPIKey / modelconfig.WithBaseURL alongside WithModel(model).
// This is the natural home for env->config mapping and matches the OpenAI SDK
// convention: core/modelconfig deliberately does NOT read env vars (verified:
// no os.Getenv in core/modelconfig), so without this the resolved model would
// have no API key / base URL and 401 in production. NewChatModel's built-in
// default BaseURL (https://api.openai.com/v1) still applies when the env var
// is unset/empty.
//
// opts is reserved for future expansion (YAGNI: not parsed today).
func openaiFactory(model string, opts map[string]any) (language.ChatModel, error) {
	configOpts := []modelconfig.Option{modelconfig.WithModel(model)}
	if v, ok := os.LookupEnv("OPENAI_API_KEY"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithAPIKey(v))
	}
	if v, ok := os.LookupEnv("OPENAI_BASE_URL"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithBaseURL(v))
	}
	return NewChatModel(configOpts...), nil
}
