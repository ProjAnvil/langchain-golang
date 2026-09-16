package gemini

import (
	"os"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// init self-registers the gemini ProviderFactory into the chatmodels registry
// so chatmodels.Resolve(ChatModelSpec{Provider: "gemini", ...}) produces a
// gemini.ChatModel — the same pattern Python langchain uses where importing
// langchain_google_genai registers the integration. No import cycle:
// langchain/chatmodels does not import partners/gemini.
func init() {
	chatmodels.RegisterProvider("gemini", geminiFactory)
}

// geminiFactory adapts the chatmodels.ProviderFactory signature to
// NewChatModel. It reads GEMINI_API_KEY / GOOGLE_API_KEY (GOOGLE wins when
// both are set, the SDK's documented precedence) and GOOGLE_GEMINI_BASE_URL
// from the environment itself (via os.LookupEnv, applying only when set and
// non-empty), matching the openai/anthropic factory pattern. NewChatModel's
// built-in defaults (model "gemini-2.5-flash", base URL
// generativelanguage.googleapis.com) still apply when nothing is set. opts is
// reserved for future expansion (not parsed today).
func geminiFactory(model string, opts map[string]any) (language.ChatModel, error) {
	configOpts := []modelconfig.Option{}
	if model != "" {
		configOpts = append(configOpts, modelconfig.WithModel(model))
	}
	if v, ok := os.LookupEnv("GEMINI_API_KEY"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithAPIKey(v))
	}
	if v, ok := os.LookupEnv("GOOGLE_API_KEY"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithAPIKey(v))
	}
	if v, ok := os.LookupEnv("GOOGLE_GEMINI_BASE_URL"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithBaseURL(v))
	}
	return NewChatModel(configOpts...), nil
}
