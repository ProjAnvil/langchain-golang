package ollama

import (
	"os"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// init self-registers the ollama ProviderFactory into the chatmodels registry
// so chatmodels.Resolve(ChatModelSpec{Provider: "ollama", ...}) produces an
// ollama.ChatModel. Importing this package activates "ollama" as a resolvable
// provider. No import cycle: langchain/chatmodels does not import
// partners/ollama.
func init() {
	chatmodels.RegisterProvider("ollama", ollamaFactory)
}

// ollamaFactory adapts the chatmodels.ProviderFactory signature to NewChatModel.
// It reads OLLAMA_HOST (the ollama client's host env var) and applies it as the
// base URL when set and non-empty. NewChatModel's built-in default BaseURL
// (http://localhost:11434) still applies when the env var is unset. Ollama needs
// no API key for local serving. opts is reserved for future expansion.
func ollamaFactory(model string, opts map[string]any) (language.ChatModel, error) {
	configOpts := []modelconfig.Option{modelconfig.WithModel(model)}
	if v, ok := os.LookupEnv("OLLAMA_HOST"); ok && v != "" {
		configOpts = append(configOpts, modelconfig.WithBaseURL(v))
	}
	return NewChatModel(configOpts...), nil
}
