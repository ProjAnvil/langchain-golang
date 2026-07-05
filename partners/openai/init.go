package openai

import (
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
// It applies the model name; API key / base URL / other settings come from
// modelconfig defaults or may be set on the returned ChatModel by the caller
// (e.g. via WithAgentModel options plumbing in a later task). opts is reserved
// for future expansion (YAGNI: not parsed today).
func openaiFactory(model string, opts map[string]any) (language.ChatModel, error) {
	return NewChatModel(modelconfig.WithModel(model)), nil
}
