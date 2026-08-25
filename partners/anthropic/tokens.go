package anthropic

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
)

// Compile-time guard: ChatModel provides the provider-specific message token
// counter the language package dispatches to (language.GetNumTokensFromMessages).
var _ language.MessageTokenCounter = ChatModel{}

// countTokensPayload is the Anthropic Messages count_tokens request body. It
// reuses the wire types from buildRequest (requestMessage/toolSpec) but omits
// generation-only fields (max_tokens, temperature, ...), mirroring what
// Python passes to client.messages.count_tokens.
type countTokensPayload struct {
	Model             string           `json:"model"`
	Messages          []requestMessage `json:"messages"`
	System            any              `json:"system,omitempty"`
	Tools             []toolSpec       `json:"tools,omitempty"`
	ContextManagement map[string]any   `json:"context_management,omitempty"`
}

// countTokensResponse is the Anthropic Messages count_tokens response body.
type countTokensResponse struct {
	InputTokens int `json:"input_tokens"`
}

// GetNumTokensFromMessages mirrors ChatAnthropic.get_num_tokens_from_messages
// (langchain_anthropic/chat_models.py:2137): it does no local counting and
// instead calls Anthropic's official token counting API
// (POST {baseURL}/messages/count_tokens) with the same message/system/tool
// formatting used for generation, returning response.input_tokens. Tools are
// included when bound via BindTools, and context_management when configured —
// matching the Python implementation.
func (m ChatModel) GetNumTokensFromMessages(msgs []messages.Message) (int, error) {
	if m.config.APIKey == "" {
		return 0, fmt.Errorf(
			"anthropic %s: count tokens requires an API key (modelconfig.WithAPIKey)",
			m.config.Model)
	}
	req, err := m.buildRequest(msgs)
	if err != nil {
		return 0, fmt.Errorf("anthropic %s: count tokens: format messages: %w", m.config.Model, err)
	}
	payload := countTokensPayload{
		Model:             req.Model,
		Messages:          req.Messages,
		System:            req.System,
		Tools:             req.Tools,
		ContextManagement: req.ContextManagement,
	}
	ctx, cancel := context.WithTimeout(context.Background(), m.config.Timeout)
	defer cancel()
	resp, err := postJSON[countTokensResponse](ctx, m.config, "/messages/count_tokens", payload)
	if err != nil {
		return 0, fmt.Errorf("anthropic %s: count tokens: %w", m.config.Model, err)
	}
	return resp.InputTokens, nil
}
