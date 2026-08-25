package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
)

// Compile-time guards: ChatModel provides the provider-specific token
// counters the language package dispatches to.
var (
	_ language.TokenCounter        = ChatModel{}
	_ language.MessageTokenCounter = ChatModel{}
)

// GetTokenIDs approximates ChatOpenAI.get_token_ids
// (chat_models/base.py:2093). Python uses tiktoken (cl100k_base/o200k_base);
// the Go port does not ship a BPE tokenizer, so this delegates to
// language.DefaultGetTokenIDs. Counts are approximate (~1 token per 4 runes);
// IDs are stable but are not tiktoken IDs.
func (m ChatModel) GetTokenIDs(text string) []int {
	return language.DefaultGetTokenIDs(text)
}

// GetNumTokens returns the approximate number of tokens in text (len of
// GetTokenIDs), mirroring BaseLanguageModel.get_num_tokens.
func (m ChatModel) GetNumTokens(text string) int {
	return len(m.GetTokenIDs(text))
}

// GetNumTokensFromMessages approximates ChatOpenAI.get_num_tokens_from_messages
// (chat_models/base.py:2103), including the per-message/per-name overhead and
// the +3 reply primer. Divergences from Python, all inherited from the
// missing tiktoken/PIL stack: token counts use the 4-rune approximation; tool
// schemas are ignored (Python warns and ignores them); images are only
// counted when the block marks detail "low" (85 tokens) — this helper never
// fetches image URLs for sizing. Like Python, it errors for models outside
// the gpt-3.5-turbo / gpt-4 / gpt-5 families.
func (m ChatModel) GetNumTokensFromMessages(msgs []messages.Message) (int, error) {
	model := m.config.Model
	tokensPerMessage := 3
	tokensPerName := 1
	switch {
	case strings.HasPrefix(model, "gpt-3.5-turbo-0301"):
		// Every message follows <|im_start|>{role/name}\n{content}<|im_end|>\n.
		tokensPerMessage = 4
		tokensPerName = -1 // if there's a name, the role is omitted
	case strings.HasPrefix(model, "gpt-3.5-turbo"),
		strings.HasPrefix(model, "gpt-4"),
		strings.HasPrefix(model, "gpt-5"):
	default:
		return 0, fmt.Errorf(
			"GetNumTokensFromMessages() is not presently implemented for model %s. "+
				"See https://platform.openai.com/docs/guides/text-generation/managing-tokens "+
				"for information on how messages are converted to tokens",
			model)
	}

	total := 0
	for _, msg := range msgs {
		total += tokensPerMessage
		total += m.GetNumTokens(openAIWireRole(msg.Role))
		if msg.Content != "" {
			total += m.GetNumTokens(msg.Content)
		}
		for _, block := range msg.ContentBlocks {
			switch typed := block.(type) {
			case messages.TextBlock:
				total += m.GetNumTokens(typed.Text)
			case messages.ImageBlock:
				// Python: detail=="low" costs 85 tokens; anything else is
				// sized via PIL/httpx (never fetched here, so ignored).
				if detail, _ := typed.Extras["detail"].(string); strings.EqualFold(detail, "low") {
					total += 85
				}
			}
		}
		if msg.Name != "" {
			total += m.GetNumTokens(msg.Name) + tokensPerName
		}
		// Tool call token counting is not documented by OpenAI; this mirrors
		// Python's approximation of counting function name + arguments.
		for _, call := range msg.ToolCalls {
			total += m.GetNumTokens(call.Name)
			if data, err := json.Marshal(call.Args); err == nil {
				total += m.GetNumTokens(string(data))
			}
		}
		if msg.ToolCallID != "" {
			// Inferred approximation from Python: tool_call_id contributes 3.
			total += 3
		}
	}
	// Every reply is primed with <|im_start|>assistant.
	total += 3
	return total, nil
}

// openAIWireRole maps a message role to the string Python's message dict
// carries ("system"/"user"/"assistant"/"tool"), which is what gets counted.
func openAIWireRole(role messages.Role) string {
	switch role {
	case messages.RoleSystem:
		return "system"
	case messages.RoleHuman:
		return "user"
	case messages.RoleAI:
		return "assistant"
	case messages.RoleTool:
		return "tool"
	default:
		return string(role)
	}
}
