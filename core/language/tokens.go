package language

import (
	"hash/fnv"

	"github.com/projanvil/langchain-golang/core/messages"
)

// TokenCounter is implemented by models that expose a model-specific
// tokenizer, mirroring Python's BaseLanguageModel.get_token_ids
// (language_models/base.py:419).
type TokenCounter interface {
	GetTokenIDs(text string) []int
}

// MessageTokenCounter is implemented by models that count tokens across chat
// messages with provider-specific overhead rules, mirroring Python's
// BaseLanguageModel.get_num_tokens_from_messages
// (language_models/base.py:450).
type MessageTokenCounter interface {
	GetNumTokensFromMessages(msgs []messages.Message) (int, error)
}

// approximateCharsPerToken mirrors the chars-per-token heuristic used by
// messages.CountTokensApproximately (core/messages/trim.go).
const approximateCharsPerToken = 4

// DefaultGetTokenIDs approximates Python's fallback get_token_ids
// (language_models/base.py:104, a GPT-2 tokenizer) without shipping a BPE
// tokenizer: text is split into chunks of approximateCharsPerToken runes and
// each chunk is mapped to a deterministic non-negative FNV-1a 31-bit ID.
// Counts are approximate; IDs are stable but are NOT real tokenizer IDs.
// Models with a real tokenizer should implement TokenCounter.
func DefaultGetTokenIDs(text string) []int {
	runes := []rune(text)
	if len(runes) == 0 {
		return []int{}
	}
	ids := make([]int, 0, (len(runes)+approximateCharsPerToken-1)/approximateCharsPerToken)
	for start := 0; start < len(runes); start += approximateCharsPerToken {
		end := start + approximateCharsPerToken
		if end > len(runes) {
			end = len(runes)
		}
		h := fnv.New32a()
		_, _ = h.Write([]byte(string(runes[start:end])))
		ids = append(ids, int(h.Sum32()&0x7FFFFFFF))
	}
	return ids
}

// GetTokenIDs returns the token IDs for text, dispatching to the model's
// TokenCounter implementation when available and falling back to
// DefaultGetTokenIDs otherwise.
func GetTokenIDs(model any, text string) []int {
	if counter, ok := model.(TokenCounter); ok {
		return counter.GetTokenIDs(text)
	}
	return DefaultGetTokenIDs(text)
}

// GetNumTokens returns the number of tokens in text, mirroring Python's
// BaseLanguageModel.get_num_tokens (language_models/base.py:433, len of
// get_token_ids). Useful for checking if an input fits a context window.
func GetNumTokens(model any, text string) int {
	return len(GetTokenIDs(model, text))
}

// GetNumTokensFromMessages sums the token count of each message rendered via
// messages.BufferString, mirroring Python's base implementation
// (language_models/base.py:450-485): per-message role prefixes are included
// and tool schemas are ignored (Python warns and ignores them). Models with
// provider-specific overhead rules should implement MessageTokenCounter.
func GetNumTokensFromMessages(model any, msgs []messages.Message) (int, error) {
	if counter, ok := model.(MessageTokenCounter); ok {
		return counter.GetNumTokensFromMessages(msgs)
	}
	total := 0
	for _, msg := range msgs {
		total += GetNumTokens(model, messages.BufferString([]messages.Message{msg}))
	}
	return total, nil
}
