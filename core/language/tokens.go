package language

import (
	"sync"

	"github.com/tiktoken-go/tokenizer"

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

// getGPT2Tokenizer lazily caches the GPT-2 (r50k_base) BPE tokenizer used by
// DefaultGetTokenIDs. The tiktoken-go vocabulary is embedded in the library,
// so constructing it does no network I/O; sync.OnceValues builds it once per
// process. It requests R50kBase rather than GPT2Enc because tiktoken-go
// v0.8.1's Get does not register the "gpt2" spelling (ForModel(GPT2) routes
// to it and fails); r50k_base is the identical GPT-2 vocabulary.
var getGPT2Tokenizer = sync.OnceValues(func() (tokenizer.Codec, error) {
	return tokenizer.Get(tokenizer.R50kBase)
})

// DefaultGetTokenIDs mirrors Python's fallback get_token_ids
// (language_models/base.py:98-104): models without a tokenizer of their own
// get token IDs from the real GPT-2 BPE tokenizer (gpt2 = r50k_base, the
// vocabulary behind transformers' GPT2TokenizerFast). Counts are true BPE
// counts ("hello world" is 2 tokens). Models with a provider-specific
// tokenizer should implement TokenCounter.
func DefaultGetTokenIDs(text string) []int {
	codec, err := getGPT2Tokenizer()
	if err != nil {
		// Unreachable in practice: R50kBase is a compile-time constant whose
		// vocabulary ships embedded in tiktoken-go, so Get cannot fail here.
		// Panicking surfaces the programming error (e.g. a future encoding
		// constant change) instead of silently returning fabricated IDs.
		panic("language: failed to load embedded GPT-2 tokenizer: " + err.Error())
	}
	ids, _, err := codec.Encode(text)
	if err != nil {
		// Equally unreachable: encoding with the embedded r50k_base codec
		// cannot fail for arbitrary text. Panic rather than swallow.
		panic("language: GPT-2 tokenizer failed to encode: " + err.Error())
	}
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
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
