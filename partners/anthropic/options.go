package anthropic

import (
	"strings"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

const (
	topPKey          = "anthropic_top_p"
	topKKey          = "anthropic_top_k"
	stopSequencesKey = "anthropic_stop_sequences"
	streamUsageKey   = "anthropic_stream_usage"
)

// WithBetaHeaders sets the anthropic-beta request header to enable beta
// features. Multiple beta flags are comma-joined per Anthropic's convention,
// e.g. WithBetaHeaders("interleaved-thinking-2025-05-14",
// "prompt-caching-2024-07-31").
func WithBetaHeaders(betas ...string) modelconfig.Option {
	return modelconfig.WithHeader("anthropic-beta", strings.Join(betas, ","))
}

// WithTopP sets the Anthropic top_p sampling parameter. Anthropic disallows
// top_p when extended thinking is enabled, so ChatModel omits it in that case.
func WithTopP(topP float64) modelconfig.Option {
	return modelconfig.WithExtra(topPKey, topP)
}

// WithTopK sets the Anthropic top_k sampling parameter. Anthropic disallows
// top_k when extended thinking is enabled, so ChatModel omits it in that case.
func WithTopK(topK int) modelconfig.Option {
	return modelconfig.WithExtra(topKKey, topK)
}

// WithStopSequences sets the Anthropic stop_sequences request parameter:
// custom strings that stop generation when the model produces them. Mirrors
// Python ChatAnthropic's stop_sequences constructor field (alias "stop",
// langchain_anthropic/chat_models.py:944); it applies to both streaming and
// non-streaming requests.
func WithStopSequences(sequences []string) modelconfig.Option {
	return modelconfig.WithExtra(stopSequencesKey, append([]string(nil), sequences...))
}

// WithStreamUsage controls whether the streaming path yields a terminal
// usage-only chunk when the API reports final usage on message_delta.
// Mirrors Python ChatAnthropic's stream_usage constructor field
// (langchain_anthropic/chat_models.py:996, default true): with it on,
// consumers aggregating over the yielded chunks see usage_metadata like
// Python's _stream (chat_models.py:1702-1725); turning it off restores the
// callback-only surface.
func WithStreamUsage(enabled bool) modelconfig.Option {
	return modelconfig.WithExtra(streamUsageKey, enabled)
}
