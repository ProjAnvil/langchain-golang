package anthropic

import (
	"strings"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

const (
	topPKey = "anthropic_top_p"
	topKKey = "anthropic_top_k"
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
