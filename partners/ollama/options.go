package ollama

import (
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// Extra-config keys used to carry Ollama-specific options through the
// provider-neutral modelconfig.Config.
const (
	reasoningKey = "ollama_reasoning"
	formatKey    = "ollama_format"
	keepAliveKey = "ollama_keep_alive"
	topPKey      = "ollama_top_p"
	topKKey      = "ollama_top_k"
	numCtxKey    = "ollama_num_ctx"
	seedKey      = "ollama_seed"
	stopKey      = "ollama_stop"
)

// WithReasoning sets the Ollama think request parameter. Pass true to enable
// reasoning (captured separately from content), false to disable it, or a
// provider-supported level string such as "low", "medium", or "high".
func WithReasoning(think any) modelconfig.Option {
	return modelconfig.WithExtra(reasoningKey, think)
}

// WithFormat sets the Ollama format request parameter, e.g. "json" or a raw
// JSON-schema object.
func WithFormat(format any) modelconfig.Option {
	return modelconfig.WithExtra(formatKey, format)
}

// WithJSONMode enables Ollama JSON output mode (format="json").
func WithJSONMode() modelconfig.Option {
	return modelconfig.WithExtra(formatKey, "json")
}

// WithKeepAlive sets how long the model stays loaded in memory after the
// request, e.g. "5m".
func WithKeepAlive(keepAlive any) modelconfig.Option {
	return modelconfig.WithExtra(keepAliveKey, keepAlive)
}

// WithTopP sets the options.top_p sampling parameter.
func WithTopP(topP float64) modelconfig.Option {
	return modelconfig.WithExtra(topPKey, topP)
}

// WithTopK sets the options.top_k sampling parameter.
func WithTopK(topK int) modelconfig.Option {
	return modelconfig.WithExtra(topKKey, topK)
}

// WithNumCtx sets the options.num_ctx context window size.
func WithNumCtx(numCtx int) modelconfig.Option {
	return modelconfig.WithExtra(numCtxKey, numCtx)
}

// WithSeed sets the options.seed generation seed.
func WithSeed(seed int) modelconfig.Option {
	return modelconfig.WithExtra(seedKey, seed)
}

// WithStop sets the options.stop stop sequences.
func WithStop(stop []string) modelconfig.Option {
	return modelconfig.WithExtra(stopKey, stop)
}

// samplingOptionKeys maps extra-config keys to Ollama options object keys.
var samplingOptionKeys = map[string]string{
	topPKey:   "top_p",
	topKKey:   "top_k",
	numCtxKey: "num_ctx",
	seedKey:   "seed",
	stopKey:   "stop",
}
