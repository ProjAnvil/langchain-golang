package messages

import (
	"encoding/json"
	"fmt"
	"math"
)

// CountTokensOption configures CountTokensApproximately.
type CountTokensOption func(*countTokensOptions)

type countTokensOptions struct {
	charsPerToken         float64
	extraTokensPerMessage float64
	countName             bool
	tokensPerImage        int
	usageMetadataScaling  bool
}

// WithCharsPerToken sets the characters-per-token divisor (default 4.0).
func WithCharsPerToken(v float64) CountTokensOption {
	return func(o *countTokensOptions) { o.charsPerToken = v }
}

// WithExtraTokensPerMessage sets the per-message token overhead (default 3.0).
func WithExtraTokensPerMessage(v float64) CountTokensOption {
	return func(o *countTokensOptions) { o.extraTokensPerMessage = v }
}

// WithCountName enables counting a message's Name field (default true,
// mirroring Python's count_name=True).
func WithCountName(enabled bool) CountTokensOption {
	return func(o *countTokensOptions) { o.countName = enabled }
}

// WithUsageMetadataScaling calibrates the estimate against provider-reported
// usage (default false), mirroring Python's `use_usage_metadata_scaling`: when
// enabled, all AI messages share one response_metadata["model_provider"], and
// at least one AI message reports usage_metadata.total_tokens, the estimate is
// multiplied by `total_tokens / approx_tokens_up_to_that_AI_message` (taken
// from the most recent such AI message), clamped to [1.0, 1.25].
func WithUsageMetadataScaling(enabled bool) CountTokensOption {
	return func(o *countTokensOptions) { o.usageMetadataScaling = enabled }
}

// CountTokensApproximately estimates the token count of messages, mirroring
// Python's `count_tokens_approximately` (utils.py:2239): each message
// contributes ceil(chars/chars_per_token) + extra_tokens_per_message where
// chars cover string content, text blocks, the OpenAI-wire role, and
// (optionally) the name; AI tool calls and tool call IDs are counted; images
// contribute a fixed penalty. Python keeps a float total and applies a single
// math.Ceil at the very end (utils.py:2395); per-message counts are rounded
// up so individual message counts add up to the total (utils.py:2361-2364).
// This is the "approximate" token counter used by TrimMessages when no
// counter is given. WithUsageMetadataScaling additionally calibrates the
// estimate against provider-reported usage_metadata.
func CountTokensApproximately(msgs []Message, opts ...CountTokensOption) int {
	o := countTokensOptions{charsPerToken: 4.0, extraTokensPerMessage: 3.0, countName: true, tokensPerImage: 85}
	for _, opt := range opts {
		opt(&o)
	}

	tokenCount := 0.0
	// Usage-metadata scaling state (mirrors Python's ai_model_provider /
	// invalid_model_provider / last_ai_total_tokens / approx_at_last_ai).
	var (
		aiProvider      string
		invalidProvider bool
		lastAITotal     int
		hasLastAITotal  bool
		approxAtLastAI  float64
	)
	for _, m := range msgs {
		messageChars := 0
		if m.Content != "" {
			messageChars += len(m.Content)
		}
		for _, block := range m.ContentBlocks {
			switch typed := block.(type) {
			case TextBlock:
				messageChars += len(typed.Text)
			case ImageBlock:
				// Fixed penalty per image content block (added directly to the
				// total, as in Python).
				tokenCount += float64(o.tokensPerImage)
			default:
				// Conservative estimate for unknown block types, mirroring
				// Python's len(repr(block)).
				messageChars += len(fmt.Sprintf("%v", block))
			}
		}
		// Stringified tool calls count only when content is a plain string
		// (no content blocks), mirroring Python's exclusion of Anthropic
		// format where tool calls are already part of the content.
		if len(m.ContentBlocks) == 0 && len(m.ToolCalls) > 0 {
			if b, err := json.Marshal(m.ToolCalls); err == nil {
				messageChars += len(b)
			}
		}
		if m.Role == RoleTool && m.ToolCallID != "" {
			messageChars += len(m.ToolCallID)
		}
		messageChars += len(approxOpenAIRole(m.Role))
		if o.countName && m.Name != "" {
			messageChars += len(m.Name)
		}
		tokenCount += math.Ceil(float64(messageChars)/o.charsPerToken) + o.extraTokensPerMessage

		if o.usageMetadataScaling && m.Role == RoleAI {
			// Python tracks response_metadata["model_provider"]: the first
			// observed value (even a missing one) is adopted while the
			// accumulated provider is still unset; once set, any differing
			// value (including missing) invalidates scaling.
			provider, _ := m.ResponseMetadata["model_provider"].(string)
			if aiProvider == "" {
				aiProvider = provider
			} else if provider != aiProvider {
				invalidProvider = true
			}
			// A TotalTokens > 0 stands in for Python's
			// `usage_metadata and isinstance(total_tokens, int)` presence check
			// (Go's UsageMetadata is a value struct, not an optional).
			if m.UsageMetadata.TotalTokens > 0 {
				lastAITotal = m.UsageMetadata.TotalTokens
				hasLastAITotal = true
				approxAtLastAI = tokenCount
			}
		}
	}

	if o.usageMetadataScaling &&
		len(msgs) > 1 &&
		!invalidProvider &&
		aiProvider != "" &&
		hasLastAITotal &&
		approxAtLastAI > 0 {
		factor := float64(lastAITotal) / approxAtLastAI
		tokenCount *= math.Min(1.25, math.Max(1.0, factor))
	}
	// Round up once at the end in case extra_tokens_per_message is a float.
	return int(math.Ceil(tokenCount))
}

// approxOpenAIRole maps a message role to the string Python's
// _get_message_openai_role returns ("system"/"user"/"assistant"/"tool"),
// whose length feeds the character count.
func approxOpenAIRole(role Role) string {
	switch role {
	case RoleHuman:
		return "user"
	case RoleAI:
		return "assistant"
	default:
		return string(role)
	}
}

// TrimStrategy selects which end of the message list to keep.
type TrimStrategy string

const (
	// TrimStrategyLast keeps the trailing messages within the token budget.
	TrimStrategyLast TrimStrategy = "last"
	// TrimStrategyFirst keeps the leading messages within the token budget.
	TrimStrategyFirst TrimStrategy = "first"
)

// TrimMessagesOptions configures TrimMessages.
type TrimMessagesOptions struct {
	MaxTokens     int
	Strategy      TrimStrategy
	TokenCounter  func([]Message) int
	IncludeSystem bool
	StartOn       []Role
	EndOn         []Role
	// AllowPartial is accepted for API parity but not implemented: the Go port
	// always trims whole messages (no partial content truncation).
	AllowPartial bool
}

// TrimMessages trims a message list to fit within MaxTokens, mirroring
// Python's `trim_messages` (without partial-content splitting). The default
// token counter is CountTokensApproximately.
func TrimMessages(msgs []Message, opts TrimMessagesOptions) ([]Message, error) {
	if opts.MaxTokens < 0 {
		return nil, fmt.Errorf("messages: TrimMessages: max_tokens must be non-negative, got %d", opts.MaxTokens)
	}
	strategy := opts.Strategy
	if strategy == "" {
		strategy = TrimStrategyLast
	}
	if strategy != TrimStrategyFirst && strategy != TrimStrategyLast {
		return nil, fmt.Errorf("messages: TrimMessages: invalid strategy %q", strategy)
	}
	counter := opts.TokenCounter
	if counter == nil {
		counter = func(msgs []Message) int { return CountTokensApproximately(msgs) }
	}

	out := append([]Message(nil), msgs...)

	// end_on: drop everything after the last occurrence of an end_on role.
	if len(opts.EndOn) > 0 {
		last := -1
		for i, m := range out {
			if roleIn(m.Role, opts.EndOn) {
				last = i
			}
		}
		if last >= 0 {
			out = out[:last+1]
		}
	}

	if strategy == TrimStrategyFirst {
		for len(out) > 0 && counter(out) > opts.MaxTokens {
			out = out[:len(out)-1]
		}
		return out, nil
	}

	// strategy == last
	var system Message
	hasSystem := false
	if opts.IncludeSystem && len(out) > 0 && out[0].Role == RoleSystem {
		system = out[0]
		hasSystem = true
		out = out[1:]
	}
	for len(out) > 0 && counter(out) > opts.MaxTokens {
		out = out[1:]
	}
	if len(opts.StartOn) > 0 {
		first := -1
		for i, m := range out {
			if roleIn(m.Role, opts.StartOn) {
				first = i
				break
			}
		}
		if first >= 0 {
			out = out[first:]
		}
	}
	if hasSystem {
		out = append([]Message{system}, out...)
	}
	return out, nil
}

func roleIn(r Role, roles []Role) bool {
	for _, want := range roles {
		if r == want {
			return true
		}
	}
	return false
}
