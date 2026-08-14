package messages

import (
	"encoding/json"
	"fmt"
	"math"
)

// CountTokensOption configures CountTokensApproximately.
type CountTokensOption func(*countTokensOptions)

type countTokensOptions struct {
	charsPerToken        float64
	extraTokensPerMessage float64
	countName            bool
	tokensPerImage       int
}

// WithCharsPerToken sets the characters-per-token divisor (default 4.0).
func WithCharsPerToken(v float64) CountTokensOption {
	return func(o *countTokensOptions) { o.charsPerToken = v }
}

// WithExtraTokensPerMessage sets the per-message token overhead (default 3.0).
func WithExtraTokensPerMessage(v float64) CountTokensOption {
	return func(o *countTokensOptions) { o.extraTokensPerMessage = v }
}

// WithCountName enables counting a message's Name field (default false).
func WithCountName(enabled bool) CountTokensOption {
	return func(o *countTokensOptions) { o.countName = enabled }
}

// CountTokensApproximately estimates the token count of messages, mirroring
// Python's `count_tokens_approximately`: string content and roles contribute
// `len/chars_per_token` + `extra_tokens_per_message`, AI tool calls and tool
// call IDs are counted, and images contribute a fixed penalty. This is the
// "approximate" token counter used by TrimMessages when no counter is given.
func CountTokensApproximately(msgs []Message, opts ...CountTokensOption) int {
	o := countTokensOptions{charsPerToken: 4.0, extraTokensPerMessage: 3.0, tokensPerImage: 85}
	for _, opt := range opts {
		opt(&o)
	}
	ceil := func(n int) int { return int(math.Ceil(float64(n) / o.charsPerToken)) }

	total := 0
	for _, m := range msgs {
		total += int(o.extraTokensPerMessage)
		if m.Content != "" {
			total += ceil(len(m.Content))
		}
		if o.countName && m.Name != "" {
			total += ceil(len(m.Name))
		}
		for _, tc := range m.ToolCalls {
			total += ceil(len(tc.Name))
			if b, err := json.Marshal(tc.Args); err == nil {
				total += ceil(len(b))
			}
		}
		if m.Role == RoleTool && m.ToolCallID != "" {
			total += ceil(len(m.ToolCallID))
		}
		// Fixed penalty per image content block.
		for _, block := range m.ContentBlocks {
			if _, ok := block.(ImageBlock); ok {
				total += o.tokensPerImage
			}
		}
	}
	return total
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
