package middleware

import (
	"context"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/modelprofiles"
)

const DefaultSummaryPrompt = `<role>
Context Extraction Assistant
</role>

The conversation history below will be replaced with the context you extract.
Respond ONLY with the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>`

// DefaultMessagesToKeep is the number of trailing messages retained when no
// explicit Keep policy resolves (mirrors Python's `_DEFAULT_MESSAGES_TO_KEEP`).
const DefaultMessagesToKeep = 20

// DefaultTrimTokensToSummarize is the default token budget applied to the
// messages handed to the summarizer, mirroring Python's
// `_DEFAULT_TRIM_TOKEN_LIMIT`. Set SummarizationMiddleware.TrimTokensToSummarize
// to a value <= 0 to disable this pre-trim entirely (Python's `None`).
const DefaultTrimTokensToSummarize = 4000

// TriggerClause expresses an AND condition across whichever fields are
// non-zero. Combine multiple clauses in SummarizationMiddleware.Trigger for
// OR semantics across clauses, mirroring Python's TriggerClause/ContextSize
// union.
type TriggerClause struct {
	Tokens   int
	Messages int
	// Fraction triggers summarization once token usage reaches this fraction
	// of the model's max input tokens (resolved via ModelProfileProvider on
	// SummarizationMiddleware.Model). Ignored if no profile is resolvable.
	Fraction float64
}

// KeepPolicy describes how much trailing context to retain after
// summarization. Only one of Messages, Tokens, or Fraction should be set;
// when more than one is set, Messages takes precedence, then Tokens, then
// Fraction, mirroring the priority Python encodes via a single ContextSize
// tuple.
type KeepPolicy struct {
	Messages int
	Tokens   int
	// Fraction retains this fraction of the model's max input tokens' worth
	// of trailing messages. Falls back to DefaultMessagesToKeep if no model
	// profile is resolvable.
	Fraction float64
}

// ModelProfileProvider is implemented by core/language.ChatModel (and any
// other model type) that can report modelprofiles.Profile metadata. It's
// duck-typed here (rather than importing core/language.ChatModel directly) so
// SummarizationMiddleware.Model can hold any model-like value.
type ModelProfileProvider interface {
	ModelProfile() modelprofiles.Profile
}

type SummarizerFunc func(prompt string, messagesToSummarize []messages.Message) (string, error)

type SummarizationMiddleware struct {
	Trigger       []TriggerClause
	Keep          KeepPolicy
	TokenCounter  TokenCounter
	SummaryPrompt string
	Summarize     SummarizerFunc
	// Model, if set and implementing ModelProfileProvider, resolves
	// Fraction-based Trigger/Keep values against the model's max input
	// tokens. Optional: fraction clauses are simply skipped/ignored (not an
	// error) if no profile is resolvable, unlike Python which raises eagerly
	// at construction time.
	Model any
	// TrimTokensToSummarize caps the token budget of messages handed to
	// Summarize, keeping the most recent messages that fit and dropping any
	// leading non-human messages so the trimmed window starts on a human
	// message (mirrors Python's `trim_messages(..., start_on="human",
	// strategy="last")`). A value <= 0 disables trimming entirely.
	TrimTokensToSummarize int
}

func NewSummarizationMiddleware(summarize SummarizerFunc) *SummarizationMiddleware {
	return &SummarizationMiddleware{
		Trigger:               []TriggerClause{{Messages: 50}},
		Keep:                  KeepPolicy{Messages: 20},
		SummaryPrompt:         DefaultSummaryPrompt,
		Summarize:             summarize,
		TrimTokensToSummarize: DefaultTrimTokensToSummarize,
	}
}

func (m *SummarizationMiddleware) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	msgs, ok := messagesFromState(state)
	if !ok || len(msgs) == 0 {
		return nil, nil
	}
	if !m.shouldTrigger(msgs) {
		return nil, nil
	}
	if m.Summarize == nil {
		return nil, fmt.Errorf("summarization requires a summarizer function")
	}
	keepStart := m.keepStart(msgs)
	if keepStart <= 0 {
		return nil, nil
	}
	toSummarize := cloneMessages(msgs[:keepStart])
	kept := cloneMessages(msgs[keepStart:])
	trimmed := m.trimForSummary(toSummarize)
	var summary string
	if len(trimmed) == 0 {
		summary = "Previous conversation was too long to summarize."
	} else {
		prompt := strings.ReplaceAll(m.summaryPrompt(), "{messages}", bufferString(trimmed))
		s, err := m.Summarize(prompt, trimmed)
		if err != nil {
			return nil, err
		}
		summary = s
	}
	summaryMessage := messages.Human(summary)
	summaryMessage.ResponseMetadata = map[string]any{"summary": true}
	next := append([]messages.Message{summaryMessage}, kept...)
	return map[string]any{"messages": next}, nil
}

func (m *SummarizationMiddleware) shouldTrigger(msgs []messages.Message) bool {
	clauses := m.Trigger
	if len(clauses) == 0 {
		clauses = []TriggerClause{{Messages: 50}}
	}
	needsTokens := false
	for _, clause := range clauses {
		if clause.Tokens > 0 || clause.Fraction > 0 {
			needsTokens = true
			break
		}
	}
	tokens := 0
	if needsTokens {
		tokens = m.countTokens(msgs)
	}
	maxInputTokens, hasMaxInputTokens := m.maxInputTokens()
	for _, clause := range clauses {
		matches := true
		if clause.Messages > 0 && len(msgs) < clause.Messages {
			matches = false
		}
		if matches && clause.Tokens > 0 && tokens < clause.Tokens &&
			!m.shouldSummarizeBasedOnReportedTokens(msgs, float64(clause.Tokens)) {
			matches = false
		}
		if matches && clause.Fraction > 0 {
			if !hasMaxInputTokens {
				matches = false
			} else if tokens < fractionThreshold(maxInputTokens, clause.Fraction) &&
				!m.shouldSummarizeBasedOnReportedTokens(msgs, float64(fractionThreshold(maxInputTokens, clause.Fraction))) {
				matches = false
			}
		}
		if matches {
			return true
		}
	}
	return false
}

func (m *SummarizationMiddleware) keepStart(msgs []messages.Message) int {
	if m.Keep.Messages > 0 {
		start := len(msgs) - m.Keep.Messages
		if start < 0 {
			return 0
		}
		return start
	}
	if m.Keep.Tokens > 0 {
		return m.tokenBudgetCutoff(msgs, m.Keep.Tokens)
	}
	if m.Keep.Fraction > 0 {
		if maxInputTokens, ok := m.maxInputTokens(); ok {
			return m.tokenBudgetCutoff(msgs, fractionThreshold(maxInputTokens, m.Keep.Fraction))
		}
		// Model profile unavailable: fall back to the default message-count
		// retention, mirroring Python's `_determine_cutoff_index` fallback.
	}
	start := len(msgs) - DefaultMessagesToKeep
	if start < 0 {
		return 0
	}
	return start
}

// tokenBudgetCutoff returns the smallest start index whose suffix (msgs[i:])
// fits within budget tokens, counted message-by-message from the end.
func (m *SummarizationMiddleware) tokenBudgetCutoff(msgs []messages.Message, budget int) int {
	total := 0
	for i := len(msgs) - 1; i >= 0; i-- {
		total += m.countTokens([]messages.Message{msgs[i]})
		if total > budget {
			return i + 1
		}
	}
	return 0
}

// fractionThreshold converts a fraction of a model's max input tokens into an
// absolute token count, clamped to at least 1 (mirroring Python's clause
// evaluation for `("fraction", value)`).
func fractionThreshold(maxInputTokens int, fraction float64) int {
	threshold := int(float64(maxInputTokens) * fraction)
	if threshold <= 0 {
		threshold = 1
	}
	return threshold
}

// maxInputTokens resolves the model's max input token limit from Model, if
// set and implementing ModelProfileProvider. Returns ok=false if Model is
// unset, doesn't implement ModelProfileProvider, or the profile has no usable
// max_input_tokens entry.
func (m *SummarizationMiddleware) maxInputTokens() (int, bool) {
	provider, ok := m.Model.(ModelProfileProvider)
	if !ok {
		return 0, false
	}
	profile := provider.ModelProfile()
	if profile == nil {
		return 0, false
	}
	value, ok := profile[modelprofiles.FieldMaxInputTokens]
	if !ok {
		return 0, false
	}
	switch v := value.(type) {
	case int:
		return v, v > 0
	case int64:
		return int(v), v > 0
	case float64:
		return int(v), v > 0
	default:
		return 0, false
	}
}

// resolveTokenCounter mirrors Python's _get_approximate_token_counter
// (summarization.py:208-216): when the user did not override TokenCounter, an
// anthropic-chat model (per LLMType) selects the 3.3 chars/token counter;
// anything else uses the default ApproximateTokenCount. A nil model or one
// without LLMType also uses the default (Python falls through the same way).
func resolveTokenCounter(model any, userOverride TokenCounter) TokenCounter {
	if userOverride != nil {
		return userOverride
	}
	if lt, ok := model.(LLMTypeProvider); ok && strings.HasPrefix(lt.LLMType(), "anthropic-chat") {
		const anthropicCharsPerToken = 3.3
		return func(msgs []messages.Message) int {
			return ApproximateTokenCountCharsPerToken(msgs, anthropicCharsPerToken)
		}
	}
	return ApproximateTokenCount
}

// shouldSummarizeBasedOnReportedTokens mirrors Python's
// _should_summarize_based_on_reported_tokens (summarization.py:561-581): when
// the last AI message carries usage_metadata.total_tokens at/above threshold
// AND its response_metadata.model_provider matches the model's LLMType prefix,
// summarization triggers even if the estimated count is under threshold.
func (m *SummarizationMiddleware) shouldSummarizeBasedOnReportedTokens(msgs []messages.Message, threshold float64) bool {
	var lastAI *messages.Message
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleAI {
			lastAI = &msgs[i]
			break
		}
	}
	if lastAI == nil || lastAI.UsageMetadata.TotalTokens == 0 {
		return false
	}
	if float64(lastAI.UsageMetadata.TotalTokens) < threshold {
		return false
	}
	messageProvider, _ := lastAI.ResponseMetadata["model_provider"].(string)
	if messageProvider == "" {
		return false
	}
	modelLLMType := ""
	if lt, ok := m.Model.(LLMTypeProvider); ok {
		modelLLMType = lt.LLMType()
	}
	return providerMatches(messageProvider, modelLLMType)
}

// providerMatches mirrors Python's _provider_matches (summarization.py:103-109):
// the message's model_provider must match the model's identity. Go LLMType
// values are "<provider>-chat" (e.g. "anthropic-chat") while model_provider is
// the bare provider ("anthropic"), so compare by prefix (and exact as a fallback).
func providerMatches(messageProvider, modelLLMType string) bool {
	if modelLLMType == "" {
		return false
	}
	if strings.HasPrefix(modelLLMType, messageProvider) {
		return true
	}
	return messageProvider == modelLLMType
}

func (m *SummarizationMiddleware) countTokens(msgs []messages.Message) int {
	return resolveTokenCounter(m.Model, m.TokenCounter)(msgs)
}

func (m *SummarizationMiddleware) summaryPrompt() string {
	if m.SummaryPrompt == "" {
		return DefaultSummaryPrompt
	}
	return m.SummaryPrompt
}

func bufferString(msgs []messages.Message) string {
	var b strings.Builder
	for _, msg := range msgs {
		b.WriteString(string(msg.Role))
		b.WriteString(": ")
		b.WriteString(msg.Content)
		b.WriteString("\n")
	}
	return b.String()
}

// trimForSummary applies TrimTokensToSummarize to msgs before they're handed
// to Summarize, mirroring Python's `_trim_messages_for_summary`: keep the
// most recent messages that fit within the token budget (always retaining a
// leading system message, if any), then drop any leading non-human messages
// so the result starts on a human message. Returns msgs unchanged if
// TrimTokensToSummarize is <= 0.
func (m *SummarizationMiddleware) trimForSummary(msgs []messages.Message) []messages.Message {
	if m.TrimTokensToSummarize <= 0 {
		return msgs
	}
	return dropUntilHuman(m.tokenBudgetTrimFromEnd(msgs, m.TrimTokensToSummarize))
}

// tokenBudgetTrimFromEnd keeps the most recent messages whose combined token
// count fits within budget, always preserving a leading system message
// regardless of budget (mirrors `trim_messages(..., strategy="last",
// include_system=True)`).
func (m *SummarizationMiddleware) tokenBudgetTrimFromEnd(msgs []messages.Message, budget int) []messages.Message {
	if len(msgs) == 0 {
		return msgs
	}
	rest := msgs
	var system *messages.Message
	if msgs[0].Role == messages.RoleSystem {
		s := msgs[0]
		system = &s
		rest = msgs[1:]
	}
	start := m.tokenBudgetCutoff(rest, budget)
	out := append([]messages.Message{}, rest[start:]...)
	if system != nil {
		out = append([]messages.Message{*system}, out...)
	}
	return out
}

// dropUntilHuman drops leading non-human, non-system messages so the
// returned slice starts on a human message (after an optional leading system
// message), mirroring `trim_messages(..., start_on="human")`. Returns nil if
// no human message is present (only a leading system message, if any, is
// kept).
func dropUntilHuman(msgs []messages.Message) []messages.Message {
	if len(msgs) == 0 {
		return nil
	}
	start := 0
	hasSystem := msgs[0].Role == messages.RoleSystem
	if hasSystem {
		start = 1
	}
	for i := start; i < len(msgs); i++ {
		if msgs[i].Role == messages.RoleHuman {
			if hasSystem {
				return append(append([]messages.Message{}, msgs[0]), msgs[i:]...)
			}
			return msgs[i:]
		}
	}
	if hasSystem {
		return msgs[:1]
	}
	return nil
}
