package middleware

import (
	"context"
	"regexp"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/messages"
)

type PIIMiddleware struct {
	PIIType            string
	Strategy           RedactionStrategy
	Detector           Detector
	ApplyToInput       bool
	ApplyToOutput      bool
	ApplyToToolResults bool
	rule               ResolvedRedactionRule
}

type PIIOption func(*PIIMiddleware)

func NewPIIMiddleware(piiType string, opts ...PIIOption) (*PIIMiddleware, error) {
	m := &PIIMiddleware{
		PIIType:       piiType,
		Strategy:      RedactionRedact,
		ApplyToInput:  true,
		ApplyToOutput: false,
	}
	for _, opt := range opts {
		opt(m)
	}
	rule, err := (RedactionRule{
		PIIType:  m.PIIType,
		Strategy: m.Strategy,
		Detector: m.Detector,
	}).Resolve()
	if err != nil {
		return nil, err
	}
	m.rule = rule
	m.PIIType = rule.PIIType
	m.Strategy = rule.Strategy
	m.Detector = rule.Detector
	return m, nil
}

func WithPIIStrategy(strategy RedactionStrategy) PIIOption {
	return func(m *PIIMiddleware) {
		m.Strategy = strategy
	}
}

func WithPIIDetector(detector Detector) PIIOption {
	return func(m *PIIMiddleware) {
		m.Detector = detector
	}
}

func WithPIIApplyToInput(apply bool) PIIOption {
	return func(m *PIIMiddleware) {
		m.ApplyToInput = apply
	}
}

func WithPIIApplyToOutput(apply bool) PIIOption {
	return func(m *PIIMiddleware) {
		m.ApplyToOutput = apply
	}
}

func WithPIIApplyToToolResults(apply bool) PIIOption {
	return func(m *PIIMiddleware) {
		m.ApplyToToolResults = apply
	}
}

func (m *PIIMiddleware) Name() string {
	return "PIIMiddleware[" + m.PIIType + "]"
}

func (m *PIIMiddleware) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	if !m.ApplyToInput && !m.ApplyToToolResults {
		return nil, nil
	}
	msgs, ok := messagesFromState(state)
	if !ok || len(msgs) == 0 {
		return nil, nil
	}

	next := cloneMessages(msgs)
	modified := false
	if m.ApplyToInput {
		for i := len(next) - 1; i >= 0; i-- {
			if next[i].Role != messages.RoleHuman || next[i].Content == "" {
				continue
			}
			updated, matches, err := m.rule.Apply(next[i].Content)
			if err != nil {
				return nil, err
			}
			if len(matches) > 0 {
				next[i].Content = updated
				modified = true
			}
			break
		}
	}
	if m.ApplyToToolResults {
		lastAI := -1
		for i := len(next) - 1; i >= 0; i-- {
			if next[i].Role == messages.RoleAI {
				lastAI = i
				break
			}
		}
		if lastAI >= 0 {
			for i := lastAI + 1; i < len(next); i++ {
				if next[i].Role != messages.RoleTool || next[i].Content == "" {
					continue
				}
				updated, matches, err := m.rule.Apply(next[i].Content)
				if err != nil {
					return nil, err
				}
				if len(matches) > 0 {
					next[i].Content = updated
					modified = true
				}
			}
		}
	}
	if !modified {
		return nil, nil
	}
	return map[string]any{"messages": next}, nil
}

func (m *PIIMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	if !m.ApplyToOutput {
		return nil, nil
	}
	msgs, ok := messagesFromState(state)
	if !ok || len(msgs) == 0 {
		return nil, nil
	}
	next := cloneMessages(msgs)
	for i := len(next) - 1; i >= 0; i-- {
		if next[i].Role != messages.RoleAI {
			continue
		}
		updated, changed, err := m.redactAIMessage(next[i])
		if err != nil {
			return nil, err
		}
		if !changed {
			return nil, nil
		}
		next[i] = updated
		return map[string]any{"messages": next}, nil
	}
	return nil, nil
}

func (m *PIIMiddleware) redactAIMessage(message messages.Message) (messages.Message, bool, error) {
	changed := false
	if message.Content != "" {
		updated, matches, err := m.rule.Apply(message.Content)
		if err != nil {
			return messages.Message{}, false, err
		}
		if len(matches) > 0 {
			message.Content = updated
			changed = true
		}
	}
	for i, call := range message.ToolCalls {
		args, argChanged, err := m.redactMapStrings(call.Args)
		if err != nil {
			return messages.Message{}, false, err
		}
		if argChanged {
			message.ToolCalls[i].Args = args
			changed = true
		}
	}
	for i, call := range message.InvalidToolCalls {
		args, argChanged, err := m.redactMapStrings(call.Args)
		if err != nil {
			return messages.Message{}, false, err
		}
		if argChanged {
			message.InvalidToolCalls[i].Args = args
			changed = true
		}
	}
	return message, changed, nil
}

func (m *PIIMiddleware) redactMapStrings(input map[string]any) (map[string]any, bool, error) {
	if input == nil {
		return nil, false, nil
	}
	out := cloneAnyMap(input)
	changed := false
	for key, value := range out {
		text, ok := value.(string)
		if !ok || text == "" {
			continue
		}
		updated, matches, err := m.rule.Apply(text)
		if err != nil {
			return nil, false, err
		}
		if len(matches) > 0 {
			out[key] = updated
			changed = true
		}
	}
	return out, changed, nil
}

func messagesFromState(state map[string]any) ([]messages.Message, bool) {
	if state == nil {
		return nil, false
	}
	msgs, ok := state["messages"].([]messages.Message)
	return msgs, ok
}

// defaultStreamLookback is the floor for the trailing buffer when no
// patterns are supplied. The brief specifies the lookback as "the longest
// pattern length"; when patterns ARE supplied we use exactly that (no floor),
// so the brief's BoundaryStraddle test — whose full text is shorter than 128
// chars — actually emits redacted output rather than swallowing it whole.
// 128 mirrors Python's _DEFAULT_STREAM_LOOKBACK; we only fall back to it for
// the degenerate empty-patterns case so the transformer stays a no-op
// redactor rather than dividing by zero.
const defaultStreamLookback = 128

// PIIStreamTransformer is a WrapModelStreamHook that redacts PII from
// streaming model deltas using a lookback buffer. It exists for the case the
// batch PIIMiddleware cannot cover: a PII pattern split across two text-delta
// chunks (e.g. "SSN" + "-123") would escape per-chunk redaction because
// neither chunk alone matches the regex.
//
// The transformer holds back a trailing tail of size lookback between deltas;
// the next delta is concatenated with the tail before redaction runs, so
// straddling patterns are caught. The held tail is emitted on Flush (or, in
// the streaming-agent pipeline, implicitly via the model_end full-text call —
// see below).
//
// Task 3.1's WrapModelStreamHook contract calls the SAME composed
// DeltaTransform instance multiple times per model call: once per text-delta,
// once on the content-block-finish event's fully-assembled block text, and
// once on the assembled model_end message. A naive append-only buffer would
// re-process the full text on the latter calls and corrupt it (duplicating
// the leading fragment). The transformer detects these terminal/full-text
// calls — the incoming text re-delivers the raw text accumulated from prior
// deltas (length >= accumulated raw bytes AND prefix matches rawSeen) — and
// returns the freshly-redacted FULL text with no trailing withhold (so the
// final message is never truncated). The held tail is PRESERVED across a
// terminal call rather than discarded: a coincidental-prefix delta that
// falsely triggers the terminal branch would otherwise lose the buffered tail
// and leak any PII straddling the reset boundary. In the genuine terminal
// path (finish/model_end) no further deltas arrive, so the preserved tail is
// simply unused. Regex PII redaction is idempotent, so re-redacting the
// assembled text is safe.
//
// Mirrors Python's _PIIStreamTransformer (langchain_v1/langchain/agents/
// middleware/pii.py), scoped to the Go middleware streaming surface.
//
// Scope note: the per-call state is a SINGLE buffer (not keyed by content
// block index, unlike Python). The Go port's only streaming surface is the
// streamChunkBridge legacy bridge, which emits exactly one text content block
// at index 0; the single buffer covers that path. Multi-block native-v3
// streaming (where text deltas for blocks 0 and 1 interleave) is out of scope
// for this port and would need per-index buffers to handle correctly.
type PIIStreamTransformer struct {
	patterns []*regexp.Regexp
	lookback int

	// lastState + flushMu let Flush() reach into the most-recent closure's
	// state. They are only touched under flushMu. In the streaming-agent
	// path Flush is never called (the model_end full-text call resets the
	// buffer via the multi-call reset branch), so this is a convenience for
	// direct/test usage where a caller wires up one transform and consumes
	// only per-delta output.
	flushMu  sync.Mutex
	lastState *piiStreamState
}

// piiStreamState is the per-call mutable state closed over by the
// DeltaTransform returned from TransformModelStream. A fresh state per model
// call keeps the transformer safe to reuse across many calls (and across
// concurrent agents on the same PIIStreamTransformer).
type piiStreamState struct {
	patterns []*regexp.Regexp
	lookback int

	// mu guards held (and rawSeen) so that a concurrent Flush() from another
	// goroutine does not race with the delta path. The streaming-agent
	// pipeline drives the delta path single-threaded and never calls Flush,
	// so in normal use mu is uncontended; this protects direct/test callers
	// that MAY run Flush concurrently with a streaming producer.
	mu sync.Mutex

	// held is the trailing tail of the post-redaction buffer held back from
	// the previous delta. Concatenating the next delta onto it lets the
	// regex see straddling patterns as one continuous string. Guarded by mu.
	held string
	// rawSeen is the concatenation of every raw text input the delta path
	// has observed so far this call. The full-text model_end / finish call
	// delivers the entire assembled message; it will start with rawSeen (or
	// equal it), which is the signature used to detect the terminal/reset
	// case. Guarded by mu.
	rawSeen string
}

// NewPIIStreamTransformer builds a streaming PII redactor from a list of
// regex patterns. Each match on the (buffered) stream text is replaced with
// the literal "[REDACTED]". The lookback window is the longest pattern source
// length — a pattern can only straddle a chunk boundary if at least
// (pattern-length - 1) chars are held back, so using the pattern source
// length keeps the buffer minimal while still catching cross-boundary matches
// for fixed-shape regexes. Callers needing a larger window for variable-
// length patterns (e.g. `\d+`) can include a longer sentinel regex.
//
// Mirrors Python's _PIIStreamTransformer constructor.
func NewPIIStreamTransformer(patterns []string) *PIIStreamTransformer {
	compiled := make([]*regexp.Regexp, 0, len(patterns))
	lookback := 0
	for _, p := range patterns {
		re := regexp.MustCompile(p)
		compiled = append(compiled, re)
		if n := len(p); n > lookback {
			lookback = n
		}
	}
	if lookback == 0 {
		// No patterns supplied: keep the transformer usable (Flush/lookback
		// queries still work) but it redacts nothing.
		lookback = defaultStreamLookback
	}
	return &PIIStreamTransformer{patterns: compiled, lookback: lookback}
}

// Lookback returns the size (in bytes) of the trailing buffer held back
// between deltas. Exposed for tests and observability.
func (x *PIIStreamTransformer) Lookback() int { return x.lookback }

// Patterns returns the compiled regex patterns (for diagnostics/tests).
func (x *PIIStreamTransformer) Patterns() []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(x.patterns))
	copy(out, x.patterns)
	return out
}

// TransformModelStream implements WrapModelStreamHook. It returns a fresh
// DeltaTransform with its own buffer state so the same PIIStreamTransformer
// is safe to reuse across model calls (and across goroutines — each call
// gets an independent state, so there is no shared mutable state to guard).
//
// The returned transform, per call:
//  1. Terminal/full-text detection: if the incoming text re-delivers the raw
//     text accumulated from prior deltas (the signature of a content-block-
//     finish or model_end full-text call from Task 3.1's invokeModelStreaming
//     path — incoming length >= accumulated raw bytes AND text starts with
//     rawSeen), return the freshly-redacted FULL text with no trailing
//     withhold. The held tail is PRESERVED (not cleared) so that a false-
//     positive reset on a coincidental-prefix delta does not lose buffered
//     content; the next delta still concatenates with the pre-reset tail.
//  2. Per-delta path: concatenate the held tail with the incoming text,
//     redact, hold back the trailing `lookback` bytes, and emit the prefix.
//
// next is the inner transform in the WrapModelStreamHook composition chain
// (Task 3.1).
func (x *PIIStreamTransformer) TransformModelStream(next DeltaTransform) DeltaTransform {
	state := &piiStreamState{
		patterns: x.patterns,
		lookback: x.lookback,
	}
	x.flushMu.Lock()
	x.lastState = state
	x.flushMu.Unlock()
	return func(text string) string {
		return next(state.apply(text))
	}
}

// Flush emits any held-back tail at stream end. Streaming-agent callers do
// not need to call this — Task 3.1's invokeModelStreaming applies the
// transform to the full assembled model_end text, which hits the terminal
// branch (see TransformModelStream) and emits the cleanly-redacted full text.
// Flush is provided for direct callers that consume only the per-delta output.
//
// Flush is safe to call concurrently with the delta path: both serialize on
// the per-call state's mutex. It is safe to call at most once per stream;
// calling it twice returns the held tail again only if new deltas have
// arrived in between.
func (x *PIIStreamTransformer) Flush() string {
	// Flush operates on the receiver's patterns but needs access to the
	// state held by the closure returned from TransformModelStream. Because
	// Flush is a method on the transformer (not the closure), it tracks a
	// sentinel state used only when the caller is using Flush directly
	// rather than going through the streaming pipeline. Tests that exercise
	// Flush pair it with a single active closure, so we route through the
	// last state.
	x.flushMu.Lock()
	last := x.lastState
	x.flushMu.Unlock()
	if last == nil {
		return ""
	}
	return last.flush()
}

// apply runs the per-call lookback machinery. See TransformModelStream for
// the contract.
//
// apply is guarded by s.mu so a concurrent Flush cannot race with the delta
// path's held read/write. The streaming pipeline runs the delta path single-
// threaded, so the lock is uncontended in normal use.
func (s *piiStreamState) apply(text string) string {
	if text == "" {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	// Terminal detection: the incoming text is the assembled full text from
	// a content-block-finish or model_end event (Task 3.1's multi-call
	// pattern). It re-delivers everything the delta path has emitted so far,
	// so its length is >= the accumulated raw delta bytes and it begins with
	// rawSeen. On a terminal call the transform returns the freshly-redacted
	// FULL text with no trailing withhold — otherwise the final message would
	// be truncated by up to `lookback` bytes (the held tail would never be
	// released, since the streaming pipeline does not call Flush).
	//
	// held is INTENTIONALLY PRESERVED (not cleared) on a terminal. This is
	// the fix for the false-positive reset bug: if a per-delta call happens
	// to start with the accumulated rawSeen (rare for token streams, but a
	// real correctness gap), discarding held would lose the buffered tail and
	// any PII straddling the reset boundary would leak. Preserving held means
	// the NEXT delta still concatenates with the pre-reset tail, so straddling
	// patterns are caught. In the genuine terminal path (finish/model_end) no
	// further deltas arrive for this call, so the preserved held is simply
	// unused (harmless).
	if s.rawSeen != "" && len(text) >= len(s.rawSeen) && strings.HasPrefix(text, s.rawSeen) {
		// Do NOT clear held — see the preserve-held rationale above.
		// Keep rawSeen as `text` so a subsequent terminal call (e.g.
		// model_end after content-block-finish) re-fires rather than being
		// mistaken for a delta, and so a post-reset delta does not
		// false-match the old (shorter) prefix.
		s.rawSeen = text
		return s.redact(text)
	}

	combined := s.held + text
	s.rawSeen += text
	redacted := s.redact(combined)

	var emit, held string
	if len(redacted) > s.lookback {
		emit = redacted[:len(redacted)-s.lookback]
		held = redacted[len(redacted)-s.lookback:]
	} else {
		emit = ""
		held = redacted
	}
	s.held = held
	return emit
}

// flush emits and clears the held tail. After flush, the state is empty.
// flush is guarded by s.mu so it cannot race with a concurrent apply.
func (s *piiStreamState) flush() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.held
	s.held = ""
	return out
}

// redact applies every compiled pattern to text, replacing each match with
// the literal "[REDACTED]". Patterns are applied sequentially; later
// patterns see the output of earlier ones, but since "[REDACTED]" contains
// no characters that could re-trigger a PII regex, ordering is immaterial.
func (s *piiStreamState) redact(text string) string {
	out := text
	for _, re := range s.patterns {
		out = re.ReplaceAllString(out, "[REDACTED]")
	}
	return out
}
