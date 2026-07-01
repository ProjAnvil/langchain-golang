package middleware

import (
	"context"

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
