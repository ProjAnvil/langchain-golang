package language

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/prompts"
	"github.com/projanvil/langchain-golang/core/ratelimiters"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/modelprofiles"
)

// LLM is the common interface for text-in, text-out language models.
type LLM interface {
	runnables.Runnable[string, string]
	ModelProfile() modelprofiles.Profile
}

// FakeLLM is a deterministic text model for tests and standard conformance.
type FakeLLM struct {
	mu           sync.Mutex
	responses    []string
	responseIdx  int
	streamChunks []string
	rateLimiter  ratelimiters.RateLimiter
	profile      modelprofiles.Profile
}

// FakeLLMOption configures a FakeLLM.
type FakeLLMOption func(*FakeLLM)

// NewFakeLLM creates a deterministic text model.
func NewFakeLLM(opts ...FakeLLMOption) *FakeLLM {
	model := &FakeLLM{}
	for _, opt := range opts {
		opt(model)
	}
	return model
}

// WithLLMResponses configures fixed responses returned by Invoke.
func WithLLMResponses(responses ...string) FakeLLMOption {
	return func(model *FakeLLM) {
		model.responses = append([]string(nil), responses...)
	}
}

// WithLLMStreamChunks configures chunks returned by Stream.
func WithLLMStreamChunks(chunks ...string) FakeLLMOption {
	return func(model *FakeLLM) {
		model.streamChunks = append([]string(nil), chunks...)
	}
}

// WithLLMRateLimiter configures a limiter acquired before invoke/stream calls.
func WithLLMRateLimiter(limiter ratelimiters.RateLimiter) FakeLLMOption {
	return func(model *FakeLLM) {
		model.rateLimiter = limiter
	}
}

// WithLLMModelProfile configures explicit model profile metadata.
func WithLLMModelProfile(profile modelprofiles.Profile) FakeLLMOption {
	return func(model *FakeLLM) {
		model.profile = cloneProfile(profile)
	}
}

// Invoke returns the next configured response or an echo response.
func (m *FakeLLM) Invoke(ctx context.Context, input string, opts ...runnables.Option) (string, error) {
	if err := m.acquireRateLimit(ctx); err != nil {
		return "", err
	}
	cfg := runnables.NewConfig(opts...)
	if err := emitLLMEvent(ctx, cfg, callbacks.EventLLMStart, input, nil, nil); err != nil {
		return "", err
	}

	m.mu.Lock()
	response := "fake response"
	if len(m.responses) > 0 {
		response = m.responses[m.responseIdx%len(m.responses)]
		m.responseIdx++
	} else if input != "" {
		response = "fake response: " + input
	}
	m.mu.Unlock()

	if err := emitLLMEvent(ctx, cfg, callbacks.EventLLMEnd, nil, response, nil); err != nil {
		return "", err
	}
	return response, nil
}

// Batch invokes the model for all inputs while preserving order.
func (m *FakeLLM) Batch(ctx context.Context, inputs []string, opts ...runnables.Option) ([]string, error) {
	outputs := make([]string, len(inputs))
	errs := make([]error, len(inputs))

	var wg sync.WaitGroup
	for i, input := range inputs {
		wg.Add(1)
		go func(i int, input string) {
			defer wg.Done()
			outputs[i], errs[i] = m.Invoke(ctx, input, opts...)
		}(i, input)
	}
	wg.Wait()

	return outputs, errors.Join(errs...)
}

// Stream returns configured chunks or a single Invoke response.
func (m *FakeLLM) Stream(ctx context.Context, input string, opts ...runnables.Option) (runnables.Stream[string], error) {
	m.mu.Lock()
	chunks := append([]string(nil), m.streamChunks...)
	m.mu.Unlock()

	if len(chunks) > 0 {
		if err := m.acquireRateLimit(ctx); err != nil {
			return nil, err
		}
		cfg := runnables.NewConfig(opts...)
		if err := emitLLMEvent(ctx, cfg, callbacks.EventLLMStart, input, nil, nil); err != nil {
			return nil, err
		}
		return newLLMCallbackStream(cfg, runnables.NewSliceStream(chunks)), nil
	}

	response, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]string{response}), nil
}

// InputSchema returns the LLM input schema.
func (m *FakeLLM) InputSchema() schema.Schema {
	return schema.String("text prompt")
}

// OutputSchema returns the LLM output schema.
func (m *FakeLLM) OutputSchema() schema.Schema {
	return schema.String("text completion")
}

// ModelProfile returns explicit profile metadata or a text-only profile.
func (m *FakeLLM) ModelProfile() modelprofiles.Profile {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.profile) > 0 {
		return cloneProfile(m.profile)
	}
	return modelprofiles.Profile{
		"text_inputs":  true,
		"text_outputs": true,
	}
}

func (m *FakeLLM) acquireRateLimit(ctx context.Context) error {
	m.mu.Lock()
	limiter := m.rateLimiter
	m.mu.Unlock()
	if limiter == nil {
		return nil
	}
	_, err := limiter.Acquire(ctx, true)
	return err
}

// PromptValueString converts prompt-shaped values to the string form expected
// by LLMs.
func PromptValueString(input any) (string, error) {
	switch typed := input.(type) {
	case string:
		return typed, nil
	case prompts.PromptValue:
		return typed.ToString(), nil
	case messages.Message:
		return messages.Text(typed), nil
	case []messages.Message:
		return messages.BufferString(typed), nil
	default:
		return "", fmt.Errorf("cannot convert %T to LLM prompt string", input)
	}
}

// PromptValueMessages converts prompt-shaped values to chat messages.
func PromptValueMessages(input any) ([]messages.Message, error) {
	switch typed := input.(type) {
	case string:
		return []messages.Message{messages.Human(typed)}, nil
	case prompts.PromptValue:
		return cloneMessages(typed.ToMessages()), nil
	case messages.Message:
		return []messages.Message{messages.Clone(typed)}, nil
	case []messages.Message:
		return cloneMessages(typed), nil
	default:
		return nil, fmt.Errorf("cannot convert %T to chat prompt messages", input)
	}
}

func emitLLMEvent(
	ctx context.Context,
	cfg runnables.Config,
	kind callbacks.EventKind,
	input any,
	output any,
	err error,
) error {
	if cfg.Callbacks.Empty() {
		return nil
	}
	event := callbacks.Event{
		Kind:     kind,
		Name:     cfg.Name,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     append([]string(nil), cfg.Tags...),
		Metadata: cloneMetadata(cfg.Metadata),
		Input:    input,
		Output:   output,
	}
	if err != nil {
		event.Error = err.Error()
	}
	return cfg.Callbacks.Emit(ctx, event)
}

type llmCallbackStream struct {
	cfg    runnables.Config
	stream runnables.Stream[string]
	ended  bool
}

func newLLMCallbackStream(cfg runnables.Config, stream runnables.Stream[string]) *llmCallbackStream {
	return &llmCallbackStream{cfg: cfg, stream: stream}
}

func (s *llmCallbackStream) Next(ctx context.Context) (string, bool, error) {
	chunk, ok, err := s.stream.Next(ctx)
	if err != nil {
		_ = emitLLMEvent(ctx, s.cfg, callbacks.EventLLMError, nil, nil, err)
		return "", false, err
	}
	if !ok {
		if !s.ended {
			s.ended = true
			if eventErr := emitLLMEvent(ctx, s.cfg, callbacks.EventLLMEnd, nil, nil, nil); eventErr != nil {
				return "", false, eventErr
			}
		}
		return "", false, nil
	}
	if s.cfg.Callbacks.Empty() {
		return chunk, true, nil
	}
	if eventErr := s.cfg.Callbacks.Emit(ctx, callbacks.Event{
		Kind:     callbacks.EventLLMStream,
		Name:     s.cfg.Name,
		RunID:    s.cfg.RunID,
		ParentID: s.cfg.ParentID,
		Tags:     append([]string(nil), s.cfg.Tags...),
		Metadata: cloneMetadata(s.cfg.Metadata),
		Chunk:    chunk,
	}); eventErr != nil {
		return "", false, eventErr
	}
	return chunk, true, nil
}

func (s *llmCallbackStream) Close() error {
	return s.stream.Close()
}

func cloneMessages(values []messages.Message) []messages.Message {
	out := make([]messages.Message, len(values))
	for i, message := range values {
		out[i] = messages.Clone(message)
	}
	return out
}
