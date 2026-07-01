package language

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/ratelimiters"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/modelprofiles"
)

// ChatModelCapabilities describes optional behavior supported by a chat model.
type ChatModelCapabilities struct {
	ToolCalling      bool
	ToolChoice       bool
	StructuredOutput bool
	JSONMode         bool
	ImageInputs      bool
	ImageURLs        bool
	AudioInputs      bool
	PDFInputs        bool
	VideoInputs      bool
	UsageMetadata    bool
	Streaming        bool
}

// ModelProfile returns a model profile-shaped view of the capabilities.
func (c ChatModelCapabilities) ModelProfile() modelprofiles.Profile {
	return modelprofiles.Profile{
		"text_inputs":         true,
		"text_outputs":        true,
		"image_inputs":        c.ImageInputs,
		"image_url_inputs":    c.ImageURLs,
		"audio_inputs":        c.AudioInputs,
		"pdf_inputs":          c.PDFInputs,
		"video_inputs":        c.VideoInputs,
		"tool_calling":        c.ToolCalling,
		"tool_choice":         c.ToolChoice,
		"structured_output":   c.StructuredOutput,
		"tool_call_streaming": c.Streaming && c.ToolCalling,
	}
}

// ChatModel is the common interface for conversational models.
type ChatModel interface {
	runnables.Runnable[[]messages.Message, messages.Message]
	BindTools(boundTools []tools.Tool) (ChatModel, error)
	Capabilities() ChatModelCapabilities
}

// FakeChatModel is a deterministic chat model for unit tests and standard
// conformance suites.
type FakeChatModel struct {
	mu           sync.Mutex
	responses    []messages.Message
	responseIdx  int
	streamChunks []messages.Message
	boundTools   []tools.Tool
	capabilities ChatModelCapabilities
	rateLimiter  ratelimiters.RateLimiter
	profile      modelprofiles.Profile
}

// FakeChatModelOption configures a FakeChatModel.
type FakeChatModelOption func(*FakeChatModel)

// NewFakeChatModel creates a deterministic chat model.
func NewFakeChatModel(opts ...FakeChatModelOption) *FakeChatModel {
	model := &FakeChatModel{
		capabilities: ChatModelCapabilities{
			Streaming:     true,
			UsageMetadata: true,
		},
	}
	for _, opt := range opts {
		opt(model)
	}
	return model
}

// WithResponses configures fixed responses returned by Invoke.
func WithResponses(responses ...messages.Message) FakeChatModelOption {
	return func(model *FakeChatModel) {
		model.responses = append([]messages.Message(nil), responses...)
	}
}

// WithStreamChunks configures chunks returned by Stream.
func WithStreamChunks(chunks ...messages.Message) FakeChatModelOption {
	return func(model *FakeChatModel) {
		model.streamChunks = append([]messages.Message(nil), chunks...)
	}
}

// WithCapabilities configures model capabilities.
func WithCapabilities(capabilities ChatModelCapabilities) FakeChatModelOption {
	return func(model *FakeChatModel) {
		model.capabilities = capabilities
	}
}

// WithRateLimiter configures a limiter acquired before invoke/stream calls.
func WithRateLimiter(limiter ratelimiters.RateLimiter) FakeChatModelOption {
	return func(model *FakeChatModel) {
		model.rateLimiter = limiter
	}
}

// WithModelProfile configures explicit model profile metadata.
func WithModelProfile(profile modelprofiles.Profile) FakeChatModelOption {
	return func(model *FakeChatModel) {
		model.profile = cloneProfile(profile)
	}
}

// Invoke returns the next configured response or an echo response.
func (m *FakeChatModel) Invoke(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (messages.Message, error) {
	if err := m.acquireRateLimit(ctx); err != nil {
		return messages.Message{}, err
	}
	cfg := runnables.NewConfig(opts...)
	if err := emitChatEvent(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return messages.Message{}, err
	}

	m.mu.Lock()

	if len(m.responses) > 0 {
		response := m.responses[m.responseIdx%len(m.responses)]
		m.responseIdx++
		m.mu.Unlock()
		if err := emitChatEvent(ctx, cfg, callbacks.EventChatModelEnd, nil, response, nil); err != nil {
			return messages.Message{}, err
		}
		return response, nil
	}

	content := "fake response"
	if len(input) > 0 {
		content = fmt.Sprintf("fake response: %s", input[len(input)-1].Content)
	}
	response := messages.AI(content)
	response.UsageMetadata = messages.UsageMetadata{
		InputTokens:  len(input),
		OutputTokens: 1,
		TotalTokens:  len(input) + 1,
	}
	m.mu.Unlock()

	if err := emitChatEvent(ctx, cfg, callbacks.EventChatModelEnd, nil, response, nil); err != nil {
		return messages.Message{}, err
	}
	return response, nil
}

// Batch invokes the model for all inputs while preserving order.
func (m *FakeChatModel) Batch(
	ctx context.Context,
	inputs [][]messages.Message,
	opts ...runnables.Option,
) ([]messages.Message, error) {
	outputs := make([]messages.Message, len(inputs))
	errs := make([]error, len(inputs))

	var wg sync.WaitGroup
	for i, input := range inputs {
		wg.Add(1)
		go func(i int, input []messages.Message) {
			defer wg.Done()
			outputs[i], errs[i] = m.Invoke(ctx, input, opts...)
		}(i, input)
	}
	wg.Wait()

	return outputs, errors.Join(errs...)
}

// Stream returns configured chunks or a single Invoke response.
func (m *FakeChatModel) Stream(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	m.mu.Lock()
	chunks := append([]messages.Message(nil), m.streamChunks...)
	m.mu.Unlock()

	if len(chunks) > 0 {
		if err := m.acquireRateLimit(ctx); err != nil {
			return nil, err
		}
		cfg := runnables.NewConfig(opts...)
		if err := emitChatEvent(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
			return nil, err
		}
		return newCallbackStream(ctx, cfg, runnables.NewSliceStream(chunks)), nil
	}

	response, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{response}), nil
}

// InputSchema returns the chat model input schema.
func (m *FakeChatModel) InputSchema() schema.Schema {
	return schema.Schema{
		"type":        "array",
		"description": "chat messages",
	}
}

// OutputSchema returns the chat model output schema.
func (m *FakeChatModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"role":    schema.String("message role"),
		"content": schema.String("message content"),
	})
}

// BindTools returns a copy of the model with the provided tools bound.
func (m *FakeChatModel) BindTools(boundTools []tools.Tool) (ChatModel, error) {
	if len(boundTools) > 0 && !m.capabilities.ToolCalling {
		return nil, fmt.Errorf("tool calling is not supported")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Construct the copy field-by-field instead of `*m` so no sync.Mutex value
	// is copied (go vet copylocks). Each copy owns a fresh zero-value mutex.
	next := &FakeChatModel{
		responses:    append([]messages.Message(nil), m.responses...),
		responseIdx:  m.responseIdx,
		streamChunks: append([]messages.Message(nil), m.streamChunks...),
		boundTools:   append([]tools.Tool(nil), boundTools...),
		capabilities: m.capabilities,
		rateLimiter:  m.rateLimiter,
		profile:      cloneProfile(m.profile),
	}
	return next, nil
}

// Capabilities returns the fake model capability declaration.
func (m *FakeChatModel) Capabilities() ChatModelCapabilities {
	return m.capabilities
}

// BoundTools returns the currently bound tools.
func (m *FakeChatModel) BoundTools() []tools.Tool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]tools.Tool(nil), m.boundTools...)
}

// ModelProfile returns explicit profile metadata or a profile derived from
// capabilities.
func (m *FakeChatModel) ModelProfile() modelprofiles.Profile {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.profile) > 0 {
		return cloneProfile(m.profile)
	}
	return m.capabilities.ModelProfile()
}

func (m *FakeChatModel) acquireRateLimit(ctx context.Context) error {
	m.mu.Lock()
	limiter := m.rateLimiter
	m.mu.Unlock()
	if limiter == nil {
		return nil
	}
	_, err := limiter.Acquire(ctx, true)
	return err
}

func emitChatEvent(
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

type callbackStream struct {
	ctx    context.Context
	cfg    runnables.Config
	stream runnables.Stream[messages.Message]
	ended  bool
}

func newCallbackStream(
	ctx context.Context,
	cfg runnables.Config,
	stream runnables.Stream[messages.Message],
) *callbackStream {
	return &callbackStream{
		ctx:    ctx,
		cfg:    cfg,
		stream: stream,
	}
}

func (s *callbackStream) Next(ctx context.Context) (messages.Message, bool, error) {
	chunk, ok, err := s.stream.Next(ctx)
	if err != nil {
		_ = emitChatEvent(ctx, s.cfg, callbacks.EventChatModelError, nil, nil, err)
		return messages.Message{}, false, err
	}
	if !ok {
		if !s.ended {
			s.ended = true
			if eventErr := emitChatEvent(ctx, s.cfg, callbacks.EventChatModelEnd, nil, nil, nil); eventErr != nil {
				return messages.Message{}, false, eventErr
			}
		}
		return messages.Message{}, false, nil
	}
	if eventErr := emitStreamEvent(ctx, s.cfg, chunk); eventErr != nil {
		return messages.Message{}, false, eventErr
	}
	return chunk, true, nil
}

func (s *callbackStream) Close() error {
	return s.stream.Close()
}

func emitStreamEvent(ctx context.Context, cfg runnables.Config, chunk messages.Message) error {
	if cfg.Callbacks.Empty() {
		return nil
	}
	return cfg.Callbacks.Emit(ctx, callbacks.Event{
		Kind:     callbacks.EventChatModelStream,
		Name:     cfg.Name,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     append([]string(nil), cfg.Tags...),
		Metadata: cloneMetadata(cfg.Metadata),
		Chunk:    chunk,
	})
}

func cloneMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return nil
	}
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		out[key] = value
	}
	return out
}

func cloneProfile(profile modelprofiles.Profile) modelprofiles.Profile {
	if profile == nil {
		return nil
	}
	out := make(modelprofiles.Profile, len(profile))
	for key, value := range profile {
		out[key] = value
	}
	return out
}
