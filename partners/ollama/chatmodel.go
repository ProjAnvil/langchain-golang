package ollama

import (
	"context"
	"encoding/json"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// ChatModel adapts LangChain chat calls to the Ollama /api/chat endpoint.
type ChatModel struct {
	config     modelconfig.Config
	boundTools []tools.Tool
	format     any
	reasoning  any
	keepAlive  any
	options    map[string]any
}

// Compile-time guard: ChatModel (value receiver) satisfies
// language.StructuredCaller so the agent's ProviderStrategy native path can
// use it.
var _ language.StructuredCaller = ChatModel{}

// NewChatModel creates an Ollama chat model adapter.
func NewChatModel(opts ...modelconfig.Option) ChatModel {
	cfg := modelconfig.New(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "llama3"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return ChatModel{
		config:    cfg,
		format:    cfg.Extra[formatKey],
		reasoning: cfg.Extra[reasoningKey],
		keepAlive: cfg.Extra[keepAliveKey],
		options:   readSamplingOptions(cfg.Extra),
	}
}

// readSamplingOptions extracts Ollama sampling parameters from the extra config
// into the options object sent with each request.
func readSamplingOptions(extra map[string]any) map[string]any {
	options := map[string]any{}
	for extraKey, optionKey := range samplingOptionKeys {
		if value, ok := extra[extraKey]; ok {
			options[optionKey] = value
		}
	}
	if len(options) == 0 {
		return nil
	}
	return options
}

// Invoke calls the Ollama chat endpoint (non-streaming).
func (m ChatModel) Invoke(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (messages.Message, error) {
	cfg := runnables.NewConfig(opts...)
	if err := emit(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return messages.Message{}, err
	}

	response, err := m.createChat(ctx, input)
	if err != nil {
		_ = emit(ctx, cfg, callbacks.EventChatModelError, nil, nil, err)
		return messages.Message{}, err
	}

	message := response.toMessage()
	if err := emit(ctx, cfg, callbacks.EventChatModelEnd, nil, message, nil); err != nil {
		return messages.Message{}, err
	}
	return message, nil
}

// Batch invokes the model for each input while preserving order.
func (m ChatModel) Batch(
	ctx context.Context,
	inputs [][]messages.Message,
	opts ...runnables.Option,
) ([]messages.Message, error) {
	runnable := runnables.NewFunc(m.Invoke, m.InputSchema(), m.OutputSchema())
	return runnable.Batch(ctx, inputs, opts...)
}

// Stream calls the Ollama chat endpoint with streaming enabled. Ollama streams
// newline-delimited JSON; the stream projects each chunk into v3 content-block
// protocol callback events alongside LangChain message chunks.
func (m ChatModel) Stream(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	cfg := runnables.NewConfig(opts...)
	if err := emit(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return nil, err
	}
	stream, err := m.createChatStream(ctx, input, cfg)
	if err != nil {
		_ = emit(ctx, cfg, callbacks.EventChatModelError, nil, nil, err)
		return nil, err
	}
	return stream, nil
}

// InputSchema returns the chat input schema.
func (m ChatModel) InputSchema() schema.Schema {
	return schema.Schema{
		"type":        "array",
		"description": "chat messages",
	}
}

// OutputSchema returns the chat output schema.
func (m ChatModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"role":    schema.String("message role"),
		"content": schema.String("message content"),
	})
}

// BindTools returns a copy of the model with function tools bound.
func (m ChatModel) BindTools(boundTools []tools.Tool) (language.ChatModel, error) {
	next := m
	next.boundTools = append([]tools.Tool(nil), boundTools...)
	return next, nil
}

// WithStructuredOutput returns a copy of the model configured for Ollama
// structured output. Ollama accepts the raw JSON schema directly in the
// request format field; the name and strict flag are not used by the API.
func (m ChatModel) WithStructuredOutput(
	name string,
	outputSchema schema.Schema,
	strict bool,
) ChatModel {
	next := m
	next.format = outputSchema
	return next
}

// InvokeStructured implements language.StructuredCaller via Ollama's native
// json_schema format — Python's DEFAULT for with_structured_output
// (langchain_ollama/chat_models.py:1448,1751 `format=response_format`). It
// configures the model for structured output (WithStructuredOutput sets the
// request `format` field to the JSON schema; name/strict are unused by the
// Ollama API) and delegates to Invoke. The model returns JSON text conforming
// to sch, matching language.InvokeStructured's "returned text is JSON" contract.
func (m ChatModel) InvokeStructured(
	ctx context.Context,
	input []messages.Message,
	sch schema.Schema,
) (messages.Message, error) {
	name := "response_format"
	if title, ok := sch["title"].(string); ok && title != "" {
		name = title
	}
	return m.WithStructuredOutput(name, sch, true).Invoke(ctx, input)
}

// Capabilities returns the adapter capability declaration.
func (m ChatModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{
		ToolCalling:      true,
		StructuredOutput: true,
		JSONMode:         true,
		ImageInputs:      true,
		UsageMetadata:    true,
		Streaming:        true,
	}
}

// LLMType reports the model's Python "_llm_type" identifier ("ollama-chat"),
// mirroring Python's BaseChatModel._llm_type attribute. Used by middleware
// (e.g. SummarizationMiddleware) for provider-aware behavior.
func (m ChatModel) LLMType() string { return "ollama-chat" }

func (m ChatModel) createChat(
	ctx context.Context,
	input []messages.Message,
) (chatResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	defer cancel()
	return postJSON[chatResponse](ctx, m.config, "/api/chat", m.buildRequest(input, false))
}

func (m ChatModel) buildRequest(input []messages.Message, stream bool) chatRequest {
	payload := chatRequest{
		Model:     m.config.Model,
		Messages:  make([]chatMessage, 0, len(input)),
		Stream:    stream,
		Tools:     make([]chatTool, 0, len(m.boundTools)),
		Think:     m.reasoning,
		Format:    m.format,
		KeepAlive: m.keepAlive,
	}
	payload.Options = m.buildOptions()

	for _, message := range input {
		payload.Messages = append(payload.Messages, toOllamaMessage(message))
	}
	for _, tool := range m.boundTools {
		payload.Tools = append(payload.Tools, chatTool{
			Type: "function",
			Function: chatToolFunction{
				Name:        tool.Name(),
				Description: tool.Description(),
				Parameters:  tool.ArgsSchema(),
			},
		})
	}
	if len(payload.Tools) == 0 {
		payload.Tools = nil
	}
	return payload
}

func (m ChatModel) buildOptions() map[string]any {
	options := map[string]any{}
	for key, value := range m.options {
		options[key] = value
	}
	if m.config.Temperature != nil {
		options["temperature"] = *m.config.Temperature
	}
	if m.config.MaxTokens != nil {
		options["num_predict"] = *m.config.MaxTokens
	}
	if len(options) == 0 {
		return nil
	}
	return options
}

type chatRequest struct {
	Model     string         `json:"model"`
	Messages  []chatMessage  `json:"messages"`
	Tools     []chatTool     `json:"tools,omitempty"`
	Stream    bool           `json:"stream"`
	Think     any            `json:"think,omitempty"`
	Format    any            `json:"format,omitempty"`
	KeepAlive any            `json:"keep_alive,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
}

type chatMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	Images     []string         `json:"images,omitempty"`
	ToolCalls  []chatMessageTool `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Thinking   string           `json:"thinking,omitempty"`
}

type chatMessageTool struct {
	Type     string                `json:"type"`
	ID       string                `json:"id,omitempty"`
	Function chatMessageToolFn      `json:"function"`
}

type chatMessageToolFn struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
}

type chatTool struct {
	Type     string           `json:"type"`
	Function chatToolFunction `json:"function"`
}

type chatToolFunction struct {
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Parameters  schema.Schema `json:"parameters,omitempty"`
}

type chatResponse struct {
	Model           string         `json:"model"`
	CreatedAt       string         `json:"created_at"`
	Message         chatResponseMessage `json:"message"`
	Done            bool           `json:"done"`
	DoneReason      string         `json:"done_reason"`
	PromptEvalCount int            `json:"prompt_eval_count"`
	EvalCount       int            `json:"eval_count"`
	TotalDuration   int64          `json:"total_duration"`
	LoadDuration    int64          `json:"load_duration"`
	Raw             map[string]any `json:"-"`
}

func (r *chatResponse) UnmarshalJSON(data []byte) error {
	type alias chatResponse
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*r = chatResponse(decoded)
	r.Raw = raw
	return nil
}

type chatResponseMessage struct {
	Role      string             `json:"role"`
	Content   string             `json:"content"`
	ToolCalls []chatResponseTool `json:"tool_calls"`
	Thinking  string             `json:"thinking"`
}

type chatResponseTool struct {
	Function chatResponseToolFn `json:"function"`
}

type chatResponseToolFn struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
}

func (r chatResponse) toMessage() messages.Message {
	text := r.Message.Content
	toolCalls, invalidToolCalls := parseToolCalls(r.Message.ToolCalls)

	message := messages.AI(text)
	message.ResponseMetadata = map[string]any{
		"model":         r.Model,
		"model_name":    r.Model,
		"created_at":    r.CreatedAt,
		"done_reason":   r.DoneReason,
		"model_provider": "ollama",
	}
	message.ToolCalls = toolCalls
	message.InvalidToolCalls = invalidToolCalls
	if r.Message.Thinking != "" {
		message.AdditionalKwargs = map[string]any{
			"reasoning_content": r.Message.Thinking,
		}
	}
	if r.PromptEvalCount != 0 || r.EvalCount != 0 {
		message.UsageMetadata = messages.UsageMetadata{
			InputTokens:  r.PromptEvalCount,
			OutputTokens: r.EvalCount,
			TotalTokens:  r.PromptEvalCount + r.EvalCount,
		}
	}
	return message
}

func emit(
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
