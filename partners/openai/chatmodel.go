package openai

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/structuredoutput"
	"github.com/projanvil/langchain-golang/core/tools"
)

const defaultBaseURL = "https://api.openai.com/v1"

// ChatModel adapts LangChain chat calls to OpenAI's Responses API.
type ChatModel struct {
	config           modelconfig.Config
	boundTools       []tools.Tool
	structuredOutput *structuredoutput.JSONSchema
}

// NewChatModel creates an OpenAI chat model adapter.
func NewChatModel(opts ...modelconfig.Option) ChatModel {
	cfg := modelconfig.New(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "gpt-4.1"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return ChatModel{config: cfg}
}

// Invoke calls the OpenAI Responses API.
func (m ChatModel) Invoke(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (messages.Message, error) {
	cfg := runnables.NewConfig(opts...)
	if err := emit(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return messages.Message{}, err
	}

	response, err := m.createResponse(ctx, input)
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

// Stream calls the OpenAI Responses API with stream enabled.
func (m ChatModel) Stream(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	cfg := runnables.NewConfig(opts...)
	if err := emit(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return nil, err
	}
	stream, err := m.createResponseStream(ctx, input, cfg)
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

// WithStructuredOutput returns a copy of the model configured for provider-native
// JSON-schema output.
func (m ChatModel) WithStructuredOutput(
	name string,
	outputSchema schema.Schema,
	strict bool,
) ChatModel {
	next := m
	cfg := structuredoutput.NewJSONSchema(name, outputSchema, strict)
	next.structuredOutput = &cfg
	return next
}

// Capabilities returns the adapter capability declaration.
func (m ChatModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{
		ToolCalling:      true,
		ToolChoice:       true,
		StructuredOutput: true,
		JSONMode:         true,
		ImageInputs:      true,
		ImageURLs:        true,
		UsageMetadata:    true,
	}
}

func (m ChatModel) createResponse(
	ctx context.Context,
	input []messages.Message,
) (responsePayload, error) {
	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	defer cancel()
	return postJSON[responsePayload](ctx, m.config, "/responses", m.buildRequest(input))
}

func (m ChatModel) buildRequest(input []messages.Message) requestPayload {
	payload := requestPayload{
		Model: m.config.Model,
		Input: make([]inputItem, 0, len(input)),
		Tools: make([]toolSpec, 0, len(m.boundTools)),
	}
	if m.config.Temperature != nil {
		payload.Temperature = m.config.Temperature
	}
	if m.config.MaxTokens != nil {
		payload.MaxOutputTokens = m.config.MaxTokens
	}
	if m.structuredOutput != nil {
		payload.Text = &textConfig{
			Format: responseFormat{
				Type:   "json_schema",
				Name:   m.structuredOutput.Name,
				Schema: m.structuredOutput.Schema,
				Strict: m.structuredOutput.Strict,
			},
		}
	}

	var instructions []string
	for _, message := range input {
		switch message.Role {
		case messages.RoleSystem:
			if message.Content != "" {
				instructions = append(instructions, message.Content)
			}
		case messages.RoleHuman:
			payload.Input = append(payload.Input, inputItem{
				Role:    "user",
				Content: message.Content,
			})
		case messages.RoleAI:
			payload.Input = append(payload.Input, inputItem{
				Role:    "assistant",
				Content: message.Content,
			})
		case messages.RoleTool:
			payload.Input = append(payload.Input, inputItem{
				Role:    "tool",
				Content: message.Content,
			})
		}
	}
	if len(instructions) > 0 {
		payload.Instructions = strings.Join(instructions, "\n")
	}
	for _, tool := range m.boundTools {
		payload.Tools = append(payload.Tools, toolSpec{
			Type:        "function",
			Name:        tool.Name(),
			Description: tool.Description(),
			Parameters:  tool.ArgsSchema(),
		})
	}
	if len(payload.Tools) == 0 {
		payload.Tools = nil
	}
	return payload
}

type requestPayload struct {
	Model           string      `json:"model"`
	Input           []inputItem `json:"input"`
	Instructions    string      `json:"instructions,omitempty"`
	Temperature     *float64    `json:"temperature,omitempty"`
	MaxOutputTokens *int        `json:"max_output_tokens,omitempty"`
	Tools           []toolSpec  `json:"tools,omitempty"`
	Text            *textConfig `json:"text,omitempty"`
	Stream          bool        `json:"stream,omitempty"`
}

type inputItem struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type toolSpec struct {
	Type        string        `json:"type"`
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Parameters  schema.Schema `json:"parameters,omitempty"`
}

type textConfig struct {
	Format responseFormat `json:"format"`
}

type responseFormat struct {
	Type   string        `json:"type"`
	Name   string        `json:"name,omitempty"`
	Schema schema.Schema `json:"schema,omitempty"`
	Strict bool          `json:"strict,omitempty"`
}

type responsePayload struct {
	ID     string       `json:"id"`
	Model  string       `json:"model"`
	Output []outputItem `json:"output"`
	Usage  usagePayload `json:"usage"`
}

type outputItem struct {
	ID        string          `json:"id"`
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	Content   []contentOutput `json:"content"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Input     string          `json:"input"`
	Raw       map[string]any  `json:"-"`
}

func (o *outputItem) UnmarshalJSON(data []byte) error {
	type alias outputItem
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*o = outputItem(decoded)
	o.Raw = raw
	return nil
}

type contentOutput struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type usagePayload struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func (r responsePayload) toMessage() messages.Message {
	var parts []string
	var toolCalls []messages.ToolCall
	var invalidToolCalls []messages.ToolCall
	for _, output := range r.Output {
		switch output.Type {
		case "message":
			if output.Role != "assistant" {
				continue
			}
			for _, content := range output.Content {
				if content.Type == "output_text" && content.Text != "" {
					parts = append(parts, content.Text)
				}
			}
		case "function_call":
			toolCall := messages.ToolCall{
				ID:   output.CallID,
				Name: output.Name,
			}
			if output.Arguments != "" {
				var args map[string]any
				if err := json.Unmarshal([]byte(output.Arguments), &args); err != nil {
					invalidToolCalls = append(invalidToolCalls, toolCall)
					continue
				}
				toolCall.Args = args
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}
	message := messages.AI(strings.Join(parts, ""))
	message.ID = r.ID
	message.ToolCalls = toolCalls
	message.InvalidToolCalls = invalidToolCalls
	message.ResponseMetadata = map[string]any{
		"model": r.Model,
	}
	message.UsageMetadata = messages.UsageMetadata{
		InputTokens:  r.Usage.InputTokens,
		OutputTokens: r.Usage.OutputTokens,
		TotalTokens:  r.Usage.TotalTokens,
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
