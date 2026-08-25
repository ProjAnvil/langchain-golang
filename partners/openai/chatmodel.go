package openai

import (
	"context"
	"encoding/json"
	"fmt"
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

// ChatModel adapts LangChain chat calls to OpenAI's APIs. By default it uses
// the Responses API; WithChatCompletions switches it to the Chat Completions
// API (the classic `/chat/completions` endpoint, Python's default).
type ChatModel struct {
	config           modelconfig.Config
	boundTools       []tools.Tool
	structuredOutput *structuredoutput.JSONSchema
	chatCompletions  bool
	reasoningEffort  string
	toolChoice       *ToolChoice
	responseFormat   map[string]any
}

// Compile-time guard: ChatModel (value receiver) satisfies
// language.StructuredCaller so the agent's ProviderStrategy native path
// (agents.invokeModel → language.InvokeStructured) can use it. A future
// refactor that drops InvokeStructured fails here.
var _ language.StructuredCaller = ChatModel{}

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
	var stream runnables.Stream[messages.Message]
	var err error
	if m.chatCompletions {
		stream, err = m.createChatCompletionsStream(ctx, input, cfg)
	} else {
		stream, err = m.createResponseStream(ctx, input, cfg)
	}
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

// WithChatCompletions returns a copy of the model that targets the Chat
// Completions API (`/chat/completions`) instead of the Responses API.
func (m ChatModel) WithChatCompletions() ChatModel {
	next := m
	next.chatCompletions = true
	return next
}

// WithReasoningEffort returns a copy of the model that sends reasoning_effort
// (low|medium|high) on Chat Completions requests, steering how hard reasoning
// models (OpenAI o-series / gpt-5 family) think before answering. Non-reasoning
// models and gateways that don't know the field ignore it.
func (m ChatModel) WithReasoningEffort(effort string) ChatModel {
	next := m
	next.reasoningEffort = effort
	return next
}

// WithToolChoice returns a copy of the model that sends tool_choice on both
// the Responses and Chat Completions APIs, mirroring Python
// bind_tools(tool_choice=...).
func (m ChatModel) WithToolChoice(choice ToolChoice) ChatModel {
	next := m
	next.toolChoice = &choice
	return next
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

// InvokeStructured implements language.StructuredCaller, producing a message
// whose text is JSON conforming to sch via OpenAI's native json_schema
// response_format. Used by the agent's ProviderStrategy native path
// (agents.invokeModel → language.InvokeStructured). It configures the model
// for structured output (deriving a name from sch's "title" if present, else
// "response_format"; strict=true) and delegates to Invoke.
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
		ToolChoice:       true,
		StructuredOutput: true,
		JSONMode:         true,
		ImageInputs:      true,
		ImageURLs:        true,
		UsageMetadata:    true,
	}
}

// LLMType reports the model's Python "_llm_type" identifier, used by
// middleware (e.g. SummarizationMiddleware) to tune provider-specific
// behavior. Mirrors Python's `BaseChatModel._llm_type` attribute.
func (m ChatModel) LLMType() string { return "openai-chat" }

func (m ChatModel) createResponse(
	ctx context.Context,
	input []messages.Message,
) (responsePayload, error) {
	if m.chatCompletions {
		return m.createChatCompletionsResponse(ctx, input)
	}
	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	defer cancel()
	resp, err := postJSON[responsePayload](ctx, m.config, "/responses", m.buildRequest(input))
	if err != nil {
		return responsePayload{}, err
	}
	// Surface misconfigured endpoints / wrong model names loudly. Gateways that
	// wrap errors in HTTP 200 bodies (e.g. {"code":500,...} or {"error":{...}})
	// decode into an all-zero responsePayload.
	if len(resp.Output) == 0 && resp.Model == "" && resp.Usage == (usagePayload{}) {
		return responsePayload{}, fmt.Errorf(
			"openai %s: response parsed but empty — likely wrong BASE_URL or unsupported model (ensure BASE_URL ends with /v1 and the model ID is valid for this endpoint)",
			m.config.Model)
	}
	return resp, nil
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
	} else if m.responseFormat != nil {
		payload.Text = &textConfig{Format: toResponseFormat(m.responseFormat)}
	}
	if m.toolChoice != nil {
		payload.ToolChoice = responsesToolChoice(m.toolChoice.value)
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
			// A blocks-only AI message (e.g. a bare custom_tool_call) emits no
			// empty assistant text item, mirroring Python's block pass-through.
			if message.Content != "" {
				payload.Input = append(payload.Input, inputItem{
					Role:    "assistant",
					Content: message.Content,
				})
			}
			// Replay custom_tool_call blocks as Responses input items
			// (chat_models/base.py:4661-4677).
			for _, block := range message.ContentBlocks {
				if ns, ok := block.(messages.NonStandardContentBlock); ok && ns.Type == "custom_tool_call" {
					payload.Input = append(payload.Input, inputItem{
						Type:   "custom_tool_call",
						ID:     stringFrom(ns.Value, "id"),
						CallID: stringFrom(ns.Value, "call_id"),
						Name:   stringFrom(ns.Value, "name"),
						Input:  stringFrom(ns.Value, "input"),
					})
				}
			}
		case messages.RoleTool:
			// A custom_tool_call_output block replaces the plain tool message
			// (chat_models/base.py:4505-4523, 4597-4601).
			replayed := false
			for _, block := range message.ContentBlocks {
				if ns, ok := block.(messages.NonStandardContentBlock); ok && ns.Type == "custom_tool_call_output" {
					payload.Input = append(payload.Input, inputItem{
						Type:   "custom_tool_call_output",
						CallID: message.ToolCallID,
						Output: stringFrom(ns.Value, "output"),
					})
					replayed = true
				}
			}
			if !replayed {
				payload.Input = append(payload.Input, inputItem{
					Role:    "tool",
					Content: message.Content,
				})
			}
		}
	}
	if len(instructions) > 0 {
		payload.Instructions = strings.Join(instructions, "\n")
	}
	for _, tool := range m.boundTools {
		if custom, ok := tool.(CustomTool); ok {
			spec := toolSpec{Type: "custom", Name: custom.name, Description: custom.description}
			if len(custom.format) > 0 {
				spec.Format = custom.format
			}
			payload.Tools = append(payload.Tools, spec)
			continue
		}
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
	ToolChoice      any         `json:"tool_choice,omitempty"`
	Text            *textConfig `json:"text,omitempty"`
	Stream          bool        `json:"stream,omitempty"`
}

type inputItem struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	Type    string `json:"type,omitempty"`
	ID      string `json:"id,omitempty"`
	CallID  string `json:"call_id,omitempty"`
	Name    string `json:"name,omitempty"`
	Input   string `json:"input,omitempty"`
	Output  string `json:"output,omitempty"`
}

type toolSpec struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  schema.Schema  `json:"parameters,omitempty"`
	Format      map[string]any `json:"format,omitempty"`
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
	InputTokens         int                  `json:"input_tokens"`
	OutputTokens        int                  `json:"output_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	InputTokensDetails  *inputTokensDetails  `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *outputTokensDetails `json:"output_tokens_details,omitempty"`
}

type inputTokensDetails struct {
	CachedTokens        int `json:"cached_tokens"`
	CacheCreationTokens int `json:"cache_creation_tokens"`
}

type outputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

func (r responsePayload) toMessage() messages.Message {
	var parts []string
	var toolCalls []messages.ToolCall
	var invalidToolCalls []messages.ToolCall
	var contentBlocks []messages.ContentBlock
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
		case "custom_tool_call":
			// chat_models/base.py:4883-4891: keep the raw item as a content
			// block and append a tool call with args {"__arg1": input}.
			contentBlocks = append(contentBlocks, messages.NonStandardContentBlock{
				Type:  "custom_tool_call",
				Value: output.Raw,
			})
			toolCalls = append(toolCalls, messages.ToolCall{
				ID:   output.CallID,
				Name: output.Name,
				Args: map[string]any{"__arg1": output.Input},
			})
		}
	}
	message := messages.AI(strings.Join(parts, ""))
	message.ID = r.ID
	message.ToolCalls = toolCalls
	message.InvalidToolCalls = invalidToolCalls
	message.ContentBlocks = contentBlocks
	message.ResponseMetadata = map[string]any{
		"model":          r.Model,
		"model_provider": "openai",
	}
	message.UsageMetadata = messages.UsageMetadata{
		InputTokens:  r.Usage.InputTokens,
		OutputTokens: r.Usage.OutputTokens,
		TotalTokens:  r.Usage.TotalTokens,
	}
	if details := r.Usage.InputTokensDetails; details != nil {
		message.UsageMetadata.InputTokenDetails = &messages.InputTokenDetails{
			CacheReadInputTokens:     details.CachedTokens,
			CacheCreationInputTokens: details.CacheCreationTokens,
		}
	}
	if details := r.Usage.OutputTokensDetails; details != nil && details.ReasoningTokens > 0 {
		message.UsageMetadata.OutputTokenDetails = &messages.OutputTokenDetails{
			ReasoningOutputTokens: details.ReasoningTokens,
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

// stringFrom extracts m[key] as a string ("" when absent or not a string).
func stringFrom(m map[string]any, key string) string {
	s, _ := m[key].(string)
	return s
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
