package anthropic

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
	"github.com/projanvil/langchain-golang/core/tools"
)

const defaultBaseURL = "https://api.anthropic.com/v1"

// ChatModel adapts LangChain chat calls to Anthropic's Messages API.
type ChatModel struct {
	config            modelconfig.Config
	boundTools        []tools.Tool
	toolStrict        *bool
	thinking          map[string]any
	toolChoice        map[string]any
	contextManagement map[string]any
	inferenceGeo      string
}

// Compile-time guard: ChatModel (value receiver) satisfies
// language.StructuredCaller so the agent's ProviderStrategy native path
// (agents.invokeModel → language.InvokeStructured) can use it.
var _ language.StructuredCaller = ChatModel{}

// NewChatModel creates an Anthropic chat model adapter.
func NewChatModel(opts ...modelconfig.Option) ChatModel {
	cfg := modelconfig.New(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "claude-sonnet-4-5"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return ChatModel{config: cfg}
}

// Invoke calls the Anthropic Messages API.
func (m ChatModel) Invoke(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (messages.Message, error) {
	cfg := runnables.NewConfig(opts...)
	if err := emit(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return messages.Message{}, err
	}

	response, err := m.createMessage(ctx, input)
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

// Stream calls the Anthropic Messages API with stream enabled.
func (m ChatModel) Stream(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	cfg := runnables.NewConfig(opts...)
	if err := emit(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return nil, err
	}
	stream, err := m.createMessageStream(ctx, input, cfg)
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

// BindTools returns a copy of the model with Anthropic tools bound.
func (m ChatModel) BindTools(boundTools []tools.Tool) (language.ChatModel, error) {
	next := m
	next.boundTools = append([]tools.Tool(nil), boundTools...)
	next.toolStrict = nil
	return next, nil
}

// BindToolsStrict returns a copy of the model with Anthropic tools bound and
// the strict tool-use schema flag set on each tool definition.
func (m ChatModel) BindToolsStrict(boundTools []tools.Tool, strict bool) (ChatModel, error) {
	next := m
	next.boundTools = append([]tools.Tool(nil), boundTools...)
	next.toolStrict = boolPtr(strict)
	return next, nil
}

// WithThinking returns a copy of the model with Anthropic extended thinking
// enabled. The map is passed through as the request "thinking" field, e.g.
// {"type":"enabled","budget_tokens":1024} or {"type":"adaptive"}.
func (m ChatModel) WithThinking(thinking map[string]any) ChatModel {
	next := m
	next.thinking = cloneAnyMap(thinking)
	return next
}

// WithToolChoice returns a copy of the model with the Anthropic tool_choice
// request parameter set, e.g. {"type":"auto"}, {"type":"any"},
// {"type":"tool","name":"search"}, or
// {"type":"auto","disable_parallel_tool_use":true}.
func (m ChatModel) WithToolChoice(choice map[string]any) ChatModel {
	next := m
	next.toolChoice = cloneAnyMap(choice)
	return next
}

// WithContextManagement returns a copy of the model with Anthropic context
// management request configuration set, e.g.
// {"edits":[{"type":"clear_tool_uses_20250919"}]}.
func (m ChatModel) WithContextManagement(contextManagement map[string]any) ChatModel {
	next := m
	next.contextManagement = cloneAnyMap(contextManagement)
	return next
}

// WithInferenceGeo returns a copy of the model with Anthropic inference_geo
// request routing set, e.g. "us".
func (m ChatModel) WithInferenceGeo(inferenceGeo string) ChatModel {
	next := m
	next.inferenceGeo = inferenceGeo
	return next
}

// Capabilities returns the adapter capability declaration.
func (m ChatModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{
		ToolCalling:   true,
		ToolChoice:    true,
		ImageInputs:   true,
		ImageURLs:     true,
		UsageMetadata: true,
		Streaming:     true,
	}
}

// LLMType reports the model's Python "_llm_type" identifier
// ("anthropic-chat"), mirroring Python's BaseChatModel._llm_type attribute.
// Used by middleware (e.g. SummarizationMiddleware) to tune provider-specific
// behavior. Matches the prefix Python's _get_approximate_token_counter checks
// (summarization.py:210).
func (m ChatModel) LLMType() string { return "anthropic-chat" }

// structuredTool adapts a JSON schema into a tools.Tool for anthropic's
// function_calling structured-output path (InvokeStructured). Only the schema
// surface (Name/Description/ArgsSchema) is used to build the request tool
// definition; Invoke is never called because the model's tool_use response is
// extracted directly, never executed. Mirrors Python's
// `convert_to_anthropic_tool(schema)` (langchain_anthropic/chat_models.py:2035).
type structuredTool struct {
	name   string
	desc   string
	inputs schema.Schema
}

func (t structuredTool) Name() string              { return t.name }
func (t structuredTool) Description() string       { return t.desc }
func (t structuredTool) ArgsSchema() schema.Schema { return t.inputs }
func (t structuredTool) Invoke(context.Context, map[string]any) (tools.Result, error) {
	return tools.Result{}, fmt.Errorf("structured output tool %q is not invocable", t.name)
}

// InvokeStructured implements language.StructuredCaller via Anthropic's
// function_calling method — Python's DEFAULT for with_structured_output
// (langchain_anthropic/chat_models.py:1943,2050 `tool_choice=tool_name`).
// It synthesizes one tool from sch, forces tool_choice to that tool, invokes,
// and returns a message whose Content is the JSON of the tool_use input args
// (matching language.InvokeStructured's "returned text is JSON" contract).
func (m ChatModel) InvokeStructured(
	ctx context.Context,
	input []messages.Message,
	sch schema.Schema,
) (messages.Message, error) {
	name := "response_format"
	if title, ok := sch["title"].(string); ok && title != "" {
		name = title
	}
	desc, _ := sch["description"].(string)
	tool := structuredTool{name: name, desc: desc, inputs: sch}

	boundAny, err := m.BindTools([]tools.Tool{tool})
	if err != nil {
		return messages.Message{}, fmt.Errorf("anthropic structured output: bind tool: %w", err)
	}
	bound, ok := boundAny.(ChatModel)
	if !ok {
		return messages.Message{}, fmt.Errorf("anthropic structured output: bound model is %T, not ChatModel", boundAny)
	}
	forced := bound.WithToolChoice(map[string]any{"type": "tool", "name": name})

	response, err := forced.Invoke(ctx, input)
	if err != nil {
		return messages.Message{}, fmt.Errorf("anthropic structured output: invoke: %w", err)
	}
	if len(response.ToolCalls) == 0 {
		return messages.Message{}, fmt.Errorf(
			"anthropic structured output: model returned no tool_call (stop_reason=%v)",
			response.ResponseMetadata["stop_reason"])
	}

	encoded, err := json.Marshal(response.ToolCalls[0].Args)
	if err != nil {
		return messages.Message{}, fmt.Errorf("anthropic structured output: encode tool input: %w", err)
	}
	out := messages.AI(string(encoded))
	out.ID = response.ID
	out.ResponseMetadata = response.ResponseMetadata
	out.UsageMetadata = response.UsageMetadata
	return out, nil
}

func (m ChatModel) createMessage(
	ctx context.Context,
	input []messages.Message,
) (messagePayload, error) {
	req, err := m.buildRequest(input)
	if err != nil {
		return messagePayload{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	defer cancel()
	resp, err := postJSON[messagePayload](ctx, m.config, "/messages", req)
	if err != nil {
		return messagePayload{}, err
	}
	// Surface misconfigured endpoints / wrong model names loudly instead of
	// returning a silently-empty message. Some gateways wrap errors in HTTP 200
	// bodies (e.g. {"code":500,"msg":"404 NOT_FOUND","success":false}) or emit
	// a standard Anthropic error object ({"type":"error",...}) with status 200;
	// in both cases the JSON decodes into an all-zero messagePayload.
	if resp.Type == "error" {
		return messagePayload{}, fmt.Errorf(
			"anthropic %s: provider returned an error response (type=%q); check BASE_URL, API key, and model",
			m.config.Model, resp.Type)
	}
	if resp.Type != "message" && len(resp.Content) == 0 && resp.StopReason == "" && resp.Model == "" {
		return messagePayload{}, fmt.Errorf(
			"anthropic %s: response parsed but empty — likely wrong BASE_URL or unsupported model (ensure BASE_URL ends with /v1 and the model ID is valid for this endpoint)",
			m.config.Model)
	}
	return resp, nil
}

func (m ChatModel) buildRequest(input []messages.Message) (requestPayload, error) {
	payload := requestPayload{
		Model:     m.config.Model,
		MaxTokens: 4096,
		Messages:  make([]requestMessage, 0, len(input)),
		Tools:     make([]toolSpec, 0, len(m.boundTools)),
	}
	if m.config.MaxTokens != nil {
		payload.MaxTokens = *m.config.MaxTokens
	}
	if m.config.Temperature != nil {
		payload.Temperature = m.config.Temperature
	}
	if topP, ok := m.config.Extra[topPKey].(float64); ok {
		payload.TopP = &topP
	}
	if topK, ok := m.config.Extra[topKKey].(int); ok {
		payload.TopK = &topK
	}

	var systemText []string
	var systemBlocks []contentBlock
	for _, message := range input {
		switch message.Role {
		case messages.RoleSystem:
			if len(message.ContentBlocks) > 0 {
				blocks, err := formatContentBlocks(message.ContentBlocks)
				if err != nil {
					return requestPayload{}, err
				}
				systemBlocks = append(systemBlocks, blocks...)
			} else if message.Content != "" {
				systemText = append(systemText, message.Content)
			}
		case messages.RoleHuman:
			content, err := formatHumanContent(message)
			if err != nil {
				return requestPayload{}, err
			}
			payload.Messages = append(payload.Messages, requestMessage{
				Role:    "user",
				Content: content,
			})
		case messages.RoleAI:
			content, err := messageToAnthropicContent(message)
			if err != nil {
				return requestPayload{}, err
			}
			payload.Messages = append(payload.Messages, requestMessage{
				Role:    "assistant",
				Content: content,
			})
		case messages.RoleTool:
			content, err := toolResultContent(message)
			if err != nil {
				return requestPayload{}, err
			}
			payload.Messages = append(payload.Messages, requestMessage{
				Role:    "user",
				Content: content,
			})
		}
	}
	if len(systemBlocks) > 0 {
		if len(systemText) > 0 {
			systemBlocks = append([]contentBlock{{Type: "text", Text: strings.Join(systemText, "\n")}}, systemBlocks...)
		}
		payload.System = systemBlocks
	} else if len(systemText) > 0 {
		payload.System = strings.Join(systemText, "\n")
	}
	for _, tool := range m.boundTools {
		payload.Tools = append(payload.Tools, toolSpec{
			Name:        tool.Name(),
			Description: tool.Description(),
			InputSchema: tool.ArgsSchema(),
			Strict:      cloneBoolPtr(m.toolStrict),
		})
	}
	if len(payload.Tools) == 0 {
		payload.Tools = nil
	}
	if m.thinking != nil {
		payload.Thinking = m.thinking
		applyThinkingConstraints(&payload)
	}
	if m.toolChoice != nil {
		payload.ToolChoice = m.toolChoice
	}
	if m.contextManagement != nil {
		payload.ContextManagement = m.contextManagement
	}
	if m.inferenceGeo != "" {
		payload.InferenceGeo = m.inferenceGeo
	}
	return payload, nil
}

type requestPayload struct {
	Model             string           `json:"model"`
	MaxTokens         int              `json:"max_tokens"`
	Messages          []requestMessage `json:"messages"`
	System            any              `json:"system,omitempty"`
	Temperature       *float64         `json:"temperature,omitempty"`
	TopP              *float64         `json:"top_p,omitempty"`
	TopK              *int             `json:"top_k,omitempty"`
	Tools             []toolSpec       `json:"tools,omitempty"`
	ToolChoice        map[string]any   `json:"tool_choice,omitempty"`
	Stream            bool             `json:"stream,omitempty"`
	Thinking          map[string]any   `json:"thinking,omitempty"`
	ContextManagement map[string]any   `json:"context_management,omitempty"`
	InferenceGeo      string           `json:"inference_geo,omitempty"`
}

type requestMessage struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type         string         `json:"type"`
	Text         string         `json:"text,omitempty"`
	ID           string         `json:"id,omitempty"`
	Name         string         `json:"name,omitempty"`
	Input        map[string]any `json:"input,omitempty"`
	ToolUseID    string         `json:"tool_use_id,omitempty"`
	Content      any            `json:"content,omitempty"`
	IsError      bool           `json:"is_error,omitempty"`
	Source       map[string]any `json:"source,omitempty"`
	CacheControl map[string]any `json:"cache_control,omitempty"`
	Thinking     string         `json:"thinking,omitempty"`
	Signature    string         `json:"signature,omitempty"`
	Data         string         `json:"data,omitempty"`
}

type toolSpec struct {
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	InputSchema schema.Schema `json:"input_schema,omitempty"`
	Strict      *bool         `json:"strict,omitempty"`
}

type messagePayload struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Role       string         `json:"role"`
	Model      string         `json:"model"`
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      usagePayload   `json:"usage"`
}

type usagePayload struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

func (u usagePayload) toUsageMetadata() messages.UsageMetadata {
	input := u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	return messages.UsageMetadata{
		InputTokens:  input,
		OutputTokens: u.OutputTokens,
		TotalTokens:  input + u.OutputTokens,
	}
}

func (r messagePayload) toMessage() messages.Message {
	var textParts []string
	var blocks []messages.ContentBlock
	var toolCalls []messages.ToolCall
	for _, content := range r.Content {
		switch content.Type {
		case "text":
			if content.Text != "" {
				textParts = append(textParts, content.Text)
			}
			blocks = append(blocks, messages.ContentBlock{
				"type": "text",
				"text": content.Text,
			})
		case "tool_use":
			block := messages.ContentBlock{
				"type": "tool_call",
				"id":   content.ID,
				"name": content.Name,
				"args": content.Input,
			}
			blocks = append(blocks, block)
			toolCalls = append(toolCalls, messages.ToolCall{
				ID:   content.ID,
				Name: content.Name,
				Args: content.Input,
			})
		case "thinking":
			blocks = append(blocks, messages.ContentBlock{
				"type":      "reasoning",
				"reasoning": content.Thinking,
				"signature": content.Signature,
			})
		case "redacted_thinking":
			block := messages.ContentBlock{
				"type": "reasoning",
				"data": content.Data,
			}
			if content.ID != "" {
				block["id"] = content.ID
			}
			blocks = append(blocks, block)
		default:
			raw, _ := json.Marshal(content)
			var block messages.ContentBlock
			_ = json.Unmarshal(raw, &block)
			blocks = append(blocks, block)
		}
	}
	message := messages.AI(strings.Join(textParts, ""))
	message.ID = r.ID
	message.ContentBlocks = blocks
	message.ToolCalls = toolCalls
	message.ResponseMetadata = map[string]any{
		"model":          r.Model,
		"model_provider": "anthropic",
		"stop_reason":    r.StopReason,
	}
	message.UsageMetadata = r.Usage.toUsageMetadata()
	return message
}

func messageToAnthropicContent(message messages.Message) ([]contentBlock, error) {
	var content []contentBlock
	if message.Content != "" {
		content = append(content, contentBlock{Type: "text", Text: message.Content})
	}
	if len(message.ContentBlocks) > 0 {
		blocks, err := formatContentBlocks(message.ContentBlocks)
		if err != nil {
			return nil, err
		}
		content = append(content, blocks...)
	}
	for _, call := range message.ToolCalls {
		content = append(content, contentBlock{
			Type:  "tool_use",
			ID:    call.ID,
			Name:  call.Name,
			Input: call.Args,
		})
	}
	if len(content) == 0 {
		content = append(content, contentBlock{Type: "text", Text: ""})
	}
	return content, nil
}

func toolResultContent(message messages.Message) ([]contentBlock, error) {
	result := contentBlock{
		Type:      "tool_result",
		ToolUseID: message.ToolCallID,
	}
	if len(message.ContentBlocks) == 0 {
		result.Content = message.Content
		return []contentBlock{result}, nil
	}

	blocks, err := formatContentBlocks(message.ContentBlocks)
	if err != nil {
		return nil, err
	}
	if message.Content != "" {
		blocks = append([]contentBlock{{Type: "text", Text: message.Content}}, blocks...)
	}
	for i := range blocks {
		if blocks[i].CacheControl != nil {
			result.CacheControl = blocks[i].CacheControl
			blocks[i].CacheControl = nil
		}
	}
	result.Content = blocks
	return []contentBlock{result}, nil
}

func applyThinkingConstraints(payload *requestPayload) {
	temperature := 1.0
	payload.Temperature = &temperature
	payload.TopP = nil
	payload.TopK = nil
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

// cloneAnyMap returns a shallow defensive copy of an arbitrary config map.
func cloneAnyMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for key, value := range m {
		out[key] = value
	}
	return out
}

func boolPtr(v bool) *bool {
	return &v
}

func cloneBoolPtr(v *bool) *bool {
	if v == nil {
		return nil
	}
	out := *v
	return &out
}
