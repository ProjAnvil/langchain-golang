package middleware

import (
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/tools"
)

type ModelRequest struct {
	Model          any
	Messages       []messages.Message
	SystemMessage  *messages.Message
	SystemPrompt   string
	ToolChoice     any
	Tools          []any
	ResponseFormat any
	State          map[string]any
	Runtime        any
	ModelSettings  map[string]any
}

type ModelResponse struct {
	Result             []messages.Message
	StructuredResponse any
}

type ExtendedModelResponse struct {
	ModelResponse ModelResponse
	Command       *Command
}

type Command struct {
	Update map[string]any
	Goto   string
	Resume any
	Graph  string
}

type ToolCall struct {
	Name string
	Args map[string]any
	ID   string
	Type string
}

func (c ToolCall) Clone() ToolCall {
	return ToolCall{
		Name: c.Name,
		Args: cloneAnyMap(c.Args),
		ID:   c.ID,
		Type: c.Type,
	}
}

type ToolCallRequest struct {
	ToolCall ToolCall
	Tool     tools.Tool
	State    map[string]any
	Runtime  any
}

type ToolCallRequestOverride func(*toolCallRequestOverride)

type toolCallRequestOverride struct {
	toolCallSet bool
	toolCall    ToolCall
	toolSet     bool
	tool        tools.Tool
	stateSet    bool
	state       map[string]any
	runtimeSet  bool
	runtime     any
}

func WithToolCall(toolCall ToolCall) ToolCallRequestOverride {
	return func(override *toolCallRequestOverride) {
		override.toolCallSet = true
		override.toolCall = toolCall
	}
}

func WithTool(tool tools.Tool) ToolCallRequestOverride {
	return func(override *toolCallRequestOverride) {
		override.toolSet = true
		override.tool = tool
	}
}

func WithToolCallState(state map[string]any) ToolCallRequestOverride {
	return func(override *toolCallRequestOverride) {
		override.stateSet = true
		override.state = state
	}
}

func WithToolCallRuntime(runtime any) ToolCallRequestOverride {
	return func(override *toolCallRequestOverride) {
		override.runtimeSet = true
		override.runtime = runtime
	}
}

func (r ToolCallRequest) Override(opts ...ToolCallRequestOverride) ToolCallRequest {
	override := toolCallRequestOverride{}
	for _, opt := range opts {
		opt(&override)
	}

	next := r
	next.ToolCall = r.ToolCall.Clone()

	if override.toolCallSet {
		next.ToolCall = override.toolCall.Clone()
	}
	if override.toolSet {
		next.Tool = override.tool
	}
	if override.stateSet {
		next.State = override.state
	}
	if override.runtimeSet {
		next.Runtime = override.runtime
	}
	return next
}

func (c Command) ValidateForWrapModelCall() error {
	if c.Goto != "" {
		return fmt.Errorf("Command goto is not supported in wrap_model_call")
	}
	if c.Resume != nil {
		return fmt.Errorf("Command resume is not supported in wrap_model_call")
	}
	if c.Graph != "" {
		return fmt.Errorf("Command graph is not supported in wrap_model_call")
	}
	return nil
}

func NewModelRequest(request ModelRequest) (ModelRequest, error) {
	if request.SystemPrompt != "" && request.SystemMessage != nil {
		return ModelRequest{}, fmt.Errorf("Cannot specify both system_prompt and system_message")
	}
	if request.SystemPrompt != "" {
		systemMessage := messages.System(request.SystemPrompt)
		request.SystemMessage = &systemMessage
		request.SystemPrompt = ""
	}
	if request.Tools == nil {
		request.Tools = []any{}
	}
	if request.State == nil {
		request.State = map[string]any{"messages": []messages.Message{}}
	}
	if request.ModelSettings == nil {
		request.ModelSettings = map[string]any{}
	}
	request.Messages = append([]messages.Message(nil), request.Messages...)
	request.Tools = append([]any(nil), request.Tools...)
	request.State = cloneAnyMap(request.State)
	request.ModelSettings = cloneAnyMap(request.ModelSettings)
	return request, nil
}

func (r ModelRequest) SystemPromptText() string {
	if r.SystemMessage == nil {
		return ""
	}
	if r.SystemMessage.Content != "" {
		return r.SystemMessage.Content
	}

	parts := make([]string, 0, len(r.SystemMessage.ContentBlocks))
	for _, block := range r.SystemMessage.ContentBlocks {
		if text, ok := block["text"].(string); ok && block["type"] == "text" {
			parts = append(parts, text)
			continue
		}
		if content, ok := block["content"].(string); ok {
			parts = append(parts, content)
		}
	}
	return strings.Join(parts, "")
}

type ModelRequestOverride func(*modelRequestOverride)

type modelRequestOverride struct {
	modelSet         bool
	model            any
	messagesSet      bool
	messages         []messages.Message
	toolsSet         bool
	tools            []any
	systemMessageSet bool
	systemMessage    *messages.Message
	systemPromptSet  bool
	systemPrompt     *string
}

func WithModel(model any) ModelRequestOverride {
	return func(override *modelRequestOverride) {
		override.modelSet = true
		override.model = model
	}
}

func WithMessages(messages []messages.Message) ModelRequestOverride {
	return func(override *modelRequestOverride) {
		override.messagesSet = true
		override.messages = cloneMessages(messages)
	}
}

func WithTools(tools []any) ModelRequestOverride {
	return func(override *modelRequestOverride) {
		override.toolsSet = true
		override.tools = append([]any(nil), tools...)
	}
}

func WithSystemMessage(message *messages.Message) ModelRequestOverride {
	return func(override *modelRequestOverride) {
		override.systemMessageSet = true
		override.systemMessage = message
	}
}

func WithSystemPrompt(prompt string) ModelRequestOverride {
	return func(override *modelRequestOverride) {
		override.systemPromptSet = true
		override.systemPrompt = &prompt
	}
}

func WithSystemPromptNone() ModelRequestOverride {
	return func(override *modelRequestOverride) {
		override.systemPromptSet = true
		override.systemPrompt = nil
	}
}

func (r ModelRequest) Override(opts ...ModelRequestOverride) (ModelRequest, error) {
	override := modelRequestOverride{}
	for _, opt := range opts {
		opt(&override)
	}
	if override.systemPromptSet && override.systemMessageSet {
		return ModelRequest{}, fmt.Errorf("Cannot specify both system_prompt and system_message")
	}

	next := r
	next.Messages = append([]messages.Message(nil), r.Messages...)
	next.Tools = append([]any(nil), r.Tools...)
	next.State = cloneAnyMap(r.State)
	next.ModelSettings = cloneAnyMap(r.ModelSettings)

	if override.modelSet {
		next.Model = override.model
	}
	if override.messagesSet {
		next.Messages = cloneMessages(override.messages)
	}
	if override.toolsSet {
		next.Tools = append([]any(nil), override.tools...)
	}
	if override.systemPromptSet {
		if override.systemPrompt == nil {
			next.SystemMessage = nil
		} else {
			systemMessage := messages.System(*override.systemPrompt)
			next.SystemMessage = &systemMessage
		}
		next.SystemPrompt = ""
	}
	if override.systemMessageSet {
		next.SystemMessage = override.systemMessage
		next.SystemPrompt = ""
	}

	return next, nil
}

func cloneAnyMap(input map[string]any) map[string]any {
	if len(input) == 0 {
		return map[string]any{}
	}
	cloned := make(map[string]any, len(input))
	for key, value := range input {
		cloned[key] = value
	}
	return cloned
}

func cloneMessages(input []messages.Message) []messages.Message {
	if input == nil {
		return nil
	}
	out := make([]messages.Message, len(input))
	for i, msg := range input {
		out[i] = msg
		out[i].ContentBlocks = append([]messages.ContentBlock(nil), msg.ContentBlocks...)
		out[i].ToolCalls = append([]messages.ToolCall(nil), msg.ToolCalls...)
		out[i].InvalidToolCalls = append([]messages.ToolCall(nil), msg.InvalidToolCalls...)
		out[i].ResponseMetadata = cloneAnyMap(msg.ResponseMetadata)
		out[i].AdditionalKwargs = cloneAnyMap(msg.AdditionalKwargs)
		out[i].ProviderNativeEvent = cloneAnyMap(msg.ProviderNativeEvent)
	}
	return out
}
