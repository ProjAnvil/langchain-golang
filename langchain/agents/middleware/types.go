package middleware

import (
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/stores"
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

// DeltaTransform rewrites a single streaming model delta (the text carried in
// a streamevents.Event content-block text-delta) and returns the replacement
// text. Used by WrapModelStreamHook for streaming redaction (e.g. PII
// lookback buffering — see Task 3.2): each delta flows through the composed
// transform before the model node emits it as a model_delta event, and the
// same transform is applied to the assembled model_end text so the two stay
// consistent.
//
// This is a bounded middleware-facing streaming surface. It is NOT langgraph
// stream modes (see the design spec Decision 3): the only streaming this port
// exposes is Agent.StreamEvents + graph.InvokeStream, and this hook lets a
// middleware observe/rewrite each model delta within that single path.
type DeltaTransform func(text string) string

// WrapModelStreamHook lets middleware observe/transform model deltas during a
// streaming model call. The model node collects every middleware implementing
// this hook (in WrapModelCall order — outermost-first) and composes them into a
// single DeltaTransform applied to each text delta before it is emitted as a
// model_delta event, and to the assembled final message text before model_end.
//
// The composition contract: TransformModelStream receives the inner transform
// (composed from middleware earlier in the chain plus an identity seed) and
// returns a new transform that wraps it. A middleware that wants to buffer
// across deltas (e.g. PII lookback) can close over mutable state in the
// returned func. When no middleware implements this hook, behavior is
// identical to today (identity transform — the streaming path is unchanged).
//
// This is the Go equivalent of Python's `_PIIStreamTransformer` delta path,
// scoped to the middleware streaming surface only.
type WrapModelStreamHook interface {
	TransformModelStream(transform DeltaTransform) DeltaTransform
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
	// Store is the agent's cross-thread KV store (Python's `InjectedStore`),
	// populated when the agent is configured with WithAgentStore. Tools that
	// need it read it explicitly (Go has no annotation-based injection). nil
	// when no store is configured.
	Store stores.BaseStore[any]
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
	storeSet    bool
	store       stores.BaseStore[any]
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

// WithStore installs a cross-thread KV store into the overridden request,
// mirroring the Store field set by CreateAgent when WithAgentStore is used.
// Useful for middleware that wants to swap (or clear) the store for the
// remainder of the tool-call chain.
func WithStore(store stores.BaseStore[any]) ToolCallRequestOverride {
	return func(override *toolCallRequestOverride) {
		override.storeSet = true
		override.store = store
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
	if override.storeSet {
		next.Store = override.store
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
