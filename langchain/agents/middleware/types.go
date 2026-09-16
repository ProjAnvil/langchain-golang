package middleware

import (
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/store"
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

// ExtendedModelResponse is a wrap_model_call layer's ModelResponse plus a
// Command carrying an additional state update (Python's
// `ExtendedModelResponse`, middleware/types.py). Only the Command's Update is
// honored: the model node applies it on top of the node's default
// messages/structured_response update (middleware keys win conflicts, matching
// Python's `commands.extend` ordering in factory._build_commands), and a
// Command carrying Goto/Resume/Graph is rejected — see
// ValidateForWrapModelCall. A nil Command means no additional update.
type ExtendedModelResponse struct {
	ModelResponse ModelResponse
	Command       *Command
}

// ModelCallResult mirrors Python's `ModelCallResult` union
// (middleware/types.py:313: `ModelResponse | AIMessage | ExtendedModelResponse`).
// A model-call handler result may be any of:
//   - ModelResponse / *ModelResponse — passed through unchanged
//   - ExtendedModelResponse / *ExtendedModelResponse — unwrapped to its
//     embedded ModelResponse, with its Command surfaced for the model node
//     to accumulate and apply
//   - messages.Message with Role == messages.RoleAI — the AIMessage short
//     form, normalized to ModelResponse{Result: [msg]}
//
// It is returned by agents.WrapModelCallResultHook and normalized by
// NormalizeModelCallResult.
type ModelCallResult = any

// NormalizeModelCallResult mirrors Python's `factory._normalize_to_model_response`
// (factory.py:177) plus the command extraction of `_to_composed_result`
// (factory.py:258): a bare AI message becomes ModelResponse{Result: [msg]}
// and an ExtendedModelResponse unwraps to its embedded ModelResponse while
// surfacing its Command, so the inner composition boundary always sees a
// ModelResponse and the caller can accumulate the Commands returned by each
// wrap_model_call layer (inner-first) instead of dropping them. The returned
// Command is nil for every non-extended result shape.
func NormalizeModelCallResult(result ModelCallResult) (ModelResponse, *Command, error) {
	switch r := result.(type) {
	case ModelResponse:
		return r, nil, nil
	case *ModelResponse:
		if r == nil {
			return ModelResponse{}, nil, fmt.Errorf("middleware: nil *ModelResponse is not a valid ModelCallResult")
		}
		return *r, nil, nil
	case ExtendedModelResponse:
		return r.ModelResponse, r.Command, nil
	case *ExtendedModelResponse:
		if r == nil {
			return ModelResponse{}, nil, fmt.Errorf("middleware: nil *ExtendedModelResponse is not a valid ModelCallResult")
		}
		return r.ModelResponse, r.Command, nil
	case messages.Message:
		if r.Role != messages.RoleAI {
			return ModelResponse{}, nil, fmt.Errorf("middleware: ModelCallResult message must have role %q, got %q", messages.RoleAI, r.Role)
		}
		return ModelResponse{Result: []messages.Message{r}}, nil, nil
	default:
		return ModelResponse{}, nil, fmt.Errorf("middleware: unsupported ModelCallResult type %T", result)
	}
}

// Command mirrors Python's langgraph Command as produced by agent middleware
// hooks (before_model jump control, and wrap_model_call results via
// ExtendedModelResponse). In a wrap_model_call result only Update is applied
// by the model node; Goto/Resume/Graph are rejected there — see
// ValidateForWrapModelCall.
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

// LLMTypeProvider is implemented by language.ChatModel values that can report
// their Python "_llm_type" identifier (e.g. "anthropic-chat", "ollama-chat",
// "openai-chat"). SummarizationMiddleware duck-types it to tune the
// approximate token counter per provider family, mirroring Python's
// `_get_approximate_token_counter(model)` check on `model._llm_type`
// (langchain_v1/langchain/agents/middleware/summarization.py:208-216). It is
// optional: models that do not implement it fall back to the default counter,
// just as Python falls through when `_llm_type` is unrecognized.
type LLMTypeProvider interface {
	LLMType() string
}

// ToolProvider is implemented by middleware that contribute tools to the
// agent's ToolNode and default model bindings, mirroring Python's
// AgentMiddleware.tools attribute (middleware/types.py:395) collected by
// create_agent (factory.py:1005: `middleware_tools = [t for m in middleware
// for t in getattr(m, "tools", [])]`, merged ahead of the caller's tools at
// factory.py:1054-1055).
//
// The method is deliberately named ProvidedTools rather than Tools: several
// middleware (TodoListMiddleware, FilesystemFileSearchMiddleware,
// ShellToolMiddleware) already expose their tools as a public `Tools` field,
// Go forbids a method and a field sharing a name, and the field must stay
// non-breaking — those middleware simply return the field from the method.
type ToolProvider interface {
	// ProvidedTools returns the tools this middleware registers. The slice is
	// not copied; CreateAgent treats it as read-only.
	ProvidedTools() []tools.Tool
}

// StateField describes one state key a middleware contributes to the agent's
// state schema. It is the middleware-package mirror of agents.StateField:
// the middleware package cannot import the agents package (which imports
// this one) without an import cycle, so the small struct is duplicated here
// and converted at collection time.
type StateField struct {
	// Name is the state key, e.g. "todos".
	Name string
	// Reducer is the merge strategy for successive writes to Name. Nil means
	// last-write-wins (channels.LastValueReducer), Python's LastValue channel.
	Reducer channels.Reducer
}

// StateSchemaContributor is implemented by middleware that contribute state
// keys to the agent's state schema, mirroring Python's
// AgentMiddleware.state_schema attribute (middleware/types.py:395-401) merged
// by create_agent (factory.py:1150-1156: middleware schemas first in
// registration order, the caller's state_schema last so it wins any field
// conflict).
type StateSchemaContributor interface {
	// StateSchema returns the state fields this middleware declares. The
	// slice is not copied; CreateAgent treats it as read-only.
	StateSchema() []StateField
}

// TracePolicyConfig mirrors langgraph's TracePolicy transforms without
// importing the graph package (which this package cannot import): a
// middleware providing it controls traced payloads for everything inside its
// wrap_model_call wrapping layer — langchain 1.3.15's middleware
// trace_policy (#38910), applied on the emit side so every tracer (LangSmith,
// console, a future OTel bridge) observes the scrubbed payloads.
//
// Fail-closed: a transform that panics drops the corresponding payload rather
// than leaking it — a deliberate divergence from upstream's fail-open
// behavior (see DIVERGENCES.md); the feature's motivation is PII/compliance.
type TracePolicyConfig struct {
	// ProcessInputs transforms start-kind event payloads (inputs). Nil leaves
	// inputs untouched.
	ProcessInputs func(value any) any
	// ProcessOutputs transforms end-kind event payloads (outputs). Nil leaves
	// outputs untouched.
	ProcessOutputs func(value any) any
}

// TracePolicyProvider is the optional hook middleware implement to scrub
// traced payloads before any tracer observes them (PII redaction at the emit
// side; installed via callbacks.NewPayloadPolicyManager by the model node).
// The policy covers everything the middleware's wrap_model_call layer
// encloses — inner middleware and the model itself — and nested middleware
// policies compose like onion layers: the outermost policy is installed into
// the context first and therefore applied last.
type TracePolicyProvider interface {
	// TracePolicy returns the middleware's payload transforms. Both nil means
	// no policy.
	TracePolicy() TracePolicyConfig
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
	// Store is the agent's cross-thread semantic store (Python's
	// `InjectedStore` / langgraph BaseStore), populated when the agent is
	// configured with WithAgentStore. Tools that need it read it explicitly
	// (Go has no annotation-based injection). nil when no store is configured.
	Store store.Store
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
	store       store.Store
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

// WithStore installs a cross-thread semantic store into the overridden
// request, mirroring the Store field set by CreateAgent when WithAgentStore is
// used. Useful for middleware that wants to swap (or clear) the store for the
// remainder of the tool-call chain.
func WithStore(s store.Store) ToolCallRequestOverride {
	return func(override *toolCallRequestOverride) {
		override.storeSet = true
		override.store = s
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

// ValidateForWrapModelCall reports whether c is an update-only Command usable
// as a wrap_model_call return value. Routing controls are not supported
// there, mirroring factory._build_commands (factory.py:210-216): goto,
// resume, and graph each yield an error, while an empty or update-only
// Command is valid. The model node enforces this before applying the
// command's Update.
func (c Command) ValidateForWrapModelCall() error {
	if c.Goto != "" {
		return fmt.Errorf("middleware: Command goto is not supported in wrap_model_call middleware. Use the jump_to state field with before_model/after_model hooks instead")
	}
	if c.Resume != nil {
		return fmt.Errorf("middleware: Command resume is not supported in wrap_model_call middleware")
	}
	if c.Graph != "" {
		return fmt.Errorf("middleware: Command graph is not supported in wrap_model_call middleware")
	}
	return nil
}

func NewModelRequest(request ModelRequest) (ModelRequest, error) {
	if request.SystemPrompt != "" && request.SystemMessage != nil {
		return ModelRequest{}, fmt.Errorf("cannot specify both system_prompt and system_message")
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
		m := messages.BlockToMap(block)
		if text, ok := m["text"].(string); ok && m["type"] == "text" {
			parts = append(parts, text)
			continue
		}
		if content, ok := m["content"].(string); ok {
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
	// Task 6 additions (mirror Python's _ModelRequestOverrides, types.py:72-82):
	toolChoiceSet     bool
	toolChoice        any
	responseFormatSet bool
	responseFormat    any
	modelSettingsSet  bool
	modelSettings     map[string]any
	stateSet          bool
	state             map[string]any
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

// WithToolChoice sets the request's tool_choice. The model node
// applies it at bind time on every model call: an overridden value replaces
// the structured-output "any" default (a ToolStrategy with declared
// structured tools still forces "any", mirroring factory.py:1388; a
// ProviderStrategy-effective call binds no tool_choice at all, mirroring
// factory.py:1366-1371 — see effectiveBindToolChoice in create_agent.go).
func WithToolChoice(choice any) ModelRequestOverride {
	return func(o *modelRequestOverride) { o.toolChoiceSet = true; o.toolChoice = choice }
}

// WithResponseFormat replaces the request's response_format for the
// remainder of the model-call chain. rf may be a ToolStrategy,
// ProviderStrategy, AutoStrategy, or a raw JSON-schema map (normalized like
// Python's factory.py:1327-1331). The model node re-derives the effective
// strategy from it per call — including ToolStrategy↔ProviderStrategy
// switches and AutoStrategy re-resolution against the current model — and a
// ToolStrategy rf may only narrow to structured tools declared in the
// agent's original response format (factory.py:1375-1385). See
// resolveEffectiveResponseFormat in create_agent.go.
func WithResponseFormat(rf any) ModelRequestOverride {
	return func(o *modelRequestOverride) { o.responseFormatSet = true; o.responseFormat = rf }
}

// WithModelSettings replaces the request's
// model_settings. The model node merges them over the effective strategy's
// kwargs on every model call and passes the result to the model's bind
// (settings win conflicts, mirroring factory.py:1361's
// `{**kwargs, **request.model_settings}`) — see mergedBindSettings /
// prepareModelCall in create_agent.go.
func WithModelSettings(settings map[string]any) ModelRequestOverride {
	return func(o *modelRequestOverride) { o.modelSettingsSet = true; o.modelSettings = settings }
}

func WithState(state map[string]any) ModelRequestOverride {
	return func(o *modelRequestOverride) { o.stateSet = true; o.state = state }
}

func (r ModelRequest) Override(opts ...ModelRequestOverride) (ModelRequest, error) {
	override := modelRequestOverride{}
	for _, opt := range opts {
		opt(&override)
	}
	if override.systemPromptSet && override.systemMessageSet {
		return ModelRequest{}, fmt.Errorf("cannot specify both system_prompt and system_message")
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
	if override.toolChoiceSet {
		next.ToolChoice = override.toolChoice
	}
	if override.responseFormatSet {
		next.ResponseFormat = override.responseFormat
	}
	if override.modelSettingsSet {
		next.ModelSettings = cloneAnyMap(override.modelSettings)
	}
	if override.stateSet {
		next.State = cloneAnyMap(override.state)
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
