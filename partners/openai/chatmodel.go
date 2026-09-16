package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
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
	"github.com/projanvil/langchain-golang/partners/internal/providerutil"
)

const defaultBaseURL = "https://api.openai.com/v1"

// ChatModel adapts LangChain chat calls to OpenAI's APIs. By default it uses
// the Responses API; WithChatCompletions switches it to the Chat Completions
// API (the classic `/chat/completions` endpoint, Python's default).
type ChatModel struct {
	config             modelconfig.Config
	boundTools         []tools.Tool
	structuredOutput   *structuredoutput.JSONSchema
	chatCompletions    bool
	reasoningEffort    string
	toolChoice         *ToolChoice
	parallelToolCalls  *bool
	responseFormat     map[string]any
	streamChunkTimeout time.Duration
	// Sampling knobs mirroring Python BaseChatOpenAI's optional fields
	// (chat_models/base.py:753-781, 963). All are sent on the Chat
	// Completions API; only top_p also applies to the Responses API
	// (stop is dropped there, chat_models/base.py:4284-4286).
	topP             *float64
	stop             []string
	seed             *int
	presencePenalty  *float64
	frequencyPenalty *float64
	logitBias        map[int]int
	n                *int
	logprobs         *bool
	topLogprobs      *int
}

// Compile-time guard: ChatModel (value receiver) satisfies
// language.StructuredCaller so the agent's ProviderStrategy native path
// (agents.invokeModel → language.InvokeStructured) can use it. A future
// refactor that drops InvokeStructured fails here.
var _ language.StructuredCaller = ChatModel{}

// Compile-time guard: ChatModel satisfies the optional language.ToolBinder
// capability so agent code can thread bind_tools tool_choice
// (agents.bindModelTools → BindToolsWithOptions).
var _ language.ToolBinder = ChatModel{}

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
	return ChatModel{config: cfg, streamChunkTimeout: resolveStreamChunkTimeout()}
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
// It is equivalent to BindToolsWithOptions(tools, language.BindToolsOptions{}).
func (m ChatModel) BindTools(boundTools []tools.Tool) (language.ChatModel, error) {
	next := m
	next.boundTools = append([]tools.Tool(nil), boundTools...)
	return next, nil
}

// BindToolsWithOptions implements language.ToolBinder, mapping the core
// ToolChoice modes onto this adapter's serialization: "auto"→"auto",
// "none"→"none", "any"→"required" (Python bind_tools' "any"/True mapping),
// and any other non-empty string to the named function tool
// ({"type":"function","function":{"name":X}}, flattened for the Responses
// API by the existing request builders). A zero-value ToolChoice keeps any
// constructor-level WithToolChoice value. ParallelToolCalls serializes as
// `parallel_tool_calls` on both the Responses and Chat Completions paths
// (Python BaseChatOpenAI default_params, chat_models/base.py:1340-1350
// exclude_if_none family); nil keeps the provider default (field omitted).
func (m ChatModel) BindToolsWithOptions(boundTools []tools.Tool, opts language.BindToolsOptions) (language.ChatModel, error) {
	next := m
	next.boundTools = append([]tools.Tool(nil), boundTools...)
	if opts.ToolChoice != "" {
		next.toolChoice = ptrToolChoice(toolChoiceFromCore(opts.ToolChoice))
	}
	if opts.ParallelToolCalls != nil {
		next.parallelToolCalls = opts.ParallelToolCalls
	}
	return next, nil
}

// toolChoiceFromCore translates a language.ToolChoice into this package's
// ToolChoice value, reusing the provider mapping Python's bind_tools applies.
func toolChoiceFromCore(choice language.ToolChoice) ToolChoice {
	switch choice {
	case language.ToolChoiceAuto:
		return ToolChoiceAuto()
	case language.ToolChoiceNone:
		return ToolChoiceNone()
	case language.ToolChoiceAny:
		return ToolChoiceRequired()
	default:
		// Any other string names the specific tool to force.
		return ToolChoiceFunction(string(choice))
	}
}

func ptrToolChoice(choice ToolChoice) *ToolChoice {
	return &choice
}

// WithChatCompletions returns a copy of the model that targets the Chat
// Completions API (`/chat/completions`) instead of the Responses API.
func (m ChatModel) WithChatCompletions() ChatModel {
	next := m
	next.chatCompletions = true
	return next
}

// WithReasoningEffort returns a copy of the model that requests a reasoning
// effort (low|medium|high), steering how hard reasoning models (OpenAI o-series
// / gpt-5 family) think before answering. Chat Completions requests carry it
// as the reasoning_effort scalar; Responses requests carry it as
// reasoning: {"effort": ...} (base.py). Non-reasoning models and gateways that
// don't know the field ignore it.
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

// WithTopP returns a copy of the model that samples from the top-p
// probability mass, mirroring Python ChatOpenAI(top_p=...)
// (chat_models/base.py:781). Sent on both the Responses and Chat Completions
// APIs (chat_models/base.py:1343).
func (m ChatModel) WithTopP(topP float64) ChatModel {
	next := m
	next.topP = &topP
	return next
}

// WithStop returns a copy of the model with default stop sequences, mirroring
// Python ChatOpenAI(stop=...) (chat_models/base.py:963). Sent on the Chat
// Completions API only: Python drops stop from Responses payloads because the
// Responses API has no stop parameter (chat_models/base.py:4284-4286).
func (m ChatModel) WithStop(stop ...string) ChatModel {
	next := m
	next.stop = append([]string(nil), stop...)
	return next
}

// WithSeed returns a copy of the model with a generation seed, mirroring
// Python ChatOpenAI(seed=...) (chat_models/base.py:759). Chat Completions
// only.
func (m ChatModel) WithSeed(seed int) ChatModel {
	next := m
	next.seed = &seed
	return next
}

// WithPresencePenalty returns a copy of the model that penalizes repeated
// tokens, mirroring Python ChatOpenAI(presence_penalty=...)
// (chat_models/base.py:753). Chat Completions only.
func (m ChatModel) WithPresencePenalty(penalty float64) ChatModel {
	next := m
	next.presencePenalty = &penalty
	return next
}

// WithFrequencyPenalty returns a copy of the model that penalizes repeated
// tokens according to frequency, mirroring Python
// ChatOpenAI(frequency_penalty=...) (chat_models/base.py:756). Chat
// Completions only.
func (m ChatModel) WithFrequencyPenalty(penalty float64) ChatModel {
	next := m
	next.frequencyPenalty = &penalty
	return next
}

// WithLogitBias returns a copy of the model that biases the likelihood of
// specific token IDs, mirroring Python ChatOpenAI(logit_bias=...)
// (chat_models/base.py:772). Chat Completions only. JSON keys are the token
// IDs as strings, per the OpenAI API shape.
func (m ChatModel) WithLogitBias(bias map[int]int) ChatModel {
	next := m
	next.logitBias = bias
	return next
}

// WithN returns a copy of the model that requests N chat completions per
// prompt, mirroring Python ChatOpenAI(n=...) (chat_models/base.py:778). Chat
// Completions only. Like the rest of this adapter, only the first choice is
// surfaced.
func (m ChatModel) WithN(n int) ChatModel {
	next := m
	next.n = &n
	return next
}

// WithLogprobs returns a copy of the model that requests log probabilities,
// mirroring Python ChatOpenAI(logprobs=...) (chat_models/base.py:762). Chat
// Completions only. The returned logprobs live in the raw response, which
// this adapter does not yet surface on the message.
func (m ChatModel) WithLogprobs(enabled bool) ChatModel {
	next := m
	next.logprobs = &enabled
	return next
}

// WithTopLogprobs returns a copy of the model that requests the K most likely
// tokens per position, mirroring Python ChatOpenAI(top_logprobs=...)
// (chat_models/base.py:765); logprobs must be enabled for it to take effect.
// Chat Completions only.
func (m ChatModel) WithTopLogprobs(k int) ChatModel {
	next := m
	next.topLogprobs = &k
	return next
}

// WithStreamChunkTimeout returns a copy of the model with a per-chunk
// wall-clock timeout on streaming, mirroring Python
// ChatOpenAI(stream_chunk_timeout=...). 0 disables; negative values are
// rejected and fall back to the env/default with a WARNING (Python's field
// validator), matching the env-var path.
func (m ChatModel) WithStreamChunkTimeout(d time.Duration) ChatModel {
	next := m
	if d < 0 {
		fallback := resolveStreamChunkTimeout()
		slog.Warn("openai: invalid negative stream_chunk_timeout; falling back (pass 0 to disable)",
			slog.String("value", d.String()),
			slog.String("fallback", fallback.String()))
		next.streamChunkTimeout = fallback
		return next
	}
	next.streamChunkTimeout = d
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

// Capabilities returns the adapter capability declaration. Every flag is
// exercised by both API paths (Responses and Chat Completions) in Invoke and
// Stream alike: image/url/audio inputs serialize through the shared request
// builders, usage metadata is extracted on invoke and on the final stream
// chunks (stream_options.include_usage / response.completed), and streaming
// is fully implemented for both APIs.
func (m ChatModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{
		ToolCalling:      true,
		ToolChoice:       true,
		StructuredOutput: true,
		JSONMode:         true,
		ImageInputs:      true,
		ImageURLs:        true,
		AudioInputs:      true,
		UsageMetadata:    true,
		Streaming:        true,
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
	request, err := m.buildRequest(input)
	if err != nil {
		return responsePayload{}, err
	}
	resp, err := postJSON[responsePayload](ctx, m.config, "/responses", request)
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

func (m ChatModel) buildRequest(input []messages.Message) (requestPayload, error) {
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
	// top_p is forwarded to the Responses API (chat_models/base.py:1343).
	// stop and the other Chat Completions sampling knobs are deliberately
	// absent: Python drops stop before the Responses call (base.py:4284-4286)
	// and never sets the CC-only knobs on this path.
	if m.topP != nil {
		payload.TopP = m.topP
	}
	if m.reasoningEffort != "" {
		payload.Reasoning = &reasoningConfig{Effort: m.reasoningEffort}
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
	if m.parallelToolCalls != nil {
		payload.ParallelToolCalls = m.parallelToolCalls
	}

	var instructions []string
	for _, message := range input {
		switch message.Role {
		case messages.RoleSystem:
			if message.Content != "" {
				instructions = append(instructions, message.Content)
			}
		case messages.RoleHuman:
			// Structured content blocks (text/image/audio) become Responses
			// input content parts; plain text stays a bare string so payloads
			// do not grow a parts array for text-only conversations.
			if len(message.ContentBlocks) > 0 {
				parts, err := responsesContentParts(message.ContentBlocks)
				if err != nil {
					return requestPayload{}, err
				}
				payload.Input = append(payload.Input, inputItem{
					Role:    "user",
					Content: parts,
				})
				continue
			}
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
	return payload, nil
}

type requestPayload struct {
	Model             string           `json:"model"`
	Input             []inputItem      `json:"input"`
	Instructions      string           `json:"instructions,omitempty"`
	Temperature       *float64         `json:"temperature,omitempty"`
	MaxOutputTokens   *int             `json:"max_output_tokens,omitempty"`
	TopP              *float64         `json:"top_p,omitempty"`
	Tools             []toolSpec       `json:"tools,omitempty"`
	ToolChoice        any              `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	Text              *textConfig      `json:"text,omitempty"`
	Stream            bool             `json:"stream,omitzero"`
	Reasoning         *reasoningConfig `json:"reasoning,omitempty"`
}

// reasoningConfig carries the Responses API reasoning controls. Effort maps
// from WithReasoningEffort (base.py emits {"reasoning": {"effort": ...}} on
// the Responses path, unlike the Chat Completions reasoning_effort scalar).
type reasoningConfig struct {
	Effort string `json:"effort,omitempty"`
}

type inputItem struct {
	// Content is a bare string for plain messages or a []any of structured
	// input content parts (input_text/input_image/input_audio) for
	// multimodal human messages.
	Role    string `json:"role,omitempty"`
	Content any    `json:"content,omitempty"`
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
	Strict bool          `json:"strict,omitzero"`
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
	message.UsageMetadata = r.Usage.toUsageMetadata()
	return message
}

// toUsageMetadata converts a Responses-shaped usage payload (also produced by
// normalizing Chat Completions usage) into the core UsageMetadata, including
// cached-input and reasoning-output details when the payload carries them.
func (u usagePayload) toUsageMetadata() messages.UsageMetadata {
	md := messages.UsageMetadata{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
	}
	if details := u.InputTokensDetails; details != nil {
		md.InputTokenDetails = &messages.InputTokenDetails{
			CacheReadInputTokens:     details.CachedTokens,
			CacheCreationInputTokens: details.CacheCreationTokens,
		}
	}
	if details := u.OutputTokensDetails; details != nil && details.ReasoningTokens > 0 {
		md.OutputTokenDetails = &messages.OutputTokenDetails{
			ReasoningOutputTokens: details.ReasoningTokens,
		}
	}
	return md
}

func emit(
	ctx context.Context,
	cfg runnables.Config,
	kind callbacks.EventKind,
	input any,
	output any,
	err error,
) error {
	return providerutil.Emit(ctx, cfg, kind, input, output, err)
}

// stringFrom extracts m[key] as a string ("" when absent or not a string).
func stringFrom(m map[string]any, key string) string {
	s, _ := m[key].(string)
	return s
}

func cloneMetadata(metadata map[string]any) map[string]any {
	return providerutil.CloneMetadata(metadata)
}
