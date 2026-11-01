package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/modelprofiles"
	"github.com/projanvil/langchain-golang/partners/internal/providerutil"
)

const defaultBaseURL = "https://api.anthropic.com/v1"

// ChatModel adapts LangChain chat calls to Anthropic's Messages API.
type ChatModel struct {
	config            modelconfig.Config
	boundTools        []tools.Tool
	toolStrict        *bool
	thinking          map[string]any
	reasoningEffort   string
	toolChoice        map[string]any
	parallelToolCalls *bool
	contextManagement map[string]any
	inferenceGeo      string
}

// Compile-time guard: ChatModel (value receiver) satisfies
// language.StructuredCaller so the agent's ProviderStrategy native path
// (agents.invokeModel → language.InvokeStructured) can use it.
var _ language.StructuredCaller = ChatModel{}

// Compile-time guard: ChatModel satisfies the optional language.ToolBinder
// capability so agent code can thread bind_tools tool_choice
// (agents.bindModelTools → BindToolsWithOptions).
var _ language.ToolBinder = ChatModel{}

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
// It is equivalent to BindToolsWithOptions(tools, language.BindToolsOptions{}).
func (m ChatModel) BindTools(boundTools []tools.Tool) (language.ChatModel, error) {
	next := m
	next.boundTools = append([]tools.Tool(nil), boundTools...)
	next.toolStrict = nil
	return next, nil
}

// BindToolsWithOptions implements language.ToolBinder, mapping the core
// ToolChoice modes onto the Messages API tool_choice shape: "auto"→
// {"type":"auto"}, "any"→{"type":"any"}, and any other non-empty string to
// the named tool {"type":"tool","name":X} — mirroring Python
// langchain-anthropic's bind_tools tool_choice conversion. "none" is NOT
// supported by the Messages API (there is no none tool_choice type; the way
// to express it is binding no tools), so it fails loudly rather than
// silently misrouting. A zero-value ToolChoice keeps any constructor-level
// WithToolChoice value. ParallelToolCalls is expressed as
// disable_parallel_tool_use inside tool_choice (inverted semantics,
// langchain-anthropic's payload assembly): when set, a missing tool_choice
// is synthesized as {"type":"auto"} so the flag always reaches the wire;
// nil leaves tool_choice exactly as bound.
func (m ChatModel) BindToolsWithOptions(boundTools []tools.Tool, opts language.BindToolsOptions) (language.ChatModel, error) {
	next := m
	next.boundTools = append([]tools.Tool(nil), boundTools...)
	next.toolStrict = nil
	if opts.ToolChoice != "" {
		choice, err := toolChoiceFromCore(opts.ToolChoice)
		if err != nil {
			return nil, err
		}
		next.toolChoice = choice
	}
	if opts.ParallelToolCalls != nil {
		next.parallelToolCalls = opts.ParallelToolCalls
	}
	return next, nil
}

// toolChoiceFromCore translates a language.ToolChoice into the Anthropic
// tool_choice request object.
func toolChoiceFromCore(choice language.ToolChoice) (map[string]any, error) {
	switch choice {
	case language.ToolChoiceAuto:
		return map[string]any{"type": "auto"}, nil
	case language.ToolChoiceAny:
		return map[string]any{"type": "any"}, nil
	case language.ToolChoiceNone:
		return nil, fmt.Errorf("anthropic: tool_choice %q is not supported by the Messages API (bind no tools instead)", choice)
	default:
		// Any other string names the specific tool to force.
		return map[string]any{"type": "tool", "name": string(choice)}, nil
	}
}

// BindToolsStrict returns a copy of the model with Anthropic tools bound and
// the strict tool-use schema flag set on each tool definition.
func (m ChatModel) BindToolsStrict(boundTools []tools.Tool, strict bool) (ChatModel, error) {
	next := m
	next.boundTools = append([]tools.Tool(nil), boundTools...)
	next.toolStrict = new(strict)
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

// validEffortLevels mirrors Python ChatAnthropic's reasoning_effort field type
// Literal["max", "xhigh", "high", "medium", "low"] (alias "effort",
// langchain_anthropic/chat_models.py:1523). The set is model-independent: the
// per-model levels in the profile advertise support but are not enforced
// client-side.
var validEffortLevels = []string{"max", "xhigh", "high", "medium", "low"}

// opus5Prefix names the model family whose members reject disabled thinking at
// xhigh/max effort (chat_models.py:1851-1863 checks
// model.startswith("claude-opus-5")).
const opus5Prefix = "claude-opus-5"

// WithEffort returns a copy of the model with the reasoning-effort level set
// (Python ChatAnthropic's `effort`/`reasoning_effort` constructor fields, #41536).
// The level is emitted as output_config.effort on the request. When no explicit
// thinking is set and the model's profile declares xhigh support (Opus 4.7+,
// Sonnet 5 and newer), thinking defaults to
// {"type":"adaptive","display":"summarized"} so the model actually reasons at
// the requested level. The level is validated at request-build time — an
// illegal value errors with Python's exact pydantic literal_error text, since
// a Go option chain cannot fail at construction.
func (m ChatModel) WithEffort(effort string) ChatModel {
	next := m
	next.reasoningEffort = effort
	return next
}

// Effort returns the resolved reasoning-effort level, or "" when unset. It is
// the Go counterpart of Python's `effort` property alias for `reasoning_effort`
// (chat_models.py:1611-1613): it exposes only the explicitly configured value
// (the profile's documented default is never resolved client-side).
func (m ChatModel) Effort() string { return m.reasoningEffort }

// assertValidEffort mirrors the effort parts of Python's
// _assert_valid_model_configuration (chat_models.py:1793-1863), which runs
// before message conversion on every request:
//
//   - the level must be one of the Literal values — Python enforces this at
//     construction via pydantic's literal_error; Go surfaces the same message
//     here because WithEffort cannot fail;
//   - claude-opus-5* models reject thinking={"type":"disabled"} combined with
//     xhigh/max effort ("permits disabled thinking only through `high` effort").
func (m ChatModel) assertValidEffort() error {
	if m.reasoningEffort == "" {
		return nil
	}
	if !slices.Contains(validEffortLevels, m.reasoningEffort) {
		return fmt.Errorf(
			"anthropic %s: effort %q is invalid: Input should be 'max', 'xhigh', 'high', 'medium' or 'low'",
			m.config.Model, m.reasoningEffort)
	}
	if strings.HasPrefix(m.config.Model, opus5Prefix) &&
		m.thinking != nil && m.thinking["type"] == "disabled" &&
		(m.reasoningEffort == "xhigh" || m.reasoningEffort == "max") {
		// Python's message verbatim (chat_models.py:1857-1862) except the
		// trailing period, dropped per Go error-string convention (staticcheck
		// ST1005).
		return fmt.Errorf(
			"`thinking={\"type\": \"disabled\"}` is not supported for %s with "+
				"`output_config.effort='%s'`; use adaptive thinking, omit `thinking`, "+
				"or set effort to `high` or below",
			m.config.Model, m.reasoningEffort)
	}
	return nil
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

// messagesEndpoint is the Messages API endpoint a request posts to. When the
// payload requires beta flags (in-place tool-change system turns), Python
// routes through the SDK's beta namespace (chat_models.py:2177-2190), whose
// endpoint is "/v1/messages?beta=true" — same path plus the beta query flag
// (anthropic-sdk-python resources/beta/messages/messages.py:1130-1131) — with
// the flags carried by the anthropic-beta header.
func messagesEndpoint(requiredBetas []string) string {
	if len(requiredBetas) > 0 {
		return "/messages?beta=true"
	}
	return "/messages"
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
	resp, err := postJSON[messagePayload](ctx, m.configWithRequiredBetas(req.requiredBetas), messagesEndpoint(req.requiredBetas), req)
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
	if err := m.assertValidEffort(); err != nil {
		return requestPayload{}, err
	}
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
	if stops, ok := m.config.Extra[stopSequencesKey].([]string); ok && len(stops) > 0 {
		payload.StopSequences = append([]string(nil), stops...)
	}

	midSystem, midTools := m.midConversationGates()

	var (
		systemText   []string
		systemBlocks []contentBlock
		// systemClaimed mirrors Python's `system is not None`
		// (chat_models.py:751): once the top-level system field holds the
		// leading run or a hoisted message, a further hoist on the
		// profile-enabled path is an error.
		systemClaimed bool
		// pending collects a run of system messages deferred for in-place
		// emission, mirroring Python's pending_system.
		pending []messages.Message
		// nonSystemSeen distinguishes the leading system run (hoisted without
		// warning) from mid-conversation ones.
		nonSystemSeen bool
	)

	// hoistSystemRun folds a run of system messages into the top-level system
	// field. allowToolChanges mirrors Python's preserve_tool_changes: only the
	// leading run (gated on the profile's mid_conversation_tools) may carry
	// tool-change blocks; a hoisted mid-conversation message never does.
	hoistSystemRun := func(run []messages.Message, allowToolChanges, warn bool) error {
		if midSystem && systemClaimed {
			return fmt.Errorf(
				"anthropic %s: received multiple non-consecutive system messages (this model's profile allows mid-conversation system messages; place extra system messages last or before an AI message)",
				m.config.Model)
		}
		for _, message := range run {
			if message.Content != "" {
				systemText = append(systemText, message.Content)
			}
			if len(message.ContentBlocks) > 0 {
				blocks, err := formatHoistedSystemBlocks(message.ContentBlocks, allowToolChanges, m.config.Model)
				if err != nil {
					return err
				}
				systemBlocks = append(systemBlocks, blocks...)
			}
		}
		systemClaimed = true
		if warn {
			warnSystemMessageHoisted(m.config.Model)
		}
		return nil
	}

	for i := 0; i < len(input); {
		message := input[i]
		if message.Role != messages.RoleSystem {
			nonSystemSeen = true
			// Flush deferred system messages ahead of the turn they precede:
			// before an assistant turn they go in place
			// (chat_models.py:1043-1046), otherwise they are hoisted with a
			// warning (chat_models.py:1047-1054).
			if len(pending) > 0 {
				if message.Role == messages.RoleAI {
					turns, err := inPlaceSystemTurns(pending, midTools, m.config.Model)
					if err != nil {
						return requestPayload{}, err
					}
					payload.Messages = append(payload.Messages, turns...)
				} else if err := hoistSystemRun(pending, false, true); err != nil {
					return requestPayload{}, err
				}
				pending = nil
			}
			switch message.Role {
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
			i++
			continue
		}
		// Gather the run of consecutive system messages (Python's
		// _merge_messages coalesces them into one message, so placement
		// decisions are per run).
		j := i
		for j < len(input) && input[j].Role == messages.RoleSystem {
			j++
		}
		run := input[i:j]
		i = j
		switch {
		case !nonSystemSeen:
			// Leading run: hoisted into the top-level system field without a
			// warning; tool-change blocks survive only when the profile
			// declares mid_conversation_tools.
			if err := hoistSystemRun(run, midTools, false); err != nil {
				return requestPayload{}, err
			}
		case midSystem && (len(pending) > 0 || previousTurnAllowsSystem(payload.Messages)):
			pending = append(pending, run...)
		default:
			if err := hoistSystemRun(run, false, true); err != nil {
				return requestPayload{}, err
			}
		}
	}
	// A trailing run of deferred system messages is emitted in place
	// (chat_models.py:1063-1065).
	if len(pending) > 0 {
		turns, err := inPlaceSystemTurns(pending, midTools, m.config.Model)
		if err != nil {
			return requestPayload{}, err
		}
		payload.Messages = append(payload.Messages, turns...)
	}
	if len(systemBlocks) > 0 {
		if len(systemText) > 0 {
			systemBlocks = append([]contentBlock{{Type: "text", Text: strings.Join(systemText, "\n")}}, systemBlocks...)
		}
		payload.System = systemBlocks
	} else if len(systemText) > 0 {
		payload.System = strings.Join(systemText, "\n")
	}
	payload.requiredBetas = requiredBetaHeaders(payload.Messages, splitBetaHeader(m.config.Headers[anthropicBetaHeader]))
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
	}
	// Effort → output_config (#41536, chat_models.py:1970-1998). Python merges
	// the constructor output_config dict and then applies effort on top; effort
	// is the only Go source, so the emitted map is exactly {"effort": level}.
	// The adaptive-thinking default (chat_models.py:2005-2010) fires only when
	// no explicit thinking is set and the model's profile declares xhigh
	// support — sending the adaptive+summarized shape to an older model (Opus
	// 4.5/4.6) is rejected with "adaptive thinking is not supported on this
	// model".
	if m.reasoningEffort != "" {
		payload.OutputConfig = map[string]any{"effort": m.reasoningEffort}
		if payload.Thinking == nil && m.profileSupportsXHighEffort() {
			payload.Thinking = map[string]any{"type": "adaptive", "display": "summarized"}
		}
	}
	if payload.Thinking != nil {
		applyThinkingConstraints(&payload)
	}
	if m.toolChoice != nil {
		payload.ToolChoice = m.toolChoice
	}
	if m.parallelToolCalls != nil {
		choice := payload.ToolChoice
		if choice == nil {
			choice = map[string]any{"type": "auto"}
		} else {
			// Copy so the constructor-level toolChoice map is never mutated.
			choice = cloneAnyMap(choice)
		}
		choice["disable_parallel_tool_use"] = !*m.parallelToolCalls
		payload.ToolChoice = choice
	}
	if m.contextManagement != nil {
		payload.ContextManagement = m.contextManagement
	}
	if m.inferenceGeo != "" {
		payload.InferenceGeo = m.inferenceGeo
	}
	return payload, nil
}

// anthropicProviderID is the models.dev provider ID keying the modelprofiles
// bundle (modelprofiles/profiles.json).
const anthropicProviderID = "anthropic"

// Beta headers required to send tool-change system blocks
// (chat_models.py:545-549 documents both as REQUIRED for
// tool_addition/tool_removal).
const (
	// toolChangeBeta enables tool_addition/tool_removal blocks referencing
	// already-bound tools (tool_reference).
	toolChangeBeta = "mid-conversation-tool-changes-2026-07-01"
	// inlineToolsBeta enables defining a tool inline in a tool_addition block
	// (tool_definition); it supersedes toolChangeBeta when both apply.
	inlineToolsBeta = "inline-tools-2026-09-15"
)

// requiredBetaHeaders mirrors Python's beta auto-append decision
// (chat_models.py:2140-2161): scan the in-place system turns (payload messages
// with role "system" — the top-level system field is NOT scanned) for
// tool-change blocks. An inline tool_definition requires the inline-tools beta
// (superseding the reference beta even when both block kinds are present);
// otherwise any tool_addition/tool_removal block requires the tool-changes
// beta, unless the inline beta is already caller-supplied. explicit holds the
// caller-supplied anthropic-beta flags. Returns the flags to append, if any.
func requiredBetaHeaders(messages []requestMessage, explicit []string) []string {
	hasToolChange := false
	hasInlineDefinition := false
	for _, message := range messages {
		if message.Role != "system" {
			continue
		}
		for _, block := range message.Content {
			switch block.Type {
			case "tool_addition", "tool_removal":
				hasToolChange = true
				if toolType, _ := block.Tool["type"].(string); toolType == "tool_definition" {
					hasInlineDefinition = true
				}
			}
		}
	}
	switch {
	case hasInlineDefinition:
		if !slices.Contains(explicit, inlineToolsBeta) {
			return []string{inlineToolsBeta}
		}
	case hasToolChange && !slices.Contains(explicit, inlineToolsBeta):
		if !slices.Contains(explicit, toolChangeBeta) {
			return []string{toolChangeBeta}
		}
	}
	return nil
}

// configWithRequiredBetas returns a copy of the model config with the required
// beta flags merged into the anthropic-beta header — the Go equivalent of
// Python appending to payload["betas"] (chat_models.py:2159-2161): the
// anthropic SDK turns that list into the comma-joined anthropic-beta header
// (anthropic-sdk-python resources/beta/messages/messages.py:1122-1124), which
// is exactly what WithBetaHeaders sets. Caller-supplied flags are preserved
// (append-if-absent), and the caller's config map is never mutated.
func (m ChatModel) configWithRequiredBetas(required []string) modelconfig.Config {
	cfg := m.config
	if len(required) == 0 {
		return cfg
	}
	merged := make(map[string]string, len(cfg.Headers)+1)
	maps.Copy(merged, cfg.Headers)
	flags := splitBetaHeader(merged[anthropicBetaHeader])
	for _, flag := range required {
		if !slices.Contains(flags, flag) {
			flags = append(flags, flag)
		}
	}
	merged[anthropicBetaHeader] = strings.Join(flags, ",")
	cfg.Headers = merged
	return cfg
}

// anthropicBetaHeader is the HTTP header carrying beta feature flags.
const anthropicBetaHeader = "anthropic-beta"

// splitBetaHeader splits a comma-joined anthropic-beta header value into its
// flags, dropping empties.
func splitBetaHeader(header string) []string {
	if header == "" {
		return nil
	}
	var flags []string
	for flag := range strings.SplitSeq(header, ",") {
		if flag = strings.TrimSpace(flag); flag != "" {
			flags = append(flags, flag)
		}
	}
	return flags
}

// midConversationGates reports whether the model's profile allows mid-
// conversation system messages and mid-conversation tool changes (Python
// #41180, langchain-anthropic chat_models.py:1901-1906 reads the same keys off
// self.profile). Model IDs are matched exactly, like Python's registry
// lookup; a profile that omits the keys allows neither, preserving the legacy
// hoisting behavior — that includes older and dated IDs in the bundle (e.g.
// claude-sonnet-4-5 and claude-sonnet-4-5-20250929 are both present in
// profiles.json but carry no mid_conversation keys) and IDs absent from the
// bundle entirely.
func (m ChatModel) midConversationGates() (system, tools bool) {
	profile, ok := modelprofiles.Lookup(anthropicProviderID, m.config.Model)
	if !ok {
		return false, false
	}
	return profile.Supports(modelprofiles.FieldMidConversationSystemMessages),
		profile.Supports(modelprofiles.FieldMidConversationTools)
}

// profileSupportsXHighEffort mirrors Python's gate `"xhigh" in
// _reasoning_effort_levels(self.profile)` (chat_models.py:2008): only models
// whose profile declares the xhigh level (Opus 4.7+, Sonnet 5 and newer) accept
// the adaptive+summarized thinking shape. Exact-ID lookup like
// midConversationGates — IDs absent from the bundle, or carrying no levels,
// report false, so older models keep their pre-effort payloads.
func (m ChatModel) profileSupportsXHighEffort() bool {
	profile, ok := modelprofiles.Lookup(anthropicProviderID, m.config.Model)
	if !ok {
		return false
	}
	return slices.Contains(profile.ReasoningEffortLevels(), "xhigh")
}

// previousTurnAllowsSystem mirrors Python's _previous_turn_allows_system
// (chat_models.py:668-680): a system message may be sent in place after a
// user turn (human or folded tool results), or after an assistant turn that
// ends with a server-side tool result block (any "*_tool_result" type). After
// a plain assistant turn it must be hoisted.
func previousTurnAllowsSystem(messages []requestMessage) bool {
	if len(messages) == 0 {
		return false
	}
	last := messages[len(messages)-1]
	if last.Role == "user" {
		return true
	}
	if last.Role != "assistant" || len(last.Content) == 0 {
		return false
	}
	return strings.HasSuffix(last.Content[len(last.Content)-1].Type, "_tool_result")
}

// inPlaceSystemTurns formats a run of deferred system messages into wire
// turns, mirroring Python's _merge_messages coalescing plus
// _format_in_place_system_messages (chat_models.py:329,679-707): the run
// becomes one system turn whose content is a block array (string contents
// become text blocks), except a single message with string content, which
// keeps the bare string. A run whose blocks all narrow away is omitted
// entirely rather than sent as an empty system turn.
func inPlaceSystemTurns(run []messages.Message, allowToolChanges bool, model string) ([]requestMessage, error) {
	if len(run) == 1 && len(run[0].ContentBlocks) == 0 {
		text := run[0].Content
		return []requestMessage{{Role: "system", Text: &text}}, nil
	}
	var blocks []contentBlock
	for _, message := range run {
		if message.Content != "" {
			blocks = append(blocks, contentBlock{Type: "text", Text: message.Content})
		}
		if len(message.ContentBlocks) > 0 {
			formatted, err := formatInPlaceSystemBlocks(message.ContentBlocks, allowToolChanges, model)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, formatted...)
		}
	}
	if len(blocks) == 0 {
		return nil, nil
	}
	return []requestMessage{{Role: "system", Content: blocks}}, nil
}

// warnSystemMessageHoisted mirrors Python's _warn_system_message_hoisted
// (chat_models.py:661-671): a non-leading SystemMessage was folded into the
// top-level system field and now applies to the entire conversation.
func warnSystemMessageHoisted(model string) {
	slog.Warn("anthropic: a non-leading system message was hoisted into the top-level "+
		"system field and now applies to the entire conversation; to send it in place, "+
		"use a model whose profile sets mid_conversation_system_messages and place it "+
		"after a human or tool message, either last or before an AI message",
		slog.String("model", model))
}

// warnToolChangeBlockDropped mirrors Python's tool-change drop warning
// (chat_models.py:623-637) extended with the mid_conversation_tools profile
// gate: tool_addition/tool_removal blocks only reach the wire on a system
// message sent in place on a model whose profile allows mid-conversation tool
// changes.
func warnToolChangeBlockDropped(blockType, model string) {
	slog.Warn("anthropic: tool-change block dropped: tool_addition/tool_removal blocks "+
		"are only valid on a system message sent in place (after a human or tool "+
		"message, either last or before an AI message) on a model whose profile sets "+
		"mid_conversation_tools",
		slog.String("block_type", blockType),
		slog.String("model", model))
}

// warnUnrecognizedSystemBlockDropped mirrors Python's unrecognized-block
// warning (chat_models.py:635-642): Anthropic accepts `text` in any system
// message plus tool_addition/tool_removal on a mid-conversation one, so any
// other block type is dropped rather than forwarded to an API rejection.
func warnUnrecognizedSystemBlockDropped(blockType, model string) {
	slog.Warn("anthropic: unrecognized system content block dropped: Anthropic accepts "+
		"text in any system message, plus tool_addition and tool_removal on a "+
		"mid-conversation one",
		slog.String("block_type", blockType),
		slog.String("model", model))
}

type requestPayload struct {
	Model             string           `json:"model"`
	MaxTokens         int              `json:"max_tokens"`
	Messages          []requestMessage `json:"messages"`
	System            any              `json:"system,omitempty"`
	Temperature       *float64         `json:"temperature,omitempty"`
	TopP              *float64         `json:"top_p,omitempty"`
	TopK              *int             `json:"top_k,omitempty"`
	StopSequences     []string         `json:"stop_sequences,omitempty"`
	Tools             []toolSpec       `json:"tools,omitempty"`
	ToolChoice        map[string]any   `json:"tool_choice,omitempty"`
	Stream            bool             `json:"stream,omitzero"`
	Thinking          map[string]any   `json:"thinking,omitempty"`
	OutputConfig      map[string]any   `json:"output_config,omitempty"`
	ContextManagement map[string]any   `json:"context_management,omitempty"`
	InferenceGeo      string           `json:"inference_geo,omitempty"`

	// requiredBetas carries beta flags the payload's in-place tool-change
	// system turns require; senders merge them into the anthropic-beta header
	// and route to the beta endpoint. Unexported, so never serialized.
	requiredBetas []string
}

// requestMessage is one turn of the messages array. Content carries block
// content. An in-place system turn whose source message had string content
// carries the bare string in Text instead — Python's _format_messages emits
// {"role":"system","content":"..."} for a single string SystemMessage sent in
// place (chat_models.py:1044) — which MarshalJSON renders as the JSON string.
type requestMessage struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
	Text    *string        `json:"-"`
}

// MarshalJSON renders the message, preferring the bare string form when Text
// is set.
func (m requestMessage) MarshalJSON() ([]byte, error) {
	if m.Text != nil {
		return json.Marshal(struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{m.Role, *m.Text})
	}
	type requestMessageFields requestMessage
	return json.Marshal(requestMessageFields(m))
}

type contentBlock struct {
	Type         string         `json:"type"`
	Text         string         `json:"text,omitempty"`
	ID           string         `json:"id,omitempty"`
	Name         string         `json:"name,omitempty"`
	Input        map[string]any `json:"input,omitempty"`
	Tool         map[string]any `json:"tool,omitempty"`
	ToolUseID    string         `json:"tool_use_id,omitempty"`
	Content      any            `json:"content,omitempty"`
	IsError      bool           `json:"is_error,omitzero"`
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
	InputTokens              int                  `json:"input_tokens"`
	OutputTokens             int                  `json:"output_tokens"`
	CacheReadInputTokens     int                  `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int                  `json:"cache_creation_input_tokens"`
	CacheCreation            cacheCreationPayload `json:"cache_creation"`
}

// cacheCreationPayload models the newer Anthropic usage.cache_creation object,
// which breaks prompt-cache creation tokens down by TTL: 5-minute and 1-hour
// ephemeral caches.
type cacheCreationPayload struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
}

// total returns the cache-creation tokens attributable to the specific
// ephemeral TTLs.
func (c cacheCreationPayload) total() int {
	return c.Ephemeral5mInputTokens + c.Ephemeral1hInputTokens
}

// merge fills zero fields of the receiver with non-zero values from next.
// Streaming message_delta events only carry the usage fields that changed
// (older API shapes report just output_tokens there), so zero values in next
// must not clobber what message_start already reported.
func (u *usagePayload) merge(next usagePayload) {
	if next.InputTokens != 0 {
		u.InputTokens = next.InputTokens
	}
	if next.OutputTokens != 0 {
		u.OutputTokens = next.OutputTokens
	}
	if next.CacheReadInputTokens != 0 {
		u.CacheReadInputTokens = next.CacheReadInputTokens
	}
	if next.CacheCreationInputTokens != 0 {
		u.CacheCreationInputTokens = next.CacheCreationInputTokens
	}
	if next.CacheCreation != (cacheCreationPayload{}) {
		u.CacheCreation = next.CacheCreation
	}
}

// isZero reports whether the payload carries no token counts at all; some
// gateways emit empty usage objects on message_delta.
func (u usagePayload) isZero() bool {
	return u == usagePayload{}
}

// toUsageMetadata mirrors Python's _create_usage_metadata
// (langchain_anthropic/chat_models.py:2356): Anthropic's input_tokens excludes
// cached tokens, so cache_read and cache_creation counts are added back to get
// the true input total. When the newer usage.cache_creation TTL breakdown
// reports specific ephemeral tokens, Python zeroes the generic cache_creation
// count to avoid double counting; core's messages.InputTokenDetails has no
// ephemeral_5m/1h fields (only the flat cache_read_input_tokens /
// cache_creation_input_tokens ints), so the closest expression here folds the
// specific ephemeral total into CacheCreationInputTokens.
func (u usagePayload) toUsageMetadata() messages.UsageMetadata {
	cacheCreation := u.CacheCreationInputTokens
	if specific := u.CacheCreation.total(); specific > 0 {
		cacheCreation = specific
	}
	input := u.InputTokens + u.CacheReadInputTokens + cacheCreation
	meta := messages.UsageMetadata{
		InputTokens:  input,
		OutputTokens: u.OutputTokens,
		TotalTokens:  input + u.OutputTokens,
	}
	if u.CacheReadInputTokens != 0 || cacheCreation != 0 {
		meta.InputTokenDetails = &messages.InputTokenDetails{
			CacheReadInputTokens:     u.CacheReadInputTokens,
			CacheCreationInputTokens: cacheCreation,
		}
	}
	return meta
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
			blocks = append(blocks, messages.TextBlock{
				Text: content.Text,
			})
		case "tool_use":
			blocks = append(blocks, messages.ToolCallBlock{
				ID:   content.ID,
				Name: content.Name,
				Args: content.Input,
			})
			toolCalls = append(toolCalls, messages.ToolCall{
				ID:   content.ID,
				Name: content.Name,
				Args: content.Input,
			})
		case "thinking":
			blocks = append(blocks, messages.ReasoningBlock{
				Reasoning: content.Thinking,
				Extras:    map[string]any{"signature": content.Signature},
			})
		case "redacted_thinking":
			extras := map[string]any{"data": content.Data}
			if content.ID != "" {
				extras["id"] = content.ID
			}
			blocks = append(blocks, messages.ReasoningBlock{
				Extras: extras,
			})
		default:
			raw, _ := json.Marshal(content)
			var m map[string]any
			_ = json.Unmarshal(raw, &m)
			blocks = append(blocks, messages.ParseContentBlock(m))
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
	return providerutil.Emit(ctx, cfg, kind, input, output, err)
}

func cloneMetadata(metadata map[string]any) map[string]any {
	return providerutil.CloneMetadata(metadata)
}

// cloneAnyMap returns a shallow defensive copy of an arbitrary config map.
func cloneAnyMap(m map[string]any) map[string]any {
	return maps.Clone(m)
}

func cloneBoolPtr(v *bool) *bool {
	if v == nil {
		return nil
	}
	out := *v
	return &out
}
