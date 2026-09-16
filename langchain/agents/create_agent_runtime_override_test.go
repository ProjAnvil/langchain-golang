package agents

// Runtime-override wiring tests (T11 + T13): middleware runtime overrides of
// response_format / tool_choice / model_settings reaching the model BIND (not
// just the request), per-call AutoStrategy re-resolution under DynamicModel,
// ProviderStrategy binding on the streaming path, and cache-key separation
// between different effective response formats.
//
// Python reference: langchain_v1 factory.py `_get_bound_model`
// (factory.py:1323-1404) — every model call (streaming and non-streaming
// alike) re-normalizes request.response_format, merges
// ProviderStrategy.to_model_kwargs() with request.model_settings, and passes
// final_tools + tool_choice into bind_tools.

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/core/caches"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/modelprofiles"
)

// errNoMoreResponses mirrors sequenceModel's exhaustion error.
func errNoMoreResponses(call int) error {
	return fmt.Errorf("bindRecordingModel: no more responses (call %d)", call)
}

// bindRecordingModel is a language.ChatModel test double that implements the
// optional capabilities the runtime-override wiring detects —
// language.ToolBinder and ModelSettingsBinder — and records every capability
// call (BindModelSettings, BindToolsWithOptions/BindTools, Invoke, Stream) so
// tests can assert what the agent's model node actually threaded into the
// model at bind time.
type bindRecordingModel struct {
	mu        sync.Mutex
	responses []messages.Message
	idx       int
	caps      language.ChatModelCapabilities
	profile   modelprofiles.Profile // optional explicit profile for AutoStrategy re-resolution

	settings   []map[string]any
	bindCalls  []language.BindToolsOptions
	boundTools [][]string
	invokes    int
	streams    int
}

func newBindRecordingModel(caps language.ChatModelCapabilities, responses ...messages.Message) *bindRecordingModel {
	return &bindRecordingModel{responses: responses, caps: caps}
}

func (m *bindRecordingModel) nextResponse() (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.idx >= len(m.responses) {
		return messages.Message{}, errNoMoreResponses(m.idx + 1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

func (m *bindRecordingModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	m.invokes++
	m.mu.Unlock()
	return m.nextResponse()
}

func (m *bindRecordingModel) Batch(_ context.Context, _ [][]messages.Message, _ ...runnables.Option) ([]messages.Message, error) {
	return nil, errNoMoreResponses(0)
}

func (m *bindRecordingModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	m.mu.Lock()
	m.streams++
	m.mu.Unlock()
	resp, err := m.nextResponse()
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{resp}), nil
}

func (m *bindRecordingModel) InputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *bindRecordingModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *bindRecordingModel) recordBind(tools []coretools.Tool, opts language.BindToolsOptions) {
	m.mu.Lock()
	defer m.mu.Unlock()
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name())
	}
	m.bindCalls = append(m.bindCalls, opts)
	m.boundTools = append(m.boundTools, names)
}

func (m *bindRecordingModel) BindTools(boundTools []coretools.Tool) (language.ChatModel, error) {
	m.recordBind(boundTools, language.BindToolsOptions{})
	return m, nil
}

func (m *bindRecordingModel) BindToolsWithOptions(boundTools []coretools.Tool, opts language.BindToolsOptions) (language.ChatModel, error) {
	m.recordBind(boundTools, opts)
	return m, nil
}

// BindModelSettings implements ModelSettingsBinder so the model node's
// per-call kwargs merge (ProviderStrategy response_format + middleware
// model_settings) is observable.
func (m *bindRecordingModel) BindModelSettings(settings map[string]any) (language.ChatModel, error) {
	m.mu.Lock()
	m.settings = append(m.settings, settings)
	m.mu.Unlock()
	return m, nil
}

func (m *bindRecordingModel) Capabilities() language.ChatModelCapabilities {
	return m.caps
}

// ModelProfile exposes the model-profile view AutoStrategy re-resolution uses
// (the SupportsProviderStrategy profile path). An explicit profile wins; the
// default is the capabilities-derived profile.
func (m *bindRecordingModel) ModelProfile() modelprofiles.Profile {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.profile != nil {
		return m.profile
	}
	return m.caps.ModelProfile()
}

func (m *bindRecordingModel) recordedSettings() []map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]map[string]any(nil), m.settings...)
}

func (m *bindRecordingModel) recordedBindCalls() []language.BindToolsOptions {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]language.BindToolsOptions(nil), m.bindCalls...)
}

func (m *bindRecordingModel) recordedBoundToolNames() [][]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([][]string(nil), m.boundTools...)
}

func (m *bindRecordingModel) invokeCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.invokes
}

func (m *bindRecordingModel) streamCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.streams
}

type responseFormatOverrideMiddleware struct{ rf any }

func (m responseFormatOverrideMiddleware) WrapModelCall(
	ctx context.Context,
	request middleware.ModelRequest,
	handler middleware.ModelHandler,
) (middleware.ModelResponse, error) {
	next, err := request.Override(middleware.WithResponseFormat(m.rf))
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	return handler(ctx, next)
}

type modelSettingsOverrideMiddleware struct{ settings map[string]any }

func (m modelSettingsOverrideMiddleware) WrapModelCall(
	ctx context.Context,
	request middleware.ModelRequest,
	handler middleware.ModelHandler,
) (middleware.ModelResponse, error) {
	next, err := request.Override(middleware.WithModelSettings(m.settings))
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	return handler(ctx, next)
}

// TestMiddlewareResponseFormatOverrideSwitchesToolToProvider proves a
// middleware runtime response_format override REPLACES the build-time
// strategy at bind time (factory.py:1323-1344): a ToolStrategy-configured
// agent whose middleware overrides to a ProviderStrategy must (a) NOT bind
// the structured-output tool anymore, (b) NOT force tool_choice "any", and
// (c) bind the ProviderStrategy response_format kwargs into the model via
// ModelSettingsBinder. The JSON text response then surfaces through the
// ProviderStrategy post-hoc parse.
func TestMiddlewareResponseFormatOverrideSwitchesToolToProvider(t *testing.T) {
	toolStrategy := NewToolStrategy(answerSchema()) // structured tool "Answer"
	providerOverride := NewProviderStrategy(answerSchema())

	model := newBindRecordingModel(
		language.ChatModelCapabilities{ToolCalling: true},
		messages.AI(`{"text": "42"}`),
	)

	agent, err := CreateAgent(model, nil,
		WithAgentResponseFormat(toolStrategy),
		WithAgentMiddleware(responseFormatOverrideMiddleware{rf: providerOverride}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("what is the answer?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["text"] != "42" {
		t.Fatalf("expected ProviderStrategy structured_response text=42 (JSON parse), got %#v", state["structured_response"])
	}

	// The ProviderStrategy response_format kwargs must reach the bind.
	settings := model.recordedSettings()
	if len(settings) != 1 {
		t.Fatalf("expected exactly one BindModelSettings call, got %d", len(settings))
	}
	rf, ok := settings[0]["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("expected merged settings to carry response_format, got %#v", settings[0])
	}
	if rf["type"] != "json_schema" {
		t.Fatalf("expected response_format type json_schema, got %#v", rf)
	}
	jsonSchema, _ := rf["json_schema"].(map[string]any)
	if jsonSchema["name"] != "Answer" {
		t.Fatalf("expected json_schema name Answer, got %#v", jsonSchema)
	}

	// The structured-output tool must NOT be bound under the provider
	// strategy, and no tool_choice may be forced.
	binds := model.recordedBindCalls()
	if len(binds) != 0 {
		t.Fatalf("expected no tool binds (structured tool filtered, no other tools), got %d: %#v / tools=%v",
			len(binds), binds, model.recordedBoundToolNames())
	}
}

// TestMiddlewareModelSettingsOverrideFlowsToBind proves middleware
// model_settings reach the model bind merged over the strategy kwargs
// (factory.py:1361: `bind_kwargs = {**kwargs, **request.model_settings}` —
// request settings win conflicts), and that the parallel_tool_calls kwarg
// maps onto language.BindToolsOptions alongside tool_choice.
func TestMiddlewareModelSettingsOverrideFlowsToBind(t *testing.T) {
	parallel := true
	model := newBindRecordingModel(
		language.ChatModelCapabilities{ToolCalling: true},
		messages.AI(`{"temperature":1,"condition":"x"}`),
	)

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)},
		WithAgentResponseFormat(NewProviderStrategy(weatherSchema())),
		WithAgentMiddleware(modelSettingsOverrideMiddleware{
			settings: map[string]any{"temperature": 0.2, "parallel_tool_calls": parallel},
		}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	settings := model.recordedSettings()
	if len(settings) != 1 {
		t.Fatalf("expected one BindModelSettings call, got %d", len(settings))
	}
	if settings[0]["temperature"] != 0.2 {
		t.Fatalf("expected middleware temperature=0.2 to reach the bind, got %#v", settings[0])
	}
	rf, ok := settings[0]["response_format"].(map[string]any)
	if !ok || rf["type"] != "json_schema" {
		t.Fatalf("expected strategy response_format kwargs merged under middleware settings, got %#v", settings[0])
	}

	binds := model.recordedBindCalls()
	if len(binds) != 1 {
		t.Fatalf("expected one tool bind, got %d", len(binds))
	}
	if binds[0].ParallelToolCalls == nil || *binds[0].ParallelToolCalls != true {
		t.Fatalf("expected parallel_tool_calls=true threaded into BindToolsOptions, got %#v", binds[0])
	}
	if binds[0].ToolChoice != "" {
		t.Fatalf("expected no tool_choice forced under ProviderStrategy, got %q", binds[0].ToolChoice)
	}
}

// TestMiddlewareToolChoiceOverrideReachesFakeChatModel is the FakeChatModel
// variant of the T6 tool_choice threading: a middleware WithToolChoice
// override must surface on the fake's BoundToolChoice() (the value its
// BindToolsWithOptions received). The response is a terminal text answer —
// FakeChatModel.BindTools returns a fresh copy per bind, so a queued tool call
// would replay forever across the loop's re-binds; one model call suffices to
// observe the recorded choice.
func TestMiddlewareToolChoiceOverrideReachesFakeChatModel(t *testing.T) {
	model := language.NewFakeChatModel(
		language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
		language.WithResponses(messages.AI("ok")),
	)

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)},
		WithAgentMiddleware(toolChoiceOverrideMiddleware{choice: "none"}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got := model.BoundToolChoice(); got != language.ToolChoiceNone {
		t.Fatalf("FakeChatModel BoundToolChoice = %q, want %q", got, language.ToolChoiceNone)
	}
}

// TestAutoStrategyDynamicModelRechecksEffectiveStrategyPerCall covers the
// audit gap "AutoStrategy 按静态 build-time model 检测，DynamicModel 换模型后不重检":
// with a DynamicModel resolver, an AutoStrategy must be re-resolved against
// the model of EACH call (factory.py:1331-1341). Call 1 lands on a
// tool-calling model (ToolStrategy: structured tool bound + tool_choice "any");
// call 2 lands on a structured-output model (ProviderStrategy: structured
// tool filtered out, response_format kwargs bound, JSON text parsed).
func TestAutoStrategyDynamicModelRechecksEffectiveStrategyPerCall(t *testing.T) {
	staticModel := language.NewFakeChatModel(
		language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
	) // only drives the build-time eager resolution (ToolStrategy setup)

	answerToolName := NewToolStrategy(answerSchema()).SchemaSpecs[0].Name
	modelA := newBindRecordingModel(
		language.ChatModelCapabilities{ToolCalling: true},
		messages.Message{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: answerToolName, Args: map[string]any{"text": "from-tool"}},
			},
		},
	)
	modelB := newBindRecordingModel(
		language.ChatModelCapabilities{StructuredOutput: true},
		messages.AI(`{"text": "from-provider"}`),
	)

	var calls atomic.Int32
	agent, err := CreateAgent(staticModel, nil,
		WithAgentResponseFormat(NewAutoStrategy(answerSchema())),
		WithAgentDynamicModel(func(_ map[string]any, _ runtime.Runtime) language.ChatModel {
			if calls.Add(1) == 1 {
				return modelA
			}
			return modelB
		}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	msgs := []messages.Message{messages.Human("what is the answer?")}

	// Call 1: tool-calling model -> ToolStrategy effective.
	state1, err := agent.InvokeWithState(t.Context(), msgs)
	if err != nil {
		t.Fatalf("invoke 1: %v", err)
	}
	if s, ok := state1["structured_response"].(map[string]any); !ok || s["text"] != "from-tool" {
		t.Fatalf("invoke 1: expected tool-strategy structured_response, got %#v", state1["structured_response"])
	}
	bindsA := modelA.recordedBindCalls()
	if len(bindsA) != 1 || bindsA[0].ToolChoice != language.ToolChoiceAny {
		t.Fatalf("invoke 1: expected one bind with tool_choice any, got %#v", bindsA)
	}
	if names := modelA.recordedBoundToolNames(); len(names) != 1 || names[0][0] != answerToolName {
		t.Fatalf("invoke 1: expected the structured tool bound, got %v", names)
	}
	if settings := modelA.recordedSettings(); len(settings) != 0 {
		t.Fatalf("invoke 1: expected no model-settings bind for ToolStrategy, got %#v", settings)
	}

	// Call 2: structured-output model -> ProviderStrategy effective (re-check).
	state2, err := agent.InvokeWithState(t.Context(), msgs)
	if err != nil {
		t.Fatalf("invoke 2: %v", err)
	}
	if s, ok := state2["structured_response"].(map[string]any); !ok || s["text"] != "from-provider" {
		t.Fatalf("invoke 2: expected provider-strategy structured_response from JSON text, got %#v", state2["structured_response"])
	}
	if binds := modelB.recordedBindCalls(); len(binds) != 0 {
		t.Fatalf("invoke 2: structured tool must be filtered from the bind under ProviderStrategy, got %#v / %v",
			binds, modelB.recordedBoundToolNames())
	}
	settingsB := modelB.recordedSettings()
	if len(settingsB) != 1 {
		t.Fatalf("invoke 2: expected the response_format kwargs bound via ModelSettingsBinder, got %#v", settingsB)
	}
	if rf, ok := settingsB[0]["response_format"].(map[string]any); !ok || rf["type"] != "json_schema" {
		t.Fatalf("invoke 2: expected response_format json_schema kwargs at the bind, got %#v", settingsB[0])
	}
}

// TestStreamingProviderStrategyBindsResponseFormat covers T13: the streaming
// path (invokeModelStreaming) must bind the ProviderStrategy response_format
// kwargs into the model like the non-streaming path does, instead of relying
// solely on the post-hoc detectStructuredOutput parse. The recorder asserts
// BindModelSettings ran BEFORE Stream, and the terminal StreamEnd state still
// carries the parsed structured_response (the retained fallback).
func TestStreamingProviderStrategyBindsResponseFormat(t *testing.T) {
	model := newBindRecordingModel(
		language.ChatModelCapabilities{Streaming: true},
		messages.AI(`{"temperature":72,"condition":"sunny"}`),
	)

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(NewProviderStrategy(weatherSchema())))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	stream, err := agent.StreamEvents(t.Context(), []messages.Message{messages.Human("weather in Tokyo?")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)

	var end *StreamEvent
	for i := range events {
		if events[i].Type == StreamEnd {
			end = &events[i]
		}
	}
	if end == nil {
		t.Fatalf("expected a terminal StreamEnd event, got %v", eventTypes(events))
	}
	if end.Err != nil {
		t.Fatalf("stream errored: %v", end.Err)
	}

	// The bind must have carried the response_format kwargs, and the call must
	// have gone through Stream (not the non-streaming Invoke fallback).
	settings := model.recordedSettings()
	if len(settings) != 1 {
		t.Fatalf("expected BindModelSettings on the streaming path, got %#v", settings)
	}
	if rf, ok := settings[0]["response_format"].(map[string]any); !ok || rf["type"] != "json_schema" {
		t.Fatalf("expected response_format json_schema kwargs bound before Stream, got %#v", settings[0])
	}
	if model.streamCount() != 1 {
		t.Fatalf("expected exactly one Stream call, got %d", model.streamCount())
	}
	if model.invokeCount() != 0 {
		t.Fatalf("expected zero Invoke calls on the streaming path, got %d", model.invokeCount())
	}

	// Post-hoc parse fallback retained: the streamed JSON surfaces as
	// structured_response in the terminal state.
	structured, ok := end.State["structured_response"].(map[string]any)
	if !ok || structured["condition"] != "sunny" {
		t.Fatalf("expected structured_response.condition=sunny in the terminal state, got %#v", end.State["structured_response"])
	}
}

// TestCacheKeyDistinguishesRuntimeEffectiveStrategy verifies the cache-key
// semantics after the runtime strategy normalization participates: two model
// calls with identical prompts but DIFFERENT effective response formats (Auto
// re-resolved per DynamicModel call) must produce different cache keys, so
// the second call is not served the first call's response (Python: the cache
// sits inside the chat-model layer and keys on the serialized BOUND model —
// langchain_core caches.py lookup(prompt, llm_string) — so different bind
// kwargs can never collide; the Go port reproduces this by hashing the
// request's ModelSettings, which the model node seeds with the effective
// strategy's kwargs, plus ToolChoice).
func TestCacheKeyDistinguishesRuntimeEffectiveStrategy(t *testing.T) {
	cache, err := caches.NewInMemoryCache()
	if err != nil {
		t.Fatalf("NewInMemoryCache: %v", err)
	}
	staticModel := language.NewFakeChatModel(
		language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
	)

	modelA := newBindRecordingModel(
		language.ChatModelCapabilities{ToolCalling: true},
		messages.AI("plain text answer"),
	)
	modelB := newBindRecordingModel(
		language.ChatModelCapabilities{StructuredOutput: true},
		messages.AI(`{"text": "structured answer"}`),
	)

	var calls atomic.Int32
	agent, err := CreateAgent(staticModel, []coretools.Tool{newEchoTool(t)},
		WithAgentResponseFormat(NewAutoStrategy(answerSchema())),
		WithAgentDynamicModel(func(_ map[string]any, _ runtime.Runtime) language.ChatModel {
			if calls.Add(1) == 1 {
				return modelA
			}
			return modelB
		}),
		WithAgentCache(cache),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	msgs := []messages.Message{messages.Human("same question")}

	if _, err := agent.Invoke(t.Context(), msgs); err != nil {
		t.Fatalf("invoke 1: %v", err)
	}

	state2, err := agent.InvokeWithState(t.Context(), msgs)
	if err != nil {
		t.Fatalf("invoke 2: a false cache hit served the tool-format response to the provider-format call: %v", err)
	}
	structured, ok := state2["structured_response"].(map[string]any)
	if !ok || structured["text"] != "structured answer" {
		t.Fatalf("expected invoke 2 to run the provider-format model call and parse its JSON, got %#v", state2["structured_response"])
	}
	if modelB.invokeCount() != 1 {
		t.Fatalf("expected invoke 2 to reach the model (distinct cache key), invokes=%d", modelB.invokeCount())
	}
}

// TestMiddlewareToolStrategyOverrideRequiresUpfrontDeclaredSpecs mirrors
// factory.py:1375-1385: middleware may narrow a ToolStrategy override to a
// subset of the structured tools declared at build time, but introducing a
// structured tool that was NOT declared upfront is a hard error.
func TestMiddlewareToolStrategyOverrideRequiresUpfrontDeclaredSpecs(t *testing.T) {
	model := newBindRecordingModel(
		language.ChatModelCapabilities{ToolCalling: true},
		messages.AI("never reached"),
	)

	agent, err := CreateAgent(model, nil,
		WithAgentResponseFormat(NewProviderStrategy(weatherSchema())),
		WithAgentMiddleware(responseFormatOverrideMiddleware{rf: NewToolStrategy(answerSchema())}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	_, err = agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected an error for a ToolStrategy override introducing an undeclared structured tool")
	}
	if !strings.Contains(err.Error(), "wasn't declared in the original response format") {
		t.Fatalf("expected the undeclared-structured-tool error, got %v", err)
	}
}

// TestMiddlewareToolStrategyOverrideNarrowsBoundStructuredTools mirrors
// factory.py:1375-1385: a middleware ToolStrategy override that narrows to a
// declared subset must bind ONLY the retained structured tools to the model
// (the effective strategy's own narrow set, like the AutoStrategy synthesized
// path) and detect structured output against that narrow set alone. A model
// call into the EXCLUDED structured tool therefore no longer surfaces its
// args as structured_response; under Go's routing the loop just returns to
// the model (the excluded call is not dispatched like an unknown client tool
// — buildRouteStructuredOnly/buildRouteAfterModel key structured calls off
// the build-time bindings), so the model re-answers through the retained
// tool.
func TestMiddlewareToolStrategyOverrideNarrowsBoundStructuredTools(t *testing.T) {
	buildTime := NewToolStrategy(schema.Schema{"oneOf": []any{weatherSchema(), locationSchema()}})
	narrowed := NewToolStrategy(weatherSchema())

	model := newBindRecordingModel(
		language.ChatModelCapabilities{ToolCalling: true},
		// Call 1: the model (incorrectly) answers through the EXCLUDED
		// structured tool.
		messages.Message{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "location_schema", Args: map[string]any{"city": "Paris", "country": "France"}},
			},
		},
		// Call 2: it answers through the RETAINED structured tool.
		messages.Message{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_2", Name: "weather_schema", Args: map[string]any{"temperature": 72, "condition": "sunny"}},
			},
		},
	)

	agent, err := CreateAgent(model, nil,
		WithAgentResponseFormat(buildTime),
		WithAgentMiddleware(responseFormatOverrideMiddleware{rf: narrowed}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("weather in Tokyo?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	// Only the retained structured tool may be bound; the excluded one must
	// not leak from the build-time set into the per-call bind.
	binds := model.recordedBoundToolNames()
	if len(binds) == 0 {
		t.Fatal("expected the structured tool(s) to be bound at least once")
	}
	for _, names := range binds {
		if len(names) != 1 || names[0] != "weather_schema" {
			t.Fatalf("expected only the retained structured tool bound, got %v", binds)
		}
	}

	// The excluded-tool call must NOT produce a structured_response for the
	// location schema: the run only ends through the retained tool, after a
	// second model call.
	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["condition"] != "sunny" {
		t.Fatalf("expected structured_response from the retained weather_schema call, got %#v", state["structured_response"])
	}
	if model.invokeCount() != 2 {
		t.Fatalf("expected the excluded-tool call to loop back to the model (2 invokes), got %d", model.invokeCount())
	}
}

// compile-time guard: the recorder really implements the capabilities under test.
var (
	_ language.ToolBinder  = (*bindRecordingModel)(nil)
	_ ModelSettingsBinder  = (*bindRecordingModel)(nil)
	_ modelProfileProvider = (*bindRecordingModel)(nil)
	_ language.ChatModel   = (*bindRecordingModel)(nil)
)
