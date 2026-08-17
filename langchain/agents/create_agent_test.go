package agents

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/caches"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/prompts"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	storepkg "github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/partners/openai"
)

// sequenceModel is a minimal test double implementing language.ChatModel. It
// returns Responses in order and, unlike language.FakeChatModel, returns
// itself from BindTools (rather than a fresh copy) so response/invocation
// state is shared across the repeated bind+invoke cycles CreateAgent's model
// node performs on every loop iteration.
type sequenceModel struct {
	mu          sync.Mutex
	responses   []messages.Message
	idx         int
	boundTools  []coretools.Tool
	invocations [][]messages.Message
}

func (m *sequenceModel) Invoke(ctx context.Context, input []messages.Message, opts ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.invocations = append(m.invocations, append([]messages.Message(nil), input...))
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("sequenceModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

func (m *sequenceModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		out[i], err = m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *sequenceModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return nil, fmt.Errorf("sequenceModel: streaming not supported")
}

func (m *sequenceModel) InputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *sequenceModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *sequenceModel) BindTools(boundTools []coretools.Tool) (language.ChatModel, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.boundTools = boundTools
	return m, nil
}

func (m *sequenceModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

func newEchoTool(t *testing.T) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewSimple("echo", "echoes its input", func(_ context.Context, input string) (coretools.Result, error) {
		return coretools.Result{Content: "echo:" + input}, nil
	})
	if err != nil {
		t.Fatalf("new echo tool: %v", err)
	}
	return tool
}

func TestCreateAgentToolLoop(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
		messages.AI("done"),
	}}

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) != 4 {
		t.Fatalf("expected 4 messages, got %d: %#v", len(out), out)
	}
	if out[0].Role != messages.RoleHuman {
		t.Fatalf("message 0 role mismatch: %v", out[0].Role)
	}
	if out[1].Role != messages.RoleAI || len(out[1].ToolCalls) != 1 {
		t.Fatalf("message 1 mismatch: %#v", out[1])
	}
	if out[2].Role != messages.RoleTool || out[2].Content != "echo:hi" {
		t.Fatalf("message 2 mismatch: %#v", out[2])
	}
	if out[3].Role != messages.RoleAI || out[3].Content != "done" {
		t.Fatalf("message 3 mismatch: %#v", out[3])
	}
	if len(model.invocations) != 2 {
		t.Fatalf("expected 2 model invocations, got %d", len(model.invocations))
	}
}

// TestCreateAgentDictToolSpecs covers the dict tool form (Python's
// `tools: Sequence[... | dict]`): WithAgentToolSpecs converts each
// map[string]any spec into a tool whose name/description/args schema come from
// the dict, and binds them to the model alongside the positional tool list.
func TestCreateAgentDictToolSpecs(t *testing.T) {
	specs := []map[string]any{
		{
			"name":        "get_weather",
			"description": "Get the weather for a location",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"location": map[string]any{"type": "string"}},
				"required":   []any{"location"},
			},
		},
		{
			"name":        "get_time",
			"description": "Get the current time",
		},
	}

	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateAgent(model, nil, WithAgentToolSpecs(specs...))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(model.boundTools) != 2 {
		t.Fatalf("expected 2 dict-spec tools bound to the model, got %d", len(model.boundTools))
	}
	if got := model.boundTools[0].Name(); got != "get_weather" {
		t.Fatalf("expected tool 0 name %q, got %q", "get_weather", got)
	}
	if got := model.boundTools[0].Description(); got != "Get the weather for a location" {
		t.Fatalf("expected tool 0 description %q, got %q", "Get the weather for a location", got)
	}
	if got := model.boundTools[1].Name(); got != "get_time" {
		t.Fatalf("expected tool 1 name %q, got %q", "get_time", got)
	}
	if got := model.boundTools[1].Description(); got != "Get the current time" {
		t.Fatalf("expected tool 1 description %q, got %q", "Get the current time", got)
	}
	if !reflect.DeepEqual(model.boundTools[0].ArgsSchema(), schema.Schema(specs[0]["parameters"].(map[string]any))) {
		t.Fatalf("expected tool 0 args schema to carry the dict parameters, got %#v", model.boundTools[0].ArgsSchema())
	}
}

func TestCreateAgentNoTools(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}

	agent, err := CreateAgent(model, nil)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) != 2 || out[1].Content != "hello" {
		t.Fatalf("unexpected result: %#v", out)
	}
}

func TestCreateAgentSystemPrompt(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}

	agent, err := CreateAgent(model, nil, WithAgentSystemPrompt("be nice"))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(model.invocations) != 1 {
		t.Fatalf("expected 1 invocation, got %d", len(model.invocations))
	}
	invoked := model.invocations[0]
	if len(invoked) != 2 || invoked[0].Role != messages.RoleSystem || invoked[0].Content != "be nice" {
		t.Fatalf("expected leading system message, got %#v", invoked)
	}
}

// recordingWrapModelCallMiddleware appends "<tag>:before"/"<tag>:after" to a
// shared log around the model call, used to assert middleware composition
// order (first-listed middleware is outermost).
type recordingWrapModelCallMiddleware struct {
	tag string
	log *[]string
}

func (r *recordingWrapModelCallMiddleware) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	*r.log = append(*r.log, r.tag+":before")
	resp, err := handler(ctx, request)
	*r.log = append(*r.log, r.tag+":after")
	return resp, err
}

func TestCreateAgentWrapModelCallOrdering(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}
	var log []string

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		&recordingWrapModelCallMiddleware{tag: "A", log: &log},
		&recordingWrapModelCallMiddleware{tag: "B", log: &log},
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	want := []string{"A:before", "B:before", "B:after", "A:after"}
	if len(log) != len(want) {
		t.Fatalf("log mismatch: got %v, want %v", log, want)
	}
	for i := range want {
		if log[i] != want[i] {
			t.Fatalf("log mismatch at %d: got %v, want %v", i, log, want)
		}
	}
}

func TestCreateAgentModelCallLimitMiddlewareEndsBeforeModelCall(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("should not be reached")}}
	limit := 0
	limitMW, err := middleware.NewModelCallLimitMiddleware(&limit, nil, "end")
	if err != nil {
		t.Fatalf("new model call limit middleware: %v", err)
	}

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(limitMW))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(model.invocations) != 0 {
		t.Fatalf("expected model to never be invoked, got %d invocations", len(model.invocations))
	}
	if len(out) != 2 || out[1].Role != messages.RoleAI {
		t.Fatalf("expected a limit-exceeded AI message appended, got %#v", out)
	}
}

func TestCreateAgentToolCallLimitMiddlewareEndsRun(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
		messages.AI("should not be reached"),
	}}
	limit := 0
	limitMW, err := middleware.NewToolCallLimitMiddleware("echo", &limit, nil, "end")
	if err != nil {
		t.Fatalf("new tool call limit middleware: %v", err)
	}

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)}, WithAgentMiddleware(limitMW))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(model.invocations) != 1 {
		t.Fatalf("expected exactly 1 model invocation, got %d", len(model.invocations))
	}
	if len(out) != 4 {
		t.Fatalf("expected 4 messages (human, ai-tool-call, blocked-tool-error, final ai), got %d: %#v", len(out), out)
	}
	if out[2].Role != messages.RoleTool {
		t.Fatalf("expected blocked tool call to produce a tool error message, got %#v", out[2])
	}
	if out[3].Role != messages.RoleAI {
		t.Fatalf("expected a final limit-exceeded AI message, got %#v", out[3])
	}
}

func TestCreateAgentRequiresModel(t *testing.T) {
	if _, err := CreateAgent(nil, nil); err == nil {
		t.Fatal("expected error for nil model")
	}
}

func answerSchema() schema.Schema {
	s := schema.Object(map[string]schema.Schema{
		"text": schema.String("the answer text"),
	}, "text")
	s["title"] = "Answer"
	return s
}

func TestCreateAgentToolStrategyStructuredOutput(t *testing.T) {
	strategy := NewToolStrategy(answerSchema())
	toolName := strategy.SchemaSpecs[0].Name

	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: toolName, Args: map[string]any{"text": "42"}},
			},
		},
	}}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("what is the answer?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["text"] != "42" {
		t.Fatalf("expected structured_response with text=42, got %#v", state["structured_response"])
	}

	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (human, ai tool call, tool result), got %d: %#v", len(msgs), msgs)
	}
	if msgs[2].Role != messages.RoleTool {
		t.Fatalf("expected tool message for structured output, got %#v", msgs[2])
	}
	if len(model.invocations) != 1 {
		t.Fatalf("expected exactly one model invocation, got %d", len(model.invocations))
	}
}

func TestCreateAgentToolStrategyMultipleStructuredOutputsError(t *testing.T) {
	strategy := NewToolStrategy(answerSchema())
	toolName := strategy.SchemaSpecs[0].Name

	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: toolName, Args: map[string]any{"text": "42"}},
				{ID: "call_2", Name: toolName, Args: map[string]any{"text": "43"}},
			},
		},
	}}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	var multiErr *MultipleStructuredOutputsError
	if !errors.As(err, &multiErr) {
		t.Fatalf("expected MultipleStructuredOutputsError, got %v", err)
	}
}

func TestCreateAgentProviderStrategyStructuredOutput(t *testing.T) {
	strategy := NewProviderStrategy(answerSchema(), WithStrict(true))

	model := &sequenceModel{responses: []messages.Message{
		messages.AI(`{"text": "42"}`),
	}}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("what is the answer?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["text"] != "42" {
		t.Fatalf("expected structured_response with text=42, got %#v", state["structured_response"])
	}
}

func TestCreateAgentRejectsUnsupportedResponseFormat(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hi")}}
	if _, err := CreateAgent(model, nil, WithAgentResponseFormat("not-a-strategy")); err == nil {
		t.Fatal("expected error for unsupported ResponseFormat type")
	}
}

// TestCreateAgentRawDictResponseFormat covers the raw-dict `response_format`
// overload (Python's `create_agent(response_format=dict)` form): a plain
// map[string]any JSON schema is treated as structured-output intent and
// auto-resolved against the model's capabilities (ToolCalling here), so the
// schema is bound to the model as a structured-output tool.
func TestCreateAgentRawDictResponseFormat(t *testing.T) {
	rawSchema := map[string]any{
		"type":        "object",
		"title":       "Answer",
		"description": "the answer",
		"properties":  map[string]any{"text": map[string]any{"type": "string"}},
		"required":    []any{"text"},
	}

	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(rawSchema))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(model.boundTools) != 1 {
		t.Fatalf("expected 1 structured-output tool bound to the model, got %d", len(model.boundTools))
	}
	bound := model.boundTools[0]
	if bound.Name() != "Answer" {
		t.Fatalf("expected structured-output tool name %q, got %q", "Answer", bound.Name())
	}
	if !reflect.DeepEqual(bound.ArgsSchema(), schema.Schema(rawSchema)) {
		t.Fatalf("expected bound tool args schema to carry the raw schema verbatim, got %#v", bound.ArgsSchema())
	}
}

// TestCreateAgentSchemaResponseFormat covers the same raw-schema overload for
// the schema.Schema spelling (the type the strategy constructors accept).
func TestCreateAgentSchemaResponseFormat(t *testing.T) {
	sch := answerSchema()

	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(sch))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(model.boundTools) != 1 || model.boundTools[0].Name() != "Answer" {
		t.Fatalf("expected one bound Answer tool, got %#v", model.boundTools)
	}
	if !reflect.DeepEqual(model.boundTools[0].ArgsSchema(), sch) {
		t.Fatalf("expected bound tool args schema to carry the schema, got %#v", model.boundTools[0].ArgsSchema())
	}
}

// TestCreateAgentAutoStrategyResolution proves the WithAgentResponseFormat
// dispatch resolves an AutoStrategy eagerly at CreateAgent time against the
// agent's bound model: ToolCalling models end up with a ToolStrategy-wired
// agent (structured tool call surfaces as structured_response), StructuredOutput-
// only models end up with a ProviderStrategy-wired agent (JSON text surfaces as
// structured_response), and a model declaring neither capability yields a typed
// *StructuredOutputUnsupportedError from CreateAgent itself.
func TestCreateAgentAutoStrategyResolution(t *testing.T) {
	t.Run("tool_calling_model_resolves_to_tool_strategy", func(t *testing.T) {
		toolName := NewToolStrategy(answerSchema()).SchemaSpecs[0].Name
		model := language.NewFakeChatModel(
			language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
			language.WithResponses(messages.Message{
				Role: messages.RoleAI,
				ToolCalls: []messages.ToolCall{
					{ID: "call_1", Name: toolName, Args: map[string]any{"text": "42"}},
				},
			}),
		)

		agent, err := CreateAgent(model, nil, WithAgentResponseFormat(NewAutoStrategy(answerSchema())))
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}

		state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("what is the answer?")})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		structured, ok := state["structured_response"].(map[string]any)
		if !ok || structured["text"] != "42" {
			t.Fatalf("expected ToolStrategy-resolved structured_response with text=42, got %#v", state["structured_response"])
		}
	})

	t.Run("structured_output_only_model_resolves_to_provider_strategy", func(t *testing.T) {
		model := language.NewFakeChatModel(
			language.WithCapabilities(language.ChatModelCapabilities{StructuredOutput: true}),
			language.WithResponses(messages.AI(`{"text": "42"}`)),
		)

		agent, err := CreateAgent(model, nil, WithAgentResponseFormat(NewAutoStrategy(answerSchema())))
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}

		state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("what is the answer?")})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		structured, ok := state["structured_response"].(map[string]any)
		if !ok || structured["text"] != "42" {
			t.Fatalf("expected ProviderStrategy-resolved structured_response with text=42, got %#v", state["structured_response"])
		}
	})

	t.Run("neither_capability_returns_typed_error_from_create_agent", func(t *testing.T) {
		model := language.NewFakeChatModel(
			language.WithCapabilities(language.ChatModelCapabilities{}),
		)
		_, err := CreateAgent(model, nil, WithAgentResponseFormat(NewAutoStrategy(answerSchema())))
		if err == nil {
			t.Fatal("expected error from CreateAgent for model with neither capability")
		}
		var unsupported *StructuredOutputUnsupportedError
		if !errors.As(err, &unsupported) {
			t.Fatalf("expected *StructuredOutputUnsupportedError from CreateAgent, got %T (%v)", err, err)
		}
	})

	t.Run("pointer_auto_strategy_supported", func(t *testing.T) {
		auto := NewAutoStrategy(answerSchema())
		model := language.NewFakeChatModel(
			language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
			language.WithResponses(messages.Message{
				Role: messages.RoleAI,
				ToolCalls: []messages.ToolCall{
					{ID: "call_1", Name: NewToolStrategy(answerSchema()).SchemaSpecs[0].Name, Args: map[string]any{"text": "42"}},
				},
			}),
		)
		if _, err := CreateAgent(model, nil, WithAgentResponseFormat(&auto)); err != nil {
			t.Fatalf("create agent with *AutoStrategy: %v", err)
		}
	})

	t.Run("nil_pointer_auto_strategy_is_no_op", func(t *testing.T) {
		model := language.NewFakeChatModel(
			language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
			language.WithResponses(messages.AI("done")),
		)
		var auto *AutoStrategy
		agent, err := CreateAgent(model, nil, WithAgentResponseFormat(auto))
		if err != nil {
			t.Fatalf("create agent with nil *AutoStrategy: %v", err)
		}
		state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if _, present := state["structured_response"]; present {
			t.Fatalf("expected no structured_response for nil *AutoStrategy, got %#v", state["structured_response"])
		}
	})
}

// recordingAgentLifecycleMiddleware implements BeforeAgentHook/AfterAgentHook,
// recording each call (and optionally contributing a BeforeAgent state
// update / returning an AfterAgent error) for assertions.
type recordingAgentLifecycleMiddleware struct {
	tag          string
	log          *[]string
	beforeUpdate map[string]any
	afterErr     error
}

func (r *recordingAgentLifecycleMiddleware) BeforeAgent(_ context.Context, _ map[string]any) (map[string]any, error) {
	*r.log = append(*r.log, r.tag+":before_agent")
	return r.beforeUpdate, nil
}

func (r *recordingAgentLifecycleMiddleware) AfterAgent(_ context.Context, _ map[string]any) error {
	*r.log = append(*r.log, r.tag+":after_agent")
	return r.afterErr
}

func TestCreateAgentBeforeAfterAgentHooksRunOncePerRun(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
		messages.AI("done"),
	}}
	var log []string
	lifecycle := &recordingAgentLifecycleMiddleware{
		tag:          "L",
		log:          &log,
		beforeUpdate: map[string]any{"seeded": true},
	}

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)}, WithAgentMiddleware(lifecycle))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	want := []string{"L:before_agent", "L:after_agent"}
	if len(log) != len(want) {
		t.Fatalf("expected before/after agent to run exactly once each despite the model/tools loop, got %v", log)
	}
	for i := range want {
		if log[i] != want[i] {
			t.Fatalf("log mismatch at %d: got %v, want %v", i, log, want)
		}
	}
	if state["seeded"] != true {
		t.Fatalf("expected BeforeAgent's state update to persist, got %#v", state["seeded"])
	}
}

func TestCreateAgentAfterAgentRunsOnJumpToEnd(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
	}}
	limit := 0
	toolLimitMW, err := middleware.NewToolCallLimitMiddleware("echo", &limit, nil, "end")
	if err != nil {
		t.Fatalf("new tool call limit middleware: %v", err)
	}
	var log []string
	lifecycle := &recordingAgentLifecycleMiddleware{tag: "L", log: &log}

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)}, WithAgentMiddleware(toolLimitMW, lifecycle))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	want := []string{"L:before_agent", "L:after_agent"}
	if len(log) != len(want) || log[0] != want[0] || log[1] != want[1] {
		t.Fatalf("expected AfterAgent to run once even on a jump_to \"end\" short-circuit, got %v", log)
	}
}

func TestCreateAgentAfterAgentErrorPropagates(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	sentinel := fmt.Errorf("cleanup failed")
	lifecycle := &recordingAgentLifecycleMiddleware{tag: "L", log: &[]string{}, afterErr: sentinel}

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(lifecycle))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected AfterAgent error to propagate, got %v", err)
	}
}

// interruptBeforeModelMiddleware calls graphpkg.Interrupt from BeforeModel,
// exercising the ctx.Context now threaded through every model-loop hook (see
// the package doc comment's Interrupts note). It pauses the run once per
// thread (tracked via a "confirmed" state key so the resumed re-execution of
// the "model" node doesn't interrupt again).
type interruptBeforeModelMiddleware struct{}

func (interruptBeforeModelMiddleware) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	if confirmed, _ := state["confirmed"].(bool); confirmed {
		return nil, nil
	}
	answer := graphpkg.Interrupt(ctx, "proceed with the run?")
	return map[string]any{"confirmed": answer}, nil
}

func TestCreateAgentInterruptThroughBeforeModelHook(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	saver := checkpoint.NewMemorySaver()

	agent, err := CreateAgent(model, nil,
		WithAgentMiddleware(interruptBeforeModelMiddleware{}),
		WithAgentCheckpointer(saver),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	first, err := agent.Graph.InvokeWithOptions(context.Background(),
		map[string]any{"messages": []messages.Message{messages.Human("hi")}},
		graphpkg.Options{ThreadID: "t1"},
	)
	if err != nil {
		t.Fatalf("first invoke: %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "proceed with the run?" {
		t.Fatalf("expected one pending interrupt, got %+v", first.Interrupts)
	}
	if len(model.invocations) != 0 {
		t.Fatalf("expected model to not be invoked before resume, got %d invocations", len(model.invocations))
	}

	second, err := agent.Graph.InvokeWithOptions(context.Background(), nil,
		graphpkg.Options{ThreadID: "t1", Resume: true},
	)
	if err != nil {
		t.Fatalf("resume invoke: %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if len(model.invocations) != 1 {
		t.Fatalf("expected model to be invoked exactly once after resume, got %d invocations", len(model.invocations))
	}
	out, _ := second.Values["messages"].([]messages.Message)
	if len(out) == 0 || out[len(out)-1].Content != "done" {
		t.Fatalf("expected run to complete after resume, got %#v", out)
	}
}

// TestCreateAgent_InterruptBeforeNode verifies that
// WithAgentInterruptBefore(ToolsNodeName) pauses the agent run before the
// tools node runs, that the model has run once (producing a tool call) at the
// pause, and that resuming via Agent.Graph.InvokeWithOptions with the same
// ThreadID and a nil Resume runs the tools node and the second model call to
// completion. The model must NOT be re-invoked for the already-completed first
// call on resume (the critical correctness property for interrupt_before).
func TestCreateAgent_InterruptBeforeNode(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
		messages.AI("done"),
	}}
	saver := checkpoint.NewMemorySaver()

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)},
		WithAgentCheckpointer(saver),
		WithAgentInterruptBefore(ToolsNodeName),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	// First invoke: model runs once and requests a tool call; the run pauses
	// before the tools node dispatches.
	first, err := agent.Graph.InvokeWithOptions(context.Background(),
		map[string]any{"messages": []messages.Message{messages.Human("hi")}},
		graphpkg.Options{ThreadID: "t1"},
	)
	if err != nil {
		t.Fatalf("first invoke: %v", err)
	}
	if len(first.Interrupts) != 1 {
		t.Fatalf("expected one pending interrupt before tools, got %+v", first.Interrupts)
	}
	if len(model.invocations) != 1 {
		t.Fatalf("expected model invoked once at pause, got %d", len(model.invocations))
	}
	firstMsgs, _ := first.Values["messages"].([]messages.Message)
	if len(firstMsgs) != 2 {
		t.Fatalf("expected human+AI(tool_call) = 2 messages at pause, got %d", len(firstMsgs))
	}
	// No tool result message yet (tools node did not run).
	for _, m := range firstMsgs {
		if m.Role == messages.RoleTool {
			t.Fatalf("unexpected tool message before tools node ran: %#v", m)
		}
	}

	// Resume: tools node runs, model runs again and produces the final answer.
	second, err := agent.Graph.InvokeWithOptions(context.Background(), nil,
		graphpkg.Options{ThreadID: "t1"},
	)
	if err != nil {
		t.Fatalf("resume invoke: %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	// Critical correctness check: the first model call is NOT re-run on resume.
	if len(model.invocations) != 2 {
		t.Fatalf("expected model invoked exactly twice total (once before pause, once after), got %d", len(model.invocations))
	}
	secondMsgs, _ := second.Values["messages"].([]messages.Message)
	if len(secondMsgs) != 4 {
		t.Fatalf("expected 4 final messages (human, AI(tool_call), tool, AI(done)), got %d", len(secondMsgs))
	}
	if secondMsgs[2].Role != messages.RoleTool || secondMsgs[2].Content != "echo:hi" {
		t.Fatalf("expected tool result 'echo:hi' after resume, got %#v", secondMsgs[2])
	}
	if secondMsgs[3].Content != "done" {
		t.Fatalf("expected final AI 'done', got %#v", secondMsgs[3])
	}
}

// TestCreateAgent_StoreInjectedIntoTool verifies that a store configured via
// WithAgentStore reaches each tool call as middleware.ToolCallRequest.Store,
// mirroring Python's `create_agent(store=...)` (Go has no annotation-based
// InjectedStore, so tools read the store explicitly off the request).
func TestCreateAgent_StoreInjectedIntoTool(t *testing.T) {
	st := storepkg.NewInMemoryStore()
	captured := make(chan storepkg.Store, 1)

	tool, err := coretools.NewFunc("reader", "reads the store",
		schema.Object(map[string]schema.Schema{"k": schema.String("key")}, "k"),
		func(ctx context.Context, in map[string]any) (coretools.Result, error) {
			return coretools.Result{Content: "ok"}, nil
		})
	if err != nil {
		t.Fatalf("NewFunc: %v", err)
	}

	// Wrapper that captures the store handed to each tool call.
	wrap := func(ctx context.Context, req middleware.ToolCallRequest, next middleware.ToolHandler) (messages.Message, error) {
		if req.Store == nil {
			t.Errorf("expected Store injected, got nil")
		}
		captured <- req.Store
		return next(ctx, req)
	}

	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "reader", Args: map[string]any{"k": "user:1"}},
			},
		},
		messages.AI("done"),
	}}

	agent, err := CreateAgent(
		model,
		[]coretools.Tool{tool},
		WithAgentStore(st),
		WithAgentMiddleware(storeCapturingMiddleware{fn: wrap}),
	)
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("read")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	select {
	case s := <-captured:
		if s == nil {
			t.Fatalf("captured store was nil")
		}
	case <-time.After(time.Second):
		t.Fatalf("tool never observed a store")
	}
}

type storeCapturingMiddleware struct {
	fn func(context.Context, middleware.ToolCallRequest, middleware.ToolHandler) (messages.Message, error)
}

func (m storeCapturingMiddleware) WrapToolCall(ctx context.Context, req middleware.ToolCallRequest, next middleware.ToolHandler) (messages.Message, error) {
	return m.fn(ctx, req, next)
}

// TestCreateAgent_CacheHitSkipsModel verifies that WithAgentCache wires
// core/caches into the model-call path: the same input twice must invoke the
// underlying model exactly once (the second response is served from cache),
// mirroring Python's `create_agent(cache=...)`.
func TestCreateAgent_CacheHitSkipsModel(t *testing.T) {
	cache, err := caches.NewInMemoryCache()
	if err != nil {
		t.Fatalf("NewInMemoryCache: %v", err)
	}
	calls := 0
	model := language.NewFakeChatModel(language.WithResponses(
		messages.AI("first"),
		messages.AI("second"),
	))
	agent, err := CreateAgent(model, nil,
		WithAgentCache(cache),
		WithAgentMiddleware(countModelCalls{&calls}),
	)
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	msgs := []messages.Message{messages.Human("hi")}
	if _, err := agent.Invoke(context.Background(), msgs); err != nil {
		t.Fatalf("invoke 1: %v", err)
	}
	out2, err := agent.Invoke(context.Background(), msgs)
	if err != nil {
		t.Fatalf("invoke 2: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected model called once (second served from cache), got %d", calls)
	}
	// The second Invoke must return the CACHED content ("first"), not the
	// would-be second response ("second"). This catches a regression where the
	// cache short-circuits the chain but returns the wrong value.
	last2 := out2[len(out2)-1]
	if last2.Role != messages.RoleAI || last2.Content != "first" {
		t.Fatalf("second Invoke returned %q (role %q), want cached %q",
			last2.Content, last2.Role, "first")
	}
}

// TestCreateAgent_CacheSkipsToolCallResponses verifies that a tool-call
// response is NOT cached. messages.Text drops a message's ToolCalls, so a
// cached tool call would be rebuilt as a text-only AI message; a second
// identical Invoke would then return plain text instead of routing through the
// tool, breaking a tool-calling agent. The fix: only terminal text responses
// are written to the cache.
//
// With a tool-calling agent whose first model response is a tool call:
//   - Invoke 1: model called for the tool call (NOT cached), tool runs, model
//     called again for the final text response (cached). countModelCalls == 2.
//   - Invoke 2: model called again for the tool call (NOT served from cache),
//     tool runs, model's second call is a cache hit. countModelCalls == 3.
//
// If the bug were present, the tool-call response would be cached as empty
// text and Invoke 2 would short-circuit to an empty text response without
// ever re-invoking the model (countModelCalls would stay at 2 and the tool
// would never run on the second Invoke).
func TestCreateAgent_CacheSkipsToolCallResponses(t *testing.T) {
	cache, err := caches.NewInMemoryCache()
	if err != nil {
		t.Fatalf("NewInMemoryCache: %v", err)
	}
	calls := 0
	toolCallMsg := messages.Message{
		Role: messages.RoleAI,
		ToolCalls: []messages.ToolCall{
			{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
		},
	}
	// Two full loops: each Invoke consumes one tool-call response + one final
	// text response. The second pair guards against a half-fix that only
	// skipped the Lookup (the model still needs responses to serve Invoke 2).
	model := &sequenceModel{responses: []messages.Message{
		toolCallMsg,
		messages.AI("final-1"),
		toolCallMsg,
		messages.AI("final-2"),
	}}
	echo := newEchoTool(t)
	agent, err := CreateAgent(model, []coretools.Tool{echo},
		WithAgentCache(cache),
		WithAgentMiddleware(countModelCalls{&calls}),
	)
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	msgs := []messages.Message{messages.Human("hi")}

	out1, err := agent.Invoke(context.Background(), msgs)
	if err != nil {
		t.Fatalf("invoke 1: %v", err)
	}
	if calls != 2 {
		t.Fatalf("after invoke 1: expected model chain entered twice (tool call + final), got %d", calls)
	}
	// Invoke 1 must have actually run the tool (sanity: this is a real
	// tool-calling loop, not a degenerate text-only path).
	var sawToolResult1 bool
	for _, m := range out1 {
		if m.Role == messages.RoleTool {
			sawToolResult1 = true
		}
	}
	if !sawToolResult1 {
		t.Fatalf("invoke 1 did not produce a tool-result message: %+v", out1)
	}

	out2, err := agent.Invoke(context.Background(), msgs)
	if err != nil {
		t.Fatalf("invoke 2: %v", err)
	}
	// The tool-call response was NOT cached, so Invoke 2 must re-enter the
	// model chain for the tool-call step (calls goes 2 -> 3). The final-text
	// step IS cached (deterministic tool result -> identical cache key), so
	// calls goes to 3, not 4. calls == 2 indicates the bug (tool call served
	// from cache as empty text); calls == 4 would indicate the final-text
	// response was also not cached (a different bug).
	if calls != 3 {
		t.Fatalf("after invoke 2: expected model chain entered 3 times (tool call re-invoked, final cached), got %d", calls)
	}
	// Invoke 2 must also have run the tool — the lossy-cache bug would
	// short-circuit Invoke 2 to a plain text response with no tool dispatch.
	var sawToolResult2 bool
	for _, m := range out2 {
		if m.Role == messages.RoleTool {
			sawToolResult2 = true
		}
	}
	if !sawToolResult2 {
		t.Fatalf("invoke 2 did not produce a tool-result message (tool call was served from cache as text?): %+v", out2)
	}
}

// countModelCalls is a WrapModelCallHook middleware that counts how many times
// the model-call chain is actually entered. Combined with WithAgentCache, a
// cache hit short-circuits the chain before this middleware runs, so a cached
// second call does not increment the counter.
type countModelCalls struct{ n *int }

func (m countModelCalls) WrapModelCall(ctx context.Context, req middleware.ModelRequest, next middleware.ModelHandler) (middleware.ModelResponse, error) {
	*m.n++
	return next(ctx, req)
}

// TestStreamEvents_CacheDoesNotSuppressEvents verifies that the cache is
// scoped to the non-streaming Invoke path: a StreamEvents run with a cache
// configured must still emit the normal model_delta/model_end sequence on
// every call, including a second identical StreamEvents (which would otherwise
// be a cache hit, short-circuit the handler, and produce an event-less
// completion). The fix skips the cache entirely when a stream sink is active.
//
// Two coverage angles:
//  1. A first StreamEvents call (with a cache configured) emits the normal
//     event sequence — i.e. wiring the cache did not itself suppress events.
//  2. A second identical StreamEvents call ALSO emits the normal sequence —
//     i.e. even if the first call had populated the cache, the streaming path
//     bypasses it. (Without the fix, the second call would be a cache hit and
//     emit zero model_delta/model_end events.)
func TestStreamEvents_CacheDoesNotSuppressEvents(t *testing.T) {
	cache, err := caches.NewInMemoryCache()
	if err != nil {
		t.Fatalf("NewInMemoryCache: %v", err)
	}
	// Two identical streamed responses so the model can serve both calls.
	model := &streamSequenceModel{
		responses: []messages.Message{
			messages.AI("Hi there"),
			messages.AI("Hi there"),
		},
		streamChunks: [][]messages.Message{
			{messages.AI("Hi"), messages.AI(" there")},
			{messages.AI("Hi"), messages.AI(" there")},
		},
	}
	agent, err := CreateAgent(model, nil, WithAgentCache(cache))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	msgs := []messages.Message{messages.Human("hi")}

	runAndAssert := func(label string) {
		stream, err := agent.StreamEvents(context.Background(), msgs)
		if err != nil {
			t.Fatalf("%s: StreamEvents: %v", label, err)
		}
		defer stream.Close()
		events := drainStream(t, stream)

		deltas := countType(events, StreamModelDelta)
		ends := countType(events, StreamModelEnd)
		if deltas == 0 {
			t.Fatalf("%s: expected at least one model_delta event, got none (types=%v) — cache suppressed streaming events",
				label, eventTypes(events))
		}
		if ends != 1 {
			t.Fatalf("%s: expected exactly one model_end event, got %d (types=%v)",
				label, ends, eventTypes(events))
		}
	}

	runAndAssert("first StreamEvents")
	runAndAssert("second StreamEvents (identical, would be a cache hit without the streaming guard)")
}

// TestCreateAgent_ModelString covers Task 5.7 Part B: WithAgentModel resolves
// a "provider:model" string via chatmodels.ParseModelString + chatmodels.Resolve
// into a real ChatModel, which then flows through the rest of CreateAgent as if
// the caller had passed it positionally. The positional `model` arg may be nil
// when WithAgentModel is used; ModelString OVERRIDES the positional arg when
// both are supplied (documented precedence).
//
// To avoid real HTTP and avoid colliding with the real "openai" factory the
// blank-import on create_agent.go brings into this binary, the test registers a
// TEST-ONLY factory under a unique provider name. That factory wraps
// openai.NewChatModel with a custom BaseURL pointing at a local httptest server
// (mirroring partners/openai/chatmodel_test.go's canned Responses-API shape).
func TestCreateAgent_ModelString(t *testing.T) {
	const provider = "test-openai-5x7"
	const wantContent = "Hello from test server"

	var requestCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		if r.URL.Path != "/responses" {
			t.Errorf("path: got %q want /responses", r.URL.Path)
		}
		_, _ = fmt.Fprintf(w, `{
			"id":"resp_test",
			"model":"gpt-test",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":%q}]}],
			"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
		}`, wantContent)
	}))
	defer server.Close()

	chatmodels.RegisterProvider(provider, func(model string, opts map[string]any) (language.ChatModel, error) {
		return openai.NewChatModel(
			modelconfig.WithBaseURL(server.URL),
			modelconfig.WithAPIKey("test-key"),
			modelconfig.WithModel(model),
		), nil
	})

	t.Run("nil positional model resolved from string", func(t *testing.T) {
		atomic.StoreInt32(&requestCount, 0)
		agent, err := CreateAgent(nil, nil, WithAgentModel(provider+":gpt-test"))
		if err != nil {
			t.Fatalf("CreateAgent: unexpected error: %v", err)
		}
		out, err := agent.Invoke(context.Background(), []messages.Message{
			messages.Human("hello"),
		})
		if err != nil {
			t.Fatalf("Invoke: %v", err)
		}
		if len(out) != 2 {
			t.Fatalf("expected 2 messages (human+ai), got %d: %#v", len(out), out)
		}
		if out[1].Role != messages.RoleAI || out[1].Content != wantContent {
			t.Fatalf("AI message: got role=%q content=%q want role=%q content=%q",
				out[1].Role, out[1].Content, messages.RoleAI, wantContent)
		}
		if got := atomic.LoadInt32(&requestCount); got != 1 {
			t.Fatalf("expected exactly 1 HTTP request to test server, got %d", got)
		}
	})

	t.Run("ModelString overrides positional model", func(t *testing.T) {
		// The positional fake would error if invoked. The resolved
		// (string-derived) model wins; the fake is never consulted.
		atomic.StoreInt32(&requestCount, 0)
		fake := &erroringModel{}
		agent, err := CreateAgent(fake, nil, WithAgentModel(provider+":gpt-test"))
		if err != nil {
			t.Fatalf("CreateAgent: unexpected error: %v", err)
		}
		out, err := agent.Invoke(context.Background(), []messages.Message{
			messages.Human("hello"),
		})
		if err != nil {
			t.Fatalf("Invoke: %v", err)
		}
		if len(out) != 2 || out[1].Content != wantContent {
			t.Fatalf("unexpected output: %#v", out)
		}
		if got := atomic.LoadInt32(&requestCount); got != 1 {
			t.Fatalf("expected exactly 1 HTTP request to test server (positional model should not be used), got %d", got)
		}
		if fake.invoked {
			t.Fatal("positional fake model was invoked; ModelString should override the positional arg")
		}
	})

	t.Run("invalid ModelString surfaces parse error", func(t *testing.T) {
		// Malformed string (no colon) is rejected by ParseModelString; the
		// positional nil never reaches the "model is required" check because
		// resolution fails first.
		_, err := CreateAgent(nil, nil, WithAgentModel("no-colon-here"))
		if err == nil {
			t.Fatal("expected parse error for malformed ModelString, got nil")
		}
	})

	t.Run("unknown provider surfaces Resolve error", func(t *testing.T) {
		_, err := CreateAgent(nil, nil, WithAgentModel("definitely-not-registered-xyz:m"))
		if err == nil {
			t.Fatal("expected Resolve error for unknown provider, got nil")
		}
	})

	t.Run("nil positional model with no ModelString still errors", func(t *testing.T) {
		// The reorder must not regress the existing "model is required"
		// check: with neither positional model nor ModelString, CreateAgent
		// returns the same error as before.
		_, err := CreateAgent(nil, nil)
		if err == nil {
			t.Fatal("expected error for nil positional model and no ModelString, got nil")
		}
	})
}

// erroringModel is a minimal language.ChatModel whose Invoke always errors. It
// is used by the ModelString-override subtest to assert the positional model is
// NOT consulted when ModelString is set. invoked is set in Invoke/BindTools so
// the test can fail loudly if the resolved model did NOT actually win.
type erroringModel struct {
	invoked bool
}

func (m *erroringModel) Invoke(context.Context, []messages.Message, ...runnables.Option) (messages.Message, error) {
	m.invoked = true
	return messages.Message{}, fmt.Errorf("erroringModel: Invoke should not be called when ModelString is set")
}

func (m *erroringModel) Batch(context.Context, [][]messages.Message, ...runnables.Option) ([]messages.Message, error) {
	m.invoked = true
	return nil, fmt.Errorf("erroringModel: Batch should not be called when ModelString is set")
}

func (m *erroringModel) Stream(context.Context, []messages.Message, ...runnables.Option) (runnables.Stream[messages.Message], error) {
	m.invoked = true
	return nil, fmt.Errorf("erroringModel: Stream should not be called when ModelString is set")
}

func (m *erroringModel) InputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *erroringModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *erroringModel) BindTools([]coretools.Tool) (language.ChatModel, error) {
	m.invoked = true
	return m, fmt.Errorf("erroringModel: BindTools should not be called when ModelString is set")
}

func (m *erroringModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// failBindModel is a ChatModel test double whose BindTools always fails, used
// to exercise the bind-error paths in invokeModel/invokeModelStreaming.
type failBindModel struct {
	*sequenceModel
}

func (m *failBindModel) BindTools(boundTools []coretools.Tool) (language.ChatModel, error) {
	return nil, fmt.Errorf("failBindModel: bind not supported")
}

func TestCreateAgentRecursionLimitStopsToolLoop(t *testing.T) {
	// The model requests a tool call on every invocation; without a recursion
	// limit this loops forever, so the limit must surface an error.
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c1", Name: "echo", Args: map[string]any{"tool_input": "x"}}}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c2", Name: "echo", Args: map[string]any{"tool_input": "x"}}}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c3", Name: "echo", Args: map[string]any{"tool_input": "x"}}}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c4", Name: "echo", Args: map[string]any{"tool_input": "x"}}}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c5", Name: "echo", Args: map[string]any{"tool_input": "x"}}}},
		messages.AI("done"),
	}}
	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)}, WithAgentRecursionLimit(3))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected recursion-limit error for a looping agent, got nil")
	}
}

func TestCreateAgentInterruptAfterSurfacesViaInvoke(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil,
		WithAgentCheckpointer(checkpoint.NewMemorySaver()),
		WithAgentInterruptAfter(ModelNodeName),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	// Agent.Invoke treats a paused run as a terminal failure, pointing the
	// caller at Agent.Graph for resumption.
	_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected an interrupted-run error from Invoke, got nil")
	}
}

func TestCreateAgentDictToolSpecValidation(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	t.Run("missing name", func(t *testing.T) {
		_, err := CreateAgent(model, nil, WithAgentToolSpecs(map[string]any{"description": "no name"}))
		if err == nil {
			t.Fatal("expected error for spec without a name")
		}
	})

	t.Run("parameters wrong type", func(t *testing.T) {
		_, err := CreateAgent(model, nil, WithAgentToolSpecs(map[string]any{"name": "x", "parameters": "nope"}))
		if err == nil {
			t.Fatal("expected error for non-object parameters")
		}
	})

	t.Run("schema key variants", func(t *testing.T) {
		agent, err := CreateAgent(model, nil, WithAgentToolSpecs(
			map[string]any{"name": "spec_params", "parameters": schema.Schema{"type": "object", "properties": map[string]any{}}},
			map[string]any{"name": "spec_input_schema", "input_schema": map[string]any{"type": "object"}},
			map[string]any{"name": "spec_args_schema", "args_schema": map[string]any{"type": "object"}},
		))
		if err != nil {
			t.Fatalf("create agent with schema key variants: %v", err)
		}
		out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if len(model.boundTools) != 3 {
			names := make([]string, 0, len(model.boundTools))
			for _, bt := range model.boundTools {
				names = append(names, bt.Name())
			}
			t.Fatalf("expected 3 bound dict-spec tools, got %v", names)
		}
		if len(out) == 0 {
			t.Fatal("expected output messages")
		}
	})
}

func TestCreateAgentDictToolSpecHasNoExecutable(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c1", Name: "builtin_tool", Args: map[string]any{}}}},
		messages.AI("done"),
	}}
	agent, err := CreateAgent(model, nil, WithAgentToolSpecs(map[string]any{"name": "builtin_tool"}))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	var toolMsg *messages.Message
	for i := range out {
		if out[i].Role == messages.RoleTool {
			toolMsg = &out[i]
		}
	}
	if toolMsg == nil {
		t.Fatalf("expected a tool message for the dict-spec tool call, got %#v", out)
	}
	if !strings.Contains(toolMsg.Content, "no executable implementation") {
		t.Fatalf("expected explanatory tool error, got %q", toolMsg.Content)
	}
}

func TestAsSchemaCoercion(t *testing.T) {
	if s, ok := asSchema(schema.Schema{"type": "object"}); !ok || s["type"] != "object" {
		t.Fatalf("schema.Schema coercion failed: %#v ok=%v", s, ok)
	}
	if s, ok := asSchema(map[string]any{"type": "object"}); !ok || s["type"] != "object" {
		t.Fatalf("map coercion failed: %#v ok=%v", s, ok)
	}
	if _, ok := asSchema("nope"); ok {
		t.Fatal("expected non-map value to be rejected")
	}
}

func TestToolsFromAnyRejectsNonTool(t *testing.T) {
	if _, err := toolsFromAny([]any{"not-a-tool"}); err == nil {
		t.Fatal("expected error for non-Tool element")
	}
	tools, err := toolsFromAny([]any{newEchoTool(t)})
	if err != nil || len(tools) != 1 {
		t.Fatalf("expected one tool, got %v err=%v", tools, err)
	}
}

func TestResolveJumpTargetBranches(t *testing.T) {
	cases := map[string]string{
		"":       AfterAgentNodeName,
		"end":    AfterAgentNodeName,
		"model":  ModelNodeName,
		"tools":  ToolsNodeName,
		"custom": "custom",
	}
	for jumpTo, want := range cases {
		if got := resolveJumpTarget(jumpTo, AfterAgentNodeName); got != want {
			t.Fatalf("resolveJumpTarget(%q) = %q, want %q", jumpTo, got, want)
		}
	}
}

func TestPopJumpToNilAndMissing(t *testing.T) {
	if jump, ok := popJumpTo(nil); ok || jump != "" {
		t.Fatalf("expected (\"\", false) for nil update, got (%q, %v)", jump, ok)
	}
	if jump, ok := popJumpTo(map[string]any{"other": 1}); ok || jump != "" {
		t.Fatalf("expected (\"\", false) without jump_to key, got (%q, %v)", jump, ok)
	}
	update := map[string]any{"jump_to": "end"}
	if jump, ok := popJumpTo(update); !ok || jump != "end" {
		t.Fatalf("expected (end, true), got (%q, %v)", jump, ok)
	}
	if _, stillThere := update["jump_to"]; stillThere {
		t.Fatal("jump_to must be consumed out of the update")
	}
}

func TestBeforeModelHookErrorPropagates(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncBeforeModel(func(ctx context.Context, state map[string]any) (map[string]any, error) {
			return nil, fmt.Errorf("before_model boom")
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected before_model error to propagate, got nil")
	}
}

func TestBeforeModelCommandHookErrorPropagates(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncBeforeModelCommand(func(ctx context.Context, state map[string]any) (*middleware.Command, error) {
			return nil, fmt.Errorf("command boom")
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected command hook error to propagate, got nil")
	}
}

func TestBeforeModelCommandHookNilCommandContinues(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	var calls int
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncBeforeModelCommand(func(ctx context.Context, state map[string]any) (*middleware.Command, error) {
			calls++
			return nil, nil // nil command: fall through to the normal model call
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if calls != 1 || len(out) == 0 || out[len(out)-1].Content != "done" {
		t.Fatalf("nil command should continue the run: calls=%d out=%#v", calls, out)
	}
}

func TestBeforeModelCommandHookCommandEndsRun(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncBeforeModelCommand(func(ctx context.Context, state map[string]any) (*middleware.Command, error) {
			return &middleware.Command{Update: map[string]any{"marker": "set"}, Goto: "end"}, nil
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if state["marker"] != "set" {
		t.Fatalf("expected command update in state, got %#v", state)
	}
	if len(model.invocations) != 0 {
		t.Fatalf("model must not run after a jump_to end command, got %d invocations", len(model.invocations))
	}
}

func TestBeforeModelJumpToEndShortCircuits(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncBeforeModel(func(ctx context.Context, state map[string]any) (map[string]any, error) {
			return map[string]any{"jump_to": "end", "marker": true}, nil
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if state["marker"] != true {
		t.Fatalf("expected marker in state, got %#v", state)
	}
	if len(model.invocations) != 0 {
		t.Fatalf("model must not run after jump_to end, got %d invocations", len(model.invocations))
	}
	if _, leaked := state["jump_to"]; leaked {
		t.Fatal("jump_to must not be persisted into state")
	}
}

func TestBeforeModelMessagesReshapeIsLocalOnly(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncBeforeModel(func(ctx context.Context, state map[string]any) (map[string]any, error) {
			return map[string]any{"messages": []messages.Message{messages.Human("replaced")}}, nil
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	// The model call saw the reshaped (local) message view...
	if len(model.invocations) != 1 || len(model.invocations[0]) != 1 || model.invocations[0][0].Content != "replaced" {
		t.Fatalf("model should see the reshaped messages, got %#v", model.invocations)
	}
	// ...but the committed state keeps the original history plus the reply.
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 2 || msgs[0].Content != "hi" || msgs[1].Content != "done" {
		t.Fatalf("state messages mismatch: %#v", msgs)
	}
}

func TestAfterModelHookErrorPropagates(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncAfterModel(func(ctx context.Context, state map[string]any) (map[string]any, error) {
			return nil, fmt.Errorf("after_model boom")
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected after_model error to propagate, got nil")
	}
}

func TestCreateAgentCacheHitWithDebugAndSystemPrompt(t *testing.T) {
	cache, err := caches.NewInMemoryCache()
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}
	model := &sequenceModel{responses: []messages.Message{messages.AI("cached answer")}}
	agent, err := CreateAgent(model, nil,
		WithAgentCache(cache),
		WithAgentDebug(true),
		WithAgentSystemPrompt("you are helpful"),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	for i := 0; i < 2; i++ {
		out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
		if err != nil {
			t.Fatalf("invoke %d: %v", i, err)
		}
		if len(out) == 0 || out[len(out)-1].Content != "cached answer" {
			t.Fatalf("invoke %d: unexpected output %#v", i, out)
		}
	}
	// The second identical Invoke must be served from the cache: the model's
	// single queued response was consumed by the first call, so a second model
	// call would have errored.
	if len(model.invocations) != 1 {
		t.Fatalf("expected exactly one model call (second served from cache), got %d", len(model.invocations))
	}
}

func TestHashToolsAndSettingsNonMarshable(t *testing.T) {
	if got := hashToolsAndSettings(nil, map[string]any{"fn": func() {}}); got != "" {
		t.Fatalf("expected empty hash for non-JSON-marshable settings, got %q", got)
	}
	// A non-Tool entry in the tools list is skipped, not fatal.
	a := hashToolsAndSettings([]any{"not-a-tool"}, nil)
	b := hashToolsAndSettings(nil, nil)
	if a != b {
		t.Fatalf("non-Tool entries must not affect the hash: %q vs %q", a, b)
	}
}

func TestToolResultMapVariants(t *testing.T) {
	if got := toolResultMap(messages.Message{}); got != nil {
		t.Fatalf("expected nil for empty message, got %#v", got)
	}
	if got := toolResultMap(messages.Tool("c1", "content")); got["content"] != "content" {
		t.Fatalf("expected content entry, got %#v", got)
	}
	metaOnly := messages.Message{ResponseMetadata: map[string]any{"status": "ok"}}
	got := toolResultMap(metaOnly)
	if got["status"] != "ok" {
		t.Fatalf("expected status entry, got %#v", got)
	}
	if _, hasContent := got["content"]; hasContent {
		t.Fatalf("unexpected content entry for empty content: %#v", got)
	}
}

func TestCreateAgentDuplicateToolNames(t *testing.T) {
	dup := func() coretools.Tool {
		tool, err := coretools.NewSimple("dup", "duplicate", func(_ context.Context, input string) (coretools.Result, error) {
			return coretools.Result{Content: input}, nil
		})
		if err != nil {
			t.Fatalf("new tool: %v", err)
		}
		return tool
	}
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	if _, err := CreateAgent(model, []coretools.Tool{dup(), dup()}); err == nil {
		t.Fatal("expected error for duplicate tool names")
	}
}

func TestAgentHookNodesWithDebug(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	var beforeRan, afterRan bool
	agent, err := CreateAgent(model, nil,
		WithAgentDebug(true),
		WithAgentMiddleware(
			FuncBeforeAgent(func(ctx context.Context, state map[string]any) (map[string]any, error) {
				beforeRan = true
				return map[string]any{"before": true}, nil
			}),
			FuncAfterAgent(func(ctx context.Context, state map[string]any) error {
				afterRan = true
				return nil
			}),
		),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !beforeRan || !afterRan {
		t.Fatalf("hooks did not run: before=%v after=%v", beforeRan, afterRan)
	}
	if state["before"] != true {
		t.Fatalf("expected before_agent update in state, got %#v", state)
	}
}

func TestBeforeAgentHookErrorPropagates(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncBeforeAgent(func(ctx context.Context, state map[string]any) (map[string]any, error) {
			return nil, fmt.Errorf("before_agent boom")
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected before_agent error to propagate, got nil")
	}
}

func TestModelNodeHandlesEmptyAndNonAIResponses(t *testing.T) {
	t.Run("empty result", func(t *testing.T) {
		model := &sequenceModel{responses: []messages.Message{messages.AI("unused")}}
		agent, err := CreateAgent(model, nil, WithAgentMiddleware(
			FuncWrapModelCall(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
				return middleware.ModelResponse{}, nil
			}),
		))
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		msgs, _ := state["messages"].([]messages.Message)
		if len(msgs) != 1 || msgs[0].Content != "hi" {
			t.Fatalf("expected only the human message, got %#v", msgs)
		}
	})

	t.Run("non-AI result", func(t *testing.T) {
		model := &sequenceModel{responses: []messages.Message{messages.AI("unused")}}
		agent, err := CreateAgent(model, nil, WithAgentMiddleware(
			FuncWrapModelCall(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
				return middleware.ModelResponse{Result: []messages.Message{messages.Human("odd")}}, nil
			}),
		))
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		msgs, _ := state["messages"].([]messages.Message)
		if len(msgs) != 2 || msgs[1].Content != "odd" {
			t.Fatalf("expected the non-AI message appended, got %#v", msgs)
		}
	})
}

func TestProviderStrategyInvalidJSONPropagates(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("this is not json")}}
	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(NewProviderStrategy(weatherSchema())))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected a parse error for non-JSON model output, got nil")
	}
}

func TestResponseFormatPointerVariants(t *testing.T) {
	toolStrategy := NewToolStrategy(weatherSchema())
	providerStrategy := NewProviderStrategy(weatherSchema())

	model := &sequenceModel{responses: []messages.Message{messages.AI(`{"temperature": 75, "condition": "sunny"}`)}}
	if _, err := CreateAgent(model, nil, WithAgentResponseFormat(&toolStrategy)); err != nil {
		t.Fatalf("*ToolStrategy: %v", err)
	}
	if _, err := CreateAgent(model, nil, WithAgentResponseFormat(&providerStrategy)); err != nil {
		t.Fatalf("*ProviderStrategy: %v", err)
	}
	auto := NewAutoStrategy(weatherSchema())
	if _, err := CreateAgent(model, nil, WithAgentResponseFormat(&auto)); err != nil {
		t.Fatalf("*AutoStrategy with tool-calling model: %v", err)
	}

	// A model with neither tool calling nor structured output cannot resolve
	// an AutoStrategy (nil *AutoStrategy is a no-op, non-nil errors).
	noCaps := language.NewFakeChatModel(language.WithCapabilities(language.ChatModelCapabilities{}))
	_, err := CreateAgent(noCaps, nil, WithAgentResponseFormat(&auto))
	if err == nil {
		t.Fatal("expected StructuredOutputUnsupportedError for a no-capability model")
	}
	var unsupported *StructuredOutputUnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("expected *StructuredOutputUnsupportedError, got %T: %v", err, err)
	}
}

func TestInvokeModelRejectsNonChatModel(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncWrapModelCall(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
			request.Model = 42
			return handler(ctx, request)
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected error for non-ChatModel request.Model")
	}
}

func TestInvokeModelRejectsNonTool(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		FuncWrapModelCall(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
			request.Tools = []any{"not-a-tool"}
			return handler(ctx, request)
		}),
	))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected error for non-Tool request.Tools entry")
	}
}

func TestInvokeModelBindToolsError(t *testing.T) {
	model := &failBindModel{sequenceModel: &sequenceModel{responses: []messages.Message{messages.AI("done")}}}
	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected BindTools error to propagate")
	}
}

func TestNativeStructuredCallerErrorPropagates(t *testing.T) {
	model := &nativeStructuredSequenceModel{sequenceModel: &sequenceModel{}}
	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(NewProviderStrategy(weatherSchema())))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected InvokeStructured error to propagate")
	}
	if !model.nativeCalled {
		t.Fatal("expected the native structured path to be taken")
	}
}

func TestStateSchemaNilReducerDefaultsToLastValue(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c1", Name: "echo", Args: map[string]any{"tool_input": "x"}}}},
		messages.AI("done"),
	}}
	var writes int
	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)},
		WithAgentStateFields(StateField{Name: "note"}), // nil Reducer: last write wins
		WithAgentMiddleware(
			FuncAfterModel(func(ctx context.Context, state map[string]any) (map[string]any, error) {
				writes++
				return map[string]any{"note": writes}, nil
			}),
		),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if writes != 2 {
		t.Fatalf("expected after_model to run twice, got %d", writes)
	}
	if state["note"] != 2 {
		t.Fatalf("expected last write to win for a nil-reducer field, got %#v", state["note"])
	}
}

func TestSystemPromptTemplateRenderFailureFallsBackToLiteral(t *testing.T) {
	// The template references a variable nobody supplies; with
	// missingkey=error the render fails and the resolver must fall back to the
	// literal system prompt rather than silently sending an empty one.
	tmpl, err := prompts.NewPromptTemplate("test", "You are {{.missing_var}}.")
	if err != nil {
		t.Fatalf("new prompt template: %v", err)
	}
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil,
		WithAgentSystemPrompt("literal fallback"),
		WithAgentSystemPromptTemplate(&tmpl, nil),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(model.invocations) != 1 || len(model.invocations[0]) != 2 {
		t.Fatalf("expected system+human messages, got %#v", model.invocations)
	}
	if got := model.invocations[0][0].Content; got != "literal fallback" {
		t.Fatalf("expected literal fallback system prompt, got %q", got)
	}
}
