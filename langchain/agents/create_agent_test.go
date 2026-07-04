package agents

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/caches"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/stores"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langchain/internal/agentruntime/graph"
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

// TestCreateAgent_StoreInjectedIntoTool verifies that a store configured via
// WithAgentStore reaches each tool call as middleware.ToolCallRequest.Store,
// mirroring Python's `create_agent(store=...)` (Go has no annotation-based
// InjectedStore, so tools read the store explicitly off the request).
func TestCreateAgent_StoreInjectedIntoTool(t *testing.T) {
	store := stores.NewInMemoryStore[any]()
	captured := make(chan stores.BaseStore[any], 1)

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
		WithAgentStore(store),
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
	if _, err := agent.Invoke(context.Background(), msgs); err != nil {
		t.Fatalf("invoke 2: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected model called once (second served from cache), got %d", calls)
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
