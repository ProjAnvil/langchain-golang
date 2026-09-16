package agents

import (
	"context"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// toolChoiceRecordingModel wraps sequenceModel with the optional
// language.ToolBinder capability, recording every BindToolsWithOptions call
// so tests can assert what the agent actually threaded into the model.
type toolChoiceRecordingModel struct {
	*sequenceModel
	mu     sync.Mutex
	calls  []language.BindToolsOptions
	binder bool // whether to expose ToolBinder at all
}

func newToolChoiceRecordingModel(responses []messages.Message, binder bool) *toolChoiceRecordingModel {
	return &toolChoiceRecordingModel{
		sequenceModel: &sequenceModel{responses: responses},
		binder:        binder,
	}
}

func (m *toolChoiceRecordingModel) BindToolsWithOptions(tools []coretools.Tool, opts language.BindToolsOptions) (language.ChatModel, error) {
	m.mu.Lock()
	m.calls = append(m.calls, opts)
	m.mu.Unlock()
	if _, err := m.sequenceModel.BindTools(tools); err != nil {
		return nil, err
	}
	// Return the wrapper itself (like sequenceModel.BindTools) so repeated
	// bind+invoke cycles in the agent loop keep recording on this instance.
	return m, nil
}

func (m *toolChoiceRecordingModel) bindCalls() []language.BindToolsOptions {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]language.BindToolsOptions(nil), m.calls...)
}

// TestToolStrategyForcesToolChoiceAny mirrors langchain_v1 factory.py:1380-1389
// (`tool_choice = "any" if structured_output_tools else request.tool_choice`):
// when the agent's ResponseFormat resolves to a ToolStrategy, the model-node
// request carries tool_choice="any" and invokeModel threads it into the model
// via the ToolBinder capability, so the model MUST call a tool.
func TestToolStrategyForcesToolChoiceAny(t *testing.T) {
	strategy := NewToolStrategy(answerSchema())
	toolName := strategy.SchemaSpecs[0].Name

	model := newToolChoiceRecordingModel([]messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: toolName, Args: map[string]any{"text": "42"}},
			},
		},
	}, true)

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("what is the answer?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if structured, ok := state["structured_response"].(map[string]any); !ok || structured["text"] != "42" {
		t.Fatalf("expected structured_response.text=42, got %#v", state["structured_response"])
	}

	calls := model.bindCalls()
	if len(calls) != 1 {
		t.Fatalf("expected one BindToolsWithOptions call, got %d", len(calls))
	}
	if calls[0].ToolChoice != language.ToolChoiceAny {
		t.Fatalf("ToolChoice = %q, want %q", calls[0].ToolChoice, language.ToolChoiceAny)
	}
}

// TestNoResponseFormatLeavesToolChoiceEmpty: without a ToolStrategy the
// request's tool_choice stays unset and invokeModel passes the zero value
// (provider default / "auto"-equivalent), never inventing a constraint.
func TestNoResponseFormatLeavesToolChoiceEmpty(t *testing.T) {
	model := newToolChoiceRecordingModel([]messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
		messages.AI("done"),
	}, true)

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	calls := model.bindCalls()
	if len(calls) == 0 {
		t.Fatal("expected at least one BindToolsWithOptions call")
	}
	for i, call := range calls {
		if call.ToolChoice != "" {
			t.Fatalf("call %d ToolChoice = %q, want empty", i, call.ToolChoice)
		}
	}
}

// toolChoiceOverrideMiddleware is a WrapModelCall middleware that overrides
// the request's tool_choice, mirroring Python middleware that mutates
// request.tool_choice before the inner model call.
type toolChoiceOverrideMiddleware struct {
	choice any
}

func (m toolChoiceOverrideMiddleware) WrapModelCall(
	ctx context.Context,
	request middleware.ModelRequest,
	handler middleware.ModelHandler,
) (middleware.ModelResponse, error) {
	next, err := request.Override(middleware.WithToolChoice(m.choice))
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	return handler(ctx, next)
}

// TestMiddlewareToolChoiceOverridePassedThrough proves invokeModel threads
// req.ToolChoice (which middleware may have overridden via
// ModelRequest.Override/WithToolChoice) into BindToolsWithOptions.
func TestMiddlewareToolChoiceOverridePassedThrough(t *testing.T) {
	model := newToolChoiceRecordingModel([]messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
		messages.AI("done"),
	}, true)

	agent, err := CreateAgent(
		model,
		[]coretools.Tool{newEchoTool(t)},
		WithAgentMiddleware(toolChoiceOverrideMiddleware{choice: "none"}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	calls := model.bindCalls()
	if len(calls) == 0 {
		t.Fatal("expected at least one BindToolsWithOptions call")
	}
	for i, call := range calls {
		if call.ToolChoice != language.ToolChoiceNone {
			t.Fatalf("call %d ToolChoice = %q, want %q", i, call.ToolChoice, language.ToolChoiceNone)
		}
	}
}

// TestToolStrategyModelWithoutToolBinderFallsBack: a model that does NOT
// implement language.ToolBinder keeps working under a ToolStrategy (the base
// BindTools path), it just cannot enforce the forced tool_choice.(sequenceModel
// itself provides the no-ToolBinder double; this test pins the combined
// behavior with an explicit non-binder wrapper sharing response state.)
func TestToolStrategyModelWithoutToolBinderFallsBack(t *testing.T) {
	strategy := NewToolStrategy(answerSchema())
	toolName := strategy.SchemaSpecs[0].Name

	// sequenceModel does not implement ToolBinder.
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
	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("what is the answer?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["text"] != "42" {
		t.Fatalf("expected structured_response.text=42, got %#v", state["structured_response"])
	}
	if len(model.invocations) != 1 {
		t.Fatalf("expected exactly one model invocation, got %d", len(model.invocations))
	}
}
