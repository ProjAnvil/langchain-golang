package agents

// Parity tests for the Go↔Python gap-filling round covering: return_direct
// tools (BaseTool.return_direct / _make_tools_to_model_edge), ToolStrategy
// HandleErrors retry (_handle_structured_output_error + factory.py:1204-1270),
// the default recursion limit of 9999 (factory.py:1780), and the dynamic-model
// resolver (langgraph.prebuilt's callable-model overload).

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// newDirectEchoTool builds a return-direct single-input echo tool.
func newDirectEchoTool(t *testing.T) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewSimple("direct_echo", "echoes its input, ending the run",
		func(_ context.Context, input string) (coretools.Result, error) {
			return coretools.Result{Content: "direct:" + input}, nil
		})
	if err != nil {
		t.Fatalf("new direct echo tool: %v", err)
	}
	return tool.WithReturnDirect()
}

// TestCreateAgentReturnDirectEndsLoopAfterTool covers Python's
// `_make_tools_to_model_edge` exit condition: when every executed client-side
// tool call targets a return-direct tool, the run ends after the tools node
// and the model is NOT re-invoked.
func TestCreateAgentReturnDirectEndsLoopAfterTool(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "direct_echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
	}}

	agent, err := CreateAgent(model, []coretools.Tool{newDirectEchoTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	msgs, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (human, ai tool call, tool result), got %d: %#v", len(msgs), msgs)
	}
	if msgs[2].Role != messages.RoleTool || msgs[2].ToolCallID != "call_1" || msgs[2].Content != "direct:hi" {
		t.Fatalf("expected direct_echo tool result as last message, got %#v", msgs[2])
	}
	if len(model.invocations) != 1 {
		t.Fatalf("return-direct tool must end the loop: expected exactly 1 model invocation, got %d", len(model.invocations))
	}
}

// TestCreateAgentReturnDirectMixedToolsContinueLoop covers the "all()" side of
// the same Python exit condition: a response mixing a return-direct tool with
// a normal tool does NOT exit — the loop returns to the model.
func TestCreateAgentReturnDirectMixedToolsContinueLoop(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "direct_echo", Args: map[string]any{"tool_input": "hi"}},
				{ID: "call_2", Name: "echo", Args: map[string]any{"tool_input": "yo"}},
			},
		},
		messages.AI("final answer"),
	}}

	agent, err := CreateAgent(model, []coretools.Tool{newDirectEchoTool(t), newEchoTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	msgs, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(model.invocations) != 2 {
		t.Fatalf("mixed return-direct + normal tool must loop back to the model: expected 2 model invocations, got %d", len(model.invocations))
	}
	last := msgs[len(msgs)-1]
	if last.Role != messages.RoleAI || last.Content != "final answer" {
		t.Fatalf("expected terminal AI message, got %#v", last)
	}
}

// TestCreateAgentToolStrategyHandleErrorsRetryDefault covers the default
// HandleErrors=true path (factory.py:1212-1236): a multiple-structured-outputs
// error injects one error ToolMessage per structured call (each answering its
// call, so routing loops back to the model instead of dispatching tools or
// exiting), the model sees the errors, and a corrected single structured call
// completes the run.
func TestCreateAgentToolStrategyHandleErrorsRetryDefault(t *testing.T) {
	strategy := NewToolStrategy(weatherSchema())
	toolName := strategy.SchemaSpecs[0].Name

	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: toolName, Args: map[string]any{"condition": "sunny"}},
				{ID: "call_2", Name: toolName, Args: map[string]any{"condition": "rainy"}},
			},
		},
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_3", Name: toolName, Args: map[string]any{"condition": "sunny", "temperature": 72}},
			},
		},
	}}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("weather?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["condition"] != "sunny" {
		t.Fatalf("expected structured_response with condition=sunny, got %#v", state["structured_response"])
	}
	if len(model.invocations) != 2 {
		t.Fatalf("expected the retry to re-invoke the model (2 invocations), got %d", len(model.invocations))
	}

	msgs, _ := state["messages"].([]messages.Message)
	// human, AI(2 calls), tool(err call_1), tool(err call_2), AI(1 call), tool(structured)
	if len(msgs) != 6 {
		t.Fatalf("expected 6 messages after retry + match, got %d: %#v", len(msgs), msgs)
	}
	for _, id := range []string{"call_1", "call_2"} {
		var retryMsg *messages.Message
		for i := range msgs {
			if msgs[i].Role == messages.RoleTool && msgs[i].ToolCallID == id && strings.HasPrefix(msgs[i].Content, "Error:") {
				retryMsg = &msgs[i]
			}
		}
		if retryMsg == nil {
			t.Fatalf("expected an error ToolMessage answering %s (content starting with \"Error:\"), got %#v", id, msgs)
		}
		if !strings.Contains(retryMsg.Content, "Please fix your mistakes.") {
			t.Fatalf("expected the default STRUCTURED_OUTPUT_ERROR_TEMPLATE, got %q", retryMsg.Content)
		}
		if retryMsg.Name != toolName {
			t.Fatalf("expected error ToolMessage name %q, got %q", toolName, retryMsg.Name)
		}
	}
	// The second model invocation must have seen both injected error messages.
	second := model.invocations[1]
	seenErrs := 0
	for _, m := range second {
		if m.Role == messages.RoleTool && strings.HasPrefix(m.Content, "Error:") {
			seenErrs++
		}
	}
	if seenErrs != 2 {
		t.Fatalf("expected the retried model call to see 2 error ToolMessages, got %d: %#v", seenErrs, second)
	}
}

// TestCreateAgentToolStrategyHandleErrorsCustomMessage covers the
// HandleErrors=string form: the retry ToolMessage carries the caller's
// message verbatim.
func TestCreateAgentToolStrategyHandleErrorsCustomMessage(t *testing.T) {
	strategy := NewToolStrategy(weatherSchema(), WithHandleErrors("fix it please"))
	toolName := strategy.SchemaSpecs[0].Name

	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: toolName, Args: map[string]any{"condition": "sunny"}},
				{ID: "call_2", Name: toolName, Args: map[string]any{"condition": "rainy"}},
			},
		},
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_3", Name: toolName, Args: map[string]any{"condition": "sunny"}},
			},
		},
	}}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("weather?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) < 4 || msgs[2].Content != "fix it please" || msgs[3].Content != "fix it please" {
		t.Fatalf("expected custom HandleErrors message on both retry ToolMessages, got %#v", msgs)
	}
}

// TestCreateAgentToolStrategyHandleErrorsTypeList covers the []string form
// (Python's tuple-of-exception-types): a matching Go type name retries, a
// non-matching one raises.
func TestCreateAgentToolStrategyHandleErrorsTypeList(t *testing.T) {
	multipleCall := func() []messages.Message {
		return []messages.Message{{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "weather_schema", Args: map[string]any{"condition": "sunny"}},
				{ID: "call_2", Name: "weather_schema", Args: map[string]any{"condition": "rainy"}},
			},
		}}
	}

	t.Run("matching type name retries", func(t *testing.T) {
		strategy := NewToolStrategy(weatherSchema(), WithHandleErrors([]string{"MultipleStructuredOutputsError"}))
		model := &sequenceModel{responses: append(multipleCall(), messages.Message{
			Role:      messages.RoleAI,
			ToolCalls: []messages.ToolCall{{ID: "call_3", Name: "weather_schema", Args: map[string]any{"condition": "sunny"}}},
		})}
		agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("weather?")}); err != nil {
			t.Fatalf("expected retry to absorb the multiple-outputs error, got %v", err)
		}
		if len(model.invocations) != 2 {
			t.Fatalf("expected 2 model invocations, got %d", len(model.invocations))
		}
	})

	t.Run("non-matching type name raises", func(t *testing.T) {
		strategy := NewToolStrategy(weatherSchema(), WithHandleErrors([]string{"SomeOtherError"}))
		model := &sequenceModel{responses: multipleCall()}
		agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("weather?")})
		var multiErr *MultipleStructuredOutputsError
		if !errors.As(err, &multiErr) {
			t.Fatalf("expected MultipleStructuredOutputsError to propagate, got %v", err)
		}
	})
}

// loopToolModel answers every call with an echo tool call and records only a
// counter (no per-invocation message copies), so a default-recursion-limit run
// stays cheap.
type loopToolModel struct{ calls int }

func (m *loopToolModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.calls++
	return messages.Message{
		Role:      messages.RoleAI,
		ToolCalls: []messages.ToolCall{{ID: fmt.Sprintf("call_%d", m.calls), Name: "echo", Args: map[string]any{"tool_input": "x"}}},
	}, nil
}

func (m *loopToolModel) Batch(context.Context, [][]messages.Message, ...runnables.Option) ([]messages.Message, error) {
	return nil, fmt.Errorf("loopToolModel: batch not supported")
}

func (m *loopToolModel) Stream(context.Context, []messages.Message, ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return nil, fmt.Errorf("loopToolModel: streaming not supported")
}

func (m *loopToolModel) InputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *loopToolModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *loopToolModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) { return m, nil }

func (m *loopToolModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// TestCreateAgentDefaultRecursionLimit9999 covers factory.py:1780 parity: an
// agent built without WithAgentRecursionLimit compiles with a limit of 9999
// (not the graph package's platform default of 100), so a never-ending tool
// loop trips GraphRecursionError with Limit 9999.
func TestCreateAgentDefaultRecursionLimit9999(t *testing.T) {
	if testing.Short() {
		t.Skip("exercises ~10k supersteps")
	}
	model := &loopToolModel{}
	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("loop forever")})
	var recErr *types.GraphRecursionError
	if !errors.As(err, &recErr) {
		t.Fatalf("expected GraphRecursionError, got %v", err)
	}
	if recErr.Limit != 9999 {
		t.Fatalf("expected default recursion limit 9999, got %d", recErr.Limit)
	}
}

// TestCreateAgentDynamicModel covers the callable-model overload: the resolver
// picks the model per call (here: a tool-calling model first, a different
// terminal model after), and both instances actually run.
func TestCreateAgentDynamicModel(t *testing.T) {
	first := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
	}}
	second := &sequenceModel{responses: []messages.Message{messages.AI("from-second-model")}}

	var calls atomic.Int32
	agent, err := CreateAgent(nil, []coretools.Tool{newEchoTool(t)}, WithAgentDynamicModel(
		func(state map[string]any, _ runtime.Runtime) language.ChatModel {
			if calls.Add(1) == 1 {
				return first
			}
			return second
		}))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	msgs, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if last := msgs[len(msgs)-1]; last.Content != "from-second-model" {
		t.Fatalf("expected final answer from the second model, got %#v", last)
	}
	if len(first.invocations) != 1 || len(second.invocations) != 1 {
		t.Fatalf("expected each model to run exactly once, got first=%d second=%d", len(first.invocations), len(second.invocations))
	}
}
