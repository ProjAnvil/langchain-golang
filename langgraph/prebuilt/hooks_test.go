package prebuilt

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
)

// TestCreateReactAgentPrePostModelHooks covers the deprecated-prebuilt
// pre_model_hook / post_model_hook parameters, mapped onto agents middleware:
// both hooks run around each model call, and non-messages state keys they
// return are persisted into the final state.
func TestCreateReactAgentPrePostModelHooks(t *testing.T) {
	var preRuns, postRuns atomic.Int32

	model := &reactSequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateReactAgent(model, nil,
		WithPreModelHook(func(_ context.Context, state map[string]any) (map[string]any, error) {
			preRuns.Add(1)
			return map[string]any{"pre_ran": true}, nil
		}),
		WithPostModelHook(func(_ context.Context, state map[string]any) (map[string]any, error) {
			postRuns.Add(1)
			return map[string]any{"post_ran": true}, nil
		}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if preRuns.Load() != 1 || postRuns.Load() != 1 {
		t.Fatalf("expected each hook to run once, got pre=%d post=%d", preRuns.Load(), postRuns.Load())
	}
	if state["pre_ran"] != true || state["post_ran"] != true {
		t.Fatalf("expected hook state keys to persist, got %#v", state)
	}
}

// countingModel is a minimal ChatModel that always returns a terminal message
// and counts invocations, for the dynamic-model test (a per-call swapped
// model must actually receive the call).
type countingModel struct {
	label string
	calls atomic.Int32
}

func (m *countingModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.calls.Add(1)
	return messages.AI("answer from " + m.label), nil
}

func (m *countingModel) Batch(context.Context, [][]messages.Message, ...runnables.Option) ([]messages.Message, error) {
	return nil, fmt.Errorf("countingModel: batch not supported")
}

func (m *countingModel) Stream(context.Context, []messages.Message, ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return nil, fmt.Errorf("countingModel: streaming not supported")
}

func (m *countingModel) InputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *countingModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *countingModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) { return m, nil }
func (m *countingModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// TestCreateReactAgentDynamicModel covers the deprecated-prebuilt callable
// model overload: the resolver swaps the model between calls (first call uses
// a tool-calling model, the second a different terminal model), and a nil
// static model is accepted when a resolver is configured.
func TestCreateReactAgentDynamicModel(t *testing.T) {
	first := &reactSequenceModel{responses: []messages.Message{{
		Role:      messages.RoleAI,
		ToolCalls: []messages.ToolCall{{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}}},
	}}}
	second := &countingModel{label: "second"}

	var calls atomic.Int32
	agent, err := CreateReactAgent(nil, []coretools.Tool{newPrebuiltEchoTool(t)}, WithDynamicModel(
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
	if last := msgs[len(msgs)-1]; last.Content != "answer from second" {
		t.Fatalf("expected final answer from the swapped-in model, got %#v", last)
	}
	if second.calls.Load() != 1 {
		t.Fatalf("expected the second model to receive the second call, got %d", second.calls.Load())
	}
}

func newPrebuiltEchoTool(t *testing.T) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewSimple("echo", "echoes its input",
		func(_ context.Context, input string) (coretools.Result, error) {
			return coretools.Result{Content: "echo:" + input}, nil
		})
	if err != nil {
		t.Fatalf("new echo tool: %v", err)
	}
	return tool
}
