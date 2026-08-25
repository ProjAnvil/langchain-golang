package prebuilt

import (
	"context"
	"fmt"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/langchain/tools"
)

// reactSequenceModel is a minimal language.ChatModel test double that returns
// its responses in order. It mirrors the sequenceModel in
// langchain/agents/create_agent_test.go:37 (BindTools returns the model itself
// so response state is shared across the bind+invoke cycles the agent's model
// node performs each loop iteration; language.FakeChatModel.BindTools returns
// a fresh copy and cannot sequence responses through that loop).
type reactSequenceModel struct {
	responses []messages.Message
	idx       int
}

func (m *reactSequenceModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("reactSequenceModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

func (m *reactSequenceModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
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

func (m *reactSequenceModel) Stream(context.Context, []messages.Message, ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return nil, fmt.Errorf("reactSequenceModel: streaming not supported")
}

func (m *reactSequenceModel) InputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *reactSequenceModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *reactSequenceModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}

func (m *reactSequenceModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// Mirrors the basic create_react_agent tool-loop scenario of
// tests/test_react_agent.py: the model requests a tool, the tool runs, the
// model then answers without tool calls, and the loop ends.
func TestCreateReactAgentToolLoop(t *testing.T) {
	model := &reactSequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
		messages.AI("final answer"),
	}}
	echo := funcTool(t, "echo", func(_ context.Context, args map[string]any) (tools.Result, error) {
		return tools.Result{Content: "echo:" + args["tool_input"].(string)}, nil
	})

	agent, err := CreateReactAgent(model, []coretools.Tool{echo})
	if err != nil {
		t.Fatalf("CreateReactAgent() error = %v", err)
	}
	if agent.Graph == nil {
		t.Fatal("agent.Graph is nil")
	}
	res, err := agent.Graph.Invoke(context.Background(), map[string]any{
		"messages": []messages.Message{messages.Human("say hi back")},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 4 {
		t.Fatalf("len(messages) = %d, want 4 (human, ai tool call, tool result, final ai): %+v", len(msgs), msgs)
	}
	if msgs[2].Role != messages.RoleTool || msgs[2].Content != "echo:hi" || msgs[2].ToolCallID != "call_1" {
		t.Errorf("messages[2] = %+v, want tool result echo:hi", msgs[2])
	}
	if msgs[3].Role != messages.RoleAI || msgs[3].Content != "final answer" {
		t.Errorf("messages[3] = %+v, want the final AI answer", msgs[3])
	}
}

// Python's create_react_agent(model=None) path surfaces a validation error;
// Go delegates to CreateAgent's "model is required" error
// (create_agent.go:543-545).
func TestCreateReactAgentNilModelErrors(t *testing.T) {
	_, err := CreateReactAgent(nil, nil)
	if err == nil {
		t.Fatal("CreateReactAgent(nil, nil) error = nil, want 'model is required'")
	}
}

// Agent options pass straight through to CreateAgent.
func TestCreateReactAgentPassesOptions(t *testing.T) {
	model := &reactSequenceModel{responses: []messages.Message{messages.AI("ok")}}
	agent, err := CreateReactAgent(model, nil, agents.WithAgentName("reacty"))
	if err != nil {
		t.Fatalf("CreateReactAgent() error = %v", err)
	}
	if agent.Name != "reacty" {
		t.Fatalf("agent.Name = %q, want reacty", agent.Name)
	}
}
