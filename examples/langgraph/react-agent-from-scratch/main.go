// Command react-agent-from-scratch is the Go port of
// langgraph/examples/react-agent-from-scratch.ipynb. It builds the ReAct loop
// by hand with a StateGraph instead of create_react_agent: an "agent" node
// that calls a tool-bound model with a system prompt, a "tools" node, a
// conditional edge that decides whether to continue, and a messages reducer.
//
// Usage:
//
//	go run .
//
// The model is an offline deterministic double; no API key or network.
package main

import (
	"context"
	"fmt"
	"sync"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	lctools "github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/prebuilt"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// scriptedModel is an offline ChatModel double returned from BindTools.
type scriptedModel struct {
	mu        sync.Mutex
	responses []messages.Message
	idx       int
}

func (m *scriptedModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("scriptedModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}
func (m *scriptedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
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
func (m *scriptedModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	resp, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{resp}), nil
}
func (m *scriptedModel) InputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *scriptedModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *scriptedModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) { return m, nil }
func (m *scriptedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true, Streaming: true}
}

func main() {
	ctx := context.Background()

	// The notebook's get_weather(location) placeholder tool.
	weather, err := coretools.NewSimple("get_weather", "Call to get the weather from a specific location.",
		func(_ context.Context, input string) (coretools.Result, error) {
			return coretools.Result{Content: "It's sunny in San Francisco."}, nil
		})
	if err != nil {
		fmt.Println("build get_weather tool:", err)
		return
	}
	toolList := []coretools.Tool{weather}

	// The model first requests the tool, then answers after seeing its result.
	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "get_weather", Args: map[string]any{"tool_input": "sf"}},
			},
		},
		messages.AI("It's sunny in San Francisco."),
	}}

	// call_model prepends the SystemMessage the notebook adds in its node.
	callModel := func(rt runtime.Runtime, state map[string]any) (any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		input := append([]messages.Message{
			messages.System("You are a helpful AI assistant, please respond to the users query to the best of your ability!"),
		}, msgs...)
		resp, err := model.Invoke(rt, input)
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": []messages.Message{resp}}, nil
	}

	// should_continue: tool calls -> "tools", otherwise END.
	shouldContinue := func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		if len(msgs) > 0 && len(msgs[len(msgs)-1].ToolCalls) > 0 {
			return graphpkg.To("tools"), nil
		}
		return graphpkg.To(types.END), nil
	}

	toolNode, err := lctools.NewToolNode(toolList)
	if err != nil {
		fmt.Println("build ToolNode:", err)
		return
	}

	builder := graphpkg.NewStateGraph()
	builder.AddReducer("messages", channels.MessagesReducer)
	builder.AddNode("agent", callModel)
	builder.AddNode("tools", prebuilt.ToolNode(toolNode))
	builder.AddEdge(types.START, "agent")
	builder.AddConditionalEdges("agent", shouldContinue)
	builder.AddEdge("tools", "agent")

	graph, err := builder.Compile()
	if err != nil {
		fmt.Println("compile graph:", err)
		return
	}
	result, err := graph.Invoke(ctx, map[string]any{
		"messages": []messages.Message{messages.Human("what is the weather in sf")},
	})
	if err != nil {
		fmt.Println("invoke graph:", err)
		return
	}
	out, _ := result.Values["messages"].([]messages.Message)
	fmt.Println("--- react agent from scratch ---")
	for _, m := range out {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
}
