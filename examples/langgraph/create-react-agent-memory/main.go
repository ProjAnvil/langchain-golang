// Command create-react-agent-memory is the Go port of
// langgraph/examples/create-react-agent-memory.ipynb. A MemorySaver
// checkpointer retains chat context across two turns on the same thread, so
// the second question sees the messages produced by the first.
//
// Usage:
//
//	go run .
//
// The ChatModel is a deterministic offline double: no API key or network.
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
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
)

// scriptedModel is an offline ChatModel double whose response cursor advances
// across the per-iteration re-binds CreateAgent performs.
type scriptedModel struct {
	mu          sync.Mutex
	responses   []messages.Message
	idx         int
	invocations [][]messages.Message
}

func (m *scriptedModel) Invoke(_ context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.invocations = append(m.invocations, append([]messages.Message(nil), input...))
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

func stateMessages(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

func printHistory(label string, msgs []messages.Message) {
	fmt.Println(label)
	for _, m := range msgs {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
}

func main() {
	ctx := context.Background()

	weather, err := coretools.NewFunc(
		"get_weather",
		"Use this to get weather information.",
		schema.Object(map[string]schema.Schema{
			"location": schema.String("the location"),
		}, "location"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			location, _ := input["location"].(string)
			return coretools.Result{Content: "It might be cloudy in nyc for " + location}, nil
		},
	)
	if err != nil {
		fmt.Println("build get_weather tool:", err)
		return
	}

	// Two scripted answers: one per turn. The checkpointer supplies the
	// accumulated message history to the model on the second turn.
	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "get_weather", Args: map[string]any{"location": "NYC"}},
			},
		},
		messages.AI("It might be cloudy in nyc."),
		messages.AI("NYC is known for its museums, Broadway and Central Park."),
	}}
	// MemorySaver is the Go counterpart of Python's MemorySaver checkpointer.
	agent, err := agents.CreateAgent(model, []coretools.Tool{weather},
		agents.WithAgentCheckpointer(checkpoint.NewMemorySaver()),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}
	cfg := graphpkg.Options{ThreadID: "1"}

	// Turn 1: "What's the weather in NYC?"
	first, err := agent.Graph.InvokeWithOptions(ctx,
		map[string]any{"messages": []messages.Message{messages.Human("What's the weather in NYC?")}}, cfg)
	if err != nil {
		fmt.Println("first invoke:", err)
		return
	}
	msgs1 := stateMessages(first.Values)
	printHistory("--- after turn 1 ---", msgs1)
	fmt.Printf("messages retained: %d\n\n", len(msgs1))

	// Turn 2: "What's it known for?" on the same thread. The earlier
	// human/tool/AI messages are still present in the restored state.
	second, err := agent.Graph.InvokeWithOptions(ctx,
		map[string]any{"messages": []messages.Message{messages.Human("What's it known for?")}}, cfg)
	if err != nil {
		fmt.Println("second invoke:", err)
		return
	}
	msgs2 := stateMessages(second.Values)
	printHistory("--- after turn 2 (same thread) ---", msgs2)
	fmt.Printf("messages retained: %d\n", len(msgs2))
	fmt.Printf("model calls total: %d\n", len(model.invocations))
}
