// Command create-react-agent is the Go port of
// langgraph/examples/create-react-agent.ipynb. It builds a ReAct agent around
// an offline ChatModel double, drives one model -> tool -> model loop, then
// drives a follow-up question that needs no tool call.
//
// Usage:
//
//	go run .
//
// No API key or external service is required: the model below is a
// deterministic in-process double standing in for a provider chat model.
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
)

// scriptedModel is an offline ChatModel double. BindTools returns the
// receiver itself so the response cursor advances across the per-iteration
// re-binds CreateAgent performs (see examples/langchain/quickstart for the rationale).
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

func printMessages(label string, msgs []messages.Message) {
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

	// The notebook's get_weather(city: Literal["nyc", "sf"]) becomes a
	// structured tool with one string argument and a deterministic body.
	weather, err := coretools.NewFunc(
		"get_weather",
		"Use this to get weather information.",
		schema.Object(map[string]schema.Schema{
			"city": schema.String("the city, nyc or sf"),
		}, "city"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			city, _ := input["city"].(string)
			switch city {
			case "nyc":
				return coretools.Result{Content: "It might be cloudy in nyc"}, nil
			case "sf":
				return coretools.Result{Content: "It's always sunny in sf"}, nil
			default:
				return coretools.Result{}, fmt.Errorf("unknown city %q", city)
			}
		},
	)
	if err != nil {
		fmt.Println("build get_weather tool:", err)
		return
	}

	// Turn 1: "what is the weather in sf" -> the model requests get_weather,
	// the ToolNode runs it, and the model produces the final answer.
	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "get_weather", Args: map[string]any{"city": "sf"}},
			},
		},
		messages.AI("It's always sunny in sf."),
	}}
	agent, err := agents.CreateAgent(model, []coretools.Tool{weather})
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}
	out, err := agent.Invoke(ctx, []messages.Message{messages.Human("what is the weather in sf")})
	if err != nil {
		fmt.Println("invoke:", err)
		return
	}
	printMessages("--- what is the weather in sf ---", out)
	fmt.Printf("model calls: %d\n\n", len(model.invocations))

	// Turn 2: "who built you?" -> no tool call, the model answers directly.
	chatModel := &scriptedModel{responses: []messages.Message{
		messages.AI("I am an offline demo agent built with langgraph-go."),
	}}
	chatAgent, err := agents.CreateAgent(chatModel, []coretools.Tool{weather})
	if err != nil {
		fmt.Println("create chat agent:", err)
		return
	}
	out2, err := chatAgent.Invoke(ctx, []messages.Message{messages.Human("who built you?")})
	if err != nil {
		fmt.Println("invoke chat:", err)
		return
	}
	printMessages("--- who built you? ---", out2)
}
