// Command create-react-agent-system-prompt is the Go port of
// langgraph/examples/create-react-agent-system-prompt.ipynb. It sets a
// literal system prompt on create_react_agent and answers a weather question.
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
)

// scriptedModel is an offline ChatModel double. It records every input so the
// example can show that the configured system prompt reaches the model.
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

func main() {
	ctx := context.Background()

	weather, err := coretools.NewFunc(
		"get_weather",
		"Use this to get weather information.",
		schema.Object(map[string]schema.Schema{
			"city": schema.String("the city, nyc or sf"),
		}, "city"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			city, _ := input["city"].(string)
			if city == "nyc" {
				return coretools.Result{Content: "It might be cloudy in nyc"}, nil
			}
			return coretools.Result{}, fmt.Errorf("unknown city %q", city)
		},
	)
	if err != nil {
		fmt.Println("build get_weather tool:", err)
		return
	}

	// The notebook's prompt = "Respond in Italian" is WithAgentSystemPrompt.
	// The scripted model answers in Italian to make the effect visible.
	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "get_weather", Args: map[string]any{"city": "nyc"}},
			},
		},
		messages.AI("Potrebbe essere nuvoloso a New York."),
	}}
	agent, err := agents.CreateAgent(model, []coretools.Tool{weather},
		agents.WithAgentSystemPrompt("Respond in Italian"),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}
	out, err := agent.Invoke(ctx, []messages.Message{messages.Human("What's the weather in NYC?")})
	if err != nil {
		fmt.Println("invoke:", err)
		return
	}

	// Show that the system prompt was actually installed ahead of the
	// conversation before the first model call.
	fmt.Println("--- first model input prefix ---")
	if len(model.invocations) > 0 {
		for _, m := range model.invocations[0] {
			if m.Role == messages.RoleSystem {
				fmt.Printf("  system: %s\n", messages.Text(m))
				break
			}
		}
	}
	fmt.Println("--- final messages ---")
	for _, m := range out {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
}
