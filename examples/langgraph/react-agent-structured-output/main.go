// Command react-agent-structured-output is the Go port of
// langgraph/examples/react-agent-structured-output.ipynb. It builds a graph
// that forces the model to answer through a schema tool: the agent calls the
// weather tool, then emits a "WeatherResponse" tool call whose arguments are
// parsed into a typed Go struct by the respond node.
//
// Usage:
//
//	go run .
//
// The model is an offline deterministic double; no API key or network.
package main

import (
	"context"
	"encoding/json"
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

// WeatherResponse is the Go equivalent of the notebook's pydantic model.
type WeatherResponse struct {
	Temperature   float64 `json:"temperature"`
	WindDirection string  `json:"wind_direction"`
	WindSpeed     float64 `json:"wind_speed"`
}

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

	// The notebook's get_weather(city: Literal["nyc", "sf"]).
	weather, err := coretools.NewFunc(
		"get_weather",
		"Use this to get weather information.",
		schema.Object(map[string]schema.Schema{
			"city": schema.String("the city, nyc or sf"),
		}, "city"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			city, _ := input["city"].(string)
			if city == "sf" {
				return coretools.Result{Content: "It is 75 degrees and sunny in SF, with 3 mph winds in the South-East direction"}, nil
			}
			return coretools.Result{Content: "It is cloudy in NYC"}, nil
		},
	)
	if err != nil {
		fmt.Println("build get_weather tool:", err)
		return
	}

	// The scripted model first calls get_weather, then answers through the
	// "WeatherResponse" schema tool with tool-choice forcing (offline stub).
	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "c1", Name: "get_weather", Args: map[string]any{"city": "sf"}},
			},
		},
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "c2", Name: "WeatherResponse", Args: map[string]any{
					"temperature":    75.0,
					"wind_direction": "SE",
					"wind_speed":     3.0,
				}},
			},
		},
	}}

	callModel := func(rt runtime.Runtime, state map[string]any) (any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		resp, err := model.Invoke(rt, msgs)
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": []messages.Message{resp}}, nil
	}

	// respond constructs the typed struct from the schema tool call's args
	// and appends the ToolMessage providers require after such a call.
	respond := func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		last := msgs[len(msgs)-1]
		call := last.ToolCalls[0]
		raw, err := json.Marshal(call.Args)
		if err != nil {
			return nil, err
		}
		var answer WeatherResponse
		if err := json.Unmarshal(raw, &answer); err != nil {
			return nil, err
		}
		toolMsg := messages.Tool(call.ID, "Here is your structured response")
		return map[string]any{
			"final_response": answer,
			"messages":       []messages.Message{toolMsg},
		}, nil
	}

	// should_continue routes a WeatherResponse tool call to respond; any
	// other tool call goes through the ToolNode.
	shouldContinue := func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		last := msgs[len(msgs)-1]
		if len(last.ToolCalls) == 1 && last.ToolCalls[0].Name == "WeatherResponse" {
			return graphpkg.To("respond"), nil
		}
		return graphpkg.To("tools"), nil
	}

	toolNode, err := lctools.NewToolNode([]coretools.Tool{weather})
	if err != nil {
		fmt.Println("build ToolNode:", err)
		return
	}

	builder := graphpkg.NewStateGraph()
	builder.AddReducer("messages", channels.MessagesReducer)
	builder.AddNode("agent", callModel)
	builder.AddNode("respond", respond)
	builder.AddNode("tools", prebuilt.ToolNode(toolNode))
	builder.AddEdge(types.START, "agent")
	builder.AddConditionalEdges("agent", shouldContinue)
	builder.AddEdge("tools", "agent")
	builder.AddEdge("respond", types.END)

	graph, err := builder.Compile()
	if err != nil {
		fmt.Println("compile graph:", err)
		return
	}
	result, err := graph.Invoke(ctx, map[string]any{
		"messages": []messages.Message{messages.Human("what's the weather in SF?")},
	})
	if err != nil {
		fmt.Println("invoke graph:", err)
		return
	}
	answer, _ := result.Values["final_response"].(WeatherResponse)
	fmt.Println("--- structured output (tool-call strategy) ---")
	fmt.Printf("final_response: %+v\n", answer)
	out, _ := result.Values["messages"].([]messages.Message)
	for _, m := range out {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
}
