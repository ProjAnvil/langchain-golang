// Command tool-calling-errors is the Go port of
// langgraph/examples/tool-calling-errors.ipynb. It explores what happens when a
// tool raises: ToolNode's default error ToolMessage, a custom error message,
// opting out of error handling, and a ReAct agent that recovers from a failed
// tool call on the next turn.
//
// Usage:
//
//	go run .
//
// All models are offline deterministic doubles; no API key or network.
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
	lctools "github.com/projanvil/langchain-golang/langchain/tools"
)

// scriptedModel is an offline ChatModel double returned from BindTools so its
// response cursor advances across the agent loop's re-binds.
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

// getWeather mirrors the notebook tool: it raises on improper input.
func getWeather(_ context.Context, input map[string]any) (coretools.Result, error) {
	location, _ := input["location"].(string)
	switch location {
	case "san francisco":
		return coretools.Result{}, fmt.Errorf("Input queries must be proper nouns")
	case "San Francisco":
		return coretools.Result{Content: "It's 60 degrees and foggy."}, nil
	default:
		return coretools.Result{}, fmt.Errorf("Invalid input.")
	}
}

func weatherTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"get_weather",
		"Call to get the current weather.",
		schema.Object(map[string]schema.Schema{
			"location": schema.String("the location"),
		}, "location"),
		getWeather,
	)
}

func printToolResult(label string, msgs []messages.Message, err error) {
	fmt.Println(label)
	if err != nil {
		fmt.Printf("  error propagated: %v\n", err)
		return
	}
	for _, m := range msgs {
		status := ""
		if m.ResponseMetadata != nil {
			status, _ = m.ResponseMetadata["status"].(string)
		}
		fmt.Printf("  %-8s status=%q content=%q\n", m.Role, status, messages.Text(m))
	}
}

func main() {
	ctx := context.Background()

	weather, err := weatherTool()
	if err != nil {
		fmt.Println("build get_weather tool:", err)
		return
	}
	badCall := messages.Message{
		Role: messages.RoleAI,
		ToolCalls: []messages.ToolCall{
			{ID: "tool_call_id", Name: "get_weather", Args: map[string]any{"location": "san francisco"}},
		},
	}

	// Default error handling (handle_tool_errors=True): the tool exception is
	// converted into an error ToolMessage instead of failing the run.
	defaultNode, err := lctools.NewToolNode([]coretools.Tool{weather})
	if err != nil {
		fmt.Println("build default ToolNode:", err)
		return
	}
	defaultMsgs, err := defaultNode.Invoke(ctx, []messages.Message{badCall}, nil)
	printToolResult("--- default error handling ---", defaultMsgs, err)

	// Custom error message, the Go counterpart of
	// ToolNode(..., handle_tool_errors="Can't use 42 ...").
	customNode, err := lctools.NewToolNode([]coretools.Tool{weather},
		lctools.WithHandleToolErrors(func(err error) (string, bool) {
			return "Input queries must be proper nouns; use a capitalized city name.", true
		}))
	if err != nil {
		fmt.Println("build custom ToolNode:", err)
		return
	}
	customMsgs, err := customNode.Invoke(ctx, []messages.Message{badCall}, nil)
	printToolResult("--- custom error message ---", customMsgs, err)

	// handle_tool_errors=False: return ("", false) so the error propagates.
	strictNode, err := lctools.NewToolNode([]coretools.Tool{weather},
		lctools.WithHandleToolErrors(func(err error) (string, bool) { return "", false }))
	if err != nil {
		fmt.Println("build strict ToolNode:", err)
		return
	}
	strictMsgs, err := strictNode.Invoke(ctx, []messages.Message{badCall}, nil)
	printToolResult("--- handle_tool_errors=False ---", strictMsgs, err)

	// A ReAct agent recovers: the first call fails, the model sees the error
	// ToolMessage, retries with a proper noun, and answers.
	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "c1", Name: "get_weather", Args: map[string]any{"location": "san francisco"}},
			},
		},
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "c2", Name: "get_weather", Args: map[string]any{"location": "San Francisco"}},
			},
		},
		messages.AI("It's 60 degrees and foggy in San Francisco."),
	}}
	agent, err := agents.CreateAgent(model, []coretools.Tool{weather})
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}
	history, err := agent.Invoke(ctx, []messages.Message{messages.Human("what is the weather in san francisco?")})
	if err != nil {
		fmt.Println("agent invoke:", err)
		return
	}
	fmt.Println("--- agent recovering from a tool error ---")
	for _, m := range history {
		status := ""
		if m.ResponseMetadata != nil {
			status, _ = m.ResponseMetadata["status"].(string)
		}
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s status=%q %s\n", m.Role, status, messages.Text(m))
	}
}
