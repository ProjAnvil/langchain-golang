// Command create-react-agent-hitl is the Go port of
// langgraph/examples/create-react-agent-hitl.ipynb. A MemorySaver checkpointer
// plus an interrupt before the tools node lets a human inspect (and edit) the
// pending tool call before it executes.
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

// scriptedModel is an offline ChatModel double that returns itself from
// BindTools so the response cursor advances across the agent loop's re-binds.
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

	var toolExecutions int
	weather, err := coretools.NewFunc(
		"get_weather",
		"Use this to get weather information from a given location.",
		schema.Object(map[string]schema.Schema{
			"location": schema.String("the location"),
		}, "location"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			toolExecutions++
			location, _ := input["location"].(string)
			return coretools.Result{Content: "It's always sunny in " + location}, nil
		},
	)
	if err != nil {
		fmt.Println("build get_weather tool:", err)
		return
	}

	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ID:   "ai-tool-call",
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "get_weather", Args: map[string]any{"location": "SF, CA"}},
			},
		},
		messages.AI("It's always sunny in San Francisco."),
	}}

	// interrupt_before tools + a checkpointer makes the pause durable and
	// resumable (the Go equivalent of the notebook's interrupt_before=["tools"]).
	agent, err := agents.CreateAgent(model, []coretools.Tool{weather},
		agents.WithAgentCheckpointer(checkpoint.NewMemorySaver()),
		agents.WithAgentInterruptBefore(agents.ToolsNodeName),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}
	cfg := graphpkg.Options{ThreadID: "42"}

	// Turn 1: run until the tools node, then pause before executing anything.
	first, err := agent.Graph.InvokeWithOptions(ctx,
		map[string]any{"messages": []messages.Message{messages.Human("what is the weather in SF, CA?")}}, cfg)
	if err != nil {
		fmt.Println("first invoke:", err)
		return
	}
	fmt.Printf("paused with %d interrupt(s); tool executions so far: %d\n", len(first.Interrupts), toolExecutions)
	paused, _ := first.Values["messages"].([]messages.Message)
	printMessages("--- state at pause ---", paused)

	// Inspect the planned next step from the checkpoint.
	snap, err := agent.Graph.GetState(ctx, checkpoint.Config{ThreadID: "42"})
	if err != nil {
		fmt.Println("get state:", err)
		return
	}
	fmt.Println("snapshot next:", snap.Next)

	// The notebook edits the pending tool call before resuming. Locate the
	// last AI message and replace its tool-call arguments.
	for i := len(paused) - 1; i >= 0; i-- {
		if paused[i].Role == messages.RoleAI && len(paused[i].ToolCalls) > 0 {
			edited := paused[i]
			edited.ToolCalls = append([]messages.ToolCall(nil), edited.ToolCalls...)
			edited.ToolCalls[0].Args = map[string]any{"location": "San Francisco"}
			// asNode is the node whose update is being made; routing from
			// "model" keeps "tools" as the next planned step on resume, the
			// Go equivalent of Python's update_state without as_node.
			if _, err := agent.Graph.UpdateState(ctx, checkpoint.Config{ThreadID: "42"},
				map[string]any{"messages": []messages.Message{edited}}, agents.ModelNodeName); err != nil {
				fmt.Println("update state:", err)
				return
			}
			fmt.Println("edited pending tool call location -> San Francisco")
			break
		}
	}

	// Turn 2: resume the same thread with nil input. A state update re-arms
	// the boundary interrupt once, so this resume pauses again at tools with
	// the edited call pending; a second resume crosses it and executes.
	second, err := agent.Graph.InvokeWithOptions(ctx, nil, cfg)
	if err != nil {
		fmt.Println("resume invoke:", err)
		return
	}
	fmt.Printf("after first resume: interrupts %d; tool executions %d\n",
		len(second.Interrupts), toolExecutions)

	finalResult := second
	if len(second.Interrupts) > 0 {
		finalResult, err = agent.Graph.InvokeWithOptions(ctx, nil, cfg)
		if err != nil {
			fmt.Println("second resume invoke:", err)
			return
		}
	}
	fmt.Printf("finished; interrupts now %d; model calls total %d; tool executions %d\n",
		len(finalResult.Interrupts), len(model.invocations), toolExecutions)
	final, _ := finalResult.Values["messages"].([]messages.Message)
	printMessages("--- final state ---", final)
}
