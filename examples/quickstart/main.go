// Command quickstart is the smallest agents.CreateAgent walkthrough: it
// builds an agent around a scripted offline model, drives one full
// model -> tool -> model loop with Invoke, and replays the same loop through
// Agent.StreamEvents so the event stream can be observed live.
//
// Usage:
//
//	go run ./examples/quickstart
//
// No API key is required: the model below is a deterministic in-process
// double, standing in for a partner chat model (e.g. partners/openai).
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

// scriptedModel is a minimal offline ChatModel double (the same pattern the
// agents test suite uses): it returns Responses in order and returns ITSELF
// from BindTools, so its response cursor advances across the per-iteration
// re-binds CreateAgent performs. language.FakeChatModel would not: its
// BindTools hands back a fresh copy whose cursor never writes back.
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

	// One offline tool. With a real provider model this is where you would
	// expose your own business logic to the agent.
	echo, err := coretools.NewSimple("echo", "echoes its input back",
		func(_ context.Context, input string) (coretools.Result, error) {
			return coretools.Result{Content: "echo:" + input}, nil
		})
	if err != nil {
		fmt.Println("build echo tool:", err)
		return
	}

	// Scripted run: the model first requests the echo tool, then answers.
	invokeModel := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hello agent"}},
			},
		},
		messages.AI("The tool replied: echo:hello agent"),
	}}

	agent, err := agents.CreateAgent(invokeModel, []coretools.Tool{echo},
		agents.WithAgentSystemPrompt("You are a helpful offline demo agent."),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}

	out, err := agent.Invoke(ctx, []messages.Message{messages.Human("say hi through the echo tool")})
	if err != nil {
		fmt.Println("invoke:", err)
		return
	}
	fmt.Println("--- Invoke (full message history) ---")
	for _, m := range out {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}

	// Streaming run: a fresh scripted model; StreamEvents surfaces node
	// lifecycle, tool dispatch, model deltas, and the terminal event.
	streamModel := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_2", Name: "echo", Args: map[string]any{"tool_input": "streamed"}},
			},
		},
		messages.AI("Streamed tool reply: echo:streamed"),
	}}
	streamAgent, err := agents.CreateAgent(streamModel, []coretools.Tool{echo})
	if err != nil {
		fmt.Println("create stream agent:", err)
		return
	}

	fmt.Println("--- StreamEvents ---")
	stream, err := streamAgent.StreamEvents(ctx, []messages.Message{messages.Human("stream it")})
	if err != nil {
		fmt.Println("stream events:", err)
		return
	}
	defer func() { _ = stream.Close() }()
	for {
		ev, ok, err := stream.Next(ctx)
		if err != nil {
			fmt.Println("stream next:", err)
			return
		}
		if !ok {
			break
		}
		switch ev.Type {
		case agents.StreamNodeStart, agents.StreamNodeEnd:
			fmt.Printf("  %-10s node=%s\n", ev.Type, ev.Node)
		case agents.StreamModelDelta:
			if ev.Text != "" {
				fmt.Printf("  %-10s text=%q\n", ev.Type, ev.Text)
			}
		case agents.StreamModelEnd:
			fmt.Printf("  %-10s assembled=%q\n", ev.Type, ev.Message.Content)
		case agents.StreamToolStart:
			fmt.Printf("  %-10s tool=%s args=%v\n", ev.Type, ev.ToolName, ev.ToolArgs)
		case agents.StreamToolEnd:
			fmt.Printf("  %-10s tool=%s result=%v\n", ev.Type, ev.ToolName, ev.ToolResult)
		case agents.StreamEnd:
			if ev.Err != nil {
				fmt.Printf("  %-10s error=%v\n", ev.Type, ev.Err)
			} else if ev.Message != nil {
				fmt.Printf("  %-10s final=%q\n", ev.Type, ev.Message.Content)
			}
		}
	}
}
