// Command hitl-interrupt demonstrates human-in-the-loop pausing with
// agents.CreateAgent: a MemorySaver checkpointer plus
// WithAgentInterruptBefore(ToolsNodeName) makes the run pause before any tool
// executes. The first invoke returns the paused state (the model has already
// requested a tool call, and the tool has NOT run); a second invoke with nil
// input on the same thread resumes the run, executes the tool, and finishes.
//
// Usage:
//
//	go run ./examples/hitl-interrupt
//
// No API key is required: the model is a deterministic in-process double.
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
// BindTools so its response cursor advances across the agent loop's
// per-iteration re-binds (see examples/quickstart for the full rationale).
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
func (m *scriptedModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}

func (m *scriptedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
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

	var approvals int
	// A tool a human should sign off on before it runs.
	charge, err := coretools.NewSimple("charge_card", "charges the customer's credit card",
		func(_ context.Context, input string) (coretools.Result, error) {
			approvals++
			return coretools.Result{Content: "charged " + input}, nil
		})
	if err != nil {
		fmt.Println("build charge tool:", err)
		return
	}

	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "charge_card", Args: map[string]any{"tool_input": "$120"}},
			},
		},
		messages.AI("Your card has been charged $120."),
	}}

	// The checkpointer is what makes the pause durable and resumable; without
	// it the same interrupt_before option would pause with no way back.
	saver := checkpoint.NewMemorySaver()
	agent, err := agents.CreateAgent(model, []coretools.Tool{charge},
		agents.WithAgentCheckpointer(saver),
		agents.WithAgentInterruptBefore(agents.ToolsNodeName),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}

	// --- Turn 1: run until the tools node, then pause. ---
	first, err := agent.Graph.InvokeWithOptions(ctx,
		map[string]any{"messages": []messages.Message{messages.Human("charge my card $120")}},
		graphpkg.Options{ThreadID: "order-1234"},
	)
	if err != nil {
		fmt.Println("first invoke:", err)
		return
	}
	fmt.Printf("paused with %d interrupt(s) before %q; tool executions so far: %d\n",
		len(first.Interrupts), agents.ToolsNodeName, approvals)
	pausedMsgs, _ := first.Values["messages"].([]messages.Message)
	printMessages("--- state at pause ---", pausedMsgs)

	// In a real application a human reviews the pending tool call here and
	// decides whether to resume. Boundary interrupts resume with a nil
	// Resume value (the Go equivalent of Python's invoke(None, config)).

	// --- Turn 2: resume the same thread; the tools node runs, then the
	// model produces the final answer. The first model call is NOT replayed. ---
	second, err := agent.Graph.InvokeWithOptions(ctx, nil,
		graphpkg.Options{ThreadID: "order-1234"},
	)
	if err != nil {
		fmt.Println("resume invoke:", err)
		return
	}
	fmt.Printf("resumed; interrupts now %d; model calls total %d; tool executions %d\n",
		len(second.Interrupts), len(model.invocations), approvals)
	finalMsgs, _ := second.Values["messages"].([]messages.Message)
	printMessages("--- final state ---", finalMsgs)
}
