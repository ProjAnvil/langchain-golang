// Command tool-calling is the Go port of langgraph/examples/tool-calling.ipynb.
// It demonstrates the core tool-calling surfaces the notebook exercises:
// defining a tool, invoking it directly, turning a tool call into a ToolMessage
// with ToolNode, driving a ReAct agent through a tool, forcing tool choice,
// disabling parallel tool calls, and return-direct tools.
//
// Usage:
//
//	go run .
//
// Every model is an offline deterministic double; no API key or network.
package main

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	lctools "github.com/projanvil/langchain-golang/langchain/tools"
	"sync"
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

// multiply mirrors the notebook's @tool multiply(a: int, b: int) -> int.
func multiplyTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"multiply",
		"Multiply two numbers.",
		schema.Object(map[string]schema.Schema{
			"a": schema.Integer("first operand"),
			"b": schema.Integer("second operand"),
		}, "a", "b"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			a, _ := input["a"].(int)
			b, _ := input["b"].(int)
			return coretools.Result{Content: fmt.Sprint(a * b)}, nil
		},
	)
}

func main() {
	ctx := context.Background()

	multiply, err := multiplyTool()
	if err != nil {
		fmt.Println("build multiply tool:", err)
		return
	}

	// Cell 6: invoke the tool directly with its arguments -> 42.
	direct, err := multiply.Invoke(ctx, map[string]any{"a": 6, "b": 7})
	if err != nil {
		fmt.Println("direct invoke:", err)
		return
	}
	fmt.Println("--- direct tool invoke ---")
	fmt.Printf("multiply(6, 7) = %s\n\n", direct.Content)

	// Cell 8: a tool call (name + args) is dispatched by ToolNode into a
	// ToolMessage, the Go equivalent of multiply.invoke(tool_call_dict).
	toolNode, err := lctools.NewToolNode([]coretools.Tool{multiply})
	if err != nil {
		fmt.Println("build ToolNode:", err)
		return
	}
	callMsg := messages.Message{
		Role: messages.RoleAI,
		ToolCalls: []messages.ToolCall{
			{ID: "1", Name: "multiply", Args: map[string]any{"a": 42, "b": 7}},
		},
	}
	toolMsgs, err := toolNode.Invoke(ctx, []messages.Message{callMsg}, nil)
	if err != nil {
		fmt.Println("ToolNode invoke:", err)
		return
	}
	fmt.Println("--- ToolNode: tool call -> ToolMessage ---")
	for _, m := range toolMsgs {
		fmt.Printf("  %-8s id=%s result=%s\n", m.Role, m.ToolCallID, messages.Text(m))
	}
	fmt.Println()

	// Cells 13/79: a ReAct agent built around the multiply tool.
	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "multiply", Args: map[string]any{"a": 42, "b": 7}},
			},
		},
		messages.AI("42 x 7 = 294"),
	}}
	agent, err := agents.CreateAgent(model, []coretools.Tool{multiply})
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}
	history, err := agent.Invoke(ctx, []messages.Message{messages.Human("what's 42 x 7?")})
	if err != nil {
		fmt.Println("agent invoke:", err)
		return
	}
	fmt.Println("--- create_react_agent ---")
	for _, m := range history {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
	fmt.Println()

	// Cell 59: bind_tools(tool_choice={type:tool,name:greet}) -> a forced
	// tool choice. Go expresses this with BindToolsOptions.ToolChoice.
	fake := language.NewFakeChatModel(
		language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
		language.WithResponses(messages.AI("forced greet")),
	)
	forced, err := fake.BindToolsWithOptions([]coretools.Tool{multiply}, language.BindToolsOptions{
		ToolChoice: language.ToolChoiceAny,
	})
	if err != nil {
		fmt.Println("bind forced tool:", err)
		return
	}
	_ = forced
	fmt.Println("--- forced tool choice ---")
	fmt.Printf("bound tool choice: %s (any = model must call a tool)\n", fake.BoundToolChoice())

	// Cell 62: parallel_tool_calls=False, the Go BindToolsOptions field.
	parallelOff := false
	if _, err := fake.BindToolsWithOptions([]coretools.Tool{multiply}, language.BindToolsOptions{
		ParallelToolCalls: &parallelOff,
	}); err != nil {
		fmt.Println("bind parallel_tool_calls=false:", err)
		return
	}
	fmt.Println("parallel_tool_calls=false accepted")

	// Cells 56/59: return_direct tools end the agent loop right after the
	// tool runs, without a second model call.
	greet, err := coretools.NewSimple("greet", "Greet user.",
		func(_ context.Context, input string) (coretools.Result, error) {
			return coretools.Result{Content: "Hello " + input + "!"}, nil
		})
	if err != nil {
		fmt.Println("build greet tool:", err)
		return
	}
	greet = greet.WithReturnDirect()
	rdModel := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_g", Name: "greet", Args: map[string]any{"tool_input": "Ada"}},
			},
		},
	}}
	rdAgent, err := agents.CreateAgent(rdModel, []coretools.Tool{greet})
	if err != nil {
		fmt.Println("create return-direct agent:", err)
		return
	}
	rdOut, err := rdAgent.Invoke(ctx, []messages.Message{messages.Human("greet Ada")})
	if err != nil {
		fmt.Println("return-direct invoke:", err)
		return
	}
	fmt.Println("--- return_direct tool ---")
	fmt.Printf("tool is return-direct: %v\n", coretools.IsReturnDirect(greet))
	fmt.Printf("model calls: %d; last message: %s\n", len(rdModel.responses), messages.Text(rdOut[len(rdOut)-1]))
}
