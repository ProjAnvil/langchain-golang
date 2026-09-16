// Command mcp-tools connects an agent to MCP server tools: it builds an
// in-process MCP server (mcp-go), attaches through partners/mcp's aggregated
// MCPConfig (the same New + LoadTools entry point used for stdio/HTTP
// fleets), and lets a scripted offline agent call a discovered tool by its
// namespaced name. No subprocess or network is involved.
//
// To point this at a real server instead, swap the InProcessServer entry for
// a StdioServer{Command: "..."} or HTTPServer{URL: "..."} in MCPConfig below;
// everything after New is transport-agnostic.
//
// Usage:
//
//	go run ./examples/mcp-tools
package main

import (
	"context"
	"fmt"
	"sync"

	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	langchainmcp "github.com/projanvil/langchain-golang/partners/mcp"
)

// scriptedModel: offline ChatModel double, self-returning from BindTools
// (see examples/quickstart for the rationale).
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

// newCalcServer builds an in-process MCP server exposing two tools. In a
// deployment this server would be a separate binary speaking MCP over stdio
// or streamable HTTP.
func newCalcServer() *mcpserver.MCPServer {
	s := mcpserver.NewMCPServer("calc", "1.0.0")

	s.AddTool(
		mcp.NewTool("add",
			mcp.WithDescription("add two integers"),
			mcp.WithNumber("a", mcp.Required(), mcp.Description("first addend")),
			mcp.WithNumber("b", mcp.Required(), mcp.Description("second addend")),
		),
		func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			a, _ := req.RequireFloat("a")
			b, _ := req.RequireFloat("b")
			return mcp.NewToolResultText(fmt.Sprintf("%g", a+b)), nil
		},
	)

	s.AddTool(
		mcp.NewTool("echo",
			mcp.WithDescription("echo the input text"),
			mcp.WithString("text", mcp.Required()),
		),
		func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			text, _ := req.RequireString("text")
			return mcp.NewToolResultText(text), nil
		},
	)

	return s
}

func main() {
	ctx := context.Background()

	// Connect the fleet. InProcessServer keeps everything offline; the same
	// MCPConfig map mixes transports freely ({"calc": InProcessServer{...},
	// "remote": HTTPServer{URL: "https://...mcp"}}).
	adapter, err := langchainmcp.New(ctx, langchainmcp.MCPConfig{
		Servers: map[string]langchainmcp.Server{
			"calc": &langchainmcp.InProcessServer{Server: newCalcServer()},
		},
	})
	if err != nil {
		fmt.Println("mcp new:", err)
		return
	}
	defer func() { _ = adapter.Close() }()

	// Discover tools: names are namespaced {server}_{tool} so fleets with
	// colliding remote names stay distinguishable.
	toolList, err := adapter.LoadTools(ctx)
	if err != nil {
		fmt.Println("load tools:", err)
		return
	}
	fmt.Println("discovered MCP tools:")
	for _, t := range toolList {
		fmt.Printf("  %s — %s\n", t.Name(), t.Description())
	}

	// An agent over the discovered tools. The scripted model calls
	// "calc_add" (the namespaced name) and then answers.
	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "calc_add", Args: map[string]any{"a": 19, "b": 23}},
			},
		},
		messages.AI("19 + 23 = 42."),
	}}
	agent, err := agents.CreateAgent(model, toolList)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}

	out, err := agent.Invoke(ctx, []messages.Message{messages.Human("what is 19 plus 23?")})
	if err != nil {
		fmt.Println("invoke agent:", err)
		return
	}
	fmt.Println("agent run messages:")
	for _, m := range out {
		if m.Role == messages.RoleTool {
			fmt.Printf("  %-8s (call %s) -> %s\n", m.Role, m.ToolCallID, messages.Text(m))
			continue
		}
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s calls %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
}
