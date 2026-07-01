package graph_test

import (
	"context"
	"strconv"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/channels"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/graph"
)

// TestAgentLoopWithRealToolNode is an integration test proving the two
// pieces `langchain/agents` needs actually compose: a graph.StateGraph
// running a "model" node and a "tools" node backed by a real
// tools.ToolNode, wired together exactly like Python's `create_agent`
// compiles its StateGraph. This is not the full agent factory (no
// middleware chain, no structured output) — it demonstrates that the
// langgraph port is sufficient to build one on top of.
func TestAgentLoopWithRealToolNode(t *testing.T) {
	addTool, err := tools.NewFunc("add", "adds two numbers", schema.Object(nil), func(_ context.Context, args map[string]any) (tools.Result, error) {
		a := args["a"].(float64)
		b := args["b"].(float64)
		return tools.Result{Content: strconv.FormatInt(int64(a+b), 10)}, nil
	})
	if err != nil {
		t.Fatalf("NewFunc() error = %v", err)
	}
	toolNode, err := tools.NewToolNode([]tools.Tool{addTool})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	modelCalls := 0
	g := graph.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)

	g.AddNode("model", func(_ context.Context, state map[string]any) (any, error) {
		modelCalls++
		msgs, _ := state["messages"].([]messages.Message)
		if modelCalls == 1 {
			ai := messages.AI("")
			ai.ID = "ai-1"
			ai.ToolCalls = []messages.ToolCall{{ID: "call-1", Name: "add", Args: map[string]any{"a": 2.0, "b": 3.0}}}
			return map[string]any{"messages": []messages.Message{ai}}, nil
		}
		// Second pass: inspect the tool result and produce a final answer.
		var toolResult string
		for _, m := range msgs {
			if m.Role == messages.RoleTool {
				toolResult = m.Content
			}
		}
		final := messages.AI("the answer is " + toolResult)
		final.ID = "ai-2"
		return map[string]any{"messages": []messages.Message{final}}, nil
	})

	g.AddNode("tools", func(ctx context.Context, state map[string]any) (any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		results, err := toolNode.Invoke(ctx, msgs, nil)
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": results}, nil
	})

	g.AddEdge(agentruntime.START, "model")
	g.AddConditionalEdges("model", func(_ context.Context, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		if tools.HasPendingToolCalls(msgs) {
			return graph.To("tools"), nil
		}
		return graph.To(agentruntime.END), nil
	})
	g.AddEdge("tools", "model")

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	result, err := cg.Invoke(context.Background(), map[string]any{"messages": []messages.Message{}})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	msgs := result.Values["messages"].([]messages.Message)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (ai tool-call, tool result, final ai), got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != messages.RoleAI || len(msgs[0].ToolCalls) != 1 {
		t.Fatalf("unexpected first message: %+v", msgs[0])
	}
	if msgs[1].Role != messages.RoleTool || msgs[1].Content != "5" {
		t.Fatalf("unexpected tool result message: %+v", msgs[1])
	}
	if msgs[2].Role != messages.RoleAI || msgs[2].Content != "the answer is 5" {
		t.Fatalf("unexpected final message: %+v", msgs[2])
	}
	if modelCalls != 2 {
		t.Fatalf("expected model to be called twice (initial + after tool), got %d", modelCalls)
	}
}
