// Command stream-values is the Go port of
// langgraph/examples/stream-values.ipynb. It streams the full graph state after
// every superstep ("values" mode), pretty-prints the last message of each
// snapshot, and then captures the final snapshot exactly like the notebook's
// final_result cell.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func prettyLast(state map[string]any) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) == 0 {
		fmt.Println("(no messages yet)")
		return
	}
	last := msgs[len(msgs)-1]
	fmt.Printf("===== %s Message =====\n%s\n", roleName(last.Role), last.Content)
}

func roleName(role messages.Role) string {
	switch role {
	case messages.RoleHuman:
		return "Human"
	case messages.RoleAI:
		return "AI"
	case messages.RoleSystem:
		return "System"
	case messages.RoleTool:
		return "Tool"
	default:
		return string(role)
	}
}

func buildGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("draft", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"messages": []messages.Message{
			messages.AI("Let me check the weather in sf..."),
		}}, nil
	})
	g.AddNode("answer", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"messages": []messages.Message{
			messages.AI("It's always sunny in sf"),
		}}, nil
	})
	g.AddEdge(types.START, "draft")
	g.AddEdge("draft", "answer")
	g.AddEdge("answer", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}
	return cg
}

func main() {
	ctx := context.Background()
	cg := buildGraph()
	input := map[string]any{
		"messages": []messages.Message{messages.Human("what's the weather in sf")},
	}
	opts := graphpkg.StreamOptions{Modes: []graphpkg.StreamMode{graphpkg.StreamValues}}

	// Python: async for chunk in graph.astream(inputs, stream_mode="values"):
	//             chunk["messages"][-1].pretty_print()
	for chunk, err := range cg.Stream(ctx, input, opts) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		state, ok := chunk.Payload.(map[string]any)
		if !ok {
			continue
		}
		prettyLast(state)
	}

	// Python: final_result = ... ; final_result["messages"][-1].pretty_print()
	var finalResult map[string]any
	for chunk, err := range cg.Stream(ctx, input, opts) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		if state, ok := chunk.Payload.(map[string]any); ok {
			finalResult = state
		}
	}
	fmt.Println("--- final result ---")
	prettyLast(finalResult)
}
