// Command stream-updates is the Go port of
// langgraph/examples/stream-updates.ipynb. It streams a ReAct weather graph in
// "updates" mode and, for each chunk, prints the node that produced the update
// and the values it wrote.
package main

import (
	"context"
	"fmt"
	"log"
	"sort"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func lastMessage(state map[string]any) (messages.Message, bool) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) == 0 {
		return messages.Message{}, false
	}
	return msgs[len(msgs)-1], true
}

func agent(_ runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if ok && last.Role == messages.RoleTool {
		return map[string]any{"messages": []messages.Message{
			messages.AI("It's always sunny in sf"),
		}}, nil
	}
	call := messages.AI("")
	call.ID = "call-msg-1"
	call.ToolCalls = []messages.ToolCall{{
		ID:   "call_1",
		Name: "get_weather",
		Args: map[string]any{"city": "sf"},
	}}
	return map[string]any{"messages": []messages.Message{call}}, nil
}

func tools(_ runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if !ok || len(last.ToolCalls) == 0 {
		return nil, fmt.Errorf("tools node: no tool call in state")
	}
	call := last.ToolCalls[0]
	result := messages.Tool(call.ID, "It's always sunny in sf")
	result.Name = call.Name
	return map[string]any{"messages": []messages.Message{result}}, nil
}

func buildGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("agent", agent)
	g.AddNode("tools", tools)
	g.AddEdge(types.START, "agent")
	g.AddConditionalEdges("agent", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		last, ok := lastMessage(state)
		if ok && len(last.ToolCalls) > 0 {
			return graphpkg.To("tools"), nil
		}
		return graphpkg.To(types.END), nil
	})
	g.AddEdge("tools", "agent")
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

	// Python: for chunk in graph.astream(inputs, stream_mode="updates"):
	//             for node, values in chunk.items(): ...
	for chunk, err := range cg.Stream(ctx, input, graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamUpdates},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		update, ok := chunk.Payload.(map[string]any)
		if !ok {
			continue
		}
		// Go maps are unordered; sort the keys to keep notebook-like output.
		nodes := make([]string, 0, len(update))
		for node := range update {
			nodes = append(nodes, node)
		}
		sort.Strings(nodes)
		for _, node := range nodes {
			fmt.Printf("Receiving update from node: '%s'\n", node)
			fmt.Printf("%v\n\n\n", update[node])
		}
	}
}
