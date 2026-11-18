// Command stream-multiple is the Go port of
// langgraph/examples/stream-multiple.ipynb. It runs a ReAct-style weather
// graph and consumes two stream modes at once (updates and debug), printing
// the mode of every chunk just as the notebook prints the event name.
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

func lastMessage(state map[string]any) (messages.Message, bool) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) == 0 {
		return messages.Message{}, false
	}
	return msgs[len(msgs)-1], true
}

// agent is an offline stand-in for ChatOpenAI bound to the get_weather tool:
// its first turn requests the tool, its second turn returns the final answer.
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

// tools executes the requested weather tool deterministically.
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

	// Python: graph.astream(inputs, stream_mode=["updates", "debug"]).
	for chunk, err := range cg.Stream(ctx, input, graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamUpdates, graphpkg.StreamDebug},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		fmt.Printf("Receiving new event of type: %s...\n", chunk.Mode)
		fmt.Printf("%v\n\n\n", chunk.Payload)
	}
}
