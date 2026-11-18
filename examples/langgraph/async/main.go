// Command async is the Go port of langgraph/examples/async.ipynb.
// It rebuilds the notebook's agent/tool loop and drives it with Invoke and
// with Stream (updates mode), the Go analogues of Python's ainvoke/astream.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/prebuilt"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// buildApp creates a fresh graph for each run so the stub model's turn counter
// starts from zero every time (Invoke and Stream each get their own app).
func buildApp() *graphpkg.CompiledGraph {
	// search is the placeholder tool from the notebook; it returns a fixed
	// string instead of surfing the web.
	search, err := tools.NewSimple("search", "Call to surf the web.",
		func(_ context.Context, query string) (tools.Result, error) {
			return tools.Result{Content: "The answer to your question lies within (query: " + query + ")"}, nil
		})
	if err != nil {
		log.Fatalf("build search tool: %v", err)
	}
	toolNode, err := tools.NewToolNode([]tools.Tool{search})
	if err != nil {
		log.Fatalf("build tool node: %v", err)
	}

	// agent is a deterministic stand-in for the bound chat model. The first
	// turn requests the search tool; the second turn answers with the tool
	// result already present in the message history.
	turn := 0
	agent := func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		turn++
		var response messages.Message
		if turn == 1 {
			response = messages.AI("")
			response.ToolCalls = []messages.ToolCall{{
				ID:   "call_search_1",
				Name: "search",
				Args: map[string]any{"tool_input": "weather in sf"},
			}}
		} else {
			answer := "It is sunny and 18C in San Francisco."
			for _, m := range msgs {
				if m.Role == messages.RoleTool {
					answer = m.Content
					break
				}
			}
			response = messages.AI(answer)
		}
		fmt.Printf("  [agent turn %d] role=%s tool_calls=%d\n", turn, response.Role, len(response.ToolCalls))
		return map[string]any{"messages": []messages.Message{response}}, nil
	}

	// shouldContinue mirrors the notebook's should_continue router.
	shouldContinue := func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		if tools.HasPendingToolCalls(msgs) {
			return graphpkg.To("action"), nil
		}
		return graphpkg.To(types.END), nil
	}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("agent", agent)
	g.AddNode("action", prebuilt.ToolNode(toolNode))
	g.AddEdge(types.START, "agent")
	g.AddConditionalEdges("agent", shouldContinue)
	g.AddEdge("action", "agent")

	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}
	return cg
}

func initialInput() map[string]any {
	return map[string]any{
		"messages": []messages.Message{messages.Human("what is the weather in sf")},
	}
}

func printHistory(msgs []messages.Message) {
	for i, m := range msgs {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  [%d] %-5s tool_call %s(%v)\n", i, m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  [%d] %-5s %s\n", i, m.Role, m.Content)
	}
}

func main() {
	ctx := context.Background()

	// ainvoke equivalent: run to completion and print the final transcript.
	fmt.Println("=== Invoke (ainvoke equivalent) ===")
	result, err := buildApp().Invoke(ctx, initialInput())
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	msgs, _ := result.Values["messages"].([]messages.Message)
	printHistory(msgs)

	// astream(stream_mode="updates") equivalent: one chunk per node write.
	fmt.Println("=== Stream (astream updates equivalent) ===")
	for chunk, err := range buildApp().Stream(ctx, initialInput(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamUpdates},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		update, _ := chunk.Payload.(map[string]any)
		for node, value := range update {
			fmt.Printf("  node %q wrote: %v\n", node, value)
		}
	}
}
