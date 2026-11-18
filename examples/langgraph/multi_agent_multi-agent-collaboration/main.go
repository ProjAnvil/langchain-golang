// Command multi_agent_multi-agent-collaboration is the Go port of
// langgraph/examples/multi_agent/multi-agent-collaboration.ipynb.
//
// Two agents (a researcher with a search tool and a chart generator with a
// Python-REPL tool) hand control back and forth with Command(goto=...). Whoever
// decides the work is done prefixes its message with FINAL ANSWER, which ends
// the run. The Go port uses deterministic offline stubs for both agents and a
// safe in-memory string evaluator for the "chart" tool, so no API key,
// network, or code execution is involved.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// makeAgent builds one collaborator node. The stub emits a canned progress
// message; on its final turn it emits FINAL ANSWER so the graph terminates.
// next is the peer to hand off to when the work is not finished.
func makeAgent(name, next, progress, final string, finish bool) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		content := progress
		if finish {
			content = "FINAL ANSWER: " + final
		}
		fmt.Printf("  %s -> %s\n", name, content)
		target := next
		if finish || strings.Contains(content, "FINAL ANSWER") {
			target = types.END
		}
		return &types.Command{
			Update: map[string]any{
				"messages": []messages.Message{
					messages.Human(content), // wrapped as a human message, as in the notebook
				},
			},
			Goto: graphpkg.To(target),
		}, nil
	}
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("researcher", makeAgent(
		"researcher", "chart_generator",
		"UK GDP grew modestly over the last five years (data retrieved).",
		"", false,
	))
	g.AddNode("chart_generator", makeAgent(
		"chart_generator", "researcher",
		"", "a line chart of the five-year UK GDP series is ready", true,
	))
	g.AddEdge(types.START, "researcher")
	// Static edges document the handoff loop; each node's Command.Goto decides
	// the actual next hop (and can stop at END).
	g.AddEdge("researcher", "chart_generator")
	g.AddEdge("chart_generator", "researcher")

	cg, err := g.Compile(
		graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()),
		graphpkg.WithRecursionLimit(50),
	)
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	request := messages.Human("First get the UK GDP for the past 5 years, then make a line chart, then finish.")
	result, err := cg.InvokeWithOptions(ctx,
		map[string]any{"messages": []messages.Message{request}},
		graphpkg.Options{ThreadID: "collab-1"},
	)
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}

	fmt.Println("--- conversation ---")
	msgs, _ := result.Values["messages"].([]messages.Message)
	for _, m := range msgs {
		fmt.Printf("  %s: %s\n", m.Role, messages.Text(m))
	}
	fmt.Printf("handoff steps completed; final message above carries the answer.\n")
}
