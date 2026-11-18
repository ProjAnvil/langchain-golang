// Command multi_agent_hierarchical_agent_teams is the Go port of
// langgraph/examples/multi_agent/hierarchical_agent_teams.ipynb.
//
// The notebook builds three supervisor/worker graphs of increasing depth:
// a research team, a document-writing team, and a top-level graph whose two
// nodes are those teams compiled as subgraphs. Workers always report back to
// their supervisor with Command(goto="supervisor"); the supervisor routes to
// the next worker or finishes. Every LLM decision here is an offline stub, so
// the example runs with no API key and no network.
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

// makeSupervisor returns a deterministic router standing in for the notebook's
// LLM-based supervisor. It visits each member once, then returns END (the Go
// equivalent of the notebook's FINISH). The counter key is distinct per graph
// because subgraphs share their state with the parent.
func makeSupervisor(members []string, counterKey string) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		visits, _ := state[counterKey].(int)
		visits++
		target := types.END
		if visits <= len(members) {
			target = members[visits-1]
		}
		fmt.Printf("  supervisor[%s] visit %d -> %s\n", counterKey, visits, target)
		return &types.Command{
			Update: map[string]any{counterKey: visits, "next": target},
			Goto:   graphpkg.To(target),
		}, nil
	}
}

// makeWorker returns a worker node that reports a canned result back to the
// supervisor through a Command, mirroring each *_node in the notebook.
func makeWorker(name, result string) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, _ map[string]any) (any, error) {
		fmt.Printf("    worker %s produced: %s\n", name, result)
		return &types.Command{
			Update: map[string]any{
				"messages": []messages.Message{
					messages.Human(fmt.Sprintf("[%s] %s", name, result)),
				},
			},
			Goto: graphpkg.To("supervisor"),
		}, nil
	}
}

// buildTeam compiles one supervisor/worker team, matching the research_builder
// and paper_writing_builder cells of the notebook.
func buildTeam(label string, counterKey string, members map[string]string) *graphpkg.CompiledGraph {
	names := make([]string, 0, len(members))
	for name := range members {
		names = append(names, name)
	}
	// A fixed order keeps the offline run deterministic (Go map iteration is
	// randomized); the notebook's LLM chooses the order at run time.
	sortNames(names)

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("supervisor", makeSupervisor(names, counterKey))
	for _, name := range names {
		g.AddNode(name, makeWorker(name, members[name]))
		// Static fallback edge; the worker's Command.Goto takes precedence.
		g.AddEdge(name, "supervisor")
	}
	g.AddEdge(types.START, "supervisor")
	cg, err := g.Compile(graphpkg.WithRecursionLimit(50))
	if err != nil {
		log.Fatalf("compile %s team: %v", label, err)
	}
	return cg
}

func sortNames(names []string) {
	for i := 1; i < len(names); i++ {
		for j := i; j > 0 && names[j] < names[j-1]; j-- {
			names[j], names[j-1] = names[j-1], names[j]
		}
	}
}

func printConversation(label string, values map[string]any) {
	fmt.Printf("--- %s: final conversation ---\n", label)
	msgs, _ := values["messages"].([]messages.Message)
	for _, m := range msgs {
		fmt.Printf("  %s: %s\n", m.Role, messages.Text(m))
	}
}

func main() {
	ctx := context.Background()
	request := messages.Human("Research AI agents and write a brief report about them.")

	// First layer: the research team (search + web scraper).
	research := buildTeam("research", "research_visits", map[string]string{
		"search":      "found 3 recent sources on agent frameworks",
		"web_scraper": "scraped the sources into short notes",
	})

	// First layer: the document-writing team.
	writing := buildTeam("writing", "writing_visits", map[string]string{
		"doc_writer":      "drafted the report from the notes",
		"note_taker":      "created an outline with 4 sections",
		"chart_generator": "generated a summary chart from the notes",
	})

	// Second layer: a top-level supervisor whose workers ARE the two teams,
	// registered with AddSubgraph so they share the parent's state keys.
	top := graphpkg.NewStateGraph()
	top.AddReducer("messages", channels.MessagesReducer)
	top.AddNode("supervisor", makeSupervisor([]string{"research_team", "writing_team"}, "top_visits"))
	top.AddSubgraph("research_team", research)
	top.AddSubgraph("writing_team", writing)
	top.AddEdge("research_team", "supervisor")
	top.AddEdge("writing_team", "supervisor")
	top.AddEdge(types.START, "supervisor")
	cg, err := top.Compile(graphpkg.WithRecursionLimit(100))
	if err != nil {
		log.Fatalf("compile top graph: %v", err)
	}

	fmt.Println("Running hierarchical supervisor -> teams -> workers (offline stubs).")
	result, err := cg.Invoke(ctx, map[string]any{
		"messages":        []messages.Message{request},
		"top_visits":      0,
		"research_visits": 0,
		"writing_visits":  0,
	})
	if err != nil {
		log.Fatalf("invoke top graph: %v", err)
	}
	printConversation("top graph", result.Values)
	fmt.Printf("next selection: %v\n", result.Values["next"])
}
