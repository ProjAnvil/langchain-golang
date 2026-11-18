// Command branching is the Go port of langgraph/examples/branching.ipynb.
// It rebuilds the notebook's three fan-out/fan-in graph shapes with the Go
// StateGraph API and runs each one offline.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func add(letter string) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Printf("adding %s to %v\n", letter, state["aggregate"])
		return map[string]any{"aggregate": []string{letter}}, nil
	}
}

func appendUniqueReducer(existing any, update any) (any, error) {
	values := []string{}
	switch current := existing.(type) {
	case []string:
		values = append(values, current...)
	case []any:
		for _, value := range current {
			if s, ok := value.(string); ok {
				values = append(values, s)
			}
		}
	}
	switch added := update.(type) {
	case []string:
		for _, value := range added {
			if !contains(values, value) {
				values = append(values, value)
			}
		}
	case string:
		if !contains(values, added) {
			values = append(values, added)
		}
	}
	return values, nil
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func compileOrDie(label string, g *graphpkg.StateGraph) *graphpkg.CompiledGraph {
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile %s: %v", label, err)
	}
	return cg
}

func parallelJoinGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("aggregate", channels.AppendSliceReducer)
	for _, name := range []string{"a", "b", "c", "d"} {
		g.AddNode(name, add(name))
	}
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("a", "c")
	g.AddJoinEdge([]string{"b", "c"}, "d")
	g.AddEdge("d", types.END)
	return compileOrDie("parallel join graph", g)
}

func conditionalGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	// The Python example writes e from every active intermediate node. Go may
	// schedule those edges as separate tasks, so a set-like reducer keeps the
	// final aggregate equal to the notebook's single e.
	g.AddReducer("aggregate", appendUniqueReducer)
	for _, name := range []string{"a", "b", "c", "d", "e"} {
		g.AddNode(name, add(name))
	}
	g.AddEdge(types.START, "a")
	g.AddConditionalEdges("a", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		if state["which"] == "cd" {
			return graphpkg.To("c", "d"), nil
		}
		return graphpkg.To("b", "c"), nil
	})
	for _, name := range []string{"b", "c", "d"} {
		g.AddEdge(name, "e")
	}
	g.AddEdge("e", types.END)
	return compileOrDie("conditional graph", g)
}

func invokeAndPrint(ctx context.Context, label string, cg *graphpkg.CompiledGraph, input map[string]any) {
	result, err := cg.Invoke(ctx, input)
	if err != nil {
		log.Fatalf("%s: %v", label, err)
	}
	fmt.Printf("%s: %v\n", label, result.Values["aggregate"])
}

func main() {
	ctx := context.Background()

	invokeAndPrint(ctx, "parallel branches", parallelJoinGraph(), map[string]any{
		"aggregate": []string{},
	})

	cg := conditionalGraph()
	invokeAndPrint(ctx, "conditional bc", cg, map[string]any{
		"aggregate": []string{},
		"which":     "bc",
	})
	invokeAndPrint(ctx, "conditional cd", cg, map[string]any{
		"aggregate": []string{},
		"which":     "cd",
	})
}
