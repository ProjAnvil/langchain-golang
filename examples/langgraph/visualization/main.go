// Command visualization is the Go port of
// langgraph/examples/visualization.ipynb.
//
// The notebook builds a fractal StateGraph and calls
// app.get_graph().draw_mermaid() (plus PNG variants). This port emits the
// equivalent Mermaid TEXT with CompiledGraph.DrawMermaid -- no PNG rendering,
// no Mermaid API and no network. It also shows the Go options that mirror the
// notebook's draw_mermaid keyword arguments and the xray subgraph expansion
// of GetGraph(WithXrayDepth).
package main

import (
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/runnables"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// noop is a deterministic stand-in for the notebook's MyNode callable: the
// drawing only needs registered nodes, not their behavior.
func noop(_ runtime.Runtime, _ map[string]any) (any, error) {
	return nil, nil
}

// route mirrors the notebook's route(): it sends long message histories to
// END and short ones back to node_C. A path map makes the conditional targets
// statically enumerable for graph export.
func route(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs, _ := state["messages"].([]any)
	if len(msgs) > 10 {
		return graphpkg.To(types.END), nil
	}
	return graphpkg.To("node_C"), nil
}

func compileOrDie(label string, g *graphpkg.StateGraph) *graphpkg.CompiledGraph {
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile %s: %v", label, err)
	}
	return cg
}

func buildGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	// A fixed, deterministic shape stands in for the notebook's random
	// fractal tree so the emitted Mermaid is reproducible.
	for _, name := range []string{"entry_node", "node_A", "node_B", "node_C"} {
		g.AddNode(name, noop)
	}
	g.AddEdge(types.START, "entry_node")
	g.AddEdge("entry_node", "node_A")
	g.AddEdge("entry_node", "node_B")
	g.AddEdge("node_A", "node_B")
	g.AddConditionalEdgesWithPathMap("node_A", route, map[string]string{
		"node_C":  "node_C",
		types.END: types.END,
	})
	g.AddEdge("node_B", "node_C")
	g.AddEdge("node_C", types.END)
	return compileOrDie("fractal graph", g)
}

func main() {
	cg := buildGraph()

	fmt.Println("=== app.get_graph().draw_mermaid() ===")
	fmt.Println(cg.DrawMermaid())

	fmt.Println("=== DrawMermaid with Python-style options ===")
	fmt.Println(cg.DrawMermaid(
		runnables.WithStyles(false),
		runnables.WithCurveStyle("basis"),
		runnables.WithFrontmatterConfig(map[string]any{"theme": "neutral"}),
		runnables.WithNodeStyles(runnables.NodeStyles{
			Default: "fill:#fad7de",
			First:   "fill:#ffdfba",
			Last:    "fill:#baffc9",
		}),
	))

	// The notebook's xray drawing is supported via GetGraph(WithXrayDepth):
	// a subgraph node is expanded into its own nested graph block.
	child := graphpkg.NewStateGraph()
	child.AddNode("inner_1", noop)
	child.AddNode("inner_2", noop)
	child.AddEdge(types.START, "inner_1")
	child.AddEdge("inner_1", "inner_2")
	child.AddEdge("inner_2", types.END)
	childCG := compileOrDie("child graph", child)

	parent := graphpkg.NewStateGraph()
	parent.AddNode("entry_node", noop)
	parent.AddSubgraph("child", childCG)
	parent.AddEdge(types.START, "entry_node")
	parent.AddEdge("entry_node", "child")
	parent.AddEdge("child", types.END)
	parentCG := compileOrDie("parent graph", parent)

	fmt.Println("=== parent graph, subgraph as a single node ===")
	fmt.Println(parentCG.DrawMermaid())
	fmt.Println("=== parent graph, xray=-1 (expands nested subgraphs) ===")
	fmt.Println(parentCG.GetGraph(graphpkg.WithXrayDepth(-1)).DrawMermaid())
}
