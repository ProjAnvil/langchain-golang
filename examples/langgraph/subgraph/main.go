// Command subgraph is the Go port of langgraph/examples/subgraph.ipynb.
//
// It covers the notebook's two subgraph shapes:
//
//  1. A compiled subgraph is added directly as a parent node with AddSubgraph.
//     The shared key "foo" flows into and out of the child; the child-only key
//     "bar" is used inside the child. Stream with Subgraphs:true shows the
//     nested chunks, the Go counterpart of stream(..., subgraphs=True).
//  2. A subgraph whose keys are not shared with the parent is invoked from a
//     normal parent node that explicitly transforms state in and out.
package main

import (
	"context"
	"fmt"
	"log"

	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func str(state map[string]any, key string) string {
	if value, ok := state[key].(string); ok {
		return value
	}
	return ""
}

func compileOrDie(label string, g *graphpkg.StateGraph) *graphpkg.CompiledGraph {
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile %s: %v", label, err)
	}
	return cg
}

func main() {
	ctx := context.Background()

	// --- Part 1: subgraph shares the key "foo" with its parent. ---
	subgraph := graphpkg.NewStateGraph()
	subgraph.AddNode("subgraph_node_1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"bar": "bar"}, nil
	})
	subgraph.AddNode("subgraph_node_2", func(_ runtime.Runtime, state map[string]any) (any, error) {
		// "bar" is subgraph-only; "foo" is shared and flows back to the parent.
		return map[string]any{"foo": str(state, "foo") + str(state, "bar")}, nil
	})
	subgraph.AddEdge(types.START, "subgraph_node_1")
	subgraph.AddEdge("subgraph_node_1", "subgraph_node_2")
	subgraph.AddEdge("subgraph_node_2", types.END)
	subCG := compileOrDie("shared-key subgraph", subgraph)

	parent := graphpkg.NewStateGraph()
	parent.AddNode("node_1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"foo": "hi! " + str(state, "foo")}, nil
	})
	// Adding a compiled graph as a node is the Go equivalent of
	// builder.add_node("node_2", subgraph).
	parent.AddSubgraph("node_2", subCG)
	parent.AddEdge(types.START, "node_1")
	parent.AddEdge("node_1", "node_2")
	parent.AddEdge("node_2", types.END)
	parentCG := compileOrDie("shared-key parent", parent)

	res, err := parentCG.Invoke(ctx, map[string]any{"foo": "foo"})
	if err != nil {
		log.Fatalf("invoke shared-key parent: %v", err)
	}
	fmt.Printf("shared-key graph: foo=%q\n", str(res.Values, "foo"))

	fmt.Println("shared-key graph stream (updates, subgraphs=true):")
	for chunk, err := range parentCG.Stream(ctx, map[string]any{"foo": "foo"}, graphpkg.StreamOptions{
		Modes:     []graphpkg.StreamMode{graphpkg.StreamUpdates},
		Subgraphs: true,
	}) {
		if err != nil {
			log.Fatalf("stream shared-key parent: %v", err)
		}
		fmt.Printf("  namespace=%q payload=%v\n", chunk.Namespace, chunk.Payload)
	}

	// --- Part 2: subgraph state is NOT shared with the parent. ---
	isolated := graphpkg.NewStateGraph()
	isolated.AddNode("subgraph_node_1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"baz": "baz"}, nil
	})
	isolated.AddNode("subgraph_node_2", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"bar": str(state, "bar") + str(state, "baz")}, nil
	})
	isolated.AddEdge(types.START, "subgraph_node_1")
	isolated.AddEdge("subgraph_node_1", "subgraph_node_2")
	isolated.AddEdge("subgraph_node_2", types.END)
	isolatedCG := compileOrDie("isolated subgraph", isolated)

	isolatedParent := graphpkg.NewStateGraph()
	isolatedParent.AddNode("node_1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"foo": "hi! " + str(state, "foo")}, nil
	})
	isolatedParent.AddNode("node_2", func(_ runtime.Runtime, state map[string]any) (any, error) {
		out, err := isolatedCG.Invoke(ctx, map[string]any{"bar": str(state, "foo")})
		if err != nil {
			return nil, err
		}
		return map[string]any{"foo": str(out.Values, "bar")}, nil
	})
	isolatedParent.AddEdge(types.START, "node_1")
	isolatedParent.AddEdge("node_1", "node_2")
	isolatedParent.AddEdge("node_2", types.END)
	isolatedParentCG := compileOrDie("isolated parent", isolatedParent)

	res2, err := isolatedParentCG.Invoke(ctx, map[string]any{"foo": "foo"})
	if err != nil {
		log.Fatalf("invoke isolated parent: %v", err)
	}
	fmt.Printf("isolated-state graph: foo=%q\n", str(res2.Values, "foo"))
}
