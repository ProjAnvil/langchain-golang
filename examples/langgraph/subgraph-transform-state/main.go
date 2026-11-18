// Command subgraph-transform-state is the Go port of
// langgraph/examples/subgraph-transform-state.ipynb.
//
// The notebook nests three graphs whose state schemas share no keys. Each
// level therefore wraps the compiled child graph in a plain node function
// that explicitly transforms state from the parent's key to the child's key
// and back. This port reproduces exactly that pattern with CompiledGraph.Invoke
// calls inside NodeFuncs; it deliberately does not use AddSubgraph, because
// AddSubgraph would merge the child's state into the parent's instead of
// translating it.
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

func invokeAndPrint(ctx context.Context, label string, cg *graphpkg.CompiledGraph, input map[string]any, key string) {
	res, err := cg.Invoke(ctx, input)
	if err != nil {
		log.Fatalf("%s: %v", label, err)
	}
	fmt.Printf("%s: %s=%q\n", label, key, str(res.Values, key))
}

func main() {
	ctx := context.Background()

	// --- Grandchild: state key my_grandchild_key. ---
	grandchild := graphpkg.NewStateGraph()
	grandchild.AddNode("grandchild_1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"my_grandchild_key": str(state, "my_grandchild_key") + ", how are you"}, nil
	})
	grandchild.AddEdge(types.START, "grandchild_1")
	grandchild.AddEdge("grandchild_1", types.END)
	grandchildCG := compileOrDie("grandchild", grandchild)
	invokeAndPrint(ctx, "grandchild_graph.invoke", grandchildCG,
		map[string]any{"my_grandchild_key": "hi Bob"}, "my_grandchild_key")

	// --- Child: state key my_child_key; wraps the grandchild. ---
	child := graphpkg.NewStateGraph()
	child.AddNode("child_1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		grandchildInput := map[string]any{"my_grandchild_key": str(state, "my_child_key")}
		out, err := grandchildCG.Invoke(ctx, grandchildInput)
		if err != nil {
			return nil, err
		}
		return map[string]any{"my_child_key": str(out.Values, "my_grandchild_key") + " today?"}, nil
	})
	child.AddEdge(types.START, "child_1")
	child.AddEdge("child_1", types.END)
	childCG := compileOrDie("child", child)
	invokeAndPrint(ctx, "child_graph.invoke", childCG,
		map[string]any{"my_child_key": "hi Bob"}, "my_child_key")

	// --- Parent: state key my_key; wraps the child between two own nodes. ---
	parent := graphpkg.NewStateGraph()
	parent.AddNode("parent_1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"my_key": "hi " + str(state, "my_key")}, nil
	})
	parent.AddNode("child", func(_ runtime.Runtime, state map[string]any) (any, error) {
		childInput := map[string]any{"my_child_key": str(state, "my_key")}
		out, err := childCG.Invoke(ctx, childInput)
		if err != nil {
			return nil, err
		}
		return map[string]any{"my_key": str(out.Values, "my_child_key")}, nil
	})
	parent.AddNode("parent_2", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"my_key": str(state, "my_key") + " bye!"}, nil
	})
	parent.AddEdge(types.START, "parent_1")
	parent.AddEdge("parent_1", "child")
	parent.AddEdge("child", "parent_2")
	parent.AddEdge("parent_2", types.END)
	parentCG := compileOrDie("parent", parent)
	invokeAndPrint(ctx, "parent_graph.invoke", parentCG,
		map[string]any{"my_key": "Bob"}, "my_key")
}
