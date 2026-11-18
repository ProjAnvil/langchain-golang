// Command pass_private_state is the Go port of
// langgraph/examples/pass_private_state.ipynb. It shows node-scoped data being
// passed from node_1 to node_2 and consumed before node_3 runs.
package main

import (
	"context"
	"fmt"
	"log"

	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func main() {
	ctx := context.Background()

	// node_1 returns data that is private to the node_1 -> node_2 handoff.
	node1 := func(_ runtime.Runtime, state map[string]any) (any, error) {
		output := map[string]any{"private_data": "set by node_1"}
		fmt.Printf("Entered node `node_1`:\n\tInput: %v\n\tReturned: %v\n", state, output)
		return output, nil
	}

	// node_2 is the only node that reads private_data. It consumes the key by
	// overwriting it, emulating Python's private-channel handoff.
	node2 := func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Printf("Entered node `node_2`:\n\tInput: %v\n", state)
		private, _ := state["private_data"].(string)
		fmt.Printf("\tRead private_data: %q\n", private)
		output := map[string]any{"a": "set by node_2", "private_data": ""}
		fmt.Printf("\tReturned: %v\n", output)
		return output, nil
	}

	// node_3 only uses the overall state key `a`; the private data is gone.
	node3 := func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Printf("Entered node `node_3`:\n\tInput: %v\n", state)
		private, _ := state["private_data"].(string)
		fmt.Printf("\tprivate_data visible here: %q (empty means hidden)\n", private)
		output := map[string]any{"a": "set by node_3"}
		fmt.Printf("\tReturned: %v\n", output)
		return output, nil
	}

	g := graphpkg.NewStateGraph()
	g.AddNode("node_1", node1)
	g.AddNode("node_2", node2)
	g.AddNode("node_3", node3)
	g.AddEdge(types.START, "node_1")
	g.AddEdge("node_1", "node_2")
	g.AddEdge("node_2", "node_3")
	g.AddEdge("node_3", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	res, err := cg.Invoke(ctx, map[string]any{"a": "set at start"})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Println()
	fmt.Printf("Output of graph invocation: %v\n", res.Values)
}
