// Command subgraph-resume demonstrates interrupt propagation across graph
// levels: a node inside a child (sub)graph calls graph.Interrupt; the paused
// run surfaces on the PARENT graph's result with the child interrupt's
// identity intact; resuming the parent thread with Options.Resume feeds the
// answer down into the child, and both graphs run to completion across the
// two invokes.
//
// Usage:
//
//	go run ./examples/subgraph-resume
package main

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func main() {
	ctx := context.Background()

	// --- Child graph: one node that pauses for a human decision. ---
	var askRuns int
	child := graphpkg.NewStateGraph()
	child.AddNode("ask_approval", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		askRuns++
		// Interrupt pauses the child run here and returns only on resume,
		// with the value the caller fed via Options.Resume.
		answer := graphpkg.Interrupt(rt, "approve the 500 EUR transfer?")
		return map[string]any{"approval": answer}, nil
	})
	child.AddEdge(types.START, "ask_approval")
	child.AddEdge("ask_approval", types.END)
	childCG, err := child.Compile()
	if err != nil {
		fmt.Println("compile child:", err)
		return
	}

	// --- Parent graph: pre -> review(subgraph) -> settle. ---
	var settleSaw any
	parent := graphpkg.NewStateGraph()
	parent.AddNode("pre", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"order": "ORD-77"}, nil
	})
	parent.AddSubgraph("review", childCG)
	parent.AddNode("settle", func(_ runtime.Runtime, state map[string]any) (any, error) {
		settleSaw = state["approval"]
		return nil, nil
	})
	parent.AddEdge(types.START, "pre")
	parent.AddEdge("pre", "review")
	parent.AddEdge("review", "settle")
	parent.AddEdge("settle", types.END)

	saver := checkpoint.NewMemorySaver()
	parentCG, err := parent.Compile(graphpkg.WithCheckpointer(saver))
	if err != nil {
		fmt.Println("compile parent:", err)
		return
	}

	// --- Invoke 1: runs "pre", enters the subgraph, pauses at the child's
	// Interrupt. The child interrupt propagates to the parent's result. ---
	paused, err := parentCG.InvokeWithOptions(ctx,
		map[string]any{"order": "ORD-77"},
		graphpkg.Options{ThreadID: "tx-1"},
	)
	if err != nil {
		fmt.Println("first invoke:", err)
		return
	}
	fmt.Printf("paused run: %d interrupt(s)\n", len(paused.Interrupts))
	for _, intr := range paused.Interrupts {
		fmt.Printf("  interrupt id=%s ns=%s value=%v\n", intr.ID, intr.NS, intr.Value)
	}
	fmt.Printf("  ask_approval executions so far: %d\n", askRuns)

	// --- Invoke 2: nil input + Options.Resume="approved" continues the SAME
	// thread: the child's Interrupt call returns "approved", the child
	// finishes, and the parent runs "settle" to completion. ---
	done, err := parentCG.InvokeWithOptions(ctx, nil,
		graphpkg.Options{ThreadID: "tx-1", Resume: "approved"},
	)
	if err != nil {
		fmt.Println("resume invoke:", err)
		return
	}
	fmt.Printf("resumed run: interrupts=%d, ask_approval executions total=%d\n",
		len(done.Interrupts), askRuns)
	fmt.Printf("parent state: order=%v approval=%v (seen by settle node: %v)\n",
		done.Values["order"], done.Values["approval"], settleSaw)
}
