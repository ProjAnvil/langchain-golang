// Command recursion-limit is the Go port of
// langgraph/examples/recursion-limit.ipynb. It shows a bounded cyclic graph and
// how a low per-run recursion limit triggers GraphRecursionError.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// appendNode returns a NodeFunc that appends one letter to `aggregate`.
func appendNode(letter string, counter *int) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		*counter++
		agg, _ := state["aggregate"].([]string)
		fmt.Printf("  Node %s sees %v\n", letter, agg)
		return map[string]any{"aggregate": []string{letter}}, nil
	}
}

// route stops the cycle once the aggregate reaches length 7.
func route(_ runtime.Runtime, state map[string]any) ([]any, error) {
	agg, _ := state["aggregate"].([]string)
	if len(agg) < 7 {
		return graphpkg.To("b"), nil
	}
	return graphpkg.To(types.END), nil
}

// runWithLimit invokes cg with an optional per-run recursion limit and reports
// whether the run completed or hit the limit.
func runWithLimit(ctx context.Context, label string, cg *graphpkg.CompiledGraph, limit int) {
	var opts graphpkg.Options
	if limit > 0 {
		opts.RecursionLimit = limit
	}
	res, err := cg.InvokeWithOptions(ctx, map[string]any{"aggregate": []string{}}, opts)
	var recErr *types.GraphRecursionError
	if errors.As(err, &recErr) {
		fmt.Printf("%s -> Recursion Error (limit %d)\n", label, limit)
		return
	}
	if err != nil {
		log.Fatalf("%s: %v", label, err)
	}
	agg, _ := res.Values["aggregate"].([]string)
	fmt.Printf("%s -> completed, aggregate=%v (len %d)\n", label, agg, len(agg))
}

func main() {
	ctx := context.Background()

	// --- Graph 1: simple a<->b cycle -----------------------------------
	fmt.Println("=== Graph 1: a <-> b ===")
	var a1, b1 int
	g1 := graphpkg.NewStateGraph()
	g1.AddReducer("aggregate", channels.AppendSliceReducer)
	g1.AddNode("a", appendNode("A", &a1))
	g1.AddNode("b", appendNode("B", &b1))
	g1.AddEdge(types.START, "a")
	g1.AddConditionalEdges("a", route)
	g1.AddEdge("b", "a")
	cg1, err := g1.Compile()
	if err != nil {
		log.Fatalf("compile graph 1: %v", err)
	}
	runWithLimit(ctx, "default limit", cg1, 0)
	fmt.Printf("  node visits: a=%d b=%d\n", a1, b1)
	a1, b1 = 0, 0
	runWithLimit(ctx, "recursion_limit=4", cg1, 4)
	fmt.Printf("  node visits: a=%d b=%d\n", a1, b1)

	// --- Graph 2: a -> b -> (c,d) -> a with a join ----------------------
	fmt.Println("=== Graph 2: a -> b -> (c,d) -> a ===")
	var a2, b2, c2, d2 int
	g2 := graphpkg.NewStateGraph()
	g2.AddReducer("aggregate", channels.AppendSliceReducer)
	g2.AddNode("a", appendNode("A", &a2))
	g2.AddNode("b", appendNode("B", &b2))
	g2.AddNode("c", appendNode("C", &c2))
	g2.AddNode("d", appendNode("D", &d2))
	g2.AddEdge(types.START, "a")
	g2.AddConditionalEdges("a", route)
	g2.AddEdge("b", "c")
	g2.AddEdge("b", "d")
	g2.AddJoinEdge([]string{"c", "d"}, "a")
	cg2, err := g2.Compile()
	if err != nil {
		log.Fatalf("compile graph 2: %v", err)
	}
	runWithLimit(ctx, "default limit", cg2, 0)
	fmt.Printf("  node visits: a=%d b=%d c=%d d=%d\n", a2, b2, c2, d2)
	a2, b2, c2, d2 = 0, 0, 0, 0
	runWithLimit(ctx, "recursion_limit=4", cg2, 4)
	fmt.Printf("  node visits: a=%d b=%d c=%d d=%d\n", a2, b2, c2, d2)
}
