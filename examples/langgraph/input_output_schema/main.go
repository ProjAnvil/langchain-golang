// Command input_output_schema is the Go port of
// langgraph/examples/input_output_schema.ipynb. The Go SDK has no input/output
// schema arguments on NewStateGraph, so this port projects the state maps
// explicitly to reproduce the notebook's InputState/OutputState behavior.
package main

import (
	"context"
	"fmt"
	"log"

	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// projectInput mirrors InputState: only "question" crosses the graph boundary.
func projectInput(raw map[string]any) map[string]any {
	out := map[string]any{}
	if q, ok := raw["question"]; ok {
		out["question"] = q
	}
	return out
}

// projectOutput mirrors OutputState: only "answer" is returned to the caller.
func projectOutput(state map[string]any) map[string]any {
	out := map[string]any{}
	if a, ok := state["answer"]; ok {
		out["answer"] = a
	}
	return out
}

func main() {
	ctx := context.Background()

	// answerNode is the notebook's answer_node: it returns an answer plus an
	// extra key that is part of OverallState but not of the output schema.
	answerNode := func(_ runtime.Runtime, state map[string]any) (any, error) {
		question, _ := state["question"].(string)
		fmt.Printf("  [answer_node] input question=%q (keys seen: %v)\n", question, state)
		return map[string]any{"answer": "bye", "question": question}, nil
	}

	g := graphpkg.NewStateGraph()
	g.AddNode("answer_node", answerNode)
	g.AddEdge(types.START, "answer_node")
	g.AddEdge("answer_node", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	// Standard invocation.
	fmt.Println("=== invoke with InputState {question: hi} ===")
	res, err := cg.Invoke(ctx, projectInput(map[string]any{"question": "hi"}))
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Printf("  full graph state: %v\n", res.Values)
	fmt.Printf("  projected output (OutputState): %v\n", projectOutput(res.Values))

	// An extra, non-schema key is dropped before the graph runs.
	fmt.Println("=== invoke with an extra non-schema key ===")
	res, err = cg.Invoke(ctx, projectInput(map[string]any{
		"question": "hi",
		"extra":    "this is not in InputState",
	}))
	if err != nil {
		log.Fatalf("invoke extra: %v", err)
	}
	fmt.Printf("  projected output (OutputState): %v\n", projectOutput(res.Values))
}
