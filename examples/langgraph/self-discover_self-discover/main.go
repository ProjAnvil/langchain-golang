// Command self-discover_self-discover is the Go port of
// langgraph/examples/self-discover/self-discover.ipynb.
//
// Self-Discovery runs four prompt-chain steps in a fixed order: select useful
// reasoning modules, adapt them to the task, structure them into a reasoning
// pipeline, then reason with that pipeline. The Go port keeps the linear
// `select -> adapt -> structure -> reason -> END` StateGraph and its state
// keys. The four LLM prompt chains are deterministic offline stubs, so no
// LangSmith hub prompts, API key, or network are needed.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// reasoningModules is a small stand-in for the notebook's long list of
// candidate reasoning modules pulled from the prompt hub.
var reasoningModules = []string{
	"1. How could I devise an experiment to help solve that problem?",
	"4. How can I simplify the problem so that it is easier to solve?",
	"9. How can I break down this problem into smaller, more manageable parts?",
	"16. What is the core issue or problem that needs to be addressed?",
	"39. Let's make a step by step plan and implement it with good notation.",
}

// selectNode stands in for the self-discovery-select prompt chain.
func selectNode(_ runtime.Runtime, state map[string]any) (any, error) {
	selected := strings.Join(reasoningModules[:3], "\n")
	fmt.Println("select: chose 3 reasoning modules (simplify, decompose, experiment)")
	return map[string]any{"selected_modules": selected}, nil
}

// adaptNode stands in for the self-discovery-adapt prompt chain.
func adaptNode(_ runtime.Runtime, state map[string]any) (any, error) {
	task, _ := state["task_description"].(string)
	adapted := fmt.Sprintf(
		"Adapted for '%s':\n- Break the visual shape into individual line segments.\n"+
			"- Count the distinct segments and identify the closed figure.\n"+
			"- Verify the figure against the answer choices.", truncate(task, 60))
	fmt.Println("adapt: tailored the modules to the task")
	return map[string]any{"adapted_modules": adapted}, nil
}

// structureNode stands in for the self-discovery-structure prompt chain.
func structureNode(_ runtime.Runtime, state map[string]any) (any, error) {
	structure := "Reasoning pipeline: (1) decompose the path into segments -> " +
		"(2) count segments and check closure -> (3) match the resulting shape to a choice."
	fmt.Println("structure: ordered the modules into a 3-step pipeline")
	return map[string]any{"reasoning_structure": structure}, nil
}

// reasonNode stands in for the self-discovery-reasoning prompt chain.
func reasonNode(_ runtime.Runtime, state map[string]any) (any, error) {
	answer := "Answer: (J) triangle. The path traces three line segments that close into a triangle."
	fmt.Println("reason: executed the pipeline")
	return map[string]any{"answer": answer}, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddNode("select", selectNode)
	g.AddNode("adapt", adaptNode)
	g.AddNode("structure", structureNode)
	g.AddNode("reason", reasonNode)
	g.AddEdge(types.START, "select")
	g.AddEdge("select", "adapt")
	g.AddEdge("adapt", "structure")
	g.AddEdge("structure", "reason")
	g.AddEdge("reason", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	task := "This SVG path draws a shape: (A) circle (B) heptagon (C) hexagon (D) kite " +
		"(E) line (F) octagon (G) pentagon (H) rectangle (I) sector (J) triangle. Which is it?"
	result, err := cg.Invoke(ctx, map[string]any{
		"task_description":  task,
		"reasoning_modules": strings.Join(reasoningModules, "\n"),
	})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Println("--- self-discovery pipeline ---")
	fmt.Println("selected modules:\n", result.Values["selected_modules"])
	fmt.Println("adapted modules:\n", result.Values["adapted_modules"])
	fmt.Println("structure:\n", result.Values["reasoning_structure"])
	fmt.Println("answer:\n", result.Values["answer"])
}
