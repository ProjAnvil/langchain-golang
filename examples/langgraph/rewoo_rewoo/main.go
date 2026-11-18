// Command rewoo_rewoo is the Go port of langgraph/examples/rewoo/rewoo.ipynb.
//
// ReWOO separates reasoning from acting: a planner emits a list of steps with
// evidence variables (#E1, #E2, ...), a tool node executes one step at a time
// and fills the variables, and a solver produces the final answer from the
// plan and evidence. The Go port keeps the graph `plan -> tool -(loop)-> solve
// -> END` and the same step/results state. The LLM planner and solver are
// deterministic offline stubs and the tools are safe in-memory lookups, so
// nothing is fetched from the network or executed as code.
package main

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"strings"

	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// planStep is one parsed tuple of the notebook's steps list:
// (plan description, evidence name, tool name, tool input).
type planStep struct {
	Description string
	Name        string
	Tool        string
	Input       string
}

var stepPattern = regexp.MustCompile(`Plan:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*\[(.*)\]`)

// mergeResults is the reducer for the evidence map.
func mergeResults(existing any, update any) (any, error) {
	out := map[string]string{}
	if m, ok := existing.(map[string]string); ok {
		for k, v := range m {
			out[k] = v
		}
	}
	if m, ok := update.(map[string]string); ok {
		for k, v := range m {
			out[k] = v
		}
	}
	return out, nil
}

// searchStub stands in for TavilySearchResults.
func searchStub(query string) string {
	return "search evidence for '" + query + "': the winner is from Melbourne"
}

// llmStub stands in for the reasoning LLM used by the LLM tool and the solver.
func llmStub(input string) string {
	if strings.Contains(strings.ToLower(input), "hometown") || strings.Contains(strings.ToLower(input), "winner") {
		return "The hometown is Melbourne."
	}
	return "Based on the evidence, the answer is Melbourne."
}

// getPlan stands in for the LLM planner. It emits a canned plan string and
// parses it with the same regex shape as the notebook.
func getPlan(_ runtime.Runtime, state map[string]any) (any, error) {
	task, _ := state["task"].(string)
	planText := fmt.Sprintf("Plan: Search for the hometown of the winner. #E1 = Google[%s]\n"+
		"Plan: Reason about the hometown given the evidence. #E2 = LLM[What is the hometown given #E1?]", task)
	matches := stepPattern.FindAllStringSubmatch(planText, -1)
	steps := make([]planStep, 0, len(matches))
	for _, m := range matches {
		steps = append(steps, planStep{Description: m[1], Name: m[2], Tool: m[3], Input: m[4]})
	}
	fmt.Printf("plan: parsed %d step(s) from the canned plan\n", len(steps))
	for _, s := range steps {
		fmt.Printf("  %s = %s[%s]\n", s.Name, s.Tool, s.Input)
	}
	return map[string]any{"steps": steps, "plan_string": planText}, nil
}

// substituteEvidence replaces every #E key with its stored evidence, as in
// the notebook's tool_execution and solve.
func substituteEvidence(input string, results map[string]string) string {
	for k, v := range results {
		input = strings.ReplaceAll(input, k, v)
	}
	return input
}

// toolExecution executes the next unexecuted step of the plan.
func toolExecution(_ runtime.Runtime, state map[string]any) (any, error) {
	steps, _ := state["steps"].([]planStep)
	results, _ := state["results"].(map[string]string)
	idx := len(results)
	if idx >= len(steps) {
		return nil, fmt.Errorf("tool_execution: no pending steps")
	}
	step := steps[idx]
	resolved := substituteEvidence(step.Input, results)
	var observation string
	switch step.Tool {
	case "Google":
		observation = searchStub(resolved)
	case "LLM":
		observation = llmStub(resolved)
	default:
		return nil, fmt.Errorf("tool_execution: unknown tool %q", step.Tool)
	}
	fmt.Printf("tool: %s[%s] -> %s\n", step.Name, step.Tool, observation)
	return map[string]any{"results": map[string]string{step.Name: observation}}, nil
}

// solve produces the final answer from the plan and gathered evidence.
func solve(_ runtime.Runtime, state map[string]any) (any, error) {
	steps, _ := state["steps"].([]planStep)
	results, _ := state["results"].(map[string]string)
	var plan strings.Builder
	for _, s := range steps {
		plan.WriteString(fmt.Sprintf("Plan: %s\n%s = %s[%s]\n",
			s.Description, s.Name, s.Tool, substituteEvidence(s.Input, results)))
	}
	answer := llmStub(substituteEvidence("task with evidence: "+plan.String(), results))
	fmt.Printf("solve -> %s\n", answer)
	return map[string]any{"result": answer}, nil
}

// route continues to the next tool step or moves on to solve, mirroring the
// notebook's _route / _get_current_task.
func route(_ runtime.Runtime, state map[string]any) ([]any, error) {
	steps, _ := state["steps"].([]planStep)
	results, _ := state["results"].(map[string]string)
	if len(results) >= len(steps) {
		return graphpkg.To("solve"), nil
	}
	return graphpkg.To("tool"), nil
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddReducer("results", mergeResults)
	g.AddNode("plan", getPlan)
	g.AddNode("tool", toolExecution)
	g.AddNode("solve", solve)
	g.AddEdge(types.START, "plan")
	g.AddEdge("plan", "tool")
	g.AddConditionalEdges("tool", route)
	g.AddEdge("solve", types.END)
	cg, err := g.Compile(graphpkg.WithRecursionLimit(50))
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	task := "what is the exact hometown of the 2024 men's Australian Open winner"
	result, err := cg.Invoke(ctx, map[string]any{
		"task":    task,
		"results": map[string]string{},
	})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Println("--- final result ---")
	fmt.Println(result.Values["result"])
}
