// Command plan-and-execute_plan-and-execute is the Go port of
// langgraph/examples/plan-and-execute/plan-and-execute.ipynb.
//
// The notebook's graph is planner -> agent -> replan, where replan either
// sends the next step back to the agent or returns the final response. The Go
// port keeps that three-node loop and its accumulating past_steps channel:
// the planner and replanner are deterministic offline stubs, and the agent
// produces a canned observation for the current step instead of calling a
// search-backed ReAct agent.
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

// stepResult is one entry of the notebook's Annotated[List[Tuple], operator.add]
// past_steps field.
type stepResult struct {
	Task   string
	Result string
}

// planStep stands in for the LLM planner: it turns the objective into a short
// ordered list of steps.
func planStep(_ runtime.Runtime, state map[string]any) (any, error) {
	objective, _ := state["input"].(string)
	plan := []string{
		"identify the key facts needed for: " + objective,
		"retrieve and verify those facts",
		"compose the final answer",
	}
	fmt.Printf("planner produced %d steps\n", len(plan))
	return map[string]any{"plan": plan}, nil
}

// executeStep stands in for the ReAct agent executor. It consumes the first
// plan step and records a canned observation in past_steps.
func executeStep(_ runtime.Runtime, state map[string]any) (any, error) {
	plan, _ := state["plan"].([]string)
	if len(plan) == 0 {
		return nil, fmt.Errorf("execute_step: empty plan")
	}
	task := plan[0]
	done, _ := state["past_steps"].([]stepResult)
	result := fmt.Sprintf("observation for step %d", len(done)+1)
	fmt.Printf("agent executing: %s -> %s\n", task, result)
	return map[string]any{
		"past_steps": []stepResult{{Task: task, Result: result}},
	}, nil
}

// replanStep stands in for the LLM replanner. With steps remaining it returns
// an updated (shorter) plan; on the last step it returns the final response.
func replanStep(_ runtime.Runtime, state map[string]any) (any, error) {
	plan, _ := state["plan"].([]string)
	if len(plan) <= 1 {
		response := "Done: all plan steps executed for the given objective."
		fmt.Println("replanner: no steps left, returning response")
		return map[string]any{"response": response}, nil
	}
	fmt.Printf("replanner: %d step(s) remain, continuing\n", len(plan)-1)
	return map[string]any{"plan": plan[1:]}, nil
}

// shouldEnd mirrors the notebook's should_end router.
func shouldEnd(_ runtime.Runtime, state map[string]any) ([]any, error) {
	if response, _ := state["response"].(string); response != "" {
		return graphpkg.To(types.END), nil
	}
	return graphpkg.To("agent"), nil
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddReducer("past_steps", channels.AppendSliceReducer)
	g.AddNode("planner", planStep)
	g.AddNode("agent", executeStep)
	g.AddNode("replan", replanStep)
	g.AddEdge(types.START, "planner")
	g.AddEdge("planner", "agent")
	g.AddEdge("agent", "replan")
	g.AddConditionalEdges("replan", shouldEnd)
	cg, err := g.Compile(graphpkg.WithRecursionLimit(50))
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	result, err := cg.Invoke(ctx, map[string]any{
		"input":      "what is the hometown of the 2024 men's Australia Open winner?",
		"past_steps": []stepResult{},
	})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Println("--- final state ---")
	fmt.Printf("response: %v\n", result.Values["response"])
	steps, _ := result.Values["past_steps"].([]stepResult)
	for i, s := range steps {
		fmt.Printf("  %d. %s -> %s\n", i+1, s.Task, s.Result)
	}
}
