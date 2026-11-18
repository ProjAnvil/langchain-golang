// Command map-reduce is the Go port of langgraph/examples/map-reduce.ipynb.
// It fans out one node invocation per subject with types.Send and reduces the
// parallel results back into a single list with channels.AppendSliceReducer.
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

func main() {
	ctx := context.Background()

	// generateTopics is the stubbed structured-output model from the notebook:
	// a fixed subject list for the given topic.
	generateTopics := func(_ runtime.Runtime, state map[string]any) (any, error) {
		topic, _ := state["topic"].(string)
		subjects := map[string][]string{
			"animals": {"cats", "dogs", "weather"},
		}[topic]
		if subjects == nil {
			subjects = []string{"cats", "dogs"}
		}
		fmt.Printf("  [generate_topics] topic=%q -> subjects=%v\n", topic, subjects)
		return map[string]any{"subjects": subjects}, nil
	}

	// continueToJokes mirrors the notebook's Send-based fan-out edge.
	continueToJokes := func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		raw, _ := state["subjects"].([]string)
		sends := make([]any, 0, len(raw))
		for _, subject := range raw {
			sends = append(sends, types.Send{
				Node: "generate_joke",
				Arg:  map[string]any{"subject": subject},
			})
		}
		fmt.Printf("  [continue_to_jokes] fanning out %d Send(s)\n", len(sends))
		return sends, nil
	}

	// generateJoke receives one subject via the Send arg (not the shared
	// state), just like the notebook's JokeState input.
	generateJoke := func(_ runtime.Runtime, state map[string]any) (any, error) {
		subject, _ := state["subject"].(string)
		joke := "Why did the " + subject + " cross the road? To get to the other side."
		fmt.Printf("  [generate_joke] subject=%q\n", subject)
		return map[string]any{"jokes": []string{joke}}, nil
	}

	// bestJoke is the stubbed judge: pick the shortest joke deterministically.
	bestJoke := func(_ runtime.Runtime, state map[string]any) (any, error) {
		jokes, _ := state["jokes"].([]string)
		if len(jokes) == 0 {
			return map[string]any{"best_selected_joke": ""}, nil
		}
		best := jokes[0]
		for _, j := range jokes[1:] {
			if len(j) < len(best) {
				best = j
			}
		}
		fmt.Printf("  [best_joke] chose from %d jokes\n", len(jokes))
		return map[string]any{"best_selected_joke": best}, nil
	}

	g := graphpkg.NewStateGraph()
	g.AddReducer("jokes", channels.AppendSliceReducer)
	// Go dispatches one best_joke task per fanned-out generate_joke branch
	// (Python deduplicates them into one). Each branch writes the same value,
	// so a last-write-wins reducer merges them into a single final value.
	g.AddReducer("best_selected_joke", channels.LastValueReducer)
	g.AddNode("generate_topics", generateTopics)
	g.AddNode("generate_joke", generateJoke)
	g.AddNode("best_joke", bestJoke)
	g.AddEdge(types.START, "generate_topics")
	g.AddConditionalEdges("generate_topics", continueToJokes)
	g.AddEdge("generate_joke", "best_joke")
	g.AddEdge("best_joke", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	input := map[string]any{"topic": "animals"}

	fmt.Println("=== Invoke (final reduced state) ===")
	res, err := cg.Invoke(ctx, input)
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	jokes, _ := res.Values["jokes"].([]string)
	best, _ := res.Values["best_selected_joke"].(string)
	fmt.Printf("  subjects: %v\n", res.Values["subjects"])
	fmt.Printf("  jokes collected: %d\n", len(jokes))
	for i, j := range jokes {
		fmt.Printf("    [%d] %s\n", i, j)
	}
	fmt.Printf("  best_selected_joke: %s\n", best)

	fmt.Println("=== Stream (updates, one chunk per node) ===")
	for chunk, err := range cg.Stream(ctx, input, graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamUpdates},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		fmt.Printf("  chunk: %v\n", chunk.Payload)
	}
}
