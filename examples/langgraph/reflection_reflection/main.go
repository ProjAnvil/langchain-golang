// Command reflection_reflection is the Go port of
// langgraph/examples/reflection/reflection.ipynb.
//
// The notebook loops a generator and a critic: generate writes an essay,
// reflect flips the message roles and writes critique as human feedback, and
// should_continue ends the loop after three iterations. The Go port keeps the
// same graph (`generate -> reflect -> generate` with a conditional edge on
// generate) and a MessagesReducer channel. Both the essay generation and the
// critique are deterministic offline stubs; no LLM or API key is used.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// generationNode stands in for the essay LLM. It appends a progressively
// revised five-paragraph essay draft, using the critique when one exists.
func generationNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	drafts := 0
	for _, m := range msgs {
		if m.Role == messages.RoleAI {
			drafts++
		}
	}
	essay := fmt.Sprintf("[Draft %d] An essay on the little prince in modern childhood, "+
		"with a clearer thesis and deeper examples than the previous attempt.", drafts+1)
	fmt.Printf("generate -> %s\n", essay)
	return map[string]any{
		"messages": []messages.Message{messages.AI(essay)},
	}, nil
}

// reflectionNode stands in for the teacher LLM. It mirrors the notebook's
// role flip (AI messages become human, human messages become AI) and appends
// the critique as a human message for the generator.
func reflectionNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) > 0 {
		translated := []messages.Message{msgs[0]}
		for _, m := range msgs[1:] {
			switch m.Role {
			case messages.RoleAI:
				translated = append(translated, messages.Human(messages.Text(m)))
			case messages.RoleHuman:
				translated = append(translated, messages.AI(messages.Text(m)))
			default:
				translated = append(translated, m)
			}
		}
		_ = translated // the offline critic keys its feedback off the essay count
	}
	critique := "Critique: add one more historical example and tighten the concluding paragraph."
	fmt.Printf("reflect -> %s\n", critique)
	return map[string]any{
		"messages": []messages.Message{messages.Human(critique)},
	}, nil
}

// shouldContinue mirrors the notebook: end once more than six messages have
// accumulated (three generate/reflect iterations), otherwise reflect again.
func shouldContinue(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) > 6 {
		return graphpkg.To(types.END), nil
	}
	return graphpkg.To("reflect"), nil
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("generate", generationNode)
	g.AddNode("reflect", reflectionNode)
	g.AddEdge(types.START, "generate")
	g.AddConditionalEdges("generate", shouldContinue)
	g.AddEdge("reflect", "generate")
	cg, err := g.Compile(
		graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()),
		graphpkg.WithRecursionLimit(50),
	)
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	request := messages.Human("Generate an essay on the topicality of The Little Prince in modern life.")
	result, err := cg.InvokeWithOptions(ctx,
		map[string]any{"messages": []messages.Message{request}},
		graphpkg.Options{ThreadID: "reflection-1"},
	)
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Println("--- final conversation ---")
	msgs, _ := result.Values["messages"].([]messages.Message)
	for i, m := range msgs {
		fmt.Printf("%d. %s: %s\n", i+1, m.Role, messages.Text(m))
	}
}
