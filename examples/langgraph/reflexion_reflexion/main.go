// Command reflexion_reflexion is the Go port of
// langgraph/examples/reflexion/reflexion.ipynb.
//
// The notebook's graph is draft -> execute_tools -> revise, with a conditional
// edge on revise that loops back to the tool node until MAX_ITERATIONS. The
// Go port keeps that loop and its message-accumulating state. The ReAct-style
// draft/revise "LLM" calls are deterministic offline stubs that emit an
// answer, a self-critique, and search queries; the search tool is a safe
// in-memory lookup returning canned evidence. No API key or network.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

const maxIterations = 5

// draftNode stands in for first_responder. It emits a structured-looking
// answer with a reflection and search queries as a single AI message.
func draftNode(_ runtime.Runtime, state map[string]any) (any, error) {
	content := "Answer: reflection improves AI by critiquing a draft and revising it. " +
		"Reflection: the answer lacks numerical evidence. " +
		"Search queries: [benefits of self-reflection in language models]"
	fmt.Println("draft -> initial answer with critique and one search query")
	return map[string]any{
		"messages": []messages.Message{messages.AI(content)},
	}, nil
}

// toolNode stands in for the notebook's ToolNode running the generated search
// queries. It returns canned evidence as a tool message, never touching the
// network or executing code.
func toolNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	observation := "tool evidence: studies show self-refinement lifts answer quality on reasoning tasks."
	fmt.Printf("execute_tools -> %s (saw %d prior messages)\n", observation, len(msgs))
	return map[string]any{
		"messages": []messages.Message{
			{Role: messages.RoleTool, Content: observation},
		},
	}, nil
}

// reviseNode stands in for revisor. It emits a revised answer using the latest
// tool observation.
func reviseNode(_ runtime.Runtime, state map[string]any) (any, error) {
	content := "Revised answer: reflection improves AI quality [1]. References: [1] https://example.com/reflection. " +
		"Search queries: []"
	fmt.Println("revise -> revised answer with a citation")
	return map[string]any{
		"messages": []messages.Message{messages.AI(content)},
	}, nil
}

// eventLoop mirrors the notebook's event_loop: count trailing AI/tool
// messages and end once the count exceeds MAX_ITERATIONS.
func eventLoop(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	count := 0
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != messages.RoleAI && msgs[i].Role != messages.RoleTool {
			break
		}
		count++
	}
	if count > maxIterations {
		return graphpkg.To(types.END), nil
	}
	return graphpkg.To("execute_tools"), nil
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("draft", draftNode)
	g.AddNode("execute_tools", toolNode)
	g.AddNode("revise", reviseNode)
	g.AddEdge(types.START, "draft")
	g.AddEdge("draft", "execute_tools")
	g.AddEdge("execute_tools", "revise")
	g.AddConditionalEdges("revise", eventLoop)
	cg, err := g.Compile(graphpkg.WithRecursionLimit(50))
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	question := "How should we handle the climate crisis?"
	result, err := cg.Invoke(ctx, map[string]any{
		"messages": []messages.Message{messages.Human(question)},
	})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Println("--- final answer ---")
	msgs, _ := result.Values["messages"].([]messages.Message)
	if len(msgs) > 0 {
		fmt.Println(messages.Text(msgs[len(msgs)-1]))
	}
	fmt.Printf("total messages: %d\n", len(msgs))
}
