// Command pass-config-to-tools is the Go port of
// langgraph/examples/pass-config-to-tools.ipynb. Tools read the run's
// configurable user_id from their context to mutate a per-user favorites map.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/prebuilt"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// userToPets is the in-memory stand-in for the notebook's module-level dict.
var userToPets = map[string][]string{}

// userIDFromContext reads config["configurable"]["user_id"] from the tool's
// context. The graph passes runtime.Runtime as that context.
func userIDFromContext(ctx context.Context) string {
	cfg := runtime.ValuesFromContext(ctx)
	userID, _ := cfg["user_id"].(string)
	return userID
}

func buildTools() []tools.Tool {
	update, err := tools.NewSimple("update_favorite_pets", "Add the list of favorite pets (comma separated).",
		func(ctx context.Context, input string) (tools.Result, error) {
			userID := userIDFromContext(ctx)
			var pets []string
			for _, p := range strings.Split(input, ",") {
				if p = strings.TrimSpace(p); p != "" {
					pets = append(pets, p)
				}
			}
			userToPets[userID] = pets
			return tools.Result{Content: fmt.Sprintf("updated favorites for user %s: %v", userID, pets)}, nil
		})
	if err != nil {
		log.Fatalf("build update tool: %v", err)
	}

	deletePets, err := tools.NewSimple("delete_favorite_pets", "Delete the list of favorite pets.",
		func(ctx context.Context, _ string) (tools.Result, error) {
			userID := userIDFromContext(ctx)
			delete(userToPets, userID)
			return tools.Result{Content: fmt.Sprintf("deleted favorites for user %s", userID)}, nil
		})
	if err != nil {
		log.Fatalf("build delete tool: %v", err)
	}

	list, err := tools.NewSimple("list_favorite_pets", "List favorite pets if asked to.",
		func(ctx context.Context, _ string) (tools.Result, error) {
			userID := userIDFromContext(ctx)
			return tools.Result{Content: strings.Join(userToPets[userID], ", ")}, nil
		})
	if err != nil {
		log.Fatalf("build list tool: %v", err)
	}

	return []tools.Tool{update, deletePets, list}
}

// toolCallFor decides which tool the stubbed model requests on its first turn.
func toolCallFor(request string) messages.ToolCall {
	call := messages.ToolCall{Args: map[string]any{"tool_input": ""}}
	switch {
	case strings.Contains(request, "my favorite pets are"):
		call.Name = "update_favorite_pets"
		call.ID = "call_update"
		call.Args["tool_input"] = "cats, dogs"
	case strings.Contains(request, "forget"):
		call.Name = "delete_favorite_pets"
		call.ID = "call_delete"
	default:
		call.Name = "list_favorite_pets"
		call.ID = "call_list"
	}
	return call
}

// runConversation builds a fresh ReAct loop for one user request and invokes
// it with the given configurable user_id.
func runConversation(ctx context.Context, request, userID string) {
	toolNode, err := tools.NewToolNode(buildTools())
	if err != nil {
		log.Fatalf("build tool node: %v", err)
	}

	turn := 0
	agent := func(_ runtime.Runtime, state map[string]any) (any, error) {
		turn++
		var response messages.Message
		if turn == 1 {
			response = messages.AI("")
			response.ToolCalls = []messages.ToolCall{toolCallFor(request)}
		} else {
			response = messages.AI(fmt.Sprintf("Done. Current favorites for user %s: %v", userID, userToPets[userID]))
		}
		return map[string]any{"messages": []messages.Message{response}}, nil
	}
	shouldContinue := func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		if tools.HasPendingToolCalls(msgs) {
			return graphpkg.To("action"), nil
		}
		return graphpkg.To(types.END), nil
	}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("agent", agent)
	g.AddNode("action", prebuilt.ToolNode(toolNode))
	g.AddEdge(types.START, "agent")
	g.AddConditionalEdges("agent", shouldContinue)
	g.AddEdge("action", "agent")
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	// This is the Go analogue of passing config={"configurable": {...}} to
	// graph.stream(...) in the notebook.
	runCtx := runtime.ContextWithValues(ctx, map[string]any{"user_id": userID})
	if _, err := cg.Invoke(runCtx, map[string]any{
		"messages": []messages.Message{messages.Human(request)},
	}); err != nil {
		log.Fatalf("invoke: %v", err)
	}
}

func main() {
	ctx := context.Background()
	const userID = "123"

	fmt.Printf("User information prior to run: %v\n", userToPets)
	fmt.Println("--- request: my favorite pets are cats and dogs ---")
	runConversation(ctx, "my favorite pets are cats and dogs", userID)
	fmt.Printf("User information after the run: %v\n\n", userToPets)

	fmt.Printf("User information prior to run: %v\n", userToPets)
	fmt.Println("--- request: what are my favorite pets ---")
	runConversation(ctx, "what are my favorite pets", userID)
	fmt.Printf("User information after the run: %v\n\n", userToPets)

	fmt.Printf("User information prior to run: %v\n", userToPets)
	fmt.Println("--- request: please forget what i told you about my favorite animals ---")
	runConversation(ctx, "please forget what i told you about my favorite animals", userID)
	fmt.Printf("User information after the run: %v\n", userToPets)
}
