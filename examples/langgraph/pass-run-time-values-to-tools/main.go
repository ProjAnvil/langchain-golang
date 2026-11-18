// Command pass-run-time-values-to-tools is the Go port of
// langgraph/examples/pass-run-time-values-to-tools.ipynb. It injects graph
// state and a cross-thread store into tools via a ToolCallWrapper.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/prebuilt"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Context keys used by the wrapper to hand state/store to tool functions.
type injectedStateKey struct{}
type injectedStoreKey struct{}

// injectRunValues is the explicit Go replacement for Python's InjectedState /
// InjectedStore annotations: it copies the request's State and Store onto the
// context the tool sees.
func injectRunValues(ctx context.Context, req tools.ToolCallRequest, next tools.ToolHandler) (messages.Message, error) {
	c := context.WithValue(ctx, injectedStateKey{}, req.State)
	c = context.WithValue(c, injectedStoreKey{}, req.Store)
	return next(c, req)
}

// buildLoop assembles the notebook's agent/tool cycle around a ToolNode.
func buildLoop(agent graphpkg.NodeFunc, tn *tools.ToolNode, opts ...graphpkg.CompileOption) *graphpkg.CompiledGraph {
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
	g.AddNode("action", prebuilt.ToolNode(tn))
	g.AddEdge(types.START, "agent")
	g.AddConditionalEdges("agent", shouldContinue)
	g.AddEdge("action", "agent")
	cg, err := g.Compile(opts...)
	if err != nil {
		log.Fatalf("compile: %v", err)
	}
	return cg
}

// makeAgent builds a stub model that calls the given tool on its first turn
// and then answers using the tool result.
func makeAgent(toolName, request string) graphpkg.NodeFunc {
	turn := 0
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		turn++
		var response messages.Message
		if turn == 1 {
			response = messages.AI("")
			response.ToolCalls = []messages.ToolCall{{
				ID:   "call_" + toolName,
				Name: toolName,
				Args: map[string]any{"tool_input": "latest news"},
			}}
		} else {
			answer := "(no tool result)"
			msgs, _ := state["messages"].([]messages.Message)
			for _, m := range msgs {
				if m.Role == messages.RoleTool {
					answer = m.Content
					break
				}
			}
			response = messages.AI("Context from tool: " + answer)
		}
		_ = request
		return map[string]any{"messages": []messages.Message{response}}, nil
	}
}

func printFinal(res graphpkg.Result) {
	msgs, _ := res.Values["messages"].([]messages.Message)
	if len(msgs) == 0 {
		return
	}
	fmt.Printf("  final answer: %s\n", messages.Text(msgs[len(msgs)-1]))
}

func main() {
	ctx := context.Background()

	// ---------------------------------------------------------------
	// Scenario 1: InjectedState — the tool reads state["docs"].
	// ---------------------------------------------------------------
	fmt.Println("=== Scenario 1: injected state (docs) ===")
	getContext, err := tools.NewSimple("get_context", "Get relevant context for answering the question.",
		func(ctx context.Context, _ string) (tools.Result, error) {
			st, _ := ctx.Value(injectedStateKey{}).(map[string]any)
			docs, _ := st["docs"].([]string)
			fmt.Printf("  [get_context] injected docs: %v\n", docs)
			return tools.Result{Content: strings.Join(docs, "\n\n")}, nil
		})
	if err != nil {
		log.Fatalf("build get_context tool: %v", err)
	}
	tnState, err := tools.NewToolNode([]tools.Tool{getContext}, tools.WithToolCallWrapper(injectRunValues))
	if err != nil {
		log.Fatalf("build state tool node: %v", err)
	}
	cgState := buildLoop(makeAgent("get_context", "news"), tnState)
	res, err := cgState.Invoke(ctx, map[string]any{
		"messages": []messages.Message{messages.Human("what's the latest news about FooBar")},
		"docs": []string{
			"FooBar company just raised 1 Billion dollars!",
			"FooBar company was founded in 2019",
		},
	})
	if err != nil {
		log.Fatalf("invoke state scenario: %v", err)
	}
	printFinal(res)

	// ---------------------------------------------------------------
	// Scenario 2: InjectedStore + config user_id.
	// ---------------------------------------------------------------
	fmt.Println("=== Scenario 2: injected store + configurable user_id ===")
	docStore := store.NewInMemoryStore()
	for _, entry := range []struct {
		userID, key, doc string
	}{
		{"1", "doc_0", "FooBar company just raised 1 Billion dollars!"},
		{"2", "doc_1", "FooBar company was founded in 2019"},
	} {
		if err := docStore.Put(ctx, []string{"documents", entry.userID}, entry.key,
			map[string]any{"doc": entry.doc}, nil); err != nil {
			log.Fatalf("seed store: %v", err)
		}
	}

	getContextStore, err := tools.NewSimple("get_context_store", "Get relevant context from the store.",
		func(ctx context.Context, _ string) (tools.Result, error) {
			st, _ := ctx.Value(injectedStoreKey{}).(store.Store)
			if st == nil {
				return tools.Result{}, errors.New("no store injected into tool")
			}
			userID := ""
			if cfg := runtime.ValuesFromContext(ctx); cfg != nil {
				userID, _ = cfg["user_id"].(string)
			}
			items, err := st.Search(ctx, []string{"documents", userID}, store.SearchOptions{})
			if err != nil {
				return tools.Result{}, err
			}
			docs := make([]string, 0, len(items))
			for _, item := range items {
				if d, ok := item.Value["doc"].(string); ok {
					docs = append(docs, d)
				}
			}
			fmt.Printf("  [get_context_store] user_id=%s docs=%v\n", userID, docs)
			return tools.Result{Content: strings.Join(docs, "\n\n")}, nil
		})
	if err != nil {
		log.Fatalf("build get_context_store tool: %v", err)
	}
	// A fresh graph per run keeps the stub model's turn counter at zero.
	for _, userID := range []string{"1", "2"} {
		tnStore, err := tools.NewToolNode([]tools.Tool{getContextStore},
			tools.WithToolNodeStore(docStore),
			tools.WithToolCallWrapper(injectRunValues))
		if err != nil {
			log.Fatalf("build store tool node: %v", err)
		}
		cgStore := buildLoop(makeAgent("get_context_store", "news"), tnStore, graphpkg.WithStore(docStore))

		fmt.Printf("--- thread with user_id=%s ---\n", userID)
		runCtx := runtime.ContextWithValues(ctx, map[string]any{"user_id": userID})
		res, err := cgStore.Invoke(runCtx, map[string]any{
			"messages": []messages.Message{messages.Human("what's the latest news about FooBar")},
		})
		if err != nil {
			log.Fatalf("invoke store scenario: %v", err)
		}
		printFinal(res)
	}
}
