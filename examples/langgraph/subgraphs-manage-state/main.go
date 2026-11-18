// Command subgraphs-manage-state is the Go port of
// langgraph/examples/subgraphs-manage-state.ipynb.
//
// The notebook embeds a weather subgraph (compiled with
// interrupt_before=["weather_node"]) in a router graph, pauses inside the
// subgraph, then inspects and edits state with get_state / get_state_history /
// update_state at both the parent and subgraph level. This port keeps that
// control flow with a deterministic stand-in for the LLM: routing is a simple
// keyword check and the weather tool is an offline function.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func messagesOf(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

func lastHumanText(state map[string]any) string {
	msgs := messagesOf(state)
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleHuman {
			return messages.Text(msgs[i])
		}
	}
	return ""
}

func str(state map[string]any, key string) string {
	if value, ok := state[key].(string); ok {
		return value
	}
	return ""
}

func printMessages(label string, state map[string]any) {
	fmt.Println(label)
	for _, m := range messagesOf(state) {
		fmt.Printf("  %-6s %s\n", m.Role, messages.Text(m))
	}
}

func getWeather(city string) string {
	return "It's sunny in " + city + "!"
}

func detectCity(text string) string {
	lower := strings.ToLower(text)
	switch {
	case strings.Contains(lower, "sf"), strings.Contains(lower, "san francisco"):
		return "sf"
	case strings.Contains(lower, "nyc"), strings.Contains(lower, "new york"):
		return "nyc"
	default:
		return "unknown"
	}
}

// appendAI returns a new full message list, mirroring the notebook's
// {"messages": [response]} update. The graph uses LastValue channels for
// messages, so returning the full list avoids double-counting when the child's
// values are merged back into the parent by AddSubgraph.
func appendAI(state map[string]any, reply string) []messages.Message {
	msgs := append([]messages.Message(nil), messagesOf(state)...)
	return append(msgs, messages.AI(reply))
}

func modelNode(_ runtime.Runtime, state map[string]any) (any, error) {
	return map[string]any{"city": detectCity(lastHumanText(state))}, nil
}

func weatherNode(_ runtime.Runtime, state map[string]any) (any, error) {
	return map[string]any{"messages": appendAI(state, getWeather(str(state, "city")))}, nil
}

func routerNode(_ runtime.Runtime, state map[string]any) (any, error) {
	route := "other"
	if strings.Contains(strings.ToLower(lastHumanText(state)), "weather") {
		route = "weather"
	}
	return map[string]any{"route": route}, nil
}

func normalLLMNode(_ runtime.Runtime, state map[string]any) (any, error) {
	return map[string]any{"messages": appendAI(state, "I'm a normal assistant reply.")}, nil
}

func routeAfterPrediction(_ runtime.Runtime, state map[string]any) ([]any, error) {
	if str(state, "route") == "weather" {
		return graphpkg.To("weather_graph"), nil
	}
	return graphpkg.To("normal_llm_node"), nil
}

func compileOrDie(label string, g *graphpkg.StateGraph, opts ...graphpkg.CompileOption) *graphpkg.CompiledGraph {
	cg, err := g.Compile(opts...)
	if err != nil {
		log.Fatalf("compile %s: %v", label, err)
	}
	return cg
}

func invokeOrDie(label string, cg *graphpkg.CompiledGraph, input map[string]any, opts graphpkg.Options) graphpkg.Result {
	res, err := cg.InvokeWithOptions(context.Background(), input, opts)
	if err != nil {
		log.Fatalf("%s: %v", label, err)
	}
	return res
}

func main() {
	ctx := context.Background()
	saver := checkpoint.NewMemorySaver()

	// --- Weather subgraph, paused before weather_node. ---
	sub := graphpkg.NewStateGraph()
	sub.AddNode("model_node", modelNode)
	sub.AddNode("weather_node", weatherNode)
	sub.AddEdge(types.START, "model_node")
	sub.AddEdge("model_node", "weather_node")
	sub.AddEdge("weather_node", types.END)
	subCG := compileOrDie("weather subgraph", sub, graphpkg.WithInterruptBefore("weather_node"))

	// --- Router graph embedding the subgraph. ---
	router := graphpkg.NewStateGraph()
	router.AddNode("router_node", routerNode)
	router.AddNode("normal_llm_node", normalLLMNode)
	router.AddSubgraph("weather_graph", subCG)
	router.AddEdge(types.START, "router_node")
	router.AddConditionalEdgesWithPathMap("router_node", routeAfterPrediction, map[string]string{
		"weather_graph":   "weather_graph",
		"normal_llm_node": "normal_llm_node",
	})
	router.AddEdge("normal_llm_node", types.END)
	router.AddEdge("weather_graph", types.END)
	routerCG := compileOrDie("router graph", router, graphpkg.WithCheckpointer(saver))

	// Case 1 (thread 1): a non-weather query takes the normal branch.
	normal := invokeOrDie("normal query", routerCG,
		map[string]any{"messages": []messages.Message{messages.Human("hi!")}},
		graphpkg.Options{ThreadID: "1"})
	printMessages("thread 1 (normal branch) final state:", normal.Values)

	// Case 2 (thread 2): a weather query pauses before weather_node inside the
	// subgraph. get_state shows the pending task; resuming finishes the run.
	weatherCfg := checkpoint.Config{ThreadID: "2"}
	paused := invokeOrDie("weather query", routerCG,
		map[string]any{"messages": []messages.Message{messages.Human("what's the weather in sf")}},
		graphpkg.Options{ThreadID: "2"})
	fmt.Printf("thread 2: paused with %d interrupt(s)\n", len(paused.Interrupts))
	for _, intr := range paused.Interrupts {
		fmt.Printf("  interrupt value=%v ns=%q\n", intr.Value, intr.NS)
	}
	pausedState, err := routerCG.GetState(ctx, weatherCfg)
	if err != nil {
		log.Fatalf("GetState thread 2: %v", err)
	}
	fmt.Printf("thread 2 get_state: next=%v tasks=%d\n", pausedState.Next, len(pausedState.Tasks))
	printMessages("thread 2 state at pause:", pausedState.Values)

	resumed := invokeOrDie("resume thread 2", routerCG, nil, graphpkg.Options{ThreadID: "2"})
	fmt.Printf("thread 2: resumed with %d interrupt(s)\n", len(resumed.Interrupts))
	printMessages("thread 2 final state:", resumed.Values)

	history, err := routerCG.GetStateHistory(ctx, weatherCfg, checkpoint.ListOptions{})
	if err != nil {
		log.Fatalf("GetStateHistory thread 2: %v", err)
	}
	fmt.Printf("thread 2 get_state_history: %d checkpoint(s)\n", len(history))
	for i, snap := range history {
		fmt.Printf("  [%d] source=%s step=%d next=%v\n", i, snap.Metadata.Source, snap.Metadata.Step, snap.Next)
	}

	// Case 3 (thread 3): pause, then edit state at the parent level with
	// update_state(as_node="weather_graph") and continue -- the notebook's
	// "skip the subgraph and inject the final answer" pattern.
	injectCfg := checkpoint.Config{ThreadID: "3"}
	invokeOrDie("weather query thread 3", routerCG,
		map[string]any{"messages": []messages.Message{messages.Human("what's the weather in sf")}},
		graphpkg.Options{ThreadID: "3"})
	updatedCfg, err := routerCG.UpdateState(ctx, injectCfg,
		map[string]any{"messages": []messages.Message{messages.AI("rainy")}}, "weather_graph")
	if err != nil {
		log.Fatalf("UpdateState thread 3: %v", err)
	}
	fmt.Printf("thread 3 update_state produced checkpoint %s\n", updatedCfg.CheckpointID)
	afterUpdate := invokeOrDie("resume thread 3", routerCG, nil, graphpkg.Options{ThreadID: "3"})
	printMessages("thread 3 state after update and resume:", afterUpdate.Values)

	// A grandparent graph wrapping the router, matching the notebook's xray
	// visualization cell. GetGraph(WithXrayDepth(-1)) expands every subgraph.
	grandparent := graphpkg.NewStateGraph()
	grandparent.AddNode("router_node", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"to_continue": true}, nil
	})
	grandparent.AddSubgraph("graph", routerCG)
	grandparent.AddEdge(types.START, "router_node")
	grandparent.AddConditionalEdgesWithPathMap("router_node",
		func(_ runtime.Runtime, state map[string]any) ([]any, error) {
			if state["to_continue"] == true {
				return graphpkg.To("graph"), nil
			}
			return graphpkg.To(types.END), nil
		},
		map[string]string{"graph": "graph", types.END: types.END})
	grandparent.AddEdge("graph", types.END)
	grandparentCG := compileOrDie("grandparent graph", grandparent, graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	fmt.Println("grandparent graph Mermaid (xray=-1):")
	fmt.Println(grandparentCG.GetGraph(graphpkg.WithXrayDepth(-1)).DrawMermaid())
}
