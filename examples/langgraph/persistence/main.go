// Command persistence is the Go port of langgraph/examples/persistence.ipynb.
// It demonstrates the two persistence layers the notebook exercises, both
// offline:
//
//  1. Thread-scoped conversation memory: checkpoint.NewMemorySaver() persists
//     the message state between invocations that share a ThreadID, so the
//     second turn sees the first turn's messages.
//  2. Cross-thread long-term memory: graphpkg.WithStore plus
//     store.NewInMemoryStore() gives nodes a store shared across threads; a
//     node reads "memories" scoped by user_id and writes one when the user
//     asks the assistant to remember something.
//
// A deterministic in-process function replaces the chat model, so no API key
// or network is required.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func lastText(state map[string]any) string {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) == 0 {
		return ""
	}
	return messages.Text(msgs[len(msgs)-1])
}

func messageCount(state map[string]any) int {
	msgs, _ := state["messages"].([]messages.Message)
	return len(msgs)
}

func printMessages(label string, state map[string]any) {
	fmt.Println(label)
	msgs, _ := state["messages"].([]messages.Message)
	for _, m := range msgs {
		fmt.Printf("  %-6s %s\n", m.Role, messages.Text(m))
	}
}

// assistant is the offline stand-in for model.invoke(state["messages"]). Its
// reply echoes the latest user text and reports the history length, which
// makes the thread-scoped checkpoint reuse visible.
func assistant(_ runtime.Runtime, state map[string]any) (any, error) {
	reply := fmt.Sprintf("You said %q; I can see %d message(s) in this thread.",
		lastText(state), messageCount(state))
	return map[string]any{"messages": []messages.Message{messages.AI(reply)}}, nil
}

// memoryAssistant adds the notebook's store-backed long-term memory: it reads
// all memories under ("memories", user_id), stores "User name is Bob" when the
// latest message asks to remember something, and includes the recalled info in
// its reply. The user_id is carried in the state map (the Go analogue of the
// Python config["configurable"]["user_id"]).
func memoryAssistant(rt runtime.Runtime, state map[string]any) (any, error) {
	text := lastText(state)
	userID, _ := state["user_id"].(string)
	if userID == "" {
		userID = "1"
	}
	namespace := []string{"memories", userID}

	var recalled []string
	if rt.Store != nil {
		items, err := rt.Store.Search(rt, namespace, store.SearchOptions{Query: text})
		if err != nil {
			return nil, err
		}
		for _, item := range items {
			if data, ok := item.Value["data"].(string); ok {
				recalled = append(recalled, data)
			}
		}
		if strings.Contains(strings.ToLower(text), "remember") {
			if err := rt.Store.Put(rt, namespace, "name",
				map[string]any{"data": "User name is Bob"}, nil); err != nil {
				return nil, err
			}
		}
	}
	info := strings.Join(recalled, "; ")
	if info == "" {
		info = "(none yet)"
	}
	reply := fmt.Sprintf("Recalled user info: %s. You said: %s", info, text)
	return map[string]any{"messages": []messages.Message{messages.AI(reply)}}, nil
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

	// --- Part 1: thread-scoped conversation memory. ---
	threadGraph := graphpkg.NewStateGraph()
	threadGraph.AddReducer("messages", channels.MessagesReducer)
	threadGraph.AddNode("call_model", assistant)
	threadGraph.AddEdge(types.START, "call_model")
	threadGraph.AddEdge("call_model", types.END)
	threadCG := compileOrDie("thread graph", threadGraph,
		graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))

	first := invokeOrDie("turn 1", threadCG,
		map[string]any{"messages": []messages.Message{messages.Human("hi! I'm bob")}},
		graphpkg.Options{ThreadID: "1"})
	printMessages("Conversation after turn 1 (thread 1):", first.Values)

	second := invokeOrDie("turn 2", threadCG,
		map[string]any{"messages": []messages.Message{messages.Human("what's my name?")}},
		graphpkg.Options{ThreadID: "1"})
	printMessages("Conversation after turn 2 (thread 1):", second.Values)

	snap, err := threadCG.GetState(ctx, checkpoint.Config{ThreadID: "1"})
	if err != nil {
		log.Fatalf("GetState: %v", err)
	}
	fmt.Printf("GetState(thread 1): messages=%d, next=%v\n", messageCount(snap.Values), snap.Next)

	history, err := threadCG.GetStateHistory(ctx, checkpoint.Config{ThreadID: "1"}, checkpoint.ListOptions{})
	if err != nil {
		log.Fatalf("GetStateHistory: %v", err)
	}
	fmt.Printf("GetStateHistory(thread 1): %d checkpoints\n", len(history))

	// --- Part 2: cross-thread memory via the store. ---
	storeGraph := graphpkg.NewStateGraph()
	storeGraph.AddReducer("messages", channels.MessagesReducer)
	storeGraph.AddNode("call_model", memoryAssistant)
	storeGraph.AddEdge(types.START, "call_model")
	storeGraph.AddEdge("call_model", types.END)
	storeCG := compileOrDie("store graph", storeGraph,
		graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()),
		graphpkg.WithStore(store.NewInMemoryStore()))

	remember := invokeOrDie("remember turn", storeCG, map[string]any{
		"messages": []messages.Message{messages.Human("Hi! Remember: my name is Bob")},
		"user_id":  "1",
	}, graphpkg.Options{ThreadID: "1"})
	printMessages("Store graph, thread 1 (writes the memory):", remember.Values)

	recall := invokeOrDie("recall turn", storeCG, map[string]any{
		"messages": []messages.Message{messages.Human("what is my name?")},
		"user_id":  "1",
	}, graphpkg.Options{ThreadID: "2"})
	printMessages("Store graph, thread 2 (reads the same user's memory):", recall.Values)
}
