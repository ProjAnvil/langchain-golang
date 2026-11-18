// Command persistence_mongodb is the Go port of
// langgraph/examples/persistence_mongodb.ipynb.
//
// The notebook builds a create_react_agent with a MongoDBSaver (sync and
// async) and calls checkpointer.get_tuple(config) to fetch the latest
// checkpoint. This port keeps the checkpoint lifecycle but replaces the
// MongoDB connection with a deterministic in-memory saver stub: no database,
// no network and no API key are used. The weather tool is likewise an offline
// function, so the "agent" is a single node that answers the weather question
// directly.
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
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// getWeather is the offline analogue of the notebook's @tool get_weather.
func getWeather(city string) string {
	switch city {
	case "nyc":
		return "It might be cloudy in nyc"
	case "sf":
		return "It's always sunny in sf"
	default:
		return "Unknown city: " + city
	}
}

func detectCity(text string) string {
	lower := strings.ToLower(text)
	switch {
	case strings.Contains(lower, "sf"), strings.Contains(lower, "san francisco"):
		return "sf"
	case strings.Contains(lower, "nyc"), strings.Contains(lower, "new york"):
		return "nyc"
	default:
		return "sf"
	}
}

func respond(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	userText := ""
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleHuman {
			userText = messages.Text(msgs[i])
			break
		}
	}
	return map[string]any{"messages": []messages.Message{messages.AI(getWeather(detectCity(userText)))}}, nil
}

func lastAIText(state map[string]any) string {
	msgs, _ := state["messages"].([]messages.Message)
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleAI {
			return messages.Text(msgs[i])
		}
	}
	return ""
}

func printSnapshot(label string, snap graphpkg.StateSnapshot) {
	msgs, _ := snap.Values["messages"].([]messages.Message)
	fmt.Printf("%s: messages=%d, next=%v\n", label, len(msgs), snap.Next)
}

func main() {
	ctx := context.Background()

	// In the notebook this is MongoDBSaver.from_conn_string(MONGODB_URI).
	// Here a deterministic in-memory saver stub stands in for MongoDBSaver.
	saver := checkpoint.NewMemorySaver()

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("respond", respond)
	g.AddEdge(types.START, "respond")
	g.AddEdge("respond", types.END)
	cg, err := g.Compile(graphpkg.WithCheckpointer(saver))
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}

	// The notebook invokes thread_ids "1" through "4" with the same saver.
	cases := []struct{ thread, question string }{
		{"1", "what's the weather in sf"},
		{"2", "What's the weather in sf?"},
		{"3", "what's the weather in sf"},
		{"4", "what's the weather in nyc"},
	}
	for _, tc := range cases {
		res, err := cg.InvokeWithOptions(ctx,
			map[string]any{"messages": []messages.Message{messages.Human(tc.question)}},
			graphpkg.Options{ThreadID: tc.thread})
		if err != nil {
			log.Fatalf("invoke thread %s: %v", tc.thread, err)
		}
		fmt.Printf("thread %s: %s\n", tc.thread, lastAIText(res.Values))

		// checkpointer.get_tuple(config) becomes CompiledGraph.GetState.
		snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: tc.thread})
		if err != nil {
			log.Fatalf("GetState thread %s: %v", tc.thread, err)
		}
		printSnapshot("  GetState (latest checkpoint tuple)", snap)
	}

	// The notebook's checkpointer.list(config) becomes GetStateHistory.
	history, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "1"}, checkpoint.ListOptions{})
	if err != nil {
		log.Fatalf("GetStateHistory: %v", err)
	}
	fmt.Printf("GetStateHistory(thread 1): %d checkpoints\n", len(history))
	for i, snap := range history {
		fmt.Printf("  [%d] source=%s step=%d\n", i, snap.Metadata.Source, snap.Metadata.Step)
	}
}
