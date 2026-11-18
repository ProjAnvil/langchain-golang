// Command persistence_postgres is the Go port of
// langgraph/examples/persistence_postgres.ipynb.
//
// The notebook creates a PostgresSaver (sync and async), calls its one-time
// setup(), builds a create_react_agent, invokes several thread_ids, and then
// reads checkpoints with get(config), get_tuple(config) and list(config). This
// port preserves that checkpoint lifecycle but replaces the PostgreSQL
// connection with a deterministic in-memory saver stub, so no database,
// network or API key is needed. The weather tool is an offline function.
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

func main() {
	ctx := context.Background()

	// In the notebook this is PostgresSaver(pool) followed by saver.setup().
	// The in-memory saver needs no setup and no DB_URI: it is the stub that
	// stands in for PostgresSaver / AsyncPostgresSaver here.
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

	cases := []struct{ thread, question string }{
		{"1", "what's the weather in sf"},
		{"2", "what's the weather in sf"},
		{"3", "what's the weather in nyc"},
	}
	for _, tc := range cases {
		res, err := cg.InvokeWithOptions(ctx,
			map[string]any{"messages": []messages.Message{messages.Human(tc.question)}},
			graphpkg.Options{ThreadID: tc.thread})
		if err != nil {
			log.Fatalf("invoke thread %s: %v", tc.thread, err)
		}
		fmt.Printf("thread %s: %s\n", tc.thread, lastAIText(res.Values))

		// checkpointer.get(config) / get_tuple(config) -> GetState.
		snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: tc.thread})
		if err != nil {
			log.Fatalf("GetState thread %s: %v", tc.thread, err)
		}
		msgs, _ := snap.Values["messages"].([]messages.Message)
		fmt.Printf("  GetState (checkpoint): source=%s step=%d messages=%d next=%v\n",
			snap.Metadata.Source, snap.Metadata.Step, len(msgs), snap.Next)

		// checkpointer.list(config) -> GetStateHistory.
		history, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: tc.thread}, checkpoint.ListOptions{})
		if err != nil {
			log.Fatalf("GetStateHistory thread %s: %v", tc.thread, err)
		}
		fmt.Printf("  GetStateHistory: %d checkpoint(s)\n", len(history))
	}
}
