// Command persistence_redis is the Go port of
// langgraph/examples/persistence_redis.ipynb.
//
// The notebook defines its own RedisSaver / AsyncRedisSaver checkpoint saver
// classes, builds a create_react_agent with them, and reads checkpoints with
// get(config), get_tuple(config) and list(config). This port keeps the
// checkpoint lifecycle and those three read patterns, but replaces the Redis
// connection and the hand-written saver with a deterministic in-memory saver
// stub. No Redis server, network or API key is used, and the weather tool is
// an offline function.
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

	// The notebook wraps a Redis connection in RedisSaver.from_conn_info(...).
	// A deterministic in-memory saver stub stands in for that custom saver.
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
		{"2", "what's the weather in nyc"},
	}
	for _, tc := range cases {
		res, err := cg.InvokeWithOptions(ctx,
			map[string]any{"messages": []messages.Message{messages.Human(tc.question)}},
			graphpkg.Options{ThreadID: tc.thread})
		if err != nil {
			log.Fatalf("invoke thread %s: %v", tc.thread, err)
		}
		fmt.Printf("thread %s: %s\n", tc.thread, lastAIText(res.Values))

		cfg := checkpoint.Config{ThreadID: tc.thread}

		// checkpointer.get(config) -> the checkpoint's state values.
		snap, err := cg.GetState(ctx, cfg)
		if err != nil {
			log.Fatalf("GetState thread %s: %v", tc.thread, err)
		}
		msgs, _ := snap.Values["messages"].([]messages.Message)
		fmt.Printf("  get: source=%s step=%d messages=%d\n",
			snap.Metadata.Source, snap.Metadata.Step, len(msgs))

		// checkpointer.get_tuple(config) -> the full snapshot (with config).
		fmt.Printf("  get_tuple: checkpoint_id present=%t, next=%v\n",
			snap.Config.CheckpointID != "", snap.Next)

		// checkpointer.list(config) -> GetStateHistory, newest first.
		history, err := cg.GetStateHistory(ctx, cfg, checkpoint.ListOptions{})
		if err != nil {
			log.Fatalf("GetStateHistory thread %s: %v", tc.thread, err)
		}
		fmt.Printf("  list: %d checkpoint(s)\n", len(history))
	}
}
