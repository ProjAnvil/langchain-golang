// Command streaming-content is the Go port of
// langgraph/examples/streaming-content.ipynb. A single node writes text chunks
// with the StreamWriter it obtains from the node context (Python's
// get_stream_writer / StreamWriter), and those chunks are consumed in
// "custom" mode, alone and together with "updates". A third run shows how to
// filter named custom payloads, the Go counterpart of the notebook's
// astream_events custom-event filter.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

var chunks = []string{
	"Four", "score", "and", "seven", "years", "ago",
	"our", "fathers", "...",
}

// buildGraph creates a graph whose only node streams chunks via its
// StreamWriter. named controls whether the payloads are bare strings or
// structured events carrying a name, matching the two notebook node variants.
func buildGraph(named bool) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		writer := graphpkg.StreamWriterFromContext(rt)
		if writer != nil {
			for _, chunk := range chunks {
				if named {
					writer(map[string]any{
						"name": "my_custom_event",
						"data": map[string]any{"chunk": chunk},
					})
				} else {
					writer(chunk)
				}
			}
		}
		return map[string]any{"messages": []messages.Message{
			messages.AI(strings.Join(chunks, " ")),
		}}, nil
	})
	g.AddEdge(types.START, "model")
	g.AddEdge("model", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}
	return cg
}

func input() map[string]any {
	return map[string]any{
		"messages": []messages.Message{messages.Human("What are you thinking about?")},
	}
}

func main() {
	ctx := context.Background()
	cg := buildGraph(false)

	// Python: async for chunk in app.astream(..., stream_mode="custom")
	fmt.Println("--- custom ---")
	for chunk, err := range cg.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamCustom},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		fmt.Printf("%v ", chunk.Payload)
	}
	fmt.Println()

	// Python: async for chunk in app.astream(..., stream_mode=["custom", "updates"])
	fmt.Println("--- custom + updates ---")
	for chunk, err := range cg.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamCustom, graphpkg.StreamUpdates},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		fmt.Printf("[%s] %v\n", chunk.Mode, chunk.Payload)
	}

	// The notebook filters on_custom_event / name == "my_custom_event" through
	// astream_events. Go has no graph-level astream_events; named custom
	// payloads are filtered directly on the custom stream instead.
	fmt.Println("--- named custom events filtered by name ---")
	named := buildGraph(true)
	for chunk, err := range named.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamCustom},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		event, ok := chunk.Payload.(map[string]any)
		if !ok || event["name"] != "my_custom_event" {
			continue
		}
		data, _ := event["data"].(map[string]any)
		fmt.Printf("%v|", data["chunk"])
	}
	fmt.Println()
}
