// Command streaming-tokens is the Go port of
// langgraph/examples/streaming-tokens.ipynb. Two fake models (a "joke" model
// and a "poem" model) stream their text token by token from a single graph
// node. The example reads all tokens in messages mode, inspects chunk
// metadata, and then reads writer-tagged tokens in custom mode, filtering by
// the "poem" tag as the notebook does.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

var jokeTokens = []string{
	"Why did ", "the cat ", "sit on ", "the mat? ",
	"Because it ", "wanted to ", "be purr-fect.",
}
var poemTokens = []string{
	"Cats are ", "quiet in the ", "morning light, ",
	"chasing ", "shadows ", "soft and ", "bright.",
}

// emitModelTokens stands in for a tagged model's stream: each token becomes
// an AI message chunk via the manager installed by StreamMessages.
func emitModelTokens(rt runtime.Runtime, id string, tokens []string) {
	manager, ok := callbacks.ManagerFromContext(rt)
	if !ok || manager.Empty() {
		return
	}
	for _, token := range tokens {
		chunk := messages.AI(token)
		chunk.ID = id
		if err := manager.Emit(rt, callbacks.Event{Kind: callbacks.EventChatModelStream, Chunk: chunk}); err != nil {
			log.Printf("emit token: %v", err)
		}
	}
}

func callModel(rt runtime.Runtime, state map[string]any) (any, error) {
	topic, _ := state["topic"].(string)
	fmt.Println("Writing joke...")
	emitModelTokens(rt, "joke-1", jokeTokens)
	fmt.Println("\n\nWriting poem...")
	emitModelTokens(rt, "poem-1", poemTokens)

	// The notebook's second half writes (chunk, metadata) tuples carrying
	// tags. Go uses a tagged custom payload instead; model tags are not part
	// of MessageChunk.Metadata.
	if writer := graphpkg.StreamWriterFromContext(rt); writer != nil {
		for _, token := range jokeTokens {
			writer(map[string]any{"content": token, "tags": []string{"joke"}})
		}
		for _, token := range poemTokens {
			writer(map[string]any{"content": token, "tags": []string{"poem"}})
		}
	}

	return map[string]any{
		"joke":  strings.Join(jokeTokens, ""),
		"poem":  strings.Join(poemTokens, ""),
		"topic": topic,
	}, nil
}

func buildGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddNode("call_model", callModel)
	g.AddEdge(types.START, "call_model")
	g.AddEdge("call_model", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}
	return cg
}

func input() map[string]any {
	return map[string]any{"topic": "cats"}
}

func hasTag(payload map[string]any, want string) bool {
	tags, _ := payload["tags"].([]string)
	for _, tag := range tags {
		if tag == want {
			return true
		}
	}
	return false
}

func main() {
	ctx := context.Background()
	cg := buildGraph()

	// Notebook cell 11: stream_mode="messages", print all token content.
	fmt.Println("--- messages mode: all tokens ---")
	for chunk, err := range cg.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamMessages},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		mc, ok := chunk.Payload.(graphpkg.MessageChunk)
		if !ok || mc.Message.Content == "" {
			continue
		}
		fmt.Printf("%s|", mc.Message.Content)
	}
	fmt.Println()

	// Notebook cell 12: inspect metadata. Python carries model tags there;
	// Go carries langgraph_node / langgraph_step / langgraph_checkpoint_ns.
	fmt.Println("--- first chunk metadata ---")
	for chunk, err := range cg.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamMessages},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		mc, ok := chunk.Payload.(graphpkg.MessageChunk)
		if ok && mc.Message.Content != "" {
			fmt.Printf("%v\n", mc.Metadata)
			break
		}
	}

	// Notebook cell 19: stream_mode="custom", print all content.
	fmt.Println("--- custom mode: all tokens ---")
	for chunk, err := range cg.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamCustom},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		payload, ok := chunk.Payload.(map[string]any)
		if !ok {
			continue
		}
		fmt.Printf("%v|", payload["content"])
	}
	fmt.Println()

	// Notebook cell 22: keep only the tokens tagged "poem".
	fmt.Println("--- custom mode: tag == poem ---")
	for chunk, err := range cg.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamCustom},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		payload, ok := chunk.Payload.(map[string]any)
		if !ok || !hasTag(payload, "poem") {
			continue
		}
		fmt.Printf("%v|", payload["content"])
	}
	fmt.Println()
}
