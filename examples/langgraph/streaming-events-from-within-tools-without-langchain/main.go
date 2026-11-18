// Command streaming-events-from-within-tools-without-langchain is the Go port
// of the notebook of the same name. The Python version drives raw AsyncOpenAI
// from a graph node and dispatches custom events from inside a tool. This port
// replaces the provider with a deterministic fake model that emits token
// chunks through the callbacks manager installed by StreamMessages, and emits
// the tool's token stream as tagged custom payloads via StreamWriter.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func lastMessage(state map[string]any) (messages.Message, bool) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) == 0 {
		return messages.Message{}, false
	}
	return msgs[len(msgs)-1], true
}

// emitStreamChunk stands in for a raw provider's streamed delta: it sends one
// AI message chunk through the callbacks manager that StreamMessages installs
// in every node's context. The ID ties streamed chunks to the final message so
// the final message is not re-emitted as a duplicate.
func emitStreamChunk(rt runtime.Runtime, id, text string, calls []messages.ToolCall) {
	manager, ok := callbacks.ManagerFromContext(rt)
	if !ok || manager.Empty() {
		return
	}
	chunk := messages.AI(text)
	chunk.ID = id
	chunk.ToolCalls = calls
	if err := manager.Emit(rt, callbacks.Event{Kind: callbacks.EventChatModelStream, Chunk: chunk}); err != nil {
		log.Printf("emit stream chunk: %v", err)
	}
}

func callModel(rt runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if ok && last.Role == messages.RoleTool {
		for _, token := range []string{"Under the bed: ", "socks, ", "shoes ", "and dust bunnies."} {
			emitStreamChunk(rt, "m-final", token, nil)
		}
		answer := messages.AI("Under the bed: socks, shoes and dust bunnies.")
		answer.ID = "m-final"
		return map[string]any{"messages": []messages.Message{answer}}, nil
	}

	// First turn: the fake model streams a tool-call chunk and asks for
	// get_items(place=bedroom).
	call := messages.ToolCall{
		ID:   "call_1",
		Name: "get_items",
		Args: map[string]any{"place": "bedroom"},
	}
	emitStreamChunk(rt, "m-toolcall", "", []messages.ToolCall{call})
	request := messages.AI("")
	request.ID = "m-toolcall"
	request.ToolCalls = []messages.ToolCall{call}
	return map[string]any{"messages": []messages.Message{request}}, nil
}

func callTools(rt runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if !ok || len(last.ToolCalls) == 0 {
		return nil, fmt.Errorf("tools node: no tool call in state")
	}
	call := last.ToolCalls[0]
	place, _ := call.Args["place"].(string)
	items := itemsIn(place)
	writer := graphpkg.StreamWriterFromContext(rt)
	for _, token := range items {
		if writer != nil {
			// Python dispatches a custom event with a "tool_call" tag;
			// Go carries the same fields in a tagged custom payload.
			writer(map[string]any{
				"function_name":     "get_items",
				"arguments":         map[string]any{"place": place},
				"tool_output_token": token,
				"tags":              []string{"tool_call"},
			})
		}
	}
	result := messages.Tool(call.ID, strings.Join(items, ", "))
	result.Name = call.Name
	return map[string]any{"messages": []messages.Message{result}}, nil
}

func itemsIn(place string) []string {
	if strings.Contains(place, "bed") {
		return []string{"socks", "shoes", "dust bunnies"}
	}
	if strings.Contains(place, "shelf") {
		return []string{"books", "penciles", "pictures"}
	}
	return []string{"cat snacks"}
}

func buildGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", callModel)
	g.AddNode("tools", callTools)
	g.AddEdge(types.START, "model")
	g.AddConditionalEdges("model", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		last, ok := lastMessage(state)
		if ok && len(last.ToolCalls) > 0 {
			return graphpkg.To("tools"), nil
		}
		return graphpkg.To(types.END), nil
	})
	g.AddEdge("tools", "model")
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}
	return cg
}

func input() map[string]any {
	return map[string]any{
		"messages": []messages.Message{messages.Human("what's in the bedroom")},
	}
}

func main() {
	ctx := context.Background()
	cg := buildGraph()

	// Python: astream_events filters on_custom_event with the "tool_call" tag.
	// Go filters tagged custom payloads from the custom stream instead.
	fmt.Println("--- tool token stream (custom mode, tag=tool_call) ---")
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
		tags, _ := payload["tags"].([]string)
		tagged := false
		for _, tag := range tags {
			if tag == "tool_call" {
				tagged = true
			}
		}
		if !tagged {
			continue
		}
		fmt.Printf("Tool token %v\n", payload["tool_output_token"])
	}

	// The same fake model's text tokens surface in messages mode; the final
	// message is deduped against the streamed chunks by ID.
	fmt.Println("--- model token stream (messages mode) ---")
	for chunk, err := range cg.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamMessages},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		mc, ok := chunk.Payload.(graphpkg.MessageChunk)
		if !ok || mc.Message.Role != messages.RoleAI || mc.Message.Content == "" {
			continue
		}
		fmt.Printf("%s|", mc.Message.Content)
	}
	fmt.Println()
}
