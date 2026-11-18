// Command streaming-tokens-without-langchain is the Go port of
// langgraph/examples/streaming-tokens-without-langchain.ipynb. The notebook
// calls raw AsyncOpenAI from a node, wraps deltas in ChatGenerationChunk so
// they show up in messages mode, and accumulates streamed tool calls. This
// port replaces the provider with a deterministic fake ChatModel that emits
// message chunks through the installed callbacks manager.
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

func emitChunk(rt runtime.Runtime, id, text string, calls []messages.ToolCall) {
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
		for _, token := range []string{"Under the bed: ", "socks, ", "shoes and ", "dust bunnies."} {
			emitChunk(rt, "m-final", token, nil)
		}
		answer := messages.AI("Under the bed: socks, shoes and dust bunnies.")
		answer.ID = "m-final"
		return map[string]any{"messages": []messages.Message{answer}}, nil
	}
	call := messages.ToolCall{
		ID:   "call_1",
		Name: "get_items",
		Args: map[string]any{"place": "bedroom"},
	}
	// The notebook wraps a tool-call delta in an empty-content chunk.
	emitChunk(rt, "m-toolcall", "", []messages.ToolCall{call})
	request := messages.AI("")
	request.ID = "m-toolcall"
	request.ToolCalls = []messages.ToolCall{call}
	return map[string]any{"messages": []messages.Message{request}}, nil
}

func getItems(place string) string {
	if strings.Contains(place, "bed") {
		return "socks, shoes and dust bunnies"
	}
	if strings.Contains(place, "shelf") {
		return "books, penciles and pictures"
	}
	return "cat snacks"
}

func callTools(rt runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if !ok || len(last.ToolCalls) == 0 {
		return nil, fmt.Errorf("call_tools: no tool call in state")
	}
	call := last.ToolCalls[0]
	place, _ := call.Args["place"].(string)
	answer := getItems(place)
	if writer := graphpkg.StreamWriterFromContext(rt); writer != nil {
		for _, token := range strings.Split(answer, ", ") {
			writer(map[string]any{
				"function_name":     call.Name,
				"arguments":         map[string]any{"place": place},
				"tool_output_token": token,
				"tags":              []string{"tool_call"},
			})
		}
	}
	result := messages.Tool(call.ID, answer)
	result.Name = call.Name
	return map[string]any{"messages": []messages.Message{result}}, nil
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

	// Notebook cell 15: print streamed text and gather streamed tool calls.
	fmt.Println("--- messages mode: text tokens and gathered tool calls ---")
	var gathered []messages.ToolCall
	for chunk, err := range cg.Stream(ctx, input(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamMessages},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		mc, ok := chunk.Payload.(graphpkg.MessageChunk)
		if !ok || mc.Message.Role != messages.RoleAI {
			continue
		}
		if mc.Message.Content != "" {
			fmt.Printf("%s|", mc.Message.Content)
		}
		if len(mc.Message.ToolCalls) > 0 {
			gathered = append(gathered, mc.Message.ToolCalls...)
			fmt.Printf("\ngathered tool calls: %v\n", gathered)
		}
	}
	fmt.Println()

	// The tool's own token stream is available as custom chunks tagged
	// "tool_call" (the notebook dispatches these as custom events).
	fmt.Println("--- custom mode: tagged tool tokens ---")
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
		fmt.Printf("%v|", payload["tool_output_token"])
	}
	fmt.Println()
}
