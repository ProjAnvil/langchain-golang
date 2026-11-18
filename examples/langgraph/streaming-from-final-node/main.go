// Command streaming-from-final-node is the Go port of
// langgraph/examples/streaming-from-final-node.ipynb. A ReAct-style agent
// graph adds a dedicated "final" node whose model output is streamed. The
// example shows how to select only that node's tokens from a messages-mode
// stream, and the Go equivalent of the notebook's tag-based astream_events
// filter using a tagged custom payload.
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

var finalTokens = []string{"It's always sunny ", "in sf, ", "says Al Roker."}

func lastMessage(state map[string]any) (messages.Message, bool) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) == 0 {
		return messages.Message{}, false
	}
	return msgs[len(msgs)-1], true
}

func agentNode(_ runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if ok && last.Role == messages.RoleTool {
		return map[string]any{"messages": []messages.Message{
			messages.AI("It's always sunny in sf"),
		}}, nil
	}
	call := messages.AI("")
	call.ID = "agent-request"
	call.ToolCalls = []messages.ToolCall{{
		ID:   "call_1",
		Name: "get_weather",
		Args: map[string]any{"city": "sf"},
	}}
	return map[string]any{"messages": []messages.Message{call}}, nil
}

func toolNode(_ runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if !ok || len(last.ToolCalls) == 0 {
		return nil, fmt.Errorf("tools node: no tool call in state")
	}
	call := last.ToolCalls[0]
	result := messages.Tool(call.ID, "It's always sunny in sf")
	result.Name = call.Name
	return map[string]any{"messages": []messages.Message{result}}, nil
}

// finalNode is the notebook's final_model: it streams its rewrite token by
// token. The callback event carries the notebook's "final_node" tag (Go's
// messages mode does not surface tags, so the second run below demonstrates
// the equivalent filter on a tagged custom payload).
func finalNode(rt runtime.Runtime, _ map[string]any) (any, error) {
	manager, ok := callbacks.ManagerFromContext(rt)
	if ok && !manager.Empty() {
		for _, token := range finalTokens {
			chunk := messages.AI(token)
			chunk.ID = "final-answer"
			if err := manager.Emit(rt, callbacks.Event{
				Kind:  callbacks.EventChatModelStream,
				Chunk: chunk,
				Tags:  []string{"final_node"},
			}); err != nil {
				log.Printf("emit final token: %v", err)
			}
		}
	}
	if writer := graphpkg.StreamWriterFromContext(rt); writer != nil {
		for _, token := range finalTokens {
			writer(map[string]any{"content": token, "tags": []string{"final_node"}})
		}
	}
	answer := messages.AI(strings.Join(finalTokens, ""))
	answer.ID = "final-answer"
	return map[string]any{"messages": []messages.Message{answer}}, nil
}

func buildGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("agent", agentNode)
	g.AddNode("tools", toolNode)
	g.AddNode("final", finalNode)
	g.AddEdge(types.START, "agent")
	g.AddConditionalEdges("agent", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		last, ok := lastMessage(state)
		if ok && len(last.ToolCalls) > 0 {
			return graphpkg.To("tools"), nil
		}
		return graphpkg.To("final"), nil
	})
	g.AddEdge("tools", "agent")
	g.AddEdge("final", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}
	return cg
}

func input() map[string]any {
	return map[string]any{
		"messages": []messages.Message{messages.Human("what is the weather in sf")},
	}
}

func main() {
	ctx := context.Background()
	cg := buildGraph()

	// Notebook cell 14: messages mode, keep only non-empty chunks from the
	// final node (and never echo the human message).
	fmt.Println("--- messages mode, langgraph_node == final ---")
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
		if mc.Metadata["langgraph_node"] != "final" {
			continue
		}
		fmt.Printf("%s|", mc.Message.Content)
	}
	fmt.Println()

	// Notebook cell 17: filter on the "final_node" tag. Go's StreamMessages
	// metadata carries node/step/namespace but not model tags, so the same
	// idea is shown with a tagged custom payload from the final node.
	fmt.Println("--- custom mode, tag == final_node ---")
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
		for _, tag := range tags {
			if tag == "final_node" {
				fmt.Printf("%v|", payload["content"])
			}
		}
	}
	fmt.Println()
}
