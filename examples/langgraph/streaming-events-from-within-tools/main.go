// Command streaming-events-from-within-tools is the Go port of
// langgraph/examples/streaming-events-from-within-tools.ipynb. It shows three
// ways a tool can surface streamed data in Go: custom data written from the
// tool, message chunks emitted from inside the tool (filtered by the
// langgraph_node metadata), and a tool-only graph whose writer carries the
// token payload itself.
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

var items = []string{"books", "penciles", "pictures"}

func lastMessage(state map[string]any) (messages.Message, bool) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) == 0 {
		return messages.Message{}, false
	}
	return msgs[len(msgs)-1], true
}

// emitToolTokens models an LLM invoked from inside the tool: each token is
// sent through the callbacks manager that StreamMessages installed.
func emitToolTokens(rt runtime.Runtime, id string, tokens []string) {
	manager, ok := callbacks.ManagerFromContext(rt)
	if !ok || manager.Empty() {
		return
	}
	for _, token := range tokens {
		chunk := messages.AI(token)
		chunk.ID = id
		if err := manager.Emit(rt, callbacks.Event{Kind: callbacks.EventChatModelStream, Chunk: chunk}); err != nil {
			log.Printf("emit tool token: %v", err)
		}
	}
}

func agentNode(_ runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if ok && last.Role == messages.RoleTool {
		return map[string]any{"messages": []messages.Message{
			messages.AI("Here are the items: " + last.Content),
		}}, nil
	}
	call := messages.AI("")
	call.ID = "agent-request"
	call.ToolCalls = []messages.ToolCall{{
		ID:   "call_1",
		Name: "get_items",
		Args: map[string]any{"place": "office"},
	}}
	return map[string]any{"messages": []messages.Message{call}}, nil
}

func toolNode(rt runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if !ok || len(last.ToolCalls) == 0 {
		return nil, fmt.Errorf("tools node: no tool call in state")
	}
	call := last.ToolCalls[0]

	// Notebook cell 6: writer({"custom_tool_data": chunk}).
	writer := graphpkg.StreamWriterFromContext(rt)
	for _, chunk := range items {
		if writer != nil {
			writer(map[string]any{"custom_tool_data": chunk})
		}
	}
	// Notebook cell 10: a model called inside the tool streams its tokens;
	// those chunks carry the tools node's langgraph_node metadata.
	emitToolTokens(rt, "tool-inner-llm", items)

	result := messages.Tool(call.ID, strings.Join(items, ", "))
	result.Name = call.Name
	return map[string]any{"messages": []messages.Message{result}}, nil
}

// buildAgentGraph creates the agent/tools graph used by parts A and B.
func buildAgentGraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("agent", agentNode)
	g.AddNode("tools", toolNode)
	g.AddEdge(types.START, "agent")
	g.AddConditionalEdges("agent", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		last, ok := lastMessage(state)
		if ok && len(last.ToolCalls) > 0 {
			return graphpkg.To("tools"), nil
		}
		return graphpkg.To(types.END), nil
	})
	g.AddEdge("tools", "agent")
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile agent graph: %v", err)
	}
	return cg
}

func agentInput() map[string]any {
	return map[string]any{
		"messages": []messages.Message{messages.Human("what items are in the office?")},
	}
}

// callToolOnly is the notebook's cell-14 graph: one node that streams tokens
// from a tool with the StreamWriter and returns the tool result.
func callToolOnly(rt runtime.Runtime, state map[string]any) (any, error) {
	last, ok := lastMessage(state)
	if !ok || len(last.ToolCalls) == 0 {
		return nil, fmt.Errorf("call_tool: no tool call in state")
	}
	call := last.ToolCalls[0]
	place := "office"
	if p, ok := call.Args["place"].(string); ok {
		place = p
	}
	writer := graphpkg.StreamWriterFromContext(rt)
	response := ""
	for _, token := range itemsIn(place) {
		response += token
		if writer != nil {
			writer(map[string]any{"content": token})
		}
	}
	result := messages.Tool(call.ID, response)
	result.Name = call.Name
	return map[string]any{"messages": []messages.Message{result}}, nil
}

func itemsIn(place string) []string {
	if strings.Contains(place, "bed") {
		return []string{"socks", "shoes", "dust bunnies"}
	}
	return items
}

func main() {
	ctx := context.Background()
	cg := buildAgentGraph()

	// Notebook cell 8: stream_mode="custom" prints each tool-written chunk.
	fmt.Println("--- custom chunks from tool ---")
	for chunk, err := range cg.Stream(ctx, agentInput(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamCustom},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		fmt.Printf("%v\n", chunk.Payload)
	}

	// Notebook cell 11: stream_mode="messages", filtered to the tools node.
	fmt.Println("--- message chunks emitted from tools node ---")
	for chunk, err := range cg.Stream(ctx, agentInput(), graphpkg.StreamOptions{
		Modes: []graphpkg.StreamMode{graphpkg.StreamMessages},
	}) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		mc, ok := chunk.Payload.(graphpkg.MessageChunk)
		if !ok || mc.Message.Role != messages.RoleAI || mc.Message.Content == "" {
			continue
		}
		if mc.Metadata["langgraph_node"] != "tools" {
			continue
		}
		fmt.Printf("%s|", mc.Message.Content)
	}
	fmt.Println()

	// Notebook cell 16: a standalone tool node whose custom payloads carry
	// the token content directly.
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("call_tool", callToolOnly)
	g.AddEdge(types.START, "call_tool")
	g.AddEdge("call_tool", types.END)
	toolCG, err := g.Compile()
	if err != nil {
		log.Fatalf("compile tool graph: %v", err)
	}
	request := messages.AI("")
	request.ToolCalls = []messages.ToolCall{{
		ID:   "1",
		Name: "get_items",
		Args: map[string]any{"place": "bedroom"},
	}}
	fmt.Println("--- custom token payloads from tool-only graph ---")
	for chunk, err := range toolCG.Stream(ctx, map[string]any{
		"messages": []messages.Message{request},
	}, graphpkg.StreamOptions{Modes: []graphpkg.StreamMode{graphpkg.StreamCustom}}) {
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
}
