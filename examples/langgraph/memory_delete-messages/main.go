// Command memory_delete-messages is the Go port of
// langgraph/examples/memory/delete-messages.ipynb. It demonstrates both ways
// of deleting conversation messages from a MemorySaver thread: an explicit
// UpdateState call with a RemoveMessage sentinel, and a dedicated graph node
// that prunes old messages with RemoveMessage during a run.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

var idCounter int

func nextID(prefix string) string {
	idCounter++
	return fmt.Sprintf("%s-%d", prefix, idCounter)
}

// scriptedModel returns a queued response per call, standing in for the
// ChatAnthropic tool-calling model in the notebook.
type scriptedModel struct {
	responses []messages.Message
	idx       int
}

func (m *scriptedModel) invoke() messages.Message {
	if m.idx >= len(m.responses) {
		return messages.AI("No more scripted responses.")
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp
}

func aiToolCall(name, args string) messages.Message {
	msg := messages.AI("")
	msg.ID = nextID("ai")
	msg.ToolCalls = []messages.ToolCall{{
		ID:   nextID("call"),
		Name: name,
		Args: map[string]any{"query": args},
	}}
	return msg
}

func aiAnswer(text string) messages.Message {
	msg := messages.AI(text)
	msg.ID = nextID("ai")
	return msg
}

func asMessages(value any) []messages.Message {
	msgs, _ := value.([]messages.Message)
	return msgs
}

func agentNode(model *scriptedModel) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		response := model.invoke()
		// Ensure every tool call has a stable ID for the tool message.
		for i := range response.ToolCalls {
			if response.ToolCalls[i].ID == "" {
				response.ToolCalls[i].ID = nextID("call")
			}
		}
		if response.ID == "" {
			response.ID = nextID("ai")
		}
		return map[string]any{"messages": []messages.Message{response}}, nil
	}
}

// actionNode executes the queued tool calls as an in-process search tool.
func actionNode(_ runtime.Runtime, state map[string]any) (any, error) {
	last := asMessages(state["messages"])[len(asMessages(state["messages"]))-1]
	updates := []messages.Message{}
	for _, tc := range last.ToolCalls {
		result := messages.Tool(tc.ID, "It's sunny in San Francisco, but you better look out if you're a Gemini.")
		result.ID = nextID("tool")
		updates = append(updates, result)
	}
	return map[string]any{"messages": updates}, nil
}

// deleteMessagesNode removes all but the three most recent messages.
func deleteMessagesNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs := asMessages(state["messages"])
	if len(msgs) <= 3 {
		return map[string]any{}, nil
	}
	toDelete := []messages.RemoveMessage{}
	for _, m := range msgs[:len(msgs)-3] {
		toDelete = append(toDelete, messages.RemoveMessage{ID: m.ID})
	}
	return map[string]any{"messages": toDelete}, nil
}

func routeAfterAgent(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs := asMessages(state["messages"])
	last := msgs[len(msgs)-1]
	if len(last.ToolCalls) > 0 {
		return graphpkg.To("action"), nil
	}
	return graphpkg.To(types.END), nil
}

func routeAfterAgentWithDelete(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs := asMessages(state["messages"])
	last := msgs[len(msgs)-1]
	if len(last.ToolCalls) > 0 {
		return graphpkg.To("action"), nil
	}
	return graphpkg.To("delete_messages"), nil
}

func buildGraph(withDelete bool, model *scriptedModel) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("agent", agentNode(model))
	g.AddNode("action", actionNode)
	g.AddEdge(types.START, "agent")
	if withDelete {
		g.AddNode("delete_messages", deleteMessagesNode)
		g.AddConditionalEdges("agent", routeAfterAgentWithDelete)
		g.AddEdge("delete_messages", types.END)
	} else {
		g.AddConditionalEdges("agent", routeAfterAgent)
	}
	g.AddEdge("action", "agent")
	app, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile: %v", err)
	}
	return app
}

func humanMessage(text string) messages.Message {
	m := messages.Human(text)
	m.ID = nextID("human")
	return m
}

func printMessages(label string, values map[string]any) {
	fmt.Println(label)
	for _, m := range asMessages(values["messages"]) {
		kind := string(m.Role)
		if len(m.ToolCalls) > 0 {
			kind = "ai/tool_call"
		}
		fmt.Printf("  %-12s [%s] %s\n", kind, m.ID, messages.Text(m))
	}
	fmt.Println()
}

func main() {
	ctx := context.Background()

	// --- Part 1: delete a single message with UpdateState. ---
	model1 := &scriptedModel{responses: []messages.Message{
		aiToolCall("search", "weather in san francisco"),
		aiAnswer("Your name is Bob."),
		aiAnswer("Your name is Bob."),
	}}
	app := buildGraph(false, model1)
	thread := checkpoint.Config{ThreadID: "2"}
	for _, text := range []string{"hi! I'm bob", "what's my name?"} {
		if _, err := app.InvokeWithOptions(ctx,
			map[string]any{"messages": []messages.Message{humanMessage(text)}},
			graphpkg.Options{ThreadID: thread.ThreadID}); err != nil {
			log.Fatalf("invoke: %v", err)
		}
	}
	snap, err := app.GetState(ctx, thread)
	if err != nil {
		log.Fatalf("get state: %v", err)
	}
	before := asMessages(snap.Values["messages"])
	printMessages("--- messages before UpdateState (thread 2) ---", snap.Values)
	if len(before) > 0 {
		removed := []messages.MessageUpdate{messages.RemoveMessage{ID: before[0].ID}}
		cfg, err := app.UpdateState(ctx, thread, map[string]any{"messages": removed}, "agent")
		if err != nil {
			log.Fatalf("update state: %v", err)
		}
		fmt.Printf("UpdateState removed message %q; new checkpoint %q\n\n", before[0].ID, cfg.CheckpointID)
	}
	after, err := app.GetState(ctx, thread)
	if err != nil {
		log.Fatalf("get state after update: %v", err)
	}
	printMessages("--- messages after UpdateState (thread 2) ---", after.Values)

	// --- Part 2: a graph node prunes messages with RemoveMessage. ---
	model2 := &scriptedModel{responses: []messages.Message{
		aiToolCall("search", "weather in san francisco"),
		aiAnswer("Hello Bob!"),
		aiAnswer("Your name is Bob."),
	}}
	app2 := buildGraph(true, model2)
	thread2 := checkpoint.Config{ThreadID: "3"}
	for _, text := range []string{"hi! I'm bob", "what's my name?"} {
		if _, err := app2.InvokeWithOptions(ctx,
			map[string]any{"messages": []messages.Message{humanMessage(text)}},
			graphpkg.Options{ThreadID: thread2.ThreadID}); err != nil {
			log.Fatalf("invoke graph 2: %v", err)
		}
	}
	snap2, err := app2.GetState(ctx, thread2)
	if err != nil {
		log.Fatalf("get state graph 2: %v", err)
	}
	printMessages("--- final messages with delete_messages node (thread 3) ---", snap2.Values)
}
