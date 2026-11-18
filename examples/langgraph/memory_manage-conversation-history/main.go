// Command memory_manage-conversation-history is the Go port of
// langgraph/examples/memory/manage-conversation-history.ipynb. It shows how
// the messages passed to a model can be filtered inside an agent node to
// control what the model remembers, while the graph still persists the full
// transcript in a MemorySaver checkpointer.
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

type scriptedModel struct {
	responses []messages.Message
	idx       int
}

func (m *scriptedModel) invoke(_ []messages.Message) messages.Message {
	if m.idx >= len(m.responses) {
		return messages.AI("No more scripted responses.")
	}
	resp := m.responses[m.idx]
	m.idx++
	if resp.ID == "" {
		resp.ID = nextID("ai")
	}
	for i := range resp.ToolCalls {
		if resp.ToolCalls[i].ID == "" {
			resp.ToolCalls[i].ID = nextID("call")
		}
	}
	return resp
}

func aiToolCall() messages.Message {
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{{
		ID:   nextID("call"),
		Name: "search",
		Args: map[string]any{"query": "weather in san francisco"},
	}}
	return msg
}

func asMessages(value any) []messages.Message {
	msgs, _ := value.([]messages.Message)
	return msgs
}

// filterMessages mirrors the notebook helper: only the last message is shown
// to the model, so earlier turns are not recalled.
func filterMessages(msgs []messages.Message) []messages.Message {
	if len(msgs) == 0 {
		return msgs
	}
	return msgs[len(msgs)-1:]
}

// buildAgent creates the agent node. When filter is true the node applies
// filterMessages before invoking the scripted model.
func buildAgent(model *scriptedModel, filter bool) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs := asMessages(state["messages"])
		if filter {
			msgs = filterMessages(msgs)
		}
		response := model.invoke(msgs)
		return map[string]any{"messages": []messages.Message{response}}, nil
	}
}

func actionNode(_ runtime.Runtime, state map[string]any) (any, error) {
	all := asMessages(state["messages"])
	last := all[len(all)-1]
	updates := []messages.Message{}
	for _, tc := range last.ToolCalls {
		toolMsg := messages.Tool(tc.ID, "It's sunny in San Francisco, but you better look out if you're a Gemini.")
		toolMsg.ID = nextID("tool")
		updates = append(updates, toolMsg)
	}
	return map[string]any{"messages": updates}, nil
}

func routeAfterAgent(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs := asMessages(state["messages"])
	if len(msgs[len(msgs)-1].ToolCalls) > 0 {
		return graphpkg.To("action"), nil
	}
	return graphpkg.To(types.END), nil
}

func buildGraph(filter bool, model *scriptedModel) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("agent", buildAgent(model, filter))
	g.AddNode("action", actionNode)
	g.AddEdge(types.START, "agent")
	g.AddConditionalEdges("agent", routeAfterAgent)
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

func runThread(app *graphpkg.CompiledGraph, thread string) {
	ctx := context.Background()
	for _, text := range []string{"hi! I'm bob", "what's my name?"} {
		result, err := app.InvokeWithOptions(ctx,
			map[string]any{"messages": []messages.Message{humanMessage(text)}},
			graphpkg.Options{ThreadID: thread},
		)
		if err != nil {
			log.Fatalf("invoke: %v", err)
		}
		msgs := asMessages(result.Values["messages"])
		last := msgs[len(msgs)-1]
		fmt.Printf("  user: %q -> AI: %s\n", text, messages.Text(last))
	}
	snap, err := app.GetState(ctx, checkpoint.Config{ThreadID: thread})
	if err != nil {
		log.Fatalf("get state: %v", err)
	}
	fmt.Printf("  persisted messages: %d\n\n", len(asMessages(snap.Values["messages"])))
}

func main() {
	// Full history: the model sees all persisted messages and recalls the name.
	fullModel := &scriptedModel{responses: []messages.Message{
		aiToolCall(),
		messages.AI("Hello Bob!"),
		messages.AI("Your name is Bob."),
	}}
	fmt.Println("--- original graph: full message history ---")
	runThread(buildGraph(false, fullModel), "full")

	// Filtered history: the node passes only messages[-1:] to the model, so the
	// answer to "what's my name?" cannot use the earlier introduction.
	filteredModel := &scriptedModel{responses: []messages.Message{
		aiToolCall(),
		messages.AI("Hello!"),
		messages.AI("I don't know your name."),
	}}
	fmt.Println("--- modified graph: only the last message is passed to the model ---")
	runThread(buildGraph(true, filteredModel), "filtered")
}
