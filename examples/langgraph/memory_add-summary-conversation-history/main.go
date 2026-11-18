// Command memory_add-summary-conversation-history is the Go port of
// langgraph/examples/memory/add-summary-conversation-history.ipynb.
// It builds a conversation graph that summarizes and prunes old messages once
// the thread grows past six messages, with all messages and the summary kept
// in a MemorySaver checkpointer.
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

var idCounter int

// nextID returns a deterministic message ID; the Go port assigns IDs itself
// because it uses the plain MessagesReducer channel rather than the SDK's
// delta channel (which stamps IDs automatically).
func nextID(prefix string) string {
	idCounter++
	return fmt.Sprintf("%s-%d", prefix, idCounter)
}

// summaryModel is a scripted stand-in for ChatAnthropic. It recognizes the
// summarization prompt by its leading text and otherwise answers from the
// latest human message.
type summaryModel struct{}

func (summaryModel) invoke(msgs []messages.Message) messages.Message {
	last := msgs[len(msgs)-1]
	text := messages.Text(last)
	switch {
	case strings.HasPrefix(text, "Create a summary"), strings.HasPrefix(text, "This is summary"):
		answer := messages.AI("Bob is a Celtics fan who also likes the Patriots.")
		answer.ID = nextID("summary")
		return answer
	case strings.Contains(strings.ToLower(text), "name"):
		answer := messages.AI("Your name is Bob.")
		answer.ID = nextID("ai")
		return answer
	default:
		answer := messages.AI("Got it: " + text)
		answer.ID = nextID("ai")
		return answer
	}
}

func asMessages(value any) []messages.Message {
	msgs, _ := value.([]messages.Message)
	return msgs
}

// conversationNode prepends a summary-derived system message when a summary
// exists, then calls the scripted model and appends its reply.
func conversationNode(_ runtime.Runtime, state map[string]any) (any, error) {
	conversation := asMessages(state["messages"])
	summary, _ := state["summary"].(string)
	modelInput := conversation
	if summary != "" {
		system := messages.System("Summary of conversation earlier: " + summary)
		system.ID = nextID("system")
		modelInput = append([]messages.Message{system}, conversation...)
	}
	response := summaryModel{}.invoke(modelInput)
	return map[string]any{"messages": []messages.Message{response}}, nil
}

// summarizeNode writes a new summary and removes every message except the two
// most recent, mirroring the notebook's RemoveMessage list.
func summarizeNode(_ runtime.Runtime, state map[string]any) (any, error) {
	conversation := asMessages(state["messages"])
	summary, _ := state["summary"].(string)
	prompt := "Create a summary of the conversation above:"
	if summary != "" {
		prompt = "This is summary of the conversation to date: " + summary +
			"\n\nExtend the summary by taking into account the new messages above:"
	}
	request := messages.Human(prompt)
	request.ID = nextID("summary-request")
	modelInput := append(append([]messages.Message{}, conversation...), request)
	response := summaryModel{}.invoke(modelInput)

	toDelete := []messages.RemoveMessage{}
	if len(conversation) > 2 {
		for _, m := range conversation[:len(conversation)-2] {
			toDelete = append(toDelete, messages.RemoveMessage{ID: m.ID})
		}
	}
	return map[string]any{
		"summary":  messages.Text(response),
		"messages": toDelete,
	}, nil
}

func shouldContinue(_ runtime.Runtime, state map[string]any) ([]any, error) {
	if len(asMessages(state["messages"])) > 6 {
		return graphpkg.To("summarize_conversation"), nil
	}
	return graphpkg.To(types.END), nil
}

func printState(label string, values map[string]any) {
	fmt.Println(label)
	for _, m := range asMessages(values["messages"]) {
		fmt.Printf("  %-6s [%s] %s\n", m.Role, m.ID, messages.Text(m))
	}
	if summary, ok := values["summary"].(string); ok && summary != "" {
		fmt.Printf("  summary: %s\n", summary)
	}
	fmt.Println()
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("conversation", conversationNode)
	g.AddNode("summarize_conversation", summarizeNode)
	g.AddEdge(types.START, "conversation")
	g.AddConditionalEdges("conversation", shouldContinue)
	g.AddEdge("summarize_conversation", types.END)
	app, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	cfg := checkpoint.Config{ThreadID: "4"}
	turns := []string{
		"hi! I'm bob",
		"what's my name?",
		"i like the celtics!",
		"i like how much they win",
		"what's my name?",
		"what NFL team do you think I like?",
		"i like the patriots!",
	}
	for i, text := range turns {
		human := messages.Human(text)
		human.ID = nextID("human")
		result, err := app.InvokeWithOptions(ctx,
			map[string]any{"messages": []messages.Message{human}},
			graphpkg.Options{ThreadID: cfg.ThreadID},
		)
		if err != nil {
			log.Fatalf("turn %d: %v", i+1, err)
		}
		// Show only the most recent exchange, as stream_mode="updates" did
		// in the notebook.
		fmt.Printf("turn %d: user=%q\n", i+1, text)
		msgs := asMessages(result.Values["messages"])
		if len(msgs) > 0 {
			last := msgs[len(msgs)-1]
			fmt.Printf("  latest: %-6s %s\n", last.Role, messages.Text(last))
		}
		if summary, ok := result.Values["summary"].(string); ok && summary != "" {
			fmt.Printf("  summary: %s\n", summary)
		}
	}

	snap, err := app.GetState(ctx, cfg)
	if err != nil {
		log.Fatalf("get state: %v", err)
	}
	printState("--- final state of thread 4 (after summarization/pruning) ---", snap.Values)
}
