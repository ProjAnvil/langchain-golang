// Command chatbot-simulation-evaluation_agent-simulation-evaluation is the Go
// port of langgraph/examples/chatbot-simulation-evaluation/
// agent-simulation-evaluation.ipynb. A scripted airline support chatbot and a
// scripted simulated customer exchange messages in a two-node StateGraph
// until the customer says FINISHED or the message cap is reached.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

var idCounter int

func nextID(prefix string) string {
	idCounter++
	return fmt.Sprintf("%s-%d", prefix, idCounter)
}

// scriptedModel returns a queued reply per call.
type scriptedModel struct {
	replies   []string
	idx       int
	lastInput []messages.Message
}

func (m *scriptedModel) reply(input []messages.Message) string {
	m.lastInput = append([]messages.Message(nil), input...)
	if m.idx >= len(m.replies) {
		return "FINISHED"
	}
	out := m.replies[m.idx]
	m.idx++
	return out
}

func asMessages(value any) []messages.Message {
	msgs, _ := value.([]messages.Message)
	return msgs
}

// chatBotNode stands in for my_chat_bot: it prepends a support-agent system
// message and answers from a deterministic queue.
func chatBotNode(model *scriptedModel) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs := asMessages(state["messages"])
		system := messages.System("You are a customer support agent for an airline.")
		system.ID = nextID("system")
		input := append([]messages.Message{system}, msgs...)
		answer := messages.AI(model.reply(input))
		answer.ID = nextID("ai")
		return map[string]any{"messages": []messages.Message{answer}}, nil
	}
}

// swapRoles mirrors the notebook helper: AI messages become human messages
// and every other message becomes an AI message for the simulated customer.
func swapRoles(msgs []messages.Message) []messages.Message {
	out := make([]messages.Message, 0, len(msgs))
	for _, m := range msgs {
		var swapped messages.Message
		if m.Role == messages.RoleAI {
			swapped = messages.Human(messages.Text(m))
		} else {
			swapped = messages.AI(messages.Text(m))
		}
		swapped.ID = nextID("swap")
		out = append(out, swapped)
	}
	return out
}

// simulatedUserNode swaps roles, calls the scripted customer model, then flips
// the AI reply back into a HumanMessage as the notebook does.
func simulatedUserNode(model *scriptedModel) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		swapped := swapRoles(asMessages(state["messages"]))
		reply := messages.Human(model.reply(swapped))
		reply.ID = nextID("human")
		return map[string]any{"messages": []messages.Message{reply}}, nil
	}
}

func shouldContinue(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs := asMessages(state["messages"])
	if len(msgs) > 6 || messages.Text(msgs[len(msgs)-1]) == "FINISHED" {
		return graphpkg.To(types.END), nil
	}
	return graphpkg.To("chat_bot"), nil
}

func main() {
	ctx := context.Background()

	chatBot := &scriptedModel{replies: []string{
		"Hello, welcome to airline support. How can I help?",
		"I'm sorry, refunds for trips more than five years ago are not eligible.",
		"I understand, but the policy does not allow a full refund in this case.",
	}}
	customer := &scriptedModel{replies: []string{
		"Hi, I want a refund for my trip to Alaska.",
		"I want ALL my money back. The trip was five years ago!",
		"FINISHED",
	}}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("chat_bot", chatBotNode(chatBot))
	g.AddNode("user", simulatedUserNode(customer))
	g.AddEdge(types.START, "chat_bot")
	g.AddEdge("chat_bot", "user")
	g.AddConditionalEdges("user", shouldContinue)
	simulation, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	fmt.Println("=== chat simulation (offline) ===")
	finalCount := 0
	for chunk, err := range simulation.Stream(ctx,
		map[string]any{"messages": []messages.Message{}},
		graphpkg.StreamOptions{Modes: []graphpkg.StreamMode{graphpkg.StreamValues}},
	) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		state, ok := chunk.Payload.(map[string]any)
		if !ok {
			continue
		}
		msgs := asMessages(state["messages"])
		if len(msgs) == 0 {
			continue
		}
		finalCount = len(msgs)
		last := msgs[len(msgs)-1]
		fmt.Printf("  %-6s: %s\n", last.Role, messages.Text(last))
	}
	fmt.Printf("\nfinal transcript length: %d messages\n", finalCount)
}
