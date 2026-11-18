// Command chatbots_information-gather-prompting is the Go port of
// langgraph/examples/chatbots/information-gather-prompting.ipynb. It builds a
// two-chain graph: an info chain asks the user for structured prompt
// instructions and emits a tool call, then a prompt-generation chain turns
// those instructions into a final prompt template.
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

const infoTemplate = `Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool.`

const promptSystem = "Based on the following requirements, write a good prompt template:\n\n{reqs}"

var idCounter int

func nextID(prefix string) string {
	idCounter++
	return fmt.Sprintf("%s-%d", prefix, idCounter)
}

// scriptedModel returns a queued response per Invoke call, standing in for
// ChatOpenAI: the first two calls ask for more information, the third emits
// the PromptInstructions tool call, the fourth writes the prompt template.
type scriptedModel struct {
	responses []messages.Message
	idx       int
	lastInput []messages.Message
}

func (m *scriptedModel) invoke(input []messages.Message) messages.Message {
	m.lastInput = append([]messages.Message(nil), input...)
	if m.idx >= len(m.responses) {
		return messages.AI("Done.")
	}
	resp := m.responses[m.idx]
	m.idx++
	if resp.ID == "" {
		resp.ID = nextID("ai")
	}
	return resp
}

func asMessages(value any) []messages.Message {
	msgs, _ := value.([]messages.Message)
	return msgs
}

func toolCallResponse() messages.Message {
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{{
		ID:   nextID("call"),
		Name: "PromptInstructions",
		Args: map[string]any{
			"objective":    "write a RAG answer prompt",
			"variables":    []string{"context", "question"},
			"constraints":  []string{"do not invent facts"},
			"requirements": []string{"cite the context", "be concise"},
		},
	}}
	return msg
}

// infoNode prepends the system template and calls the scripted model.
func infoNode(model *scriptedModel) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		system := messages.System(infoTemplate)
		system.ID = nextID("system")
		input := append([]messages.Message{system}, asMessages(state["messages"])...)
		response := model.invoke(input)
		return map[string]any{"messages": []messages.Message{response}}, nil
	}
}

// addToolNode acknowledges the tool call with a ToolMessage, mirroring the
// notebook's add_tool_message node.
func addToolNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs := asMessages(state["messages"])
	last := msgs[len(msgs)-1]
	callID := ""
	if len(last.ToolCalls) > 0 {
		callID = last.ToolCalls[0].ID
	}
	toolMsg := messages.Tool(callID, "Prompt generated!")
	toolMsg.ID = nextID("tool")
	return map[string]any{"messages": []messages.Message{toolMsg}}, nil
}

// promptNode builds messages after the tool call and writes the final prompt
// template, mirroring get_prompt_messages + prompt_gen_chain.
func promptNode(model *scriptedModel) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs := asMessages(state["messages"])
		var reqs any
		other := []messages.Message{}
		for _, m := range msgs {
			switch {
			case m.Role == messages.RoleAI && len(m.ToolCalls) > 0:
				reqs = m.ToolCalls[0].Args
			case m.Role == messages.RoleTool:
				continue
			case reqs != nil:
				other = append(other, m)
			}
		}
		system := messages.System(strings.ReplaceAll(promptSystem, "{reqs}", fmt.Sprintf("%v", reqs)))
		system.ID = nextID("system")
		input := append([]messages.Message{system}, other...)
		response := model.invoke(input)
		return map[string]any{"messages": []messages.Message{response}}, nil
	}
}

// getState routes after the info chain: a tool call goes to add_tool_message,
// a non-human final message ends the run, otherwise ask the user again.
func getState(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs := asMessages(state["messages"])
	last := msgs[len(msgs)-1]
	if last.Role == messages.RoleAI && len(last.ToolCalls) > 0 {
		return graphpkg.To("add_tool_message"), nil
	}
	if last.Role != messages.RoleHuman {
		return graphpkg.To(types.END), nil
	}
	return graphpkg.To("info"), nil
}

func humanMessage(text string) messages.Message {
	m := messages.Human(text)
	m.ID = nextID("human")
	return m
}

func main() {
	ctx := context.Background()

	model := &scriptedModel{responses: []messages.Message{
		messages.AI("What is the objective of the prompt, and what variables should it take?"),
		messages.AI("What constraints and requirements should the output follow?"),
		toolCallResponse(),
		messages.AI("Prompt template: Given {context}, answer {question}. Cite the context and be concise."),
	}}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("info", infoNode(model))
	g.AddNode("add_tool_message", addToolNode)
	g.AddNode("prompt", promptNode(model))
	g.AddEdge(types.START, "info")
	g.AddConditionalEdges("info", getState)
	g.AddEdge("add_tool_message", "prompt")
	g.AddEdge("prompt", types.END)
	app, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	thread := checkpoint.Config{ThreadID: "info-gather-demo"}
	// The notebook's cached user responses; the loop stops once the prompt
	// generation chain has produced its output.
	for _, text := range []string{"hi!", "rag prompt", "1 rag, 2 none, 3 no, 4 no", "red", "q"} {
		fmt.Printf("User: %s\n", text)
		if text == "q" || text == "Q" {
			fmt.Println("AI: Byebye")
			break
		}
		result, err := app.InvokeWithOptions(ctx,
			map[string]any{"messages": []messages.Message{humanMessage(text)}},
			graphpkg.Options{ThreadID: thread.ThreadID},
		)
		if err != nil {
			log.Fatalf("invoke: %v", err)
		}
		msgs := asMessages(result.Values["messages"])
		last := msgs[len(msgs)-1]
		if len(last.ToolCalls) > 0 {
			fmt.Printf("AI: [tool call %s]\n", last.ToolCalls[0].Name)
		} else {
			fmt.Printf("AI: %s\n", messages.Text(last))
		}
		if strings.HasPrefix(messages.Text(last), "Prompt template:") {
			fmt.Println("Done!")
			break
		}
	}

	snap, err := app.GetState(ctx, thread)
	if err != nil {
		log.Fatalf("get state: %v", err)
	}
	fmt.Println("\n--- final transcript ---")
	for _, m := range asMessages(snap.Values["messages"]) {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-6s [tool call %s]\n", m.Role, m.ToolCalls[0].Name)
			continue
		}
		fmt.Printf("  %-6s %s\n", m.Role, messages.Text(m))
	}
}
