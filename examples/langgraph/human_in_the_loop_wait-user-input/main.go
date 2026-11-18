// Command human_in_the_loop_wait-user-input is the Go port of
// langgraph/examples/human_in_the_loop/wait-user-input.ipynb.
//
// It demonstrates graphpkg.Interrupt -- the Go equivalent of langgraph's
// interrupt() -- in the notebook's two shapes:
//
//  1. A three-step graph whose human_feedback node pauses for feedback and
//     resumes with Options{Resume: ...}.
//  2. An agent/tool loop where the "AskHuman" tool pauses for a location,
//     then a tool message carrying the human's answer is fed back to the
//     agent so it can finish.
//
// A checkpoint.NewMemorySaver() makes each pause durable and resumable. No
// API key, network or service is required: the agent is deterministic.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func str(state map[string]any, key string) string {
	if value, ok := state[key].(string); ok {
		return value
	}
	return ""
}

func messagesOf(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

func printMessages(label string, state map[string]any) {
	fmt.Println(label)
	for _, m := range messagesOf(state) {
		fmt.Printf("  %-6s %s\n", m.Role, messages.Text(m))
	}
}

func appendMessage(state map[string]any, msg messages.Message) []messages.Message {
	msgs := append([]messages.Message(nil), messagesOf(state)...)
	return append(msgs, msg)
}

func compileOrDie(label string, g *graphpkg.StateGraph) *graphpkg.CompiledGraph {
	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile %s: %v", label, err)
	}
	return cg
}

func invokeOrDie(label string, cg *graphpkg.CompiledGraph, input map[string]any, opts graphpkg.Options) graphpkg.Result {
	res, err := cg.InvokeWithOptions(context.Background(), input, opts)
	if err != nil {
		log.Fatalf("%s: %v", label, err)
	}
	return res
}

func printInterrupts(res graphpkg.Result) {
	fmt.Printf("  interrupts=%d\n", len(res.Interrupts))
	for _, intr := range res.Interrupts {
		fmt.Printf("    value=%v id=%s ns=%q\n", intr.Value, intr.ID, intr.NS)
	}
}

func main() {
	ctx := context.Background()

	// --- Part 1: pause for free-form human feedback. ---
	feedback := graphpkg.NewStateGraph()
	feedback.AddNode("step_1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		fmt.Println("  ---Step 1---")
		return nil, nil
	})
	feedback.AddNode("human_feedback", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		fmt.Println("  ---human_feedback---")
		answer := graphpkg.Interrupt(rt, "Please provide feedback:")
		return map[string]any{"user_feedback": fmt.Sprint(answer)}, nil
	})
	feedback.AddNode("step_3", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		fmt.Println("  ---Step 3---")
		return nil, nil
	})
	feedback.AddEdge(types.START, "step_1")
	feedback.AddEdge("step_1", "human_feedback")
	feedback.AddEdge("human_feedback", "step_3")
	feedback.AddEdge("step_3", types.END)
	feedbackCG := compileOrDie("feedback graph", feedback)

	fmt.Println("=== Part 1: run until human_feedback interrupts ===")
	first := invokeOrDie("first invoke", feedbackCG,
		map[string]any{"input": "hello world"}, graphpkg.Options{ThreadID: "1"})
	printInterrupts(first)
	fmt.Printf("  input=%v user_feedback=%q\n", first.Values["input"], str(first.Values, "user_feedback"))

	fmt.Println("=== Part 1: resume with the human's feedback ===")
	done := invokeOrDie("resume invoke", feedbackCG, nil,
		graphpkg.Options{ThreadID: "1", Resume: "go to step 3!"})
	printInterrupts(done)
	fmt.Printf("  input=%v user_feedback=%q\n", done.Values["input"], str(done.Values, "user_feedback"))

	state, err := feedbackCG.GetState(ctx, checkpoint.Config{ThreadID: "1"})
	if err != nil {
		log.Fatalf("GetState thread 1: %v", err)
	}
	fmt.Printf("  get_state: user_feedback=%q next=%v\n", str(state.Values, "user_feedback"), state.Next)

	// --- Part 2: agent asks a human for a location, then looks up weather. ---
	agent := graphpkg.NewStateGraph()
	agent.AddNode("agent", func(_ runtime.Runtime, state map[string]any) (any, error) {
		if state["asked"] == true {
			return map[string]any{"messages": appendMessage(state,
				messages.AI("It's sunny in "+str(state, "location")+"."))}, nil
		}
		// Stands in for the model emitting an AskHuman tool call.
		return map[string]any{"messages": appendMessage(state,
			messages.AI("Where are you?"))}, nil
	})
	agent.AddNode("ask_human", func(rt runtime.Runtime, state map[string]any) (any, error) {
		// Stands in for AskHuman.model_validate(...).question.
		location := fmt.Sprint(graphpkg.Interrupt(rt, "Where are you?"))
		msgs := appendMessage(state, messages.Tool("", "location: "+location))
		return map[string]any{
			"messages": msgs,
			"asked":    true,
			"location": location,
		}, nil
	})
	agent.AddEdge(types.START, "agent")
	agent.AddConditionalEdgesWithPathMap("agent",
		func(_ runtime.Runtime, state map[string]any) ([]any, error) {
			if state["asked"] == true {
				return graphpkg.To(types.END), nil
			}
			return graphpkg.To("ask_human"), nil
		},
		map[string]string{"ask_human": "ask_human", types.END: types.END})
	agent.AddEdge("ask_human", "agent")
	agentCG := compileOrDie("agent graph", agent)

	fmt.Println("=== Part 2: agent loop pauses at AskHuman ===")
	askPaused := invokeOrDie("agent first invoke", agentCG, map[string]any{
		"messages": []messages.Message{messages.Human("Ask the user where they are, then look up the weather there")},
	}, graphpkg.Options{ThreadID: "2"})
	printInterrupts(askPaused)
	printMessages("  messages at pause:", askPaused.Values)

	fmt.Println("=== Part 2: resume with the location ===")
	askDone := invokeOrDie("agent resume invoke", agentCG, nil,
		graphpkg.Options{ThreadID: "2", Resume: "san francisco"})
	printInterrupts(askDone)
	printMessages("  final messages:", askDone.Values)
}
