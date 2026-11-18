// Command web-navigation_web_voyager is the Go port of
// examples/web-navigation/web_voyager.ipynb.
//
// WebVoyager drives a Playwright browser through a StateGraph: an agent
// node predicts an action, a conditional router dispatches it to one of six
// tool nodes (Click, Type, Scroll, Wait, GoBack, Google), and the observation
// is written back to a scratchpad before the agent acts again. The loop ends
// when the agent predicts ANSWER.
//
// This port keeps that exact graph shape and the "Action:" decision protocol,
// but the browser is a deterministic in-memory string machine. NO browser is
// launched, NO page is rendered, and NO network request is made. Each action
// is a core/tools tool executed through langchain/tools.ToolNode.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	lctools "github.com/projanvil/langchain-golang/langchain/tools"
	checkpoint "github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// bbox is the offline stand-in for a Playwright screenshot annotation.
type bbox struct {
	Index int
	Text  string
	Type  string
}

var defaultBBoxes = []bbox{
	{0, "Search query", "input"},
	{1, "Search", "button"},
	{2, "WebVoyager paper", "link"},
}

// browserTool builds a single-input core tool over a tiny page-transform
// function. The tool input is "currentPage|arg1|arg2|..." so the ToolNode can
// execute it without any external state.
func browserTool(name, description string, transform func(page string, args []string) string) (coretools.Simple, *lctools.ToolNode) {
	tool, err := coretools.NewSimple(name, description, func(_ context.Context, input string) (coretools.Result, error) {
		parts := strings.Split(input, "|")
		page := parts[0]
		args := parts[1:]
		return coretools.Result{Content: transform(page, args)}, nil
	})
	if err != nil {
		log.Fatal(err)
	}
	node, err := lctools.NewToolNode([]lctools.Tool{tool})
	if err != nil {
		log.Fatal(err)
	}
	return tool, node
}

// buildActionNodes creates the six offline browser actions.
func buildActionNodes() map[string]*lctools.ToolNode {
	nodes := map[string]*lctools.ToolNode{}
	_, nodes["Google"] = browserTool("Google", "Navigate to the offline search results page.",
		func(_ string, _ []string) string { return "google results: WebVoyager paper; LangChain blog; XKCD" })
	_, nodes["Click"] = browserTool("Click", "Click a labelled bounding box.",
		func(page string, args []string) string {
			if len(args) == 0 {
				return page + " [invalid click]"
			}
			return page + " [clicked " + args[0] + "]"
		})
	_, nodes["Type"] = browserTool("Type", "Type text into a labelled element.",
		func(page string, args []string) string {
			text := ""
			if len(args) > 1 {
				text = args[1]
			}
			return page + " [typed " + text + "]"
		})
	_, nodes["Scroll"] = browserTool("Scroll", "Scroll the window or an element.",
		func(page string, args []string) string {
			direction := ""
			if len(args) > 1 {
				direction = args[1]
			}
			return page + " [scrolled " + direction + "]"
		})
	_, nodes["Wait"] = browserTool("Wait", "Wait for the page to settle.",
		func(page string, _ []string) string { return page + " [waited]" })
	_, nodes["GoBack"] = browserTool("GoBack", "Navigate back.",
		func(_ string, _ []string) string { return "previous page" })
	return nodes
}

func msgsOf(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

func intOf(state map[string]any, key string) int {
	v, _ := state[key].(int)
	return v
}

// nextPrediction is the offline replacement for the vision-language agent.
// It emits a fixed, useful action sequence and then a final answer.
func nextPrediction(step int, question string) messages.Message {
	switch step {
	case 0:
		return messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "act_1", Name: "Google", Args: map[string]any{},
		}}}
	case 1:
		return messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "act_2", Name: "Type", Args: map[string]any{"tool_input": "0|" + question},
		}}}
	case 2:
		return messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "act_3", Name: "Click", Args: map[string]any{"tool_input": "2"},
		}}}
	case 3:
		return messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "act_4", Name: "Scroll", Args: map[string]any{"tool_input": "WINDOW|down"},
		}}}
	default:
		return messages.AI("ANSWER: " + question + " is explained by the offline WebVoyager paper summary found on the simulated results page.")
	}
}

// formatBBoxes mirrors the notebook's format_descriptions helper.
func formatBBoxes(boxes []bbox) string {
	var b strings.Builder
	b.WriteString("\nValid Bounding Boxes:\n")
	for _, box := range boxes {
		fmt.Fprintf(&b, "%d (<%s/>): %q\n", box.Index, box.Type, box.Text)
	}
	return b.String()
}

func buildGraph() *graphpkg.CompiledGraph {
	actions := buildActionNodes()

	g := graphpkg.NewStateGraph()
	g.AddNode("agent", func(_ runtime.Runtime, state map[string]any) (any, error) {
		step := intOf(state, "steps")
		question, _ := state["input"].(string)
		prediction := nextPrediction(step, question)
		action := "ANSWER"
		args := ""
		if len(prediction.ToolCalls) > 0 {
			action = prediction.ToolCalls[0].Name
			if v, ok := prediction.ToolCalls[0].Args["tool_input"].(string); ok {
				args = v
			}
		}
		fmt.Printf("%d. %s: %s\n", step+1, action, args)
		fmt.Print(formatBBoxes(defaultBBoxes))
		return map[string]any{"prediction": prediction}, nil
	})

	// Each action node builds a tool call, runs it through its ToolNode and
	// writes the resulting page text back as the new observation.
	for name, node := range actions {
		actionName, actionNode := name, node
		g.AddNode(actionName, func(rt runtime.Runtime, state map[string]any) (any, error) {
			page, _ := state["page"].(string)
			prediction, _ := state["prediction"].(messages.Message)
			arg := ""
			if len(prediction.ToolCalls) > 0 {
				if v, ok := prediction.ToolCalls[0].Args["tool_input"].(string); ok {
					arg = v
				}
			}
			input := page
			if arg != "" {
				input += "|" + arg
			}
			call := messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
				ID: "tool_" + actionName, Name: actionName,
				Args: map[string]any{"tool_input": input},
			}}}
			results, err := actionNode.Invoke(rt, []messages.Message{call}, state)
			if err != nil {
				return nil, err
			}
			newPage := ""
			if len(results) > 0 {
				newPage = messages.Text(results[0])
			}
			observation := fmt.Sprintf("%s -> %s", actionName, newPage)
			fmt.Printf("   observation: %s\n", observation)
			updates := map[string]any{
				"page":        newPage,
				"observation": observation,
			}
			if actionName == "Google" {
				updates["bboxes"] = defaultBBoxes
			}
			return updates, nil
		})
		g.AddEdge(actionName, "update_scratchpad")
	}

	g.AddNode("update_scratchpad", func(_ runtime.Runtime, state map[string]any) (any, error) {
		scratchpad, _ := state["scratchpad"].(string)
		observation, _ := state["observation"].(string)
		step := intOf(state, "steps") + 1
		scratchpad += fmt.Sprintf("\n%d. %s", step, observation)
		return map[string]any{"scratchpad": scratchpad, "steps": step}, nil
	})

	g.AddEdge(types.START, "agent")
	g.AddEdge("update_scratchpad", "agent")
	g.AddConditionalEdges("agent", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		prediction, _ := state["prediction"].(messages.Message)
		if len(prediction.ToolCalls) == 0 {
			return graphpkg.To(types.END), nil
		}
		return graphpkg.To(prediction.ToolCalls[0].Name), nil
	})

	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile web voyager graph: %v", err)
	}
	return cg
}

func main() {
	ctx := context.Background()
	cg := buildGraph()
	question := "Could you explain the WebVoyager paper?"
	result, err := cg.InvokeWithOptions(ctx, map[string]any{
		"input":      question,
		"page":       "about:blank",
		"scratchpad": "",
		"steps":      0,
		"bboxes":     defaultBBoxes,
	}, graphpkg.Options{ThreadID: "web-voyager-thread"})
	if err != nil {
		log.Fatalf("invoke web voyager: %v", err)
	}
	prediction, _ := result.Values["prediction"].(messages.Message)
	fmt.Println("\n===== FINAL RESPONSE =====")
	fmt.Println(messages.Text(prediction))
	fmt.Printf("steps taken: %v\n", result.Values["steps"])
}
