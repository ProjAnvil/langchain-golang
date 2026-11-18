// Command customer-support_customer-support is the Go port of the advanced
// (part 4) graph in langgraph/examples/customer-support/customer-support.ipynb.
// It reproduces the multi-assistant routing pattern: a primary assistant
// delegates to four specialized assistants, each with safe and sensitive
// tools, a dialog-state stack tracks the active skill, and sensitive tool
// nodes are gated by an interrupt_before boundary.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
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

func asMessages(value any) []messages.Message {
	msgs, _ := value.([]messages.Message)
	return msgs
}

// ---------------------------------------------------------------------------
// Scripted model (offline stand-in for ChatAnthropic)
// ---------------------------------------------------------------------------

type scriptedModel struct {
	responses []messages.Message
	idx       int
}

func newModel(responses ...messages.Message) *scriptedModel {
	return &scriptedModel{responses: responses}
}

func (m *scriptedModel) invoke() messages.Message {
	if m.idx >= len(m.responses) {
		return messages.AI("I cannot help with that.")
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

func toolCall(name string) messages.Message {
	msg := messages.AI("")
	msg.ToolCalls = []messages.ToolCall{{
		ID:   nextID("call"),
		Name: name,
		Args: map[string]any{"tool_input": "demo"},
	}}
	return msg
}

// ---------------------------------------------------------------------------
// Deterministic in-memory tools (offline stand-ins for the notebook's tools)
// ---------------------------------------------------------------------------

func simpleTool(name, description, result string) coretools.Tool {
	t, err := coretools.NewSimple(name, description, func(_ context.Context, _ string) (coretools.Result, error) {
		return coretools.Result{Content: result}, nil
	})
	if err != nil {
		log.Fatalf("build tool %s: %v", name, err)
	}
	return t
}

func buildTools() (map[string]coretools.Tool, map[string][]string) {
	registry := map[string]coretools.Tool{}
	add := func(name, description, result string) {
		registry[name] = simpleTool(name, description, result)
	}
	add("search_flights", "Search flights", "FL123 SFB->JFK 9am")
	add("update_ticket_to_new_flight", "Update a ticket", "Ticket updated to FL456")
	add("cancel_ticket", "Cancel a ticket", "Ticket cancelled")
	add("search_car_rentals", "Search car rentals", "Rental car available: compact")
	add("book_car_rental", "Book a car rental", "Car rental booked")
	add("search_hotels", "Search hotels", "Hotel available: Swiss Inn")
	add("book_hotel", "Book a hotel", "Hotel booked")
	add("search_trip_recommendations", "Search excursions", "Excursion available: city tour")
	add("book_excursion", "Book an excursion", "Excursion booked")
	add("CompleteOrEscalate", "Return control to the primary assistant", "Dialog escalated")
	add("ToFlightBookingAssistant", "Delegate to flight assistant", "Delegating to flight assistant")
	add("ToBookCarRental", "Delegate to car rental assistant", "Delegating to car rental assistant")
	add("ToHotelBookingAssistant", "Delegate to hotel assistant", "Delegating to hotel assistant")
	add("ToBookExcursion", "Delegate to excursion assistant", "Delegating to excursion assistant")

	safe := map[string][]string{
		"update_flight":   {"search_flights"},
		"book_car_rental": {"search_car_rentals"},
		"book_hotel":      {"search_hotels"},
		"book_excursion":  {"search_trip_recommendations"},
	}
	return registry, safe
}

// ---------------------------------------------------------------------------
// State reducer for the dialog stack: push a skill name or pop with "pop"
// ---------------------------------------------------------------------------

func dialogStackReducer(existing any, update any) (any, error) {
	stack := []string{}
	switch v := existing.(type) {
	case []string:
		stack = append(stack, v...)
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				stack = append(stack, s)
			}
		}
	}
	switch v := update.(type) {
	case string:
		if v == "pop" {
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
		} else if v != "" {
			stack = append(stack, v)
		}
	}
	return stack, nil
}

func currentSkill(state map[string]any) string {
	switch v := state["dialog_state"].(type) {
	case []string:
		if len(v) > 0 {
			return v[len(v)-1]
		}
	case []any:
		if len(v) > 0 {
			if s, ok := v[len(v)-1].(string); ok {
				return s
			}
		}
	}
	return ""
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

func assistantNode(model *scriptedModel) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		_ = state
		return map[string]any{"messages": []messages.Message{model.invoke()}}, nil
	}
}

func makeToolNode(registry map[string]coretools.Tool) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs := asMessages(state["messages"])
		last := msgs[len(msgs)-1]
		updates := []messages.Message{}
		for _, tc := range last.ToolCalls {
			content := "Unknown tool: " + tc.Name
			if tool, ok := registry[tc.Name]; ok {
				res, err := tool.Invoke(context.Background(), tc.Args)
				if err != nil {
					content = fmt.Sprintf("Error: %v. Please fix your mistakes.", err)
				} else {
					content = res.Content
				}
			}
			toolMsg := messages.Tool(tc.ID, content)
			toolMsg.ID = nextID("tool")
			updates = append(updates, toolMsg)
		}
		return map[string]any{"messages": updates}, nil
	}
}

func entryNode(assistantName, skill string) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs := asMessages(state["messages"])
		callID := ""
		if len(msgs) > 0 && len(msgs[len(msgs)-1].ToolCalls) > 0 {
			callID = msgs[len(msgs)-1].ToolCalls[0].ID
		}
		toolMsg := messages.Tool(callID,
			"The assistant is now the "+assistantName+". Use the provided tools to assist the user; "+
				"call CompleteOrEscalate to return to the primary assistant.")
		toolMsg.ID = nextID("tool")
		return map[string]any{
			"messages":     []messages.Message{toolMsg},
			"dialog_state": skill,
		}, nil
	}
}

func fetchUserInfoNode(_ runtime.Runtime, state map[string]any) (any, error) {
	_ = state
	return map[string]any{"user_info": "passenger 342 (offline stub)"}, nil
}

// leaveSkillNode pops the dialog stack and answers the escalation tool call.
func leaveSkillNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs := asMessages(state["messages"])
	updates := []messages.Message{}
	if len(msgs) > 0 && len(msgs[len(msgs)-1].ToolCalls) > 0 {
		toolMsg := messages.Tool(msgs[len(msgs)-1].ToolCalls[0].ID,
			"Resuming dialog with the host assistant.")
		toolMsg.ID = nextID("tool")
		updates = append(updates, toolMsg)
	}
	return map[string]any{
		"dialog_state": "pop",
		"messages":     updates,
	}, nil
}

// ---------------------------------------------------------------------------
// Conditional edges
// ---------------------------------------------------------------------------

func routeToWorkflow(_ runtime.Runtime, state map[string]any) ([]any, error) {
	if skill := currentSkill(state); skill != "" {
		return graphpkg.To(skill), nil
	}
	return graphpkg.To("primary_assistant"), nil
}

var transferTargets = map[string]string{
	"ToFlightBookingAssistant": "enter_update_flight",
	"ToBookCarRental":          "enter_book_car_rental",
	"ToHotelBookingAssistant":  "enter_book_hotel",
	"ToBookExcursion":          "enter_book_excursion",
}

func routePrimary(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs := asMessages(state["messages"])
	last := msgs[len(msgs)-1]
	if len(last.ToolCalls) == 0 {
		return graphpkg.To(types.END), nil
	}
	if target, ok := transferTargets[last.ToolCalls[0].Name]; ok {
		return graphpkg.To(target), nil
	}
	return graphpkg.To("primary_assistant_tools"), nil
}

// routeSkill mirrors route_update_flight and its siblings: no tool calls end
// the run, CompleteOrEscalate leaves the skill, otherwise the tool calls are
// split between safe and sensitive tool nodes.
func routeSkill(skill string, safeNames []string) graphpkg.ConditionalEdge {
	return func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs := asMessages(state["messages"])
		last := msgs[len(msgs)-1]
		if len(last.ToolCalls) == 0 {
			return graphpkg.To(types.END), nil
		}
		for _, tc := range last.ToolCalls {
			if tc.Name == "CompleteOrEscalate" {
				return graphpkg.To("leave_skill"), nil
			}
		}
		for _, tc := range last.ToolCalls {
			found := false
			for _, safe := range safeNames {
				if tc.Name == safe {
					found = true
					break
				}
			}
			if !found {
				return graphpkg.To(skill + "_sensitive_tools"), nil
			}
		}
		return graphpkg.To(skill + "_safe_tools"), nil
	}
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

type models struct {
	primary *scriptedModel
	skills  map[string]*scriptedModel
}

func buildGraph(ms models) *graphpkg.CompiledGraph {
	registry, safe := buildTools()
	toolNode := makeToolNode(registry)

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddReducer("dialog_state", dialogStackReducer)

	g.AddNode("fetch_user_info", fetchUserInfoNode)
	g.AddNode("primary_assistant", assistantNode(ms.primary))
	g.AddNode("primary_assistant_tools", toolNode)
	g.AddNode("leave_skill", leaveSkillNode)

	entryNames := map[string]string{
		"update_flight":   "Flight Updates & Booking Assistant",
		"book_car_rental": "Car Rental Assistant",
		"book_hotel":      "Hotel Booking Assistant",
		"book_excursion":  "Excursion Booking Assistant",
	}
	entryNodes := map[string]string{
		"update_flight":   "enter_update_flight",
		"book_car_rental": "enter_book_car_rental",
		"book_hotel":      "enter_book_hotel",
		"book_excursion":  "enter_book_excursion",
	}
	for skill, model := range ms.skills {
		entry := entryNodes[skill]
		g.AddNode(entry, entryNode(entryNames[skill], skill))
		g.AddNode(skill, assistantNode(model))
		g.AddNode(skill+"_safe_tools", toolNode)
		g.AddNode(skill+"_sensitive_tools", toolNode)
		g.AddEdge(entry, skill)
		g.AddEdge(skill+"_safe_tools", skill)
		g.AddEdge(skill+"_sensitive_tools", skill)
		g.AddConditionalEdges(skill, routeSkill(skill, safe[skill]))
	}
	g.AddEdge("leave_skill", "primary_assistant")
	g.AddEdge("primary_assistant_tools", "primary_assistant")
	g.AddEdge(types.START, "fetch_user_info")
	g.AddConditionalEdges("fetch_user_info", routeToWorkflow)
	g.AddConditionalEdges("primary_assistant", routePrimary)

	app, err := g.Compile(
		graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()),
		graphpkg.WithInterruptBefore(
			"update_flight_sensitive_tools",
			"book_car_rental_sensitive_tools",
			"book_hotel_sensitive_tools",
			"book_excursion_sensitive_tools",
		),
	)
	if err != nil {
		log.Fatalf("compile: %v", err)
	}
	return app
}

func defaultModels() models {
	return models{
		primary: newModel(toolCall("ToFlightBookingAssistant"), messages.AI("Done.")),
		skills: map[string]*scriptedModel{
			"update_flight":   newModel(toolCall("search_flights"), toolCall("update_ticket_to_new_flight"), toolCall("CompleteOrEscalate")),
			"book_car_rental": newModel(toolCall("search_car_rentals"), toolCall("CompleteOrEscalate")),
			"book_hotel":      newModel(toolCall("search_hotels"), toolCall("CompleteOrEscalate")),
			"book_excursion":  newModel(toolCall("search_trip_recommendations"), toolCall("CompleteOrEscalate")),
		},
	}
}

func printMessages(label string, values map[string]any) {
	fmt.Println(label)
	fmt.Printf("  dialog_state: %v\n", values["dialog_state"])
	for _, m := range asMessages(values["messages"]) {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-6s [tool call %s]\n", m.Role, m.ToolCalls[0].Name)
			continue
		}
		text := messages.Text(m)
		if len(text) > 110 {
			text = text[:110] + "..."
		}
		fmt.Printf("  %-6s %s\n", m.Role, text)
	}
	fmt.Println()
}

func runFlightScenario() {
	fmt.Println("=== Scenario 1: flight delegation with sensitive-tool approval ===")
	app := buildGraph(defaultModels())
	ctx := context.Background()
	thread := checkpoint.Config{ThreadID: "customer-1"}

	human := messages.Human("Update my flight")
	human.ID = nextID("human")
	first, err := app.InvokeWithOptions(ctx,
		map[string]any{"messages": []messages.Message{human}},
		graphpkg.Options{ThreadID: thread.ThreadID},
	)
	if err != nil {
		log.Fatalf("first invoke: %v", err)
	}
	fmt.Printf("paused with %d interrupt(s) before the sensitive tool\n", len(first.Interrupts))
	printMessages("--- state at pause ---", first.Values)

	// Approve the pending sensitive tool by resuming the same thread with nil
	// input (the Go equivalent of Python's invoke(None, config)).
	resumed, err := app.InvokeWithOptions(ctx, nil,
		graphpkg.Options{ThreadID: thread.ThreadID},
	)
	if err != nil {
		log.Fatalf("resume: %v", err)
	}
	fmt.Printf("after approval: %d interrupt(s)\n", len(resumed.Interrupts))
	printMessages("--- final state ---", resumed.Values)
}

func runHotelScenario() {
	fmt.Println("=== Scenario 2: hotel delegation without a sensitive tool ===")
	ms := defaultModels()
	ms.primary = newModel(toolCall("ToHotelBookingAssistant"), messages.AI("Your hotel is booked. Anything else?"))
	app := buildGraph(ms)
	ctx := context.Background()
	thread := checkpoint.Config{ThreadID: "customer-2"}

	human := messages.Human("Book me a hotel")
	human.ID = nextID("human")
	result, err := app.InvokeWithOptions(ctx,
		map[string]any{"messages": []messages.Message{human}},
		graphpkg.Options{ThreadID: thread.ThreadID},
	)
	if err != nil {
		log.Fatalf("invoke hotel: %v", err)
	}
	printMessages("--- final state (hotel) ---", result.Values)
}

func main() {
	runFlightScenario()
	runHotelScenario()
}
