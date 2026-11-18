// Command extraction_retries is the Go port of
// langgraph/examples/extraction/retries.ipynb. It rebuilds the
// validate-with-retries loop: a model emits a structured tool call, a
// validator checks it (schema plus a field rule), and on failure a fallback
// model is asked again until a valid answer is produced or a maximum attempt
// count is reached.
//
// Usage:
//
//	go run .
//
// All models are offline deterministic doubles; no API key or network.
package main

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/prebuilt"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// scriptedModel is an offline ChatModel double returned from BindTools.
type scriptedModel struct {
	mu        sync.Mutex
	responses []messages.Message
	idx       int
}

func (m *scriptedModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("scriptedModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}
func (m *scriptedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		out[i], err = m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
func (m *scriptedModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	resp, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{resp}), nil
}
func (m *scriptedModel) InputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *scriptedModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *scriptedModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) { return m, nil }
func (m *scriptedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true, Streaming: true}
}

// respondTool mirrors the notebook's Respond pydantic model.
func respondTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"Respond",
		"Use to generate the response. Always use when responding to the user.",
		schema.Object(map[string]schema.Schema{
			"reason": schema.String("step-by-step justification for the answer"),
			"answer": schema.String("the answer"),
		}, "reason", "answer"),
		func(_ context.Context, _ map[string]any) (coretools.Result, error) {
			return coretools.Result{}, nil // validation-only; never executed
		},
	)
}

func respondCall(id, reason, answer string) messages.Message {
	return messages.Message{
		Role: messages.RoleAI,
		ToolCalls: []messages.ToolCall{{
			ID: id, Name: "Respond",
			Args: map[string]any{"reason": reason, "answer": answer},
		}},
	}
}

// Helpers for graph state values.
func stateMessages(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}
func stateInt(state map[string]any, key string) int {
	switch v := state[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

// addOrOverwriteMessages appends normally but replaces the whole list when a
// "finalize" update arrives, mirroring the notebook's custom reducer.
func addOrOverwriteMessages(existing any, update any) (any, error) {
	if m, ok := update.(map[string]any); ok {
		if finalized, ok := m["finalize"]; ok {
			switch f := finalized.(type) {
			case messages.Message:
				return []messages.Message{f}, nil
			case []messages.Message:
				return f, nil
			}
		}
	}
	return channels.MessagesReducer(existing, update)
}

// sumReducer mirrors Python's operator.add on attempt_number.
func sumReducer(existing any, update any) (any, error) {
	total := stateInt(map[string]any{"v": existing}, "v") + stateInt(map[string]any{"v": update}, "v")
	return total, nil
}

func lastAI(msgs []messages.Message) *messages.Message {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleAI {
			m := msgs[i]
			return &m
		}
	}
	return nil
}

func hasError(msgs []messages.Message) bool {
	for _, m := range msgs {
		if v, _ := m.AdditionalKwargs["is_error"].(bool); v {
			return true
		}
	}
	return false
}

// buildRetryGraph assembles the validate-with-retries state graph for a given
// model pair and attempt limit.
func buildRetryGraph(maxAttempts int, primary, fallback *scriptedModel, respond coretools.Tool) (*graphpkg.CompiledGraph, error) {
	// ValidationNode checks tool-call arguments against the Respond schema.
	schemaValidator, err := prebuilt.NewValidationNode([]coretools.Tool{respond})
	if err != nil {
		return nil, err
	}

	countMessages := func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"initial_num_messages": len(stateMessages(state))}, nil
	}
	llm := func(rt runtime.Runtime, state map[string]any) (any, error) {
		resp, err := primary.Invoke(rt, stateMessages(state))
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": []messages.Message{resp}, "attempt_number": 1}, nil
	}
	fallbackModel := func(rt runtime.Runtime, state map[string]any) (any, error) {
		resp, err := fallback.Invoke(rt, stateMessages(state))
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": []messages.Message{resp}, "attempt_number": 1}, nil
	}

	// validator = select generated messages | ValidationNode | field rule.
	validator := func(rt runtime.Runtime, state map[string]any) (any, error) {
		msgs := stateMessages(state)
		initial := stateInt(state, "initial_num_messages")
		if initial < 0 || initial > len(msgs) {
			initial = 0
		}
		selected := msgs[initial:]

		out, err := schemaValidator(rt, map[string]any{"messages": selected})
		if err != nil {
			return nil, err
		}
		outMap, ok := out.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("validator: unexpected output %T", out)
		}
		results := stateMessages(outMap)

		// Field-level rule, the Go stand-in for the notebook's
		// pydantic field_validator on the "answer" field.
		if !hasError(results) {
			if ai := lastAI(selected); ai != nil && len(ai.ToolCalls) > 0 {
				answer, _ := ai.ToolCalls[0].Args["answer"].(string)
				if !strings.Contains(strings.ToLower(answer), "llama") {
					errMsg := messages.Tool(ai.ToolCalls[0].ID,
						"Field 'answer' must contain 'llama' (field validator failed).")
					errMsg.Name = "Respond"
					errMsg.AdditionalKwargs = map[string]any{"is_error": true}
					results = []messages.Message{errMsg}
				}
			}
		}
		return map[string]any{"messages": results}, nil
	}

	finalizer := func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs := stateMessages(state)
		initial := stateInt(state, "initial_num_messages")
		if initial < 0 || initial > len(msgs) {
			initial = 0
		}
		ai := lastAI(msgs[initial:])
		if ai == nil {
			return nil, fmt.Errorf("finalizer: no AI message to finalize")
		}
		return map[string]any{"messages": map[string]any{"finalize": []messages.Message{*ai}}}, nil
	}

	routeValidator := func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs := stateMessages(state)
		if len(msgs) > 0 && len(msgs[len(msgs)-1].ToolCalls) > 0 {
			return graphpkg.To("validator"), nil
		}
		return graphpkg.To(types.END), nil
	}
	routeValidation := func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		if stateInt(state, "attempt_number") > maxAttempts {
			return nil, fmt.Errorf("Could not extract a valid value in %d attempts", maxAttempts)
		}
		msgs := stateMessages(state)
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == messages.RoleAI {
				break
			}
			if v, _ := msgs[i].AdditionalKwargs["is_error"].(bool); v {
				return graphpkg.To("fallback"), nil
			}
		}
		return graphpkg.To("finalizer"), nil
	}

	builder := graphpkg.NewStateGraph()
	builder.AddReducer("messages", addOrOverwriteMessages)
	builder.AddReducer("attempt_number", sumReducer)
	builder.AddNode("count_messages", countMessages)
	builder.AddNode("llm", llm)
	builder.AddNode("fallback", fallbackModel)
	builder.AddNode("validator", validator)
	builder.AddNode("finalizer", finalizer)
	builder.AddEdge(types.START, "count_messages")
	builder.AddEdge("count_messages", "llm")
	builder.AddConditionalEdges("llm", routeValidator)
	builder.AddConditionalEdges("validator", routeValidation)
	builder.AddEdge("fallback", "validator")
	builder.AddEdge("finalizer", types.END)
	return builder.Compile()
}

func main() {
	ctx := context.Background()

	respond, err := respondTool()
	if err != nil {
		fmt.Println("build Respond tool:", err)
		return
	}
	userInput := map[string]any{
		"messages":       []messages.Message{messages.Human("Does P = NP?")},
		"attempt_number": 0,
	}

	// Scenario 1: the first answer fails the field rule; the fallback model
	// supplies a corrected tool call and the graph finalizes it.
	primary := &scriptedModel{responses: []messages.Message{
		respondCall("c1", "P vs NP is an open problem.", "Unknown"),
	}}
	fallback := &scriptedModel{responses: []messages.Message{
		respondCall("c2", "P vs NP remains unresolved.",
			"Llama V3 rhymes so bright, P vs NP shines in the night."),
	}}
	graph, err := buildRetryGraph(3, primary, fallback, respond)
	if err != nil {
		fmt.Println("build retry graph:", err)
		return
	}
	result, err := graph.Invoke(ctx, userInput)
	if err != nil {
		fmt.Println("invoke retry graph:", err)
		return
	}
	fmt.Println("--- retry succeeds within max_attempts ---")
	fmt.Printf("primary model calls: %d; fallback model calls: %d\n", len(primary.responses), len(fallback.responses))
	finalMsgs := stateMessages(result.Values)
	if ai := lastAI(finalMsgs); ai != nil && len(ai.ToolCalls) > 0 {
		fmt.Printf("final reason: %v\n", ai.ToolCalls[0].Args["reason"])
		fmt.Printf("final answer: %v\n", ai.ToolCalls[0].Args["answer"])
	}

	// Scenario 2: every attempt fails; the validator router raises the
	// max-attempts error instead of finalizing.
	badPrimary := &scriptedModel{responses: []messages.Message{
		respondCall("c3", "P vs NP is an open problem.", "Unknown"),
	}}
	badFallback := &scriptedModel{responses: []messages.Message{
		respondCall("c4", "Still unresolved.", "No llama here."),
	}}
	failGraph, err := buildRetryGraph(1, badPrimary, badFallback, respond)
	if err != nil {
		fmt.Println("build failing graph:", err)
		return
	}
	_, err = failGraph.Invoke(ctx, userInput)
	fmt.Println("--- retry exhausted ---")
	if err != nil {
		fmt.Printf("error: %v\n", err)
	} else {
		fmt.Println("expected a max-attempts error but the graph succeeded")
	}
}
