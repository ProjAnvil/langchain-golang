// Command middleware-suite composes three agent middleware on one
// agents.CreateAgent agent and makes each one's effect visible:
//
//   - PIIMiddleware redacts emails out of the model's input view;
//   - SummarizationMiddleware collapses old turns into a summary once the
//     history crosses its trigger, keeping only the trailing messages;
//   - TodoListMiddleware contributes the write_todos tool and a todos state
//     key, and injects its planning system prompt into every model call.
//
// The model is an offline scripted double that records every prompt it
// receives, so the middleware effects are printed from the model's own point
// of view.
//
// Usage:
//
//	go run ./examples/middleware-suite
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
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// scriptedModel: offline ChatModel double, self-returning from BindTools
// (see examples/quickstart for the rationale). invocations records exactly
// what each model call saw — the observability point for this example.
type scriptedModel struct {
	mu          sync.Mutex
	responses   []messages.Message
	idx         int
	invocations [][]messages.Message
}

func (m *scriptedModel) Invoke(_ context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.invocations = append(m.invocations, append([]messages.Message(nil), input...))
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
func (m *scriptedModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}

func (m *scriptedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// messageText renders a message's text across Content AND ContentBlocks (the
// TodoListMiddleware appends its planning prompt as a block, which a plain
// messages.Text call would hide when Content is non-empty).
func messageText(m messages.Message) string {
	var b strings.Builder
	b.WriteString(messages.Text(m))
	for _, block := range m.ContentBlocks {
		if tb, ok := block.(messages.TextBlock); ok {
			b.WriteString(tb.Text)
		}
	}
	return b.String()
}

// tail returns the last n runes of s (the whole string when shorter).
func tail(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[len(runes)-n:])
}

func main() {
	ctx := context.Background()

	// --- Middleware 1: redact emails from the model's input view. ---
	pii, err := middleware.NewPIIMiddleware("email")
	if err != nil {
		fmt.Println("new pii middleware:", err)
		return
	}

	// --- Middleware 2: summarize once the history crosses 5 messages,
	// keeping only the trailing 2. The summarizer is an ordinary function —
	// point it at a real model (or an agents.Agent as a tool) in production. ---
	summarizerCalls := 0
	summarize := middleware.NewSummarizationMiddleware(func(_ string, msgs []messages.Message) (string, error) {
		summarizerCalls++
		return fmt.Sprintf("[summary] %d earlier customer message(s) about shipping arrangements.", len(msgs)), nil
	})
	summarize.Trigger = []middleware.TriggerClause{{Messages: 5}}
	summarize.Keep = middleware.KeepPolicy{Messages: 2}
	summarize.TrimTokensToSummarize = 0 // disable pre-trim; the demo corpus is tiny

	// --- Middleware 3: todo planning (contributes tool + state key). ---
	todo, err := middleware.NewTodoListMiddleware()
	if err != nil {
		fmt.Println("new todo middleware:", err)
		return
	}

	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{
					ID:   "call_1",
					Name: middleware.WriteTodosToolName,
					Args: map[string]any{
						"todos": []any{
							map[string]any{"content": "confirm shipping address", "status": "in_progress"},
							map[string]any{"content": "book courier pickup", "status": "pending"},
							map[string]any{"content": "send tracking link", "status": "pending"},
						},
					},
				},
			},
		},
		messages.AI("Shipping plan confirmed: three crates to Berlin next week, todos created."),
	}}

	agent, err := agents.CreateAgent(model, nil,
		agents.WithAgentMiddleware(pii, summarize, todo),
		agents.WithAgentSystemPrompt("You are a shipping coordinator."),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}

	state, err := agent.InvokeWithState(ctx, []messages.Message{
		messages.Human("I need to ship three crates to Berlin next week."),
		messages.Human("The crates weigh about 40kg each."),
		messages.Human("Send updates to me at shipper@corp-example.com, please plan this out."),
	})
	if err != nil {
		fmt.Println("invoke agent:", err)
		return
	}

	// --- What the model actually saw on each call. ---
	fmt.Println("model call 1 (email redacted by PIIMiddleware, planning prompt appended by TodoListMiddleware):")
	for _, m := range model.invocations[0] {
		if m.Role == messages.RoleSystem {
			// The system message is long; show its tail, where the
			// write_todos planning instructions were appended.
			fmt.Printf("  %-8s ...%q\n", m.Role, tail(messageText(m), 60))
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
	fmt.Println("model call 2 (old turns collapsed into the summary by SummarizationMiddleware):")
	for _, m := range model.invocations[1] {
		if m.Role == messages.RoleSystem {
			fmt.Printf("  %-8s ...%q\n", m.Role, tail(messageText(m), 60))
			continue
		}
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s\n", m.Role, m.ToolCalls[0].Name)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}

	// --- Committed state: the raw human input stays intact (redaction was
	// local to the model's view), the todos key landed in graph state. ---
	fmt.Println("final committed messages:")
	finalMsgs, _ := state["messages"].([]messages.Message)
	for _, m := range finalMsgs {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s\n", m.Role, m.ToolCalls[0].Name)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
	fmt.Printf("summarizer invocations: %d\n", summarizerCalls)
	todos, _ := state["todos"].([]middleware.Todo)
	fmt.Printf("todos state key (%d entries):\n", len(todos))
	for i, td := range todos {
		fmt.Printf("  %d. [%s] %s\n", i+1, td.Status, td.Content)
	}
}
