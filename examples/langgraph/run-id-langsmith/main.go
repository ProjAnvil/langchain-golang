// Command run-id-langsmith is the Go port of
// langgraph/examples/run-id-langsmith.ipynb. The Python notebook passes a
// run_name and tags in the RunnableConfig so LangSmith can label the traced
// run. This port keeps that idea entirely in-process: an in-memory trace
// recorder assigns a run ID, stores the name/tags, and records model and tool
// events while a ReAct agent answers a weather question.
package main

import (
	"context"
	"fmt"
	"sync"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
)

// traceEvent is one in-memory span, the local analogue of a LangSmith run.
type traceEvent struct {
	RunID   string
	Kind    string
	Name    string
	Summary string
}

// inMemoryTracer is a deterministic, offline stand-in for a LangSmith client.
type inMemoryTracer struct {
	mu     sync.Mutex
	nextID int
	events []traceEvent
}

func (t *inMemoryTracer) startRun(name string, tags []string) string {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.nextID++
	runID := fmt.Sprintf("run-%04d", t.nextID)
	t.events = append(t.events, traceEvent{
		RunID: runID, Kind: "run_start", Name: name,
		Summary: fmt.Sprintf("tags=%v", tags),
	})
	return runID
}

func (t *inMemoryTracer) record(runID, kind, name, summary string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.events = append(t.events, traceEvent{RunID: runID, Kind: kind, Name: name, Summary: summary})
}

func (t *inMemoryTracer) print() {
	t.mu.Lock()
	defer t.mu.Unlock()
	fmt.Println("\n--- in-memory trace ---")
	for _, e := range t.events {
		fmt.Printf("  %s  %-10s %-16s %s\n", e.RunID, e.Kind, e.Name, e.Summary)
	}
}

// scriptedModel is a deterministic ChatModel double. BindTools returns the
// receiver so the response cursor survives the agent's per-iteration rebinds.
type scriptedModel struct {
	mu        sync.Mutex
	responses []messages.Message
	idx       int
	tracer    *inMemoryTracer
	runID     string
}

func (m *scriptedModel) Invoke(_ context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	resp := m.responses[m.idx%len(m.responses)]
	m.idx++
	m.tracer.record(m.runID, "chat_model", "ChatOpenAI", fmt.Sprintf("input_messages=%d", len(input)))
	return resp, nil
}

func (m *scriptedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		if out[i], err = m.Invoke(ctx, in, opts...); err != nil {
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
	return language.ChatModelCapabilities{ToolCalling: true, Streaming: true}
}

func main() {
	ctx := context.Background()
	tracer := &inMemoryTracer{}

	// The notebook's config: {"run_name": "agent_007", "tags": [...]}.
	runName := "agent_007"
	tags := []string{"cats are awesome"}
	runID := tracer.startRun(runName, tags)

	model := &scriptedModel{
		responses: []messages.Message{
			{
				Role: messages.RoleAI,
				ToolCalls: []messages.ToolCall{{
					ID: "call_weather", Name: "get_weather",
					Args: map[string]any{"city": "sf"},
				}},
			},
			messages.AI("It's always sunny in sf."),
		},
		tracer: tracer,
		runID:  runID,
	}

	weather, err := coretools.NewFunc(
		"get_weather",
		"Use this to get weather information.",
		schema.Object(map[string]schema.Schema{"city": schema.String("nyc or sf")}, "city"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			city, _ := input["city"].(string)
			tracer.record(runID, "tool", "get_weather", fmt.Sprintf("city=%s", city))
			if city == "nyc" {
				return coretools.Result{Content: "It might be cloudy in nyc"}, nil
			}
			return coretools.Result{Content: "It's always sunny in sf"}, nil
		},
	)
	if err != nil {
		fmt.Println("build tool:", err)
		return
	}

	agent, err := agents.CreateAgent(model, []coretools.Tool{weather},
		agents.WithAgentName(runName))
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}

	// Stream values, mirroring graph.stream(inputs, config, stream_mode="values").
	for chunk, err := range agent.Graph.Stream(ctx,
		map[string]any{"messages": []messages.Message{messages.Human("what is the weather in sf")}},
		graphpkg.StreamOptions{Modes: []graphpkg.StreamMode{graphpkg.StreamValues}},
	) {
		if err != nil {
			fmt.Println("stream:", err)
			return
		}
		state, ok := chunk.Payload.(map[string]any)
		if !ok {
			continue
		}
		msgs, _ := state["messages"].([]messages.Message)
		if len(msgs) == 0 {
			continue
		}
		last := msgs[len(msgs)-1]
		if len(last.ToolCalls) > 0 {
			fmt.Printf("AI: [tool call %s]\n", last.ToolCalls[0].Name)
			continue
		}
		fmt.Printf("AI: %s\n", messages.Text(last))
	}

	tracer.record(runID, "run_end", runName, "completed")
	tracer.print()
}
