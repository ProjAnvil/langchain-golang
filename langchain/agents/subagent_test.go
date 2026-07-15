package agents

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// lastAIText returns the text of the last AI message in msgs, or "" if none.
// Used by the subagent tool helper to surface a nested agent's final answer.
func lastAIText(msgs []messages.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleAI {
			return messages.Text(msgs[i])
		}
	}
	return ""
}

// newSubagentTool is the Go equivalent of Python's hand-rolled subagent tool
// (see langchain_v1 test_subagent_transformer.py::test_subagents_surfaces_named_subagent):
// a coretools.Func whose body invokes a nested named *Agent via InvokeWithState
// and returns the final AI message text. There is intentionally no public
// AgentAsTool helper — Python has none; this local test helper mirrors the
// user-authored pattern.
func newSubagentTool(agent *Agent, name, inputField string) coretools.Tool {
	tool, err := coretools.NewFunc(
		name,
		"Delegate a task to the "+agent.Name+" subagent.",
		schema.Object(map[string]schema.Schema{
			inputField: schema.String("task to delegate"),
		}, inputField),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			task, _ := input[inputField].(string)
			state, err := agent.InvokeWithState(ctx, []messages.Message{messages.Human(task)})
			if err != nil {
				return coretools.Result{}, err
			}
			msgs, _ := state["messages"].([]messages.Message)
			if text := lastAIText(msgs); text != "" {
				return coretools.Result{Content: text}, nil
			}
			return coretools.Result{}, fmt.Errorf("subagent %s produced no output", agent.Name)
		},
	)
	if err != nil {
		panic(fmt.Sprintf("newSubagentTool: %v", err))
	}
	return tool
}

// nameRecorder is a middleware that records the value of NameFromContext at
// each BeforeModel call, so a test can assert which agent name a nested run
// observed.
type nameRecorder struct {
	mu   sync.Mutex
	seen []string
}

func (r *nameRecorder) BeforeModel(ctx context.Context, _ map[string]any) (map[string]any, error) {
	if name, ok := NameFromContext(ctx); ok {
		r.mu.Lock()
		r.seen = append(r.seen, name)
		r.mu.Unlock()
	}
	return nil, nil
}

// TestCreateAgent_SubagentUnderStreamingParentNoLeak guards the sink-leak fix:
// a non-streaming InvokeWithState called from a tool running inside a
// StreamEvents parent must NOT emit the nested run's model_delta events into
// the parent stream. Before the fix, the nested model node inherited the
// parent's event sink via context and streamed the inner agent's output.
func TestCreateAgent_SubagentUnderStreamingParentNoLeak(t *testing.T) {
	ctx := context.Background()

	// Inner agent's model supports streaming and emits a sentinel. Its single
	// chunk is the whole sentinel so the assertion can match one delta exactly.
	innerModel := &streamSequenceModel{
		responses: []messages.Message{messages.AI("INNER_LEAK")},
		streamChunks: [][]messages.Message{
			{messages.AI("INNER_LEAK")},
		},
	}
	innerAgent, err := CreateAgent(innerModel, nil, WithAgentName("weather_agent"))
	if err != nil {
		t.Fatalf("inner CreateAgent: %v", err)
	}
	weatherTool := newSubagentTool(innerAgent, "call_weather", "city")

	// Supervisor: first model call dispatches the tool, second is the final
	// answer. Both calls are streamed as chunks.
	supModel := &streamSequenceModel{
		responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
				{ID: "call_w", Name: "call_weather", Args: map[string]any{"city": "SF"}},
			}},
			messages.AI("done"),
		},
		streamChunks: [][]messages.Message{
			{{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
				{ID: "call_w", Name: "call_weather", Args: map[string]any{"city": "SF"}},
			}}},
			{messages.AI("done")},
		},
	}
	supervisor, err := CreateAgent(supModel, []coretools.Tool{weatherTool}, WithAgentName("supervisor"))
	if err != nil {
		t.Fatalf("supervisor CreateAgent: %v", err)
	}

	stream, err := supervisor.StreamEvents(ctx, []messages.Message{messages.Human("weather?")})
	if err != nil {
		t.Fatalf("StreamEvents: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)
	requireTerminalEnd(t, events) // sanity: the run completed

	for _, ev := range events {
		if ev.Type == StreamModelDelta && strings.Contains(ev.Text, "INNER_LEAK") {
			t.Fatalf("inner agent output leaked into parent stream as a model_delta: %q (events: %v)",
				ev.Text, eventTypes(events))
		}
	}
}
