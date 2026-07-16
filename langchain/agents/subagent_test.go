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

// TestCreateAgent_SubagentViaTool is the Go counterpart of the invocation
// pattern in Python's test_subagent_transformer.py: a supervisor delegates to
// a named inner agent via a hand-rolled tool, and the inner agent's final
// answer flows back as the tool result.
func TestCreateAgent_SubagentViaTool(t *testing.T) {
	ctx := context.Background()

	innerAgent, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{messages.AI("sunny in SF")}},
		nil, WithAgentName("weather_agent"),
	)
	if err != nil {
		t.Fatalf("inner CreateAgent: %v", err)
	}
	weatherTool := newSubagentTool(innerAgent, "call_weather", "city")

	supervisor, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
				{ID: "call_w", Name: "call_weather", Args: map[string]any{"city": "SF"}},
			}},
			messages.AI("The weather is sunny in SF"),
		}},
		[]coretools.Tool{weatherTool}, WithAgentName("supervisor"),
	)
	if err != nil {
		t.Fatalf("supervisor CreateAgent: %v", err)
	}

	out, err := supervisor.Invoke(ctx, []messages.Message{messages.Human("weather?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	var foundToolResult bool
	for _, m := range out {
		if m.Role == messages.RoleTool && strings.Contains(messages.Text(m), "sunny in SF") {
			foundToolResult = true
		}
	}
	if !foundToolResult {
		t.Fatalf("inner agent output did not surface as a tool result; messages: %v", out)
	}
	if last := out[len(out)-1]; last.Role != messages.RoleAI {
		t.Fatalf("expected last message to be AI, got %v", last.Role)
	}
}

// TestCreateAgent_SubagentNamePropagation asserts that inside a nested run,
// NameFromContext returns the inner agent's name (not the supervisor's),
// because InvokeWithState rebinds the run-name context tag. This is the
// A1-level "distinguishable subagent" property.
func TestCreateAgent_SubagentNamePropagation(t *testing.T) {
	ctx := context.Background()

	rec := &nameRecorder{}
	innerAgent, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{messages.AI("ok")}},
		nil, WithAgentName("weather_agent"), WithAgentMiddleware(rec),
	)
	if err != nil {
		t.Fatalf("inner CreateAgent: %v", err)
	}
	weatherTool := newSubagentTool(innerAgent, "call_weather", "city")

	supervisor, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
				{ID: "call_w", Name: "call_weather", Args: map[string]any{"city": "SF"}},
			}},
			messages.AI("done"),
		}},
		[]coretools.Tool{weatherTool}, WithAgentName("supervisor"),
	)
	if err != nil {
		t.Fatalf("supervisor CreateAgent: %v", err)
	}

	if _, err := supervisor.Invoke(ctx, []messages.Message{messages.Human("weather?")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.seen) == 0 {
		t.Fatalf("inner agent's BeforeModel never ran; name not recorded")
	}
	for _, name := range rec.seen {
		if name != "weather_agent" {
			t.Fatalf("inner run observed name %q, want %q (all seen: %v)", name, "weather_agent", rec.seen)
		}
	}
}

// TestCreateAgent_SubagentErrorPropagation asserts that an error from the inner
// agent surfaces through the tool into the supervisor's run as an error
// ToolMessage (via ToolNode's default HandleToolErrors), not a panic, and the
// supervisor run still completes.
func TestCreateAgent_SubagentErrorPropagation(t *testing.T) {
	ctx := context.Background()

	// Inner model with no responses: its first (and only) Invoke errors.
	innerAgent, err := CreateAgent(
		&sequenceModel{responses: nil},
		nil, WithAgentName("weather_agent"),
	)
	if err != nil {
		t.Fatalf("inner CreateAgent: %v", err)
	}
	weatherTool := newSubagentTool(innerAgent, "call_weather", "city")

	supervisor, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
				{ID: "call_w", Name: "call_weather", Args: map[string]any{"city": "SF"}},
			}},
			messages.AI("recovered"),
		}},
		[]coretools.Tool{weatherTool}, WithAgentName("supervisor"),
	)
	if err != nil {
		t.Fatalf("supervisor CreateAgent: %v", err)
	}

	out, err := supervisor.Invoke(ctx, []messages.Message{messages.Human("weather?")})
	if err != nil {
		t.Fatalf("supervisor invoke returned error (tool error should be handled): %v", err)
	}

	var foundErrToolMsg bool
	for _, m := range out {
		if m.Role == messages.RoleTool && strings.Contains(messages.Text(m), "Error:") {
			foundErrToolMsg = true
		}
	}
	if !foundErrToolMsg {
		t.Fatalf("expected an error ToolMessage from the failed subagent; messages: %v", out)
	}
}

// TestCreateAgent_SubagentNested asserts one level of nesting works: a
// supervisor delegates to a subagent that itself delegates to a sub-subagent,
// and the leaf actually runs (observed via its name during the supervisor's
// single Invoke).
func TestCreateAgent_SubagentNested(t *testing.T) {
	ctx := context.Background()

	leafRec := &nameRecorder{}
	leaf, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{messages.AI("leaf-answer")}},
		nil, WithAgentName("leaf"), WithAgentMiddleware(leafRec),
	)
	if err != nil {
		t.Fatalf("leaf CreateAgent: %v", err)
	}
	leafTool := newSubagentTool(leaf, "call_leaf", "task")

	middle, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
				{ID: "call_l", Name: "call_leaf", Args: map[string]any{"task": "go"}},
			}},
			messages.AI("middle-done"),
		}},
		[]coretools.Tool{leafTool}, WithAgentName("middle"),
	)
	if err != nil {
		t.Fatalf("middle CreateAgent: %v", err)
	}
	middleTool := newSubagentTool(middle, "call_middle", "task")

	supervisor, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
				{ID: "call_m", Name: "call_middle", Args: map[string]any{"task": "go"}},
			}},
			messages.AI("supervisor-done"),
		}},
		[]coretools.Tool{middleTool}, WithAgentName("supervisor"),
	)
	if err != nil {
		t.Fatalf("supervisor CreateAgent: %v", err)
	}

	if _, err := supervisor.Invoke(ctx, []messages.Message{messages.Human("go")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	leafRec.mu.Lock()
	defer leafRec.mu.Unlock()
	if len(leafRec.seen) == 0 {
		t.Fatalf("leaf never ran; nesting did not reach the sub-subagent")
	}
	if leafRec.seen[0] != "leaf" {
		t.Fatalf("leaf observed wrong name %q, want %q", leafRec.seen[0], "leaf")
	}
}

// TestCreateAgent_UnnamedSubagentInheritsParentName is the Go counterpart of
// Python's test_subagent_transformer.py::test_unnamed_inner_agent_surfaces_with_inherited_name:
// an inner agent built WITHOUT WithAgentName, run inside a named supervisor's
// tool, observes the PARENT's name via NameFromContext. This follows from
// withRunTags being a no-op when Agent.Name is empty — it leaves the parent's
// run-name context tag in place rather than replacing it.
func TestCreateAgent_UnnamedSubagentInheritsParentName(t *testing.T) {
	ctx := context.Background()

	rec := &nameRecorder{}
	innerAgent, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{messages.AI("ok")}},
		nil, WithAgentMiddleware(rec), // intentionally NO WithAgentName
	)
	if err != nil {
		t.Fatalf("inner CreateAgent: %v", err)
	}
	innerTool := newSubagentTool(innerAgent, "call_inner", "task")

	supervisor, err := CreateAgent(
		&sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
				{ID: "c1", Name: "call_inner", Args: map[string]any{"task": "go"}},
			}},
			messages.AI("done"),
		}},
		[]coretools.Tool{innerTool}, WithAgentName("supervisor"),
	)
	if err != nil {
		t.Fatalf("supervisor CreateAgent: %v", err)
	}

	if _, err := supervisor.Invoke(ctx, []messages.Message{messages.Human("go")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.seen) == 0 {
		t.Fatalf("inner agent's BeforeModel never ran; name not recorded")
	}
	for _, name := range rec.seen {
		if name != "supervisor" {
			t.Fatalf("unnamed inner run observed name %q, want inherited parent name %q (all seen: %v)",
				name, "supervisor", rec.seen)
		}
	}
}
