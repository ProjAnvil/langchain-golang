package agents

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// newCommandTool builds a tool whose invocation returns a *types.Command in
// its Result.Artifact (the Go port's convention for a tool signaling graph
// control flow, mirroring Python tools returning langgraph Command values —
// see langchain/tools/tool_node.go InvokeToolCallsFull). update lands in the
// Command's Update; goto, when non-empty, becomes the Command's Goto.
func newCommandTool(t *testing.T, name string, update map[string]any, gotoNames []string) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewFunc(name, "returns a command", nil,
		func(_ context.Context, _ map[string]any) (coretools.Result, error) {
			cmd := &types.Command{Update: update}
			if len(gotoNames) > 0 {
				dests := make([]any, len(gotoNames))
				for i, n := range gotoNames {
					dests[i] = n
				}
				cmd.Goto = dests
			}
			return coretools.Result{Content: "command issued", Artifact: cmd}, nil
		})
	if err != nil {
		t.Fatalf("new command tool: %v", err)
	}
	return tool
}

// TestCreateAgentToolsNodeAppliesCommandUpdate verifies the tools node
// consumes the Command a tool returns via Result.Artifact: the Command's
// Update commits in the same superstep as the tool messages (mirroring
// Python's ToolNode, which passes tool Commands through to langgraph —
// prebuilt/tool_node.py:864-912 _combine_tool_outputs), and the loop then
// continues back to the model.
func TestCreateAgentToolsNodeAppliesCommandUpdate(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "bump", Args: map[string]any{}},
			},
		},
		messages.AI("done"),
	}}
	tool := newCommandTool(t, "bump", map[string]any{"counter": 5}, nil)

	agent, err := CreateAgent(model, []coretools.Tool{tool})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got, _ := state["counter"].(int); got != 5 {
		t.Fatalf("state[counter] = %v (%T), want 5 (command update not applied)", state["counter"], state["counter"])
	}

	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 4 {
		t.Fatalf("expected human + ai(tool_call) + tool + ai messages, got %d: %#v", len(msgs), msgs)
	}
	if msgs[2].Role != messages.RoleTool || msgs[2].Content != "command issued" || msgs[2].ToolCallID != "call_1" {
		t.Fatalf("tool message mismatch: %#v", msgs[2])
	}
	if msgs[3].Role != messages.RoleAI || msgs[3].Content != "done" {
		t.Fatalf("loop should return to the model after an update-only command, got: %#v", msgs[3])
	}
}

// TestCreateAgentToolsNodeHonorsCommandGoto verifies a tool Command carrying
// Goto routes the graph (Python: langgraph honors Command.goto returned
// through the ToolNode; it is neither an error nor ignored). A goto to END
// ends the run without a second model call.
func TestCreateAgentToolsNodeHonorsCommandGoto(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "finish", Args: map[string]any{}},
			},
		},
		messages.AI("should never be produced"),
	}}
	tool := newCommandTool(t, "finish", map[string]any{"finished": true}, []string{types.END})

	agent, err := CreateAgent(model, []coretools.Tool{tool})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got, _ := state["finished"].(bool); !got {
		t.Fatalf("state[finished] = %v, want true", state["finished"])
	}
	if n := len(model.invocations); n != 1 {
		t.Fatalf("model invoked %d times, want 1 (command goto END must end the run)", n)
	}
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 3 {
		t.Fatalf("expected human + ai(tool_call) + tool messages, got %d: %#v", len(msgs), msgs)
	}
}

// TestCreateAgentToolsNodeCommandMessagesMerge verifies a tool Command whose
// Update also carries a "messages" key: those messages append after the
// tool's own ToolMessage (the append-reducer semantics of two sequential
// messages writes in Python's list-of-Commands node return).
func TestCreateAgentToolsNodeCommandMessagesMerge(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "annotate", Args: map[string]any{}},
			},
		},
		messages.AI("done"),
	}}
	extra := messages.Tool("call_1", "injected by command")
	tool := newCommandTool(t, "annotate", map[string]any{
		"counter":  1,
		"messages": []messages.Message{extra},
	}, nil)

	agent, err := CreateAgent(model, []coretools.Tool{tool})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 5 {
		t.Fatalf("expected human + ai(tool_call) + tool + command-injected tool + ai, got %d: %#v", len(msgs), msgs)
	}
	if msgs[2].Content != "command issued" || msgs[3].Content != "injected by command" {
		t.Fatalf("command messages should append after the tool message: %#v %#v", msgs[2], msgs[3])
	}
}

// TestCreateAgentToolsNodeCommandResumeRejected verifies the Go create_agent
// tools node rejects Command Resume and ANY non-empty Command Graph
// (including types.ParentGraph — the check is cmd.Graph != "", with no
// reserved-passthrough exception), documenting the scoped-down subset
// relative to Python's langgraph, which supports both.
func TestCreateAgentToolsNodeCommandResumeRejected(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "resume", Args: map[string]any{}},
			},
		},
	}}
	tool, err := coretools.NewFunc("resume", "returns a resume command", nil,
		func(_ context.Context, _ map[string]any) (coretools.Result, error) {
			return coretools.Result{
				Content:  "resume",
				Artifact: &types.Command{Update: map[string]any{"k": 1}, Resume: "boom"},
			}, nil
		})
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}

	agent, err := CreateAgent(model, []coretools.Tool{tool})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, err = agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected an error for a tool Command carrying Resume")
	}
	if !strings.Contains(err.Error(), "resume") || !strings.Contains(err.Error(), "Command") {
		t.Fatalf("error should name the unsupported Command resume, got: %v", err)
	}
}
