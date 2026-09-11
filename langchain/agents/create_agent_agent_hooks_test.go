package agents

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// countModelCallsMiddleware counts model invocations (see create_agent_test.go
// for the original; duplicated here for locality).
type agentHooksCountModelCalls struct{ calls *int32 }

func (m agentHooksCountModelCalls) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	atomic.AddInt32(m.calls, 1)
	return handler(ctx, request)
}

// TestBeforeAgentJumpToEnd verifies a before_agent hook can short-circuit the
// whole run via update["jump_to"] = "end" (factory.py:1694-1713: the
// before_agent nodes' conditional edge routes jump_to "end" to the exit
// node): no model call happens, and the hook's other state keys persist.
func TestBeforeAgentJumpToEnd(t *testing.T) {
	var modelCalls int32
	model := &sequenceModel{responses: []messages.Message{messages.AI("never")}}

	agent, err := CreateAgent(model, nil,
		WithAgentMiddleware(
			agentHooksCountModelCalls{calls: &modelCalls},
			FuncBeforeAgent(func(ctx context.Context, state map[string]any) (map[string]any, error) {
				return map[string]any{"jump_to": "end", "skipped": true}, nil
			}),
		),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if n := atomic.LoadInt32(&modelCalls); n != 0 {
		t.Fatalf("model called %d times, want 0 (jump_to end must skip the loop)", n)
	}
	if got, _ := state["skipped"].(bool); !got {
		t.Fatalf("state[skipped] = %v, want true", state["skipped"])
	}
	if _, hasJump := state["jump_to"]; hasJump {
		t.Fatalf("jump_to must not persist into state: %#v", state["jump_to"])
	}
}

// TestBeforeAgentJumpToModel verifies a before_agent jump_to "model" still
// enters the model<->tools loop (the default before_agent destination).
func TestBeforeAgentJumpToModel(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateAgent(model, nil,
		WithAgentMiddleware(
			FuncBeforeAgent(func(ctx context.Context, state map[string]any) (map[string]any, error) {
				return map[string]any{"jump_to": "model", "entered": true}, nil
			}),
		),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got, _ := state["entered"].(bool); !got {
		t.Fatalf("state[entered] = %v, want true", state["entered"])
	}
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 2 || msgs[1].Content != "done" {
		t.Fatalf("model should have run: %#v", msgs)
	}
}

// jumpBackAfterAgent implements the update-returning AfterAgent shape: on its
// first run it jumps back into the model loop; afterwards it lets the run
// finish.
type jumpBackAfterAgent struct{ ran *int32 }

func (m jumpBackAfterAgent) AfterAgent(ctx context.Context, state map[string]any) (map[string]any, error) {
	if atomic.AddInt32(m.ran, 1) == 1 {
		return map[string]any{"jump_to": "model", "after_ran": true}, nil
	}
	return map[string]any{"after_ran": true}, nil
}

// TestAfterAgentJumpBackToModel verifies an after_agent hook can jump back
// into the model<->tools loop via update["jump_to"] = "model" (factory.py:
// 1753-1776: the after_agent → END edge carries model_destination =
// loop_entry_node), and that its state update persists.
func TestAfterAgentJumpBackToModel(t *testing.T) {
	var afterRuns int32
	model := &sequenceModel{responses: []messages.Message{
		messages.AI("first"),
		messages.AI("second"),
	}}

	agent, err := CreateAgent(model, nil,
		WithAgentMiddleware(jumpBackAfterAgent{ran: &afterRuns}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if n := atomic.LoadInt32(&afterRuns); n != 2 {
		t.Fatalf("after_agent ran %d times, want 2", n)
	}
	if n := len(model.invocations); n != 2 {
		t.Fatalf("model invoked %d times, want 2", n)
	}
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 3 || msgs[1].Content != "first" || msgs[2].Content != "second" {
		t.Fatalf("messages mismatch: %#v", msgs)
	}
	if got, _ := state["after_ran"].(bool); !got {
		t.Fatalf("state[after_ran] = %v, want true", state["after_ran"])
	}
}

// TestAfterAgentUpdateWithoutJump verifies an update-returning after_agent
// hook that sets no jump_to behaves exactly like the classic error-only
// AfterAgentHook: its update commits and the run ends.
func TestAfterAgentUpdateWithoutJump(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateAgent(model, nil,
		WithAgentMiddleware(jumpAfterAgentUpdate{update: map[string]any{"cleaned": true}}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got, _ := state["cleaned"].(bool); !got {
		t.Fatalf("state[cleaned] = %v, want true", state["cleaned"])
	}
}

type jumpAfterAgentUpdate struct{ update map[string]any }

func (m jumpAfterAgentUpdate) AfterAgent(ctx context.Context, state map[string]any) (map[string]any, error) {
	return m.update, nil
}

// duplicateNamedMiddleware pairs a Name() with a marker for T12d tests.
type duplicateNamedMiddleware struct {
	name  string
	marks *int32
}

func (m duplicateNamedMiddleware) Name() string { return m.name }

func (m duplicateNamedMiddleware) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	atomic.AddInt32(m.marks, 1)
	return nil, nil
}

// TestCreateAgentRejectsDuplicateMiddlewareNames verifies duplicate middleware
// names are rejected at build time, mirroring factory.py:1080-1082
// (`len({m.name for m in middleware}) != len(middleware)` → "Please remove
// duplicate middleware instances."). Two instances of the same type share the
// type-derived name, like two instances of one Python middleware class.
func TestCreateAgentRejectsDuplicateMiddlewareNames(t *testing.T) {
	var marks int32
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	_, err := CreateAgent(model, nil, WithAgentMiddleware(
		duplicateNamedMiddleware{name: "same", marks: &marks},
		duplicateNamedMiddleware{name: "same", marks: &marks},
	))
	if err == nil {
		t.Fatal("expected duplicate middleware name error")
	}
	if !strings.Contains(err.Error(), "duplicate middleware") {
		t.Fatalf("error should mention duplicate middleware: %v", err)
	}
}

// noNameAfterModelMiddleware has no Name() method: its middleware name
// defaults to the Go type name, the analog of Python's class-name default.
type noNameAfterModelMiddleware struct{ marks *int32 }

func (m noNameAfterModelMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	atomic.AddInt32(m.marks, 1)
	return nil, nil
}

// TestCreateAgentRejectsDuplicateMiddlewareInstances verifies two DISTINCT
// instances of the same middleware type (no Name override) are also rejected,
// mirroring Python where the default name is the class name.
func TestCreateAgentRejectsDuplicateMiddlewareInstances(t *testing.T) {
	var marks int32
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	_, err := CreateAgent(model, nil, WithAgentMiddleware(
		noNameAfterModelMiddleware{marks: &marks},
		noNameAfterModelMiddleware{marks: &marks},
	))
	if err == nil {
		t.Fatal("expected duplicate middleware instance error for two instances of one type")
	}
	if !strings.Contains(err.Error(), "duplicate middleware") {
		t.Fatalf("error should mention duplicate middleware: %v", err)
	}
}

// TestCreateAgentAllowsDistinctMiddleware verifies distinct middleware names
// (including distinct Name() values and distinct function middleware) do not
// trip the duplicate check.
func TestCreateAgentAllowsDistinctMiddleware(t *testing.T) {
	var marks1, marks2, marks3 int32
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(
		duplicateNamedMiddleware{name: "one", marks: &marks1},
		duplicateNamedMiddleware{name: "two", marks: &marks2},
		FuncBeforeModel(func(ctx context.Context, state map[string]any) (map[string]any, error) { return nil, nil }),
		FuncAfterModel(func(ctx context.Context, state map[string]any) (map[string]any, error) { return nil, nil }),
	))
	if err != nil {
		t.Fatalf("distinct middleware must be accepted: %v", err)
	}
	if _, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	_ = marks3
}

// TestCreateAgentNamesAIMessages verifies the agent's Name (create_agent
// name=...) is stamped onto the model's output AIMessage, mirroring
// factory.py:1418-1419 (`output = model_.invoke(messages); if name:
// output.name = name`) — the value is the agent name, not a middleware or
// node name.
func TestCreateAgentNamesAIMessages(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}

	agent, err := CreateAgent(model, nil, WithAgentName("my-agent"))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 2 {
		t.Fatalf("messages mismatch: %#v", msgs)
	}
	if msgs[1].Name != "my-agent" {
		t.Fatalf("AI message Name = %q, want %q", msgs[1].Name, "my-agent")
	}
}

// TestCreateAgentWithoutNameLeavesAIMessageUnnamed verifies no name is set
// when create_agent received none.
func TestCreateAgentWithoutNameLeavesAIMessageUnnamed(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}

	agent, err := CreateAgent(model, nil)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	msgs, _ := state["messages"].([]messages.Message)
	if msgs[1].Name != "" {
		t.Fatalf("AI message Name = %q, want empty", msgs[1].Name)
	}
}
