package agents

import (
	"context"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// todoAgentModel returns a sequenceModel whose first response calls
// write_todos with the given todos and whose second response is a terminal
// answer, the canonical TodoListMiddleware round trip.
func todoAgentModel() *sequenceModel {
	return &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{
					ID:   "call_todo",
					Name: middleware.WriteTodosToolName,
					Args: map[string]any{
						"todos": []any{
							map[string]any{"content": "plan", "status": "completed"},
							map[string]any{"content": "execute", "status": "in_progress"},
						},
					},
				},
			},
		},
		messages.AI("all planned"),
	}}
}

// TestTodoListMiddlewarePersistsTodosToState verifies the full T12a path: the
// write_todos tool's Command lands in the agent's "todos" state key
// (PlanningState.todos, todo.py:35-42), the ToolMessage stays in messages,
// and the loop continues to the model.
func TestTodoListMiddlewarePersistsTodosToState(t *testing.T) {
	todoMW, err := middleware.NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}
	model := todoAgentModel()

	agent, err := CreateAgent(model, todoMW.Tools, WithAgentMiddleware(todoMW))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("plan this")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	todos, ok := state["todos"].([]middleware.Todo)
	if !ok {
		t.Fatalf("state[todos] = %#v, want []middleware.Todo", state["todos"])
	}
	if len(todos) != 2 || todos[0].Content != "plan" || todos[0].Status != middleware.TodoCompleted || todos[1].Status != middleware.TodoInProgress {
		t.Fatalf("todos mismatch: %#v", todos)
	}

	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) != 4 || msgs[2].Role != messages.RoleTool || msgs[2].Name != middleware.WriteTodosToolName {
		t.Fatalf("messages mismatch: %#v", msgs)
	}
	if msgs[3].Content != "all planned" {
		t.Fatalf("loop should continue to the model after write_todos: %#v", msgs[3])
	}
}

// TestCreateAgentAutoCollectsMiddlewareTools verifies T10 tool
// auto-collection: middleware ProvidedTools() are merged into the agent's
// tool set without the caller listing them (factory.py:1005, 1054-1055:
// available_tools = middleware_tools + regular_tools). The write_todos call
// must execute (todos land in state) even though CreateAgent received no
// positional tools.
func TestCreateAgentAutoCollectsMiddlewareTools(t *testing.T) {
	todoMW, err := middleware.NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}
	model := todoAgentModel()

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(todoMW))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("plan this")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if _, ok := state["todos"].([]middleware.Todo); !ok {
		t.Fatalf("state[todos] = %#v, want []middleware.Todo (middleware tools should be auto-collected)", state["todos"])
	}

	// The middleware tool must also be bound to the model: the model node's
	// request carries it (factory.py:1070-1075 default_tools).
	if len(model.boundTools) == 0 || model.boundTools[0].Name() != middleware.WriteTodosToolName {
		t.Fatalf("middleware tools should be bound to the model: %#v", model.boundTools)
	}
}

// TestCreateAgentAutoCollectsMiddlewareToolsOrder verifies user tools win on
// a name conflict with a middleware tool, mirroring Python's ToolNode dict
// registration order (middleware tools first, user tools later overwrite —
// prebuilt/tool_node.py:784) rather than erroring on the duplicate name.
func TestCreateAgentAutoCollectsMiddlewareToolsOrder(t *testing.T) {
	todoMW, err := middleware.NewTodoListMiddleware()
	if err != nil {
		t.Fatalf("new todo middleware: %v", err)
	}
	userTool, err := coretools.NewFunc(middleware.WriteTodosToolName, "user override", nil,
		func(_ context.Context, _ map[string]any) (coretools.Result, error) {
			return coretools.Result{Content: "user tool ran"}, nil
		})
	if err != nil {
		t.Fatalf("new user tool: %v", err)
	}
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: middleware.WriteTodosToolName, Args: map[string]any{}},
			},
		},
		messages.AI("done"),
	}}

	agent, err := CreateAgent(model, []coretools.Tool{userTool}, WithAgentMiddleware(todoMW))
	if err != nil {
		t.Fatalf("create agent should not fail on a user/middleware tool name conflict: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) < 3 || msgs[2].Content != "user tool ran" {
		t.Fatalf("user tool should win the name conflict: %#v", msgs)
	}
	if _, hasTodos := state["todos"]; hasTodos {
		t.Fatalf("user tool should have replaced the middleware tool (no todos state write): %#v", state["todos"])
	}
}

// TestCreateAgentAutoCollectsFilesystemFileSearchTools verifies a second
// ToolProvider middleware (FilesystemFileSearchMiddleware contributes glob +
// grep) is auto-collected: the model node binds its tools with no positional
// tools.
func TestCreateAgentAutoCollectsFilesystemFileSearchTools(t *testing.T) {
	fsMW, err := middleware.NewFilesystemFileSearchMiddleware(t.TempDir(), 1)
	if err != nil {
		t.Fatalf("new fs middleware: %v", err)
	}
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(fsMW))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	bound := make(map[string]bool, len(model.boundTools))
	for _, tool := range model.boundTools {
		bound[tool.Name()] = true
	}
	if !bound["glob_search"] || !bound["grep_search"] {
		t.Fatalf("filesystem file search tools should be auto-bound to the model: %#v", model.boundTools)
	}
}

// contributingStateMiddleware declares two state fields: "hits" with an
// appending reducer and "shadowed" with an appending reducer the caller is
// about to override.
type contributingStateMiddleware struct{ writes *int32 }

func (m contributingStateMiddleware) StateSchema() []middleware.StateField {
	concat := func(existing any, update any) (any, error) {
		if existing == nil {
			return update, nil
		}
		return append(existing.([]string), update.([]string)...), nil
	}
	return []middleware.StateField{
		{Name: "hits", Reducer: concat},
		{Name: "shadowed", Reducer: concat},
	}
}

func (m contributingStateMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	return map[string]any{
		"hits":     []string{"h" + strconv.Itoa(int(atomic.AddInt32(m.writes, 1)))},
		"shadowed": []string{"mw"},
	}, nil
}

// TestCreateAgentMergesMiddlewareStateSchema verifies middleware state-field
// contributions register with their reducers (factory.py:1150-1156), and that
// a caller's WithAgentStateFields declaration of the same key WINS the
// conflict (base_state merges last) — "shadowed" keeps last-write-wins while
// "hits" appends across two model calls.
func TestCreateAgentMergesMiddlewareStateSchema(t *testing.T) {
	var writes int32
	// One tool-calling pass then a terminal answer: two model calls, so the
	// AfterModel-driven "hits" writes append twice under the middleware's
	// custom reducer while "shadowed" (overridden by the caller's
	// last-write-wins declaration) keeps only the final value.
	echo := newEchoTool(t)
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c1", Name: "echo", Args: map[string]any{}}}},
		messages.AI("done"),
	}}

	agent, err := CreateAgent(model, []coretools.Tool{echo},
		WithAgentMiddleware(contributingStateMiddleware{writes: &writes}),
		WithAgentStateFields(StateField{Name: "shadowed"}), // nil reducer → LastValue
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	hits, ok := state["hits"].([]string)
	if !ok || len(hits) != 2 {
		t.Fatalf("state[hits] = %#v, want two appended writes (middleware reducer applied)", state["hits"])
	}
	shadowed, ok := state["shadowed"].([]string)
	if !ok || len(shadowed) != 1 || shadowed[0] != "mw" {
		t.Fatalf("state[shadowed] = %#v, want the middleware's single write (LastValue via user override replaces the concat reducer)", state["shadowed"])
	}
}
