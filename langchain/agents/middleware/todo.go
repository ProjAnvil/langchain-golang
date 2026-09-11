package middleware

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

type TodoStatus string

const (
	TodoPending    TodoStatus = "pending"
	TodoInProgress TodoStatus = "in_progress"
	TodoCompleted  TodoStatus = "completed"
)

type Todo struct {
	Content string     `json:"content"`
	Status  TodoStatus `json:"status"`
}

const WriteTodosToolName = "write_todos"

const WriteTodosSystemPrompt = "## `write_todos`\n\nYou have access to the `write_todos` tool to help you manage and plan complex objectives.\nUse this tool for complex objectives to ensure that you are tracking each necessary step."

const WriteTodosToolDescription = "Use this tool to create and manage a structured task list for your current work session. This helps you track progress and organize complex tasks."

type TodoListMiddleware struct {
	SystemPrompt    string
	ToolDescription string
	Tools           []tools.Tool
}

func NewTodoListMiddleware() (*TodoListMiddleware, error) {
	m := &TodoListMiddleware{
		SystemPrompt:    WriteTodosSystemPrompt,
		ToolDescription: WriteTodosToolDescription,
	}
	tool, err := NewWriteTodosTool(m.ToolDescription)
	if err != nil {
		return nil, err
	}
	m.Tools = []tools.Tool{tool}
	return m, nil
}

// ProvidedTools implements ToolProvider: the write_todos tool is registered
// with the agent's ToolNode automatically (factory.py:1005 collects
// AgentMiddleware.tools; the Go method name differs because the public Tools
// field must stay and Go forbids a field/method name collision).
func (m *TodoListMiddleware) ProvidedTools() []tools.Tool {
	return m.Tools
}

// StateSchema implements StateSchemaContributor: the middleware's
// PlanningState declares the "todos" key (todo.py:35-42, a NotRequired
// list[Todo] with Python's default LastValue semantics), so CreateAgent
// registers it as a state field alongside the caller's state_schema
// (factory.py:1150-1156).
func (m *TodoListMiddleware) StateSchema() []StateField {
	return []StateField{{Name: "todos"}}
}

// NewWriteTodosTool builds the write_todos tool. Mirroring Python's
// write_todos (todo.py:139-149), which returns
// Command(update={"todos": todos, "messages": [ToolMessage(...)]}), the Go
// tool signals the todos write via a *types.Command in Result.Artifact —
// the Go ToolNode derives the ToolMessage from Result.Content (the port's
// documented convention, langchain/tools/tool_node.go), so the Command's
// Update carries only "todos"; create_agent's tools node commits it to
// PlanningState.todos in the same superstep.
func NewWriteTodosTool(description string) (tools.Tool, error) {
	return tools.NewFunc(
		WriteTodosToolName,
		description,
		schema.Object(map[string]schema.Schema{
			"todos": {
				"type": "array",
				"items": schema.Object(map[string]schema.Schema{
					"content": schema.String(""),
					"status":  schema.String(""),
				}, "content", "status"),
			},
		}, "todos"),
		func(_ context.Context, input map[string]any) (tools.Result, error) {
			todos := todosFromInput(input["todos"])
			return tools.Result{
				Content: fmt.Sprintf("Updated todo list to %v", input["todos"]),
				Artifact: &types.Command{
					Update: map[string]any{"todos": todos},
				},
			}, nil
		},
	)
}

// todosFromInput normalizes the raw decoded "todos" argument (a []any of
// map[string]any) into []Todo. Values that do not match the expected shape
// are coerced field-by-field (missing keys become zero values), matching the
// lenient dict → TypedDict coercion Python performs on tool args.
func todosFromInput(raw any) []Todo {
	items, _ := raw.([]any)
	out := make([]Todo, 0, len(items))
	for _, item := range items {
		var todo Todo
		if m, ok := item.(map[string]any); ok {
			todo.Content, _ = m["content"].(string)
			if status, ok := m["status"].(string); ok {
				todo.Status = TodoStatus(status)
			}
		}
		out = append(out, todo)
	}
	return out
}

func (m *TodoListMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	systemPrompt := m.SystemPrompt
	if systemPrompt == "" {
		systemPrompt = WriteTodosSystemPrompt
	}
	var system messages.Message
	if request.SystemMessage != nil {
		system = *request.SystemMessage
		system.ContentBlocks = append([]messages.ContentBlock(nil), system.ContentBlocks...)
		system.ContentBlocks = append(system.ContentBlocks, messages.TextBlock{Text: "\n\n" + systemPrompt})
	} else {
		system = messages.System("")
		system.ContentBlocks = []messages.ContentBlock{messages.TextBlock{Text: systemPrompt}}
	}
	next, err := request.Override(WithSystemMessage(&system))
	if err != nil {
		return ModelResponse{}, err
	}
	return handler(ctx, next)
}

func (m *TodoListMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	msgs, ok := messagesFromState(state)
	if !ok || len(msgs) == 0 {
		return nil, nil
	}
	last := lastAIMessage(state)
	if last == nil || len(last.ToolCalls) == 0 {
		return nil, nil
	}
	calls := []messages.ToolCall{}
	for _, call := range last.ToolCalls {
		if call.Name == WriteTodosToolName {
			calls = append(calls, call)
		}
	}
	if len(calls) <= 1 {
		return nil, nil
	}
	errMsgs := make([]messages.Message, 0, len(calls))
	for _, call := range calls {
		errMsgs = append(errMsgs, errorToolMessage(call.ID, WriteTodosToolName, "Error: The `write_todos` tool should never be called multiple times in parallel. Please call it only once per model invocation to update the todo list."))
	}
	return map[string]any{"messages": errMsgs}, nil
}
