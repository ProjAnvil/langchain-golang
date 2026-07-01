package middleware

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
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
			return tools.Result{Content: fmt.Sprintf("Updated todo list to %v", input["todos"])}, nil
		},
	)
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
		system.ContentBlocks = append(system.ContentBlocks, messages.ContentBlock{"type": "text", "text": "\n\n" + systemPrompt})
	} else {
		system = messages.System("")
		system.ContentBlocks = []messages.ContentBlock{{"type": "text", "text": systemPrompt}}
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
