package middleware

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
)

type ToolEmulationFunc func(ToolCallRequest, string) (string, error)

type LLMToolEmulator struct {
	EmulateAll     bool
	ToolsToEmulate map[string]bool
	Emulate        ToolEmulationFunc
}

func NewLLMToolEmulator(toolNames []string, emulate ToolEmulationFunc) *LLMToolEmulator {
	m := &LLMToolEmulator{
		EmulateAll:     toolNames == nil,
		ToolsToEmulate: map[string]bool{},
		Emulate:        emulate,
	}
	for _, name := range toolNames {
		m.ToolsToEmulate[name] = true
	}
	return m
}

func (m *LLMToolEmulator) WrapToolCall(ctx context.Context, request ToolCallRequest, handler ToolHandler) (messages.Message, error) {
	toolName := request.ToolCall.Name
	if !m.EmulateAll && !m.ToolsToEmulate[toolName] {
		return handler(ctx, request)
	}
	if m.Emulate == nil {
		return messages.Message{}, fmt.Errorf("tool emulator requires an emulation function")
	}
	prompt := BuildToolEmulationPrompt(request)
	content, err := m.Emulate(request, prompt)
	if err != nil {
		return messages.Message{}, err
	}
	msg := messages.Tool(request.ToolCall.ID, content)
	msg.Name = toolName
	return msg, nil
}

func BuildToolEmulationPrompt(request ToolCallRequest) string {
	description := "No description available"
	if request.Tool != nil {
		description = request.Tool.Description()
	}
	return fmt.Sprintf("You are emulating a tool call for testing purposes.\n\nTool: %s\nDescription: %s\nArguments: %v\n\nGenerate a realistic response that this tool would return given these arguments.\nReturn ONLY the tool's output, no explanation or preamble. Introduce variation into your responses.", request.ToolCall.Name, description, request.ToolCall.Args)
}
