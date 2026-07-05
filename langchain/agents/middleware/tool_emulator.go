package middleware

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

type ToolEmulationFunc func(ToolCallRequest, string) (string, error)

// LLMToolEmulator intercepts tool calls for tools in its emulation set and
// produces a synthetic tool-result message instead of executing the real tool.
// Emulation resolution precedence:
//
//  1. If the configured Model satisfies language.ChatModel, the
//     structured-output path (language.InvokeStructured) is used — the
//     ToolEmulationFunc callback is IGNORED even when set. This is the
//     spec-mandated primary path ("structured-primary, callback-fallback").
//  2. Else if Emulate is set, the callback is used (the retained fallback).
//  3. Else a typed error is returned.
//
// When BOTH a real ChatModel and a callback are configured, the structured
// path takes precedence and the callback is ignored — this is the
// spec-mandated behavior ("callback retained as fallback").
//
// ToolCallRequest has no model field, so the ChatModel must be supplied
// explicitly via WithToolEmulatorModel. WithToolEmulatorFunc is retained for
// the callback fallback path.
type LLMToolEmulator struct {
	Model          any
	EmulateAll     bool
	ToolsToEmulate map[string]bool
	Emulate        ToolEmulationFunc
}

// LLMToolEmulatorOption configures an LLMToolEmulator.
type LLMToolEmulatorOption func(*LLMToolEmulator)

// NewLLMToolEmulator builds an emulator that intercepts calls to toolNames
// (or every tool when toolNames is nil). The variadic options shape mirrors
// NewLLMToolSelectorMiddleware; pass WithToolEmulatorFunc to install the
// callback fallback and/or WithToolEmulatorModel to install a real ChatModel.
func NewLLMToolEmulator(toolNames []string, opts ...LLMToolEmulatorOption) *LLMToolEmulator {
	m := &LLMToolEmulator{
		EmulateAll:     toolNames == nil,
		ToolsToEmulate: map[string]bool{},
	}
	for _, name := range toolNames {
		m.ToolsToEmulate[name] = true
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// WithToolEmulatorModel installs a language.ChatModel used for the
// structured-output primary path. The value is type-asserted at call time, so
// callers may pass any ChatModel implementation without the middleware package
// importing concrete provider types.
func WithToolEmulatorModel(model any) LLMToolEmulatorOption {
	return func(m *LLMToolEmulator) {
		m.Model = model
	}
}

// WithToolEmulatorFunc installs the ToolEmulationFunc callback used as the
// fallback when no real ChatModel is configured.
func WithToolEmulatorFunc(emulate ToolEmulationFunc) LLMToolEmulatorOption {
	return func(m *LLMToolEmulator) {
		m.Emulate = emulate
	}
}

func (m *LLMToolEmulator) WrapToolCall(ctx context.Context, request ToolCallRequest, handler ToolHandler) (messages.Message, error) {
	toolName := request.ToolCall.Name
	if !m.EmulateAll && !m.ToolsToEmulate[toolName] {
		return handler(ctx, request)
	}

	// Resolve the structured-primary / callback-fallback precedence. The
	// ChatModel type assertion gates the structured path; existing call sites
	// that supply no Model stay on the callback/pass-through path unchanged.
	if chatModel, ok := m.Model.(language.ChatModel); ok {
		return m.emulateViaStructured(ctx, chatModel, request, toolName)
	}
	if m.Emulate != nil {
		prompt := BuildToolEmulationPrompt(request)
		content, err := m.Emulate(request, prompt)
		if err != nil {
			return messages.Message{}, err
		}
		msg := messages.Tool(request.ToolCall.ID, content)
		msg.Name = toolName
		return msg, nil
	}
	return messages.Message{}, fmt.Errorf("tool emulator requires an emulation function")
}

// emulateViaStructured is the spec-mandated primary path: route through
// language.InvokeStructured with a schema describing a single emulated-output
// string, then unwrap the decoded output into a tool-result message.
func (m *LLMToolEmulator) emulateViaStructured(
	ctx context.Context,
	chatModel language.ChatModel,
	request ToolCallRequest,
	toolName string,
) (messages.Message, error) {
	sch := schema.Object(map[string]schema.Schema{
		"output": schema.String("The emulated tool output"),
	}, "output")

	prompt := BuildToolEmulationPrompt(request)
	input := []messages.Message{messages.Human(prompt)}

	response, err := language.InvokeStructured(ctx, chatModel, input, sch)
	if err != nil {
		return messages.Message{}, fmt.Errorf("tool emulator: structured output: %w", err)
	}

	var data map[string]any
	if err := json.Unmarshal([]byte(messages.Text(response)), &data); err != nil {
		return messages.Message{}, fmt.Errorf("tool emulator: parse structured response: %w", err)
	}
	rawOutput, ok := data["output"]
	if !ok {
		return messages.Message{}, fmt.Errorf("tool emulator: parse structured response: %w", fmt.Errorf("missing %q key", "output"))
	}
	content, ok := rawOutput.(string)
	if !ok {
		return messages.Message{}, fmt.Errorf("tool emulator: parse structured response: %w", fmt.Errorf("%q is not a string", "output"))
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
