package middleware

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

const DefaultToolSelectionSystemPrompt = "Your goal is to select the most relevant tools for answering the user's query."

type ToolSelectionRequest struct {
	AvailableTools  []tools.Tool
	SystemMessage   string
	LastUserMessage messages.Message
	Model           any
	ValidToolNames  []string
}

type ToolSelectionFunc func(ToolSelectionRequest) ([]string, error)

// LLMToolSelectorMiddleware asks an LLM to narrow the tool list before each
// model call. Selection resolution precedence:
//
//  1. If the resolved Model (m.Model, else request.Model) satisfies
//     language.ChatModel, the structured-output path (language.InvokeStructured)
//     is used — the ToolSelectionFunc callback is IGNORED even when set. This
//     is the spec-mandated primary path ("structured-primary, callback-fallback").
//  2. Else if Select is set, the callback is used (the retained fallback).
//  3. Else a typed error is returned.
//
// When BOTH a real ChatModel and a callback are configured, the structured
// path takes precedence and the callback is ignored — this is the
// spec-mandated behavior ("callback retained as fallback").
type LLMToolSelectorMiddleware struct {
	Model         any
	SystemPrompt  string
	MaxTools      *int
	AlwaysInclude []string
	Select        ToolSelectionFunc
}

type LLMToolSelectorOption func(*LLMToolSelectorMiddleware)

func NewLLMToolSelectorMiddleware(opts ...LLMToolSelectorOption) *LLMToolSelectorMiddleware {
	m := &LLMToolSelectorMiddleware{
		SystemPrompt: DefaultToolSelectionSystemPrompt,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

func WithToolSelectorModel(model any) LLMToolSelectorOption {
	return func(m *LLMToolSelectorMiddleware) {
		m.Model = model
	}
}

func WithToolSelectorSystemPrompt(prompt string) LLMToolSelectorOption {
	return func(m *LLMToolSelectorMiddleware) {
		m.SystemPrompt = prompt
	}
}

func WithToolSelectorMaxTools(maxTools int) LLMToolSelectorOption {
	return func(m *LLMToolSelectorMiddleware) {
		m.MaxTools = &maxTools
	}
}

func WithToolSelectorAlwaysInclude(names ...string) LLMToolSelectorOption {
	return func(m *LLMToolSelectorMiddleware) {
		m.AlwaysInclude = append([]string(nil), names...)
	}
}

func WithToolSelectorFunc(selectFunc ToolSelectionFunc) LLMToolSelectorOption {
	return func(m *LLMToolSelectorMiddleware) {
		m.Select = selectFunc
	}
}

func (m *LLMToolSelectorMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	selectionRequest, err := m.prepareSelectionRequest(request)
	if err != nil {
		return ModelResponse{}, err
	}
	if selectionRequest == nil {
		return handler(ctx, request)
	}

	// Resolve the structured-primary / callback-fallback precedence.
	var selectedNames []string
	if chatModel, ok := selectionRequest.Model.(language.ChatModel); ok {
		selectedNames, err = m.selectViaStructured(ctx, chatModel, selectionRequest)
		if err != nil {
			return ModelResponse{}, err
		}
	} else if m.Select != nil {
		selectedNames, err = m.Select(*selectionRequest)
		if err != nil {
			return ModelResponse{}, err
		}
	} else {
		return ModelResponse{}, fmt.Errorf("tool selection requires a selection function")
	}

	next, err := m.processSelectionResponse(selectedNames, selectionRequest.AvailableTools, selectionRequest.ValidToolNames, request)
	if err != nil {
		return ModelResponse{}, err
	}
	return handler(ctx, next)
}

// selectViaStructured is the spec-mandated primary path: route through
// language.InvokeStructured with a schema that constrains the model to the
// valid tool names. The decoded []string is fed into processSelectionResponse
// so MaxTools / AlwaysInclude / dedup / invalid-name detection stay identical
// across both paths.
func (m *LLMToolSelectorMiddleware) selectViaStructured(
	ctx context.Context,
	chatModel language.ChatModel,
	req *ToolSelectionRequest,
) ([]string, error) {
	sch := schema.Object(map[string]schema.Schema{
		"tools": {
			"type":  "array",
			"items": schema.Schema{"type": "string", "enum": req.ValidToolNames},
		},
	}, "tools")

	input := []messages.Message{
		messages.System(req.SystemMessage),
		req.LastUserMessage,
	}

	response, err := language.InvokeStructured(ctx, chatModel, input, sch)
	if err != nil {
		return nil, fmt.Errorf("tool selection: structured output: %w", err)
	}

	var data map[string]any
	if err := json.Unmarshal([]byte(messages.Text(response)), &data); err != nil {
		return nil, fmt.Errorf("tool selection: parse structured response: %w", err)
	}
	rawTools, ok := data["tools"]
	if !ok {
		return nil, fmt.Errorf("tool selection: parse structured response: %w", fmt.Errorf("missing %q key", "tools"))
	}
	arr, ok := rawTools.([]any)
	if !ok {
		return nil, fmt.Errorf("tool selection: parse structured response: %w", fmt.Errorf("%q is not an array", "tools"))
	}
	names := make([]string, 0, len(arr))
	for i, item := range arr {
		name, ok := item.(string)
		if !ok {
			return nil, fmt.Errorf("tool selection: parse structured response: %w", fmt.Errorf("tools[%d] is not a string", i))
		}
		names = append(names, name)
	}
	return names, nil
}

func (m *LLMToolSelectorMiddleware) prepareSelectionRequest(request ModelRequest) (*ToolSelectionRequest, error) {
	if len(request.Tools) == 0 {
		return nil, nil
	}

	baseTools := []tools.Tool{}
	for _, entry := range request.Tools {
		if tool, ok := entry.(tools.Tool); ok {
			baseTools = append(baseTools, tool)
		}
	}

	if len(m.AlwaysInclude) > 0 {
		availableNames := map[string]bool{}
		for _, tool := range baseTools {
			availableNames[tool.Name()] = true
		}
		missing := []string{}
		for _, name := range m.AlwaysInclude {
			if !availableNames[name] {
				missing = append(missing, name)
			}
		}
		if len(missing) > 0 {
			sort.Strings(missing)
			names := make([]string, 0, len(availableNames))
			for name := range availableNames {
				names = append(names, name)
			}
			sort.Strings(names)
			return nil, fmt.Errorf("tools in always_include not found in request: %v. Available tools: %v", missing, names)
		}
	}

	always := map[string]bool{}
	for _, name := range m.AlwaysInclude {
		always[name] = true
	}
	availableTools := []tools.Tool{}
	for _, tool := range baseTools {
		if !always[tool.Name()] {
			availableTools = append(availableTools, tool)
		}
	}
	if len(availableTools) == 0 {
		return nil, nil
	}

	systemMessage := m.SystemPrompt
	if m.MaxTools != nil {
		systemMessage += fmt.Sprintf("\nIMPORTANT: List the tool names in order of relevance, with the most relevant first. If you exceed the maximum number of tools, only the first %d will be used.", *m.MaxTools)
	}

	lastUser, ok := lastHumanMessage(request.Messages)
	if !ok {
		return nil, fmt.Errorf("no user message found in request messages")
	}

	model := m.Model
	if model == nil {
		model = request.Model
	}
	validNames := make([]string, len(availableTools))
	for i, tool := range availableTools {
		validNames[i] = tool.Name()
	}

	return &ToolSelectionRequest{
		AvailableTools:  availableTools,
		SystemMessage:   systemMessage,
		LastUserMessage: lastUser,
		Model:           model,
		ValidToolNames:  validNames,
	}, nil
}

func (m *LLMToolSelectorMiddleware) processSelectionResponse(selectedNames []string, availableTools []tools.Tool, validToolNames []string, request ModelRequest) (ModelRequest, error) {
	valid := map[string]bool{}
	for _, name := range validToolNames {
		valid[name] = true
	}

	selected := []string{}
	seen := map[string]bool{}
	invalid := []string{}
	for _, name := range selectedNames {
		if !valid[name] {
			invalid = append(invalid, name)
			continue
		}
		if seen[name] {
			continue
		}
		if m.MaxTools != nil && len(selected) >= *m.MaxTools {
			continue
		}
		selected = append(selected, name)
		seen[name] = true
	}
	if len(invalid) > 0 {
		return ModelRequest{}, fmt.Errorf("model selected invalid tools: %v", invalid)
	}

	selectedSet := map[string]bool{}
	for _, name := range selected {
		selectedSet[name] = true
	}
	always := map[string]bool{}
	for _, name := range m.AlwaysInclude {
		always[name] = true
	}

	nextTools := []any{}
	for _, tool := range availableTools {
		if selectedSet[tool.Name()] {
			nextTools = append(nextTools, tool)
		}
	}
	for _, entry := range request.Tools {
		tool, ok := entry.(tools.Tool)
		if ok && always[tool.Name()] {
			nextTools = append(nextTools, tool)
			continue
		}
		if !ok {
			nextTools = append(nextTools, entry)
		}
	}
	return request.Override(WithTools(nextTools))
}

func lastHumanMessage(msgs []messages.Message) (messages.Message, bool) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleHuman {
			return msgs[i], true
		}
	}
	return messages.Message{}, false
}
