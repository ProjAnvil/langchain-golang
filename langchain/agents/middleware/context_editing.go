package middleware

import (
	"context"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

const DefaultToolPlaceholder = "[cleared]"

type TokenCounter func([]messages.Message) int

type ContextEdit interface {
	Apply(messages []messages.Message, countTokens TokenCounter) []messages.Message
}

type ClearToolUsesEdit struct {
	Trigger         int
	ClearAtLeast    int
	Keep            int
	ClearToolInputs bool
	ExcludeTools    []string
	Placeholder     string
}

func NewClearToolUsesEdit() ClearToolUsesEdit {
	return ClearToolUsesEdit{
		Trigger:     100000,
		Keep:        3,
		Placeholder: DefaultToolPlaceholder,
	}
}

func (e ClearToolUsesEdit) Apply(input []messages.Message, countTokens TokenCounter) []messages.Message {
	out := cloneMessages(input)
	if e.Placeholder == "" {
		e.Placeholder = DefaultToolPlaceholder
	}
	if countTokens(out) <= e.Trigger {
		return out
	}

	candidates := []int{}
	for i, msg := range out {
		if msg.Role == messages.RoleTool {
			candidates = append(candidates, i)
		}
	}
	if e.Keep >= len(candidates) {
		return out
	}
	if e.Keep > 0 {
		candidates = candidates[:len(candidates)-e.Keep]
	}

	excluded := map[string]bool{}
	for _, name := range e.ExcludeTools {
		excluded[name] = true
	}
	originalTokens := countTokens(out)

	for _, idx := range candidates {
		toolMessage := out[idx]
		if contextEditingCleared(toolMessage) {
			continue
		}
		aiIdx, call, ok := findOriginatingToolCall(out[:idx], toolMessage.ToolCallID)
		if !ok {
			continue
		}
		toolName := toolMessage.Name
		if toolName == "" {
			toolName = call.Name
		}
		if excluded[toolName] {
			continue
		}

		toolMessage.Content = e.Placeholder
		toolMessage.ResponseMetadata = cloneAnyMap(toolMessage.ResponseMetadata)
		toolMessage.ResponseMetadata["context_editing"] = map[string]any{
			"cleared":  true,
			"strategy": "clear_tool_uses",
		}
		out[idx] = toolMessage

		if e.ClearToolInputs {
			out[aiIdx] = clearToolInput(out[aiIdx], toolMessage.ToolCallID)
		}

		if e.ClearAtLeast > 0 && originalTokens-countTokens(out) >= e.ClearAtLeast {
			break
		}
	}
	return out
}

type ContextEditingMiddleware struct {
	Edits       []ContextEdit
	CountTokens TokenCounter
	TokenMethod string
}

func NewContextEditingMiddleware(edits ...ContextEdit) *ContextEditingMiddleware {
	if len(edits) == 0 {
		defaultEdit := NewClearToolUsesEdit()
		edits = []ContextEdit{defaultEdit}
	}
	return &ContextEditingMiddleware{
		Edits:       edits,
		CountTokens: ApproximateTokenCount,
		TokenMethod: "approximate",
	}
}

func (m *ContextEditingMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	if len(request.Messages) == 0 {
		return handler(ctx, request)
	}
	counter := m.CountTokens
	if counter == nil {
		counter = ApproximateTokenCount
	}
	edited := cloneMessages(request.Messages)
	for _, edit := range m.Edits {
		edited = edit.Apply(edited, counter)
	}
	next, err := request.Override(WithMessages(edited))
	if err != nil {
		return ModelResponse{}, err
	}
	return handler(ctx, next)
}

func ApproximateTokenCount(msgs []messages.Message) int {
	total := 0
	for _, msg := range msgs {
		total += approximateTextTokens(msg.Content)
		for _, block := range msg.ContentBlocks {
			for _, value := range block {
				if text, ok := value.(string); ok {
					total += approximateTextTokens(text)
				}
			}
		}
		for _, call := range msg.ToolCalls {
			total += approximateTextTokens(call.Name)
			for _, value := range call.Args {
				if text, ok := value.(string); ok {
					total += approximateTextTokens(text)
				}
			}
		}
	}
	return total
}

func approximateTextTokens(text string) int {
	if text == "" {
		return 0
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return 1
	}
	return len(fields)
}

func contextEditingCleared(message messages.Message) bool {
	raw, ok := message.ResponseMetadata["context_editing"].(map[string]any)
	if !ok {
		return false
	}
	cleared, _ := raw["cleared"].(bool)
	return cleared
}

func findOriginatingToolCall(msgs []messages.Message, toolCallID string) (int, messages.ToolCall, bool) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != messages.RoleAI {
			continue
		}
		for _, call := range msgs[i].ToolCalls {
			if call.ID == toolCallID {
				return i, call, true
			}
		}
		return -1, messages.ToolCall{}, false
	}
	return -1, messages.ToolCall{}, false
}

func clearToolInput(message messages.Message, toolCallID string) messages.Message {
	message.ToolCalls = append([]messages.ToolCall(nil), message.ToolCalls...)
	cleared := false
	for i, call := range message.ToolCalls {
		if call.ID == toolCallID {
			message.ToolCalls[i].Args = map[string]any{}
			cleared = true
		}
	}
	if cleared {
		message.ResponseMetadata = cloneAnyMap(message.ResponseMetadata)
		entry, _ := message.ResponseMetadata["context_editing"].(map[string]any)
		entry = cloneAnyMap(entry)
		entry["cleared_tool_inputs"] = appendClearedID(entry["cleared_tool_inputs"], toolCallID)
		message.ResponseMetadata["context_editing"] = entry
	}
	return message
}

func appendClearedID(existing any, id string) []string {
	seen := map[string]bool{}
	out := []string{}
	if values, ok := existing.([]string); ok {
		for _, value := range values {
			if !seen[value] {
				seen[value] = true
				out = append(out, value)
			}
		}
	}
	if !seen[id] {
		out = append(out, id)
	}
	return out
}
