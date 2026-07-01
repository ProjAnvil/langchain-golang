package outputparser

import (
	"fmt"
	"sort"

	"github.com/projanvil/langchain-golang/core/messages"
)

// AnthropicToolCall is the normalized output of Anthropic tool-use parsing.
type AnthropicToolCall struct {
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name"`
	Args  map[string]any `json:"args"`
	Index int            `json:"index,omitempty"`
}

// AnthropicToolsOutputParser extracts Anthropic tool_use blocks from a message.
type AnthropicToolsOutputParser struct {
	FirstToolOnly bool
	ArgsOnly      bool
}

// ParseMessage parses tool calls from normalized ToolCalls or content blocks.
func (p AnthropicToolsOutputParser) ParseMessage(message messages.Message) (any, error) {
	calls, err := anthropicToolCallsFromMessage(message)
	if err != nil {
		return nil, err
	}
	if p.ArgsOnly {
		args := make([]map[string]any, len(calls))
		for i, call := range calls {
			args[i] = call.Args
		}
		if p.FirstToolOnly {
			if len(args) == 0 {
				return nil, nil
			}
			return args[0], nil
		}
		return args, nil
	}
	if p.FirstToolOnly {
		if len(calls) == 0 {
			return nil, nil
		}
		return calls[0], nil
	}
	return calls, nil
}

func anthropicToolCallsFromMessage(message messages.Message) ([]AnthropicToolCall, error) {
	if len(message.ToolCalls) > 0 {
		indexByID := anthropicToolUseIndexes(message.ContentBlocks)
		out := make([]AnthropicToolCall, len(message.ToolCalls))
		for i, call := range message.ToolCalls {
			out[i] = AnthropicToolCall{
				ID:    call.ID,
				Name:  call.Name,
				Args:  cloneArgs(call.Args),
				Index: indexByID[call.ID],
			}
		}
		return out, nil
	}

	out := []AnthropicToolCall{}
	for i, block := range message.ContentBlocks {
		blockType, _ := block["type"].(string)
		if blockType != "tool_use" {
			continue
		}
		name, _ := block["name"].(string)
		id, _ := block["id"].(string)
		if name == "" {
			return nil, fmt.Errorf("anthropic tool_use block missing name")
		}
		args, err := anthropicToolUseArgs(block)
		if err != nil {
			return nil, err
		}
		out = append(out, AnthropicToolCall{
			ID:    id,
			Name:  name,
			Args:  args,
			Index: i,
		})
	}
	return out, nil
}

func anthropicToolUseIndexes(blocks []messages.ContentBlock) map[string]int {
	out := map[string]int{}
	for i, block := range blocks {
		blockType, _ := block["type"].(string)
		id, _ := block["id"].(string)
		if blockType == "tool_use" && id != "" {
			out[id] = i
		}
	}
	return out
}

func anthropicToolUseArgs(block messages.ContentBlock) (map[string]any, error) {
	raw, ok := block["input"]
	if !ok || raw == nil {
		return map[string]any{}, nil
	}
	args, ok := raw.(map[string]any)
	if ok {
		return cloneArgs(args), nil
	}
	if typed, ok := raw.(messages.ContentBlock); ok {
		return cloneContentBlockArgs(typed), nil
	}
	return nil, fmt.Errorf("anthropic tool_use input must be an object")
}

func cloneContentBlockArgs(args messages.ContentBlock) map[string]any {
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make(map[string]any, len(args))
	for _, key := range keys {
		out[key] = args[key]
	}
	return out
}
