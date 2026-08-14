package messages

import (
	"encoding/json"
	"fmt"
)

// This file mirrors langchain_core/messages/utils.py's convert_to_messages and
// convert_to_openai_messages. The Go port reduces Python's MessageLikeRepresentation
// union (BaseMessage | str | dict | tuple) to the three shapes the Message struct
// can express: an existing Message, a raw string (human shorthand), or a
// map[string]any carrying "role" and "content".

// ConvertToMessages converts a sequence of message-like values into a slice of
// Messages. Each element may be:
//
//   - a Message (passed through unchanged);
//   - a string (wrapped as a human-role Message whose Content is that string); or
//   - a map[string]any with "role" and "content" keys, where "content" is either a
//     string or a list of content-block maps parsed via ParseContentBlock. String
//     elements inside a content list are treated as text blocks.
//
// It mirrors Python's convert_to_messages and raises a typed error (the Go analog
// of Python's ValueError) on unsupported inputs.
func ConvertToMessages(inputs []any) ([]Message, error) {
	out := make([]Message, 0, len(inputs))
	for i, input := range inputs {
		msg, err := convertToMessage(input)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", i, err)
		}
		out = append(out, msg)
	}
	return out, nil
}

func convertToMessage(input any) (Message, error) {
	switch value := input.(type) {
	case Message:
		return value, nil
	case *Message:
		if value == nil {
			return Message{}, fmt.Errorf("unsupported message type <nil>: expected Message, string, or map[string]any")
		}
		return *value, nil
	case string:
		return Human(value), nil
	case map[string]any:
		return messageFromDict(value)
	default:
		return Message{}, fmt.Errorf("unsupported message type %T: expected Message, string, or map[string]any", input)
	}
}

// messageFromDict builds a Message from a {"role", "content"} map. It mirrors
// Python's _create_message_from_message_type: "role" is preferred over "type",
// and content may be a string or a list of block maps (or strings, which become
// text blocks).
func messageFromDict(m map[string]any) (Message, error) {
	roleValue, ok := m["role"]
	if !ok {
		roleValue, ok = m["type"]
	}
	if !ok {
		return Message{}, fmt.Errorf("message dict must contain 'role' and 'content' keys, got %v", m)
	}
	roleStr, ok := roleValue.(string)
	if !ok {
		return Message{}, fmt.Errorf("message 'role' must be a string, got %T", roleValue)
	}
	role, err := roleFromString(roleStr)
	if err != nil {
		return Message{}, err
	}

	contentValue, ok := m["content"]
	if !ok {
		return Message{}, fmt.Errorf("message dict must contain 'role' and 'content' keys, got %v", m)
	}

	msg := Message{Role: role}
	switch content := contentValue.(type) {
	case nil:
		// Python treats a missing/None content as the empty string.
		msg.Content = ""
	case string:
		msg.Content = content
	case []any:
		blocks, err := parseContentBlocks(content)
		if err != nil {
			return Message{}, err
		}
		msg.ContentBlocks = blocks
	default:
		return Message{}, fmt.Errorf("unsupported message content type %T", contentValue)
	}

	if name, ok := m["name"].(string); ok {
		msg.Name = name
	}
	if id, ok := m["id"].(string); ok {
		msg.ID = id
	}
	if toolCallID, ok := m["tool_call_id"].(string); ok {
		msg.ToolCallID = toolCallID
	}
	if toolCalls, ok := m["tool_calls"].([]any); ok {
		msg.ToolCalls = parseToolCalls(toolCalls)
	}
	if responseMetadata, ok := m["response_metadata"].(map[string]any); ok {
		msg.ResponseMetadata = responseMetadata
	}
	if additionalKwargs, ok := m["additional_kwargs"].(map[string]any); ok {
		msg.AdditionalKwargs = additionalKwargs
	}
	return msg, nil
}

// roleFromString maps a wire role string to a Role. It accepts the same aliases
// Python's _create_message_from_message_type does for the roles the Go Message
// models (system/human/ai/tool). Python's "function" and "remove" message types
// have no Role counterpart in this port and are rejected.
func roleFromString(role string) (Role, error) {
	switch role {
	case "human", "user":
		return RoleHuman, nil
	case "ai", "assistant":
		return RoleAI, nil
	case "system", "developer":
		return RoleSystem, nil
	case "tool":
		return RoleTool, nil
	default:
		return "", fmt.Errorf(
			"unexpected message type %q: use one of 'human', 'user', 'ai', 'assistant', 'system', 'developer', or 'tool'",
			role,
		)
	}
}

func parseContentBlocks(values []any) ([]ContentBlock, error) {
	blocks := make([]ContentBlock, 0, len(values))
	for i, value := range values {
		switch block := value.(type) {
		case string:
			blocks = append(blocks, TextBlock{Text: block})
		case map[string]any:
			blocks = append(blocks, ParseContentBlock(block))
		default:
			return nil, fmt.Errorf("unsupported content block type %T at index %d", value, i)
		}
	}
	return blocks, nil
}

// parseToolCalls converts a raw tool_calls list into []ToolCall, accepting both
// the OpenAI wire shape ({"id", "type": "function", "function": {"name", "arguments"}})
// and the LangChain shape ({"id", "name", "args"}). Invalid elements are skipped.
func parseToolCalls(values []any) []ToolCall {
	out := make([]ToolCall, 0, len(values))
	for _, value := range values {
		tcMap, ok := value.(map[string]any)
		if !ok {
			continue
		}
		if fn, ok := tcMap["function"].(map[string]any); ok {
			// OpenAI format: parse the JSON-encoded arguments string.
			args := fn["arguments"]
			if argsStr, ok := args.(string); ok {
				args = parseJSONArgs(argsStr)
			}
			name, _ := fn["name"].(string)
			id, _ := tcMap["id"].(string)
			out = append(out, ToolCall{ID: id, Name: name, Args: argsAsMap(args)})
			continue
		}
		name, _ := tcMap["name"].(string)
		id, _ := tcMap["id"].(string)
		args, _ := tcMap["args"].(map[string]any)
		out = append(out, ToolCall{ID: id, Name: name, Args: args})
	}
	return out
}

// parseJSONArgs decodes an OpenAI-format arguments string. On failure it returns
// the raw string so no data is silently dropped.
func parseJSONArgs(raw string) any {
	var decoded map[string]any
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		return raw
	}
	return decoded
}

func argsAsMap(value any) map[string]any {
	if m, ok := value.(map[string]any); ok {
		return m
	}
	return nil
}

// ConvertToOpenAIMessages converts a slice of Messages into OpenAI-format dicts
// with {"role", "content"} keys. It mirrors Python's convert_to_openai_messages:
//
//   - RoleSystem -> "system", RoleHuman -> "user", RoleAI -> "assistant",
//     RoleTool -> "tool";
//   - content is the message's string content, or a list of block maps produced by
//     BlockToMap when the message carries structured ContentBlocks;
//   - "name" and "tool_call_id" are included when set;
//   - an AI message with ToolCalls emits the OpenAI tool_calls shape
//     {"type": "function", "id", "function": {"name", "arguments"}}.
//
// Unlike Python's helper, this port does not collapse all-text block content into
// a single joined string (the text_format="string" behavior); block content is
// always emitted as a list of block maps.
func ConvertToOpenAIMessages(messages []Message) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(messages))
	for i, message := range messages {
		role, err := openAIRole(message.Role)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", i, err)
		}
		oai := map[string]any{"role": role}
		if message.Name != "" {
			oai["name"] = message.Name
		}
		if message.ToolCallID != "" {
			oai["tool_call_id"] = message.ToolCallID
		}
		if message.Role == RoleAI && len(message.ToolCalls) > 0 {
			oai["tool_calls"] = openAIToolCalls(message.ToolCalls)
		}
		oai["content"] = openAIContent(message)
		out = append(out, oai)
	}
	return out, nil
}

// openAIRole maps a Role to its OpenAI wire string. It mirrors Python's
// _get_message_openai_role for the roles this port models.
func openAIRole(role Role) (string, error) {
	switch role {
	case RoleSystem:
		return "system", nil
	case RoleHuman:
		return "user", nil
	case RoleAI:
		return "assistant", nil
	case RoleTool:
		return "tool", nil
	default:
		return "", fmt.Errorf("unknown message role %q", role)
	}
}

// openAIContent returns the OpenAI content value: the message's string content,
// or a list of block maps (via BlockToMap) when structured blocks are present.
func openAIContent(message Message) any {
	if len(message.ContentBlocks) > 0 {
		blocks := make([]map[string]any, len(message.ContentBlocks))
		for i, block := range message.ContentBlocks {
			blocks[i] = BlockToMap(block)
		}
		return blocks
	}
	return message.Content
}

// openAIToolCalls converts ToolCalls to OpenAI's tool_calls shape, mirroring
// Python's _convert_to_openai_tool_calls (arguments JSON-encoded with
// ensure_ascii=False, which Go's json encoding matches natively).
func openAIToolCalls(calls []ToolCall) []map[string]any {
	out := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		out = append(out, map[string]any{
			"type": "function",
			"id":   call.ID,
			"function": map[string]any{
				"name":      call.Name,
				"arguments": toolCallArgsJSON(call.Args),
			},
		})
	}
	return out
}
