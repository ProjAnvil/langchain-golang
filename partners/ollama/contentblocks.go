package ollama

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

// newToolCallID generates a UUID v4 string. Ollama does not return tool-call
// identifiers, so we synthesize them to match the Python ChatOllama behavior.
func newToolCallID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// toOllamaMessage converts a normalized LangChain message to an Ollama chat
// message. Text content blocks are joined into the content string; image blocks
// are collected into images; AI tool calls map to OpenAI-style tool_calls.
func toOllamaMessage(message messages.Message) chatMessage {
	role := "user"
	switch message.Role {
	case messages.RoleSystem:
		role = "system"
	case messages.RoleHuman:
		role = "user"
	case messages.RoleAI:
		role = "assistant"
	case messages.RoleTool:
		role = "tool"
	}

	content, images := extractContent(message)

	out := chatMessage{
		Role:    role,
		Content: content,
		Images:  images,
	}
	if len(images) == 0 {
		out.Images = nil
	}

	if message.Role == messages.RoleAI && len(message.ToolCalls) > 0 {
		out.ToolCalls = make([]chatMessageTool, 0, len(message.ToolCalls))
		for _, call := range message.ToolCalls {
			out.ToolCalls = append(out.ToolCalls, chatMessageTool{
				Type: "function",
				ID:   call.ID,
				Function: chatMessageToolFn{
					Name:      call.Name,
					Arguments: call.Args,
				},
			})
		}
	}
	if message.Role == messages.RoleTool {
		out.ToolCallID = message.ToolCallID
	}
	if message.Role == messages.RoleAI {
		if thinking, ok := message.AdditionalKwargs["reasoning_content"].(string); ok && thinking != "" {
			out.Thinking = thinking
		}
	}
	return out
}

// extractContent flattens a message's string content and content blocks into a
// single content string plus a slice of base64 image payloads.
func extractContent(message messages.Message) (string, []string) {
	if len(message.ContentBlocks) == 0 {
		return message.Content, nil
	}

	var parts []string
	var images []string
	if message.Content != "" {
		parts = append(parts, message.Content)
	}
	for _, block := range message.ContentBlocks {
		switch block["type"] {
		case "text":
			if text, ok := block["text"].(string); ok && text != "" {
				parts = append(parts, text)
			}
		case "image", "image_url":
			if image := extractImage(block); image != "" {
				images = append(images, image)
			}
		case "tool_use":
			// AI tool-use blocks are represented via tool_calls, not content.
			continue
		}
	}
	return strings.Join(parts, "\n"), images
}

// extractImage returns the base64 payload for an image content block, handling
// data: URIs and both v0 (data/source_type) and v1 (base64) block shapes.
func extractImage(block messages.ContentBlock) string {
	if base64, ok := block["base64"].(string); ok && base64 != "" {
		return stripDataURL(base64)
	}
	if data, ok := block["data"].(string); ok && data != "" {
		return stripDataURL(data)
	}
	source, _ := block["source"].(map[string]any)
	if source != nil {
		if data, ok := source["data"].(string); ok && data != "" {
			return stripDataURL(data)
		}
	}
	if imageURL, ok := block["image_url"].(map[string]any); ok {
		if url, ok := imageURL["url"].(string); ok && url != "" {
			return stripDataURL(url)
		}
	}
	if url, ok := block["image_url"].(string); ok && url != "" {
		return stripDataURL(url)
	}
	if url, ok := block["url"].(string); ok && url != "" {
		return stripDataURL(url)
	}
	return ""
}

// stripDataURL returns the base64 payload portion of a data: URI, or the input
// unchanged if it is not a data URI.
func stripDataURL(value string) string {
	if strings.HasPrefix(value, "data:") {
		if comma := strings.Index(value, ","); comma >= 0 {
			return value[comma+1:]
		}
	}
	return value
}

// parseToolCalls converts Ollama response tool_calls into normalized tool calls.
// Ollama tool-call arguments may be a JSON object, a JSON string, or a Python
// literal; malformed arguments are recorded as invalid tool calls.
func parseToolCalls(raw []chatResponseTool) ([]messages.ToolCall, []messages.ToolCall) {
	if len(raw) == 0 {
		return nil, nil
	}
	var toolCalls []messages.ToolCall
	var invalidToolCalls []messages.ToolCall
	for _, entry := range raw {
		name := entry.Function.Name
		args, ok := parseToolCallArguments(entry.Function.Arguments, name)
		call := messages.ToolCall{
			ID:   newToolCallID(),
			Name: name,
			Args: args,
		}
		if ok {
			toolCalls = append(toolCalls, call)
		} else {
			invalidToolCalls = append(invalidToolCalls, call)
		}
	}
	return toolCalls, invalidToolCalls
}

// parseToolCallArguments normalizes Ollama tool-call arguments into a map. It
// accepts maps, JSON strings, and shallowly-nested string-encoded JSON values
// (matching the Python band-aid for inconsistent argument structure). Returns
// (nil, true) for empty arguments and (raw, false) when unparseable.
func parseToolCallArguments(arguments any, name string) (map[string]any, bool) {
	switch value := arguments.(type) {
	case nil:
		return map[string]any{}, true
	case map[string]any:
		return normalizeArgumentMap(value, name), true
	case string:
		if strings.TrimSpace(value) == "" {
			return map[string]any{}, true
		}
		var parsed map[string]any
		if err := json.Unmarshal([]byte(value), &parsed); err == nil {
			return normalizeArgumentMap(parsed, name), true
		}
		return nil, false
	default:
		return nil, false
	}
}

// normalizeArgumentMap filters out Ollama metadata fields (e.g. a functionName
// key echoing the tool name) and parses shallowly-nested JSON-encoded string
// values, mirroring the Python _parse_arguments_from_tool_call behavior.
func normalizeArgumentMap(arguments map[string]any, name string) map[string]any {
	out := make(map[string]any, len(arguments))
	for key, value := range arguments {
		if key == "functionName" && value == name {
			continue
		}
		if strValue, ok := value.(string); ok {
			if parsed, inner := tryParseJSONValue(strValue); inner {
				out[key] = parsed
				continue
			}
		}
		out[key] = value
	}
	return out
}

// tryParseJSONValue parses a JSON object or array string, reporting whether the
// value was a parsable container.
func tryParseJSONValue(value string) (any, bool) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return value, false
	}
	if trimmed[0] != '{' && trimmed[0] != '[' {
		return value, false
	}
	var parsed any
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		return value, false
	}
	return parsed, true
}
