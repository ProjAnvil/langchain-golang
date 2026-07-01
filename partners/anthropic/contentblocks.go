package anthropic

import (
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

// formatHumanContent converts a human message into Anthropic content blocks.
// Structured content blocks take precedence over the plain-text Content field.
func formatHumanContent(message messages.Message) ([]contentBlock, error) {
	if len(message.ContentBlocks) > 0 {
		return formatContentBlocks(message.ContentBlocks)
	}
	return []contentBlock{{Type: "text", Text: message.Content}}, nil
}

// formatContentBlocks converts normalized LangChain content blocks into
// Anthropic Messages API content blocks. Standard data blocks (image/file/text)
// are translated; provider-native blocks (tool_use, thinking,
// redacted_thinking, ...) pass through with their cache_control preserved.
func formatContentBlocks(blocks []messages.ContentBlock) ([]contentBlock, error) {
	out := make([]contentBlock, 0, len(blocks))
	for _, block := range blocks {
		converted, err := formatContentBlock(block)
		if err != nil {
			return nil, err
		}
		out = append(out, converted)
	}
	return out, nil
}

func formatContentBlock(block messages.ContentBlock) (contentBlock, error) {
	cacheControl, _ := block["cache_control"].(map[string]any)
	switch blockType, _ := block["type"].(string); blockType {
	case "image":
		source, err := imageSource(block)
		if err != nil {
			return contentBlock{}, err
		}
		return contentBlock{Type: "image", Source: source, CacheControl: cacheControl}, nil
	case "file":
		source, err := documentSource(block)
		if err != nil {
			return contentBlock{}, err
		}
		return contentBlock{Type: "document", Source: source, CacheControl: cacheControl}, nil
	case "text-plain":
		return contentBlock{
			Type: "document",
			Source: map[string]any{
				"type":       "text",
				"media_type": stringValue(block["mime_type"], "text/plain"),
				"data":       stringValue(block["text"], ""),
			},
			CacheControl: cacheControl,
		}, nil
	case "text":
		text, _ := block["text"].(string)
		return contentBlock{Type: "text", Text: text, CacheControl: cacheControl}, nil
	default:
		return passthroughBlock(block), nil
	}
}

// imageSource builds the Anthropic "source" object for an image content block.
func imageSource(block messages.ContentBlock) (map[string]any, error) {
	if url, ok := block["url"].(string); ok && url != "" {
		if mediaType, data, isData := parseDataURI(url); isData {
			return map[string]any{"type": "base64", "media_type": mediaType, "data": data}, nil
		}
		return map[string]any{"type": "url", "url": url}, nil
	}
	switch sourceType, _ := block["source_type"].(string); sourceType {
	case "base64":
		if _, ok := block["base64"]; ok {
			return map[string]any{
				"type":       "base64",
				"media_type": block["mime_type"],
				"data":       stringValue(block["data"], block["base64"]),
			}, nil
		}
	case "id":
		return map[string]any{"type": "file", "file_id": stringValue(block["id"], block["file_id"])}, nil
	}
	if fileID, ok := block["file_id"].(string); ok && fileID != "" {
		return map[string]any{"type": "file", "file_id": fileID}, nil
	}
	return nil, fmt.Errorf("anthropic: image content block requires url, base64, or file_id")
}

// documentSource builds the Anthropic "source" object for a file/document block.
func documentSource(block messages.ContentBlock) (map[string]any, error) {
	if url, ok := block["url"].(string); ok && url != "" {
		if mediaType, data, isData := parseDataURI(url); isData {
			return map[string]any{"type": "base64", "media_type": mediaType, "data": data}, nil
		}
		return map[string]any{"type": "url", "url": url}, nil
	}
	switch sourceType, _ := block["source_type"].(string); sourceType {
	case "text":
		return map[string]any{
			"type":       "text",
			"media_type": stringValue(block["mime_type"], "text/plain"),
			"data":       stringValue(block["text"], ""),
		}, nil
	case "id":
		return map[string]any{"type": "file", "file_id": stringValue(block["id"], block["file_id"])}, nil
	case "base64":
		return map[string]any{
			"type":       "base64",
			"media_type": stringValue(block["mime_type"], "application/pdf"),
			"data":       stringValue(block["data"], block["base64"]),
		}, nil
	}
	if _, ok := block["base64"]; ok {
		return map[string]any{
			"type":       "base64",
			"media_type": stringValue(block["mime_type"], "application/pdf"),
			"data":       stringValue(block["data"], block["base64"]),
		}, nil
	}
	if fileID, ok := block["file_id"].(string); ok && fileID != "" {
		return map[string]any{"type": "file", "file_id": fileID}, nil
	}
	return nil, fmt.Errorf("anthropic: file content block requires url, base64, text, or file_id")
}

// parseDataURI decodes a "data:<media_type>;base64,<data>" URI used for inline
// image/file payloads. Non-base64 data URIs are not supported.
func parseDataURI(uri string) (mediaType, data string, ok bool) {
	if !strings.HasPrefix(uri, "data:") {
		return "", "", false
	}
	rest := uri[len("data:"):]
	comma := strings.Index(rest, ",")
	if comma < 0 {
		return "", "", false
	}
	meta := rest[:comma]
	data = rest[comma+1:]
	mediaType = "text/plain"
	base64 := false
	for _, part := range strings.Split(meta, ";") {
		switch {
		case part == "base64":
			base64 = true
		case part == "":
		default:
			mediaType = part
		}
	}
	if !base64 {
		return "", "", false
	}
	return mediaType, data, true
}

// passthroughBlock forwards provider-native blocks (tool_use, thinking,
// redacted_thinking, ...) without reinterpreting their fields.
func passthroughBlock(block messages.ContentBlock) contentBlock {
	out := contentBlock{}
	for key, value := range block {
		setContentBlockField(&out, key, value)
	}
	return out
}

func setContentBlockField(out *contentBlock, key string, value any) {
	s, isString := value.(string)
	m, isMap := value.(map[string]any)
	switch key {
	case "type":
		if isString {
			out.Type = s
		}
	case "text":
		if isString {
			out.Text = s
		}
	case "id":
		if isString {
			out.ID = s
		}
	case "name":
		if isString {
			out.Name = s
		}
	case "input":
		if isMap {
			out.Input = m
		}
	case "tool_use_id":
		if isString {
			out.ToolUseID = s
		}
	case "thinking":
		if isString {
			out.Thinking = s
		}
	case "signature":
		if isString {
			out.Signature = s
		}
	case "data":
		if isString {
			out.Data = s
		}
	case "cache_control":
		if isMap {
			out.CacheControl = m
		}
	}
}

// stringValue returns v when it is a non-empty string, otherwise fallback.
func stringValue(v any, fallback any) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	if s, ok := fallback.(string); ok {
		return s
	}
	return ""
}
