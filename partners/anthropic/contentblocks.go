package anthropic

import (
	"errors"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

// errSkipBlock marks a content block that has no wire representation and must
// be silently dropped by formatContentBlocks. It is internal: only
// formatContentBlocks handles it; every other formatContentBlock caller
// (formatSystemBlocks reaches it only for text blocks) never sees it, so
// per-block error behavior elsewhere is unchanged.
var errSkipBlock = errors.New("anthropic: block not sendable")

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
		if errors.Is(err, errSkipBlock) {
			continue
		}
		if err != nil {
			return nil, err
		}
		out = append(out, converted)
	}
	return out, nil
}

func formatContentBlock(block messages.ContentBlock) (contentBlock, error) {
	m := messages.BlockToMap(block)
	cacheControl, _ := m["cache_control"].(map[string]any)
	switch blockType, _ := m["type"].(string); blockType {
	case "image":
		source, err := imageSource(m)
		if err != nil {
			return contentBlock{}, err
		}
		return contentBlock{Type: "image", Source: source, CacheControl: cacheControl}, nil
	case "file":
		source, err := documentSource(m)
		if err != nil {
			return contentBlock{}, err
		}
		return contentBlock{Type: "document", Source: source, CacheControl: cacheControl}, nil
	case "text-plain":
		return contentBlock{
			Type: "document",
			Source: map[string]any{
				"type":       "text",
				"media_type": stringValue(m["mime_type"], "text/plain"),
				"data":       stringValue(m["text"], ""),
			},
			CacheControl: cacheControl,
		}, nil
	case "text":
		text, _ := m["text"].(string)
		return contentBlock{Type: "text", Text: text, CacheControl: cacheControl}, nil
	case "refusal":
		// The Messages API has no refusal input block (#41582, _compat.py:125-127).
		return contentBlock{}, errSkipBlock
	default:
		return passthroughBlock(m), nil
	}
}

// imageSource builds the Anthropic "source" object for an image content block.
// Matches Python's _format_image (chat_models.py): inline "data:" URLs must be
// base64 data URIs with an image/* media type; anything else that starts with
// "data:" is rejected with an explicit error (Python raises ValueError for
// malformed inputs) rather than being sent as a broken url source. Non-data
// URLs pass through as-is.
func imageSource(m map[string]any) (map[string]any, error) {
	if url, ok := m["url"].(string); ok && url != "" {
		if strings.HasPrefix(url, "data:") {
			mediaType, data, isData := parseDataURI(url)
			if !isData {
				return nil, fmt.Errorf("anthropic: only base64 data URIs are supported for image sources: %q", url)
			}
			if !strings.HasPrefix(mediaType, "image/") {
				return nil, fmt.Errorf("anthropic: image data URI media type must be image/*, got %q", mediaType)
			}
			return map[string]any{"type": "base64", "media_type": mediaType, "data": data}, nil
		}
		return map[string]any{"type": "url", "url": url}, nil
	}
	switch sourceType, _ := m["source_type"].(string); sourceType {
	case "base64":
		if _, ok := m["base64"]; ok {
			return map[string]any{
				"type":       "base64",
				"media_type": m["mime_type"],
				"data":       stringValue(m["data"], m["base64"]),
			}, nil
		}
	case "id":
		return map[string]any{"type": "file", "file_id": stringValue(m["id"], m["file_id"])}, nil
	}
	// Typed Go blocks (messages.ImageBlock serialized via BlockToMap) carry a
	// plain "base64" key without an explicit source_type — the standard
	// LangChain image-input shape Python's _format_image accepts by computing
	// source_type from the fields. Mirror documentSource's fallback so those
	// blocks serialize instead of erroring.
	if _, ok := m["base64"]; ok {
		return map[string]any{
			"type":       "base64",
			"media_type": m["mime_type"],
			"data":       stringValue(m["data"], m["base64"]),
		}, nil
	}
	if fileID, ok := m["file_id"].(string); ok && fileID != "" {
		return map[string]any{"type": "file", "file_id": fileID}, nil
	}
	return nil, fmt.Errorf("anthropic: image content block requires url, base64, or file_id")
}

// documentSource builds the Anthropic "source" object for a file/document block.
// Inline "data:" URLs must be base64 data URIs; non-base64 data URIs are
// rejected with an explicit error (Python raises ValueError for malformed
// inputs) instead of being sent as a broken url source. Unlike images, the
// media type is not validated here because Python does not validate document
// media types either. Non-data URLs pass through as-is.
func documentSource(m map[string]any) (map[string]any, error) {
	if url, ok := m["url"].(string); ok && url != "" {
		if strings.HasPrefix(url, "data:") {
			mediaType, data, isData := parseDataURI(url)
			if !isData {
				return nil, fmt.Errorf("anthropic: only base64 data URIs are supported for document sources: %q", url)
			}
			return map[string]any{"type": "base64", "media_type": mediaType, "data": data}, nil
		}
		return map[string]any{"type": "url", "url": url}, nil
	}
	switch sourceType, _ := m["source_type"].(string); sourceType {
	case "text":
		return map[string]any{
			"type":       "text",
			"media_type": stringValue(m["mime_type"], "text/plain"),
			"data":       stringValue(m["text"], ""),
		}, nil
	case "id":
		return map[string]any{"type": "file", "file_id": stringValue(m["id"], m["file_id"])}, nil
	case "base64":
		return map[string]any{
			"type":       "base64",
			"media_type": stringValue(m["mime_type"], "application/pdf"),
			"data":       stringValue(m["data"], m["base64"]),
		}, nil
	}
	if _, ok := m["base64"]; ok {
		return map[string]any{
			"type":       "base64",
			"media_type": stringValue(m["mime_type"], "application/pdf"),
			"data":       stringValue(m["data"], m["base64"]),
		}, nil
	}
	if fileID, ok := m["file_id"].(string); ok && fileID != "" {
		return map[string]any{"type": "file", "file_id": fileID}, nil
	}
	return nil, fmt.Errorf("anthropic: file content block requires url, base64, text, or file_id")
}

// parseDataURI decodes a "data:<media_type>;base64,<data>" URI used for inline
// image/file payloads. Non-base64 data URIs are not supported: callers
// (imageSource/documentSource) turn ok=false into an explicit error, matching
// the ValueError Python raises for malformed data URI inputs.
func parseDataURI(uri string) (mediaType, data string, ok bool) {
	if !strings.HasPrefix(uri, "data:") {
		return "", "", false
	}
	rest := uri[len("data:"):]
	meta, payload, found := strings.Cut(rest, ",")
	if !found {
		return "", "", false
	}
	data = payload
	mediaType = "text/plain"
	base64 := false
	for part := range strings.SplitSeq(meta, ";") {
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

// formatSystemBlocks narrows system message content to the blocks Anthropic
// accepts in a system message — Python's _format_system_content
// (chat_models.py:578-643), which serves every call site: the leading run
// (:742), a hoisted mid-conversation message (:759, :1052), and an in-place
// system turn. Text blocks go through the standard text formatting;
// tool-change blocks (tool_addition / tool_removal) are forwarded only when
// allowToolChanges, otherwise dropped with a warning; every other block type
// is dropped with an unrecognized-block warning rather than forwarded to an
// API rejection.
func formatSystemBlocks(blocks []messages.ContentBlock, allowToolChanges bool, model string) ([]contentBlock, error) {
	out := make([]contentBlock, 0, len(blocks))
	for _, block := range blocks {
		raw := messages.BlockToMap(block)
		blockType, _ := raw["type"].(string)
		switch {
		case blockType == "tool_addition" || blockType == "tool_removal":
			if !allowToolChanges {
				warnToolChangeBlockDropped(blockType, model)
				continue
			}
			out = append(out, passthroughBlock(raw))
		case blockType != "text":
			warnUnrecognizedSystemBlockDropped(blockType, model)
		default:
			converted, err := formatContentBlock(block)
			if err != nil {
				return nil, err
			}
			out = append(out, converted)
		}
	}
	return out, nil
}

// passthroughBlock forwards provider-native blocks (tool_use, thinking,
// redacted_thinking, ...) without reinterpreting their fields.
func passthroughBlock(m map[string]any) contentBlock {
	out := contentBlock{}
	for key, value := range m {
		setContentBlockField(&out, key, value)
	}
	// #41678: server-side tool blocks echoing a direct caller are
	// rejected by the API; client tool_use passes its caller verbatim
	// (chat_models.py:834-835 vs 853-855/965-967).
	if out.Caller != nil && out.Type != "tool_use" {
		if t, _ := out.Caller["type"].(string); t == "direct" {
			out.Caller = nil
		}
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
	case "tool":
		if isMap {
			out.Tool = m
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
	case "content":
		out.Content = value
	case "caller":
		// The caller gate depends on the block type, and map iteration is
		// unordered (Type may not be set yet), so the value is staged here and
		// resolved by passthroughBlock after the loop.
		if isMap {
			out.Caller = m
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
