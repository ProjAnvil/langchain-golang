package typesafe

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Verbatim Python error messages from langchain_typesafe/_state.py. They keep
// their trailing punctuation to match the upstream text exactly.
var (
	errStateRootScalar = errors.New( //nolint:staticcheck // ST1005: verbatim Python text (_state.py:47-50)
		"TypeSafe state must be a string, object, array, BaseMessage, or sequence of BaseMessage objects.")
	errStateObjectKeys = errors.New( //nolint:staticcheck // ST1005: verbatim Python text (_state.py:20)
		"TypeSafe state object keys must be strings.")
)

// unsupportedStateValueError mirrors `f"Unsupported TypeSafe state value:
// {type(value).__name__}."` (_state.py:27-28), rendering the Go type name.
func unsupportedStateValueError(value any) error {
	return fmt.Errorf( //nolint:staticcheck // ST1005: verbatim Python text (_state.py:27)
		"Unsupported TypeSafe state value: %s.", goTypeName(value))
}

// This file mirrors langchain_typesafe/_state.py (#40542). The TypeSafe
// state may be a string, a JSON object, or a JSON array; LangChain messages
// can appear at the root or at any depth inside objects and arrays and are
// converted to OpenAI role/content JSON while the surrounding structure is
// preserved.
//
// The message mapping is implemented locally rather than reusing
// core/messages.ConvertToOpenAIMessages: that helper emits LangChain-shaped
// image blocks ({"type":"image",...}) rather than the OpenAI image_url form
// Python's convert_to_openai_messages produces (spec decision #7,
// DIVERGENCES #13). The shape follows partners/openai/contentblocks.go's
// established mapping (mime default image/jpeg for base64 images, audio as
// input_audio), implemented independently to keep the partner
// self-contained.

// SerializeState recursively converts LangChain messages inside TypeSafe
// state to JSON. It returns a string, object, or array suitable for the
// TypeSafe "state" field.
//
// It errors when the root is a JSON scalar other than a string, when an
// object carries non-string keys, or when a value cannot be represented as
// JSON (error text mirroring _state.py).
func SerializeState(state any) (any, error) {
	if state == nil {
		return nil, errStateRootScalar
	}
	switch k := reflect.ValueOf(state).Kind(); k {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return nil, errStateRootScalar
	}
	return serializeStateValue(state)
}

// serializeStateValue mirrors _serialize_state_value (_state.py:15-28).
func serializeStateValue(value any) (any, error) {
	switch v := value.(type) {
	case messages.Message:
		return messageToOpenAIDict(v)
	case *messages.Message:
		if v == nil {
			return nil, unsupportedStateValueError(value)
		}
		return messageToOpenAIDict(*v)
	case nil, string, bool, int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64, float32, float64:
		return value, nil
	case []byte:
		// Python excludes bytes/bytearray from the Sequence branch
		// (_state.py:23), so raw byte slices are unsupported values.
		return nil, unsupportedStateValueError(value)
	}

	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Map:
		if reflected.Type().Key().Kind() != reflect.String {
			return nil, errStateObjectKeys
		}
		out := make(map[string]any, reflected.Len())
		for _, key := range reflected.MapKeys() {
			serialized, err := serializeStateValue(reflected.MapIndex(key).Interface())
			if err != nil {
				return nil, err
			}
			out[key.String()] = serialized
		}
		return out, nil
	case reflect.Slice, reflect.Array:
		out := make([]any, reflected.Len())
		for i := range reflected.Len() {
			serialized, err := serializeStateValue(reflected.Index(i).Interface())
			if err != nil {
				return nil, err
			}
			out[i] = serialized
		}
		return out, nil
	default:
		return nil, unsupportedStateValueError(value)
	}
}

// goTypeName renders the value's type the way Python's type(x).__name__ does:
// the bare name without the package qualifier.
func goTypeName(value any) string {
	name := reflect.TypeOf(value).String()
	if idx := strings.LastIndex(name, "."); idx >= 0 {
		name = name[idx+1:]
	}
	return name
}

// messageToOpenAIDict maps one message onto the OpenAI role/content dict
// shape Python's convert_to_openai_messages produces for a single message:
// role always; name and tool_call_id when set; tool_calls on assistant
// messages in the OpenAI function shape; IDs omitted. Text-only block
// content collapses into a single newline-joined string; mixed content
// becomes a list of OpenAI parts (text / image_url / input_audio).
func messageToOpenAIDict(message messages.Message) (map[string]any, error) {
	role, err := openAIRole(message.Role)
	if err != nil {
		return nil, err
	}
	out := map[string]any{"role": role}
	if message.Name != "" {
		out["name"] = message.Name
	}
	if message.ToolCallID != "" {
		out["tool_call_id"] = message.ToolCallID
	}
	if message.Role == messages.RoleAI && len(message.ToolCalls) > 0 {
		out["tool_calls"] = openAIToolCalls(message.ToolCalls)
	}
	content, err := openAIContent(message)
	if err != nil {
		return nil, err
	}
	out["content"] = content
	return out, nil
}

func openAIRole(role messages.Role) (string, error) {
	switch role {
	case messages.RoleSystem:
		return "system", nil
	case messages.RoleHuman:
		return "user", nil
	case messages.RoleAI:
		return "assistant", nil
	case messages.RoleTool:
		return "tool", nil
	default:
		return "", fmt.Errorf("typesafe: unknown message role %q", string(role))
	}
}

// openAIContent mirrors convert_to_openai_messages' text_format="string"
// behavior: all-text block lists join into one string, anything else becomes
// a list of OpenAI-shaped parts.
func openAIContent(message messages.Message) (any, error) {
	if len(message.ContentBlocks) == 0 {
		return message.Content, nil
	}
	allText := true
	for _, block := range message.ContentBlocks {
		if _, ok := block.(messages.TextBlock); !ok {
			allText = false
			break
		}
	}
	if allText {
		parts := make([]string, 0, len(message.ContentBlocks))
		for _, block := range message.ContentBlocks {
			parts = append(parts, block.(messages.TextBlock).Text)
		}
		return strings.Join(parts, "\n"), nil
	}
	parts := make([]any, 0, len(message.ContentBlocks))
	for _, block := range message.ContentBlocks {
		converted, err := openAIBlock(block)
		if err != nil {
			return nil, err
		}
		parts = append(parts, converted)
	}
	return parts, nil
}

// openAIBlock converts one content block to its OpenAI Chat Completions
// shape (block_translators/openai.py): images become image_url parts with
// base64 data as a data: URI, audio becomes input_audio, and other blocks
// are rejected the way convert_to_openai_data_block rejects unsupported
// types.
func openAIBlock(block messages.ContentBlock) (any, error) {
	switch b := block.(type) {
	case messages.TextBlock:
		return map[string]any{"type": "text", "text": b.Text}, nil
	case messages.ImageBlock:
		url, err := imageURL(b)
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": url},
		}, nil
	case messages.AudioBlock:
		if b.Base64 == "" {
			return nil, fmt.Errorf("typesafe: audio content block requires base64 data")
		}
		return map[string]any{
			"type":        "input_audio",
			"input_audio": map[string]any{"data": b.Base64, "format": audioFormat(b)},
		}, nil
	case messages.NonStandardContentBlock:
		// Unknown block shapes pass through verbatim, mirroring
		// convert_to_openai_messages' pass_through_unknown_blocks=True.
		return b.Value, nil
	default:
		return nil, fmt.Errorf("typesafe: content block type %q is not supported in state messages", block.BlockType())
	}
}

// imageURL resolves an image block's source: the URL wins, then base64 data
// as a data: URI. The media type defaults to image/jpeg, matching the
// partners/openai precedent.
func imageURL(b messages.ImageBlock) (string, error) {
	if b.URL != "" {
		return b.URL, nil
	}
	if b.Base64 != "" {
		mime := b.MimeType
		if mime == "" {
			mime = "image/jpeg"
		}
		return "data:" + mime + ";base64," + b.Base64, nil
	}
	return "", fmt.Errorf("typesafe: image content block requires url or base64")
}

// audioFormat derives the input_audio format: an explicit "format" extra
// wins, then the suffix of mime_type ("audio/wav" -> "wav").
func audioFormat(b messages.AudioBlock) string {
	if format, ok := b.Extras["format"].(string); ok && format != "" {
		return format
	}
	if idx := strings.LastIndex(b.MimeType, "/"); idx >= 0 && idx+1 < len(b.MimeType) {
		return b.MimeType[idx+1:]
	}
	return b.MimeType
}

// openAIToolCalls converts tool calls to the OpenAI function shape with
// JSON-encoded arguments.
func openAIToolCalls(calls []messages.ToolCall) []any {
	out := make([]any, 0, len(calls))
	for _, call := range calls {
		out = append(out, map[string]any{
			"type": "function",
			"id":   call.ID,
			"function": map[string]any{
				"name":      call.Name,
				"arguments": argsJSON(call.Args),
			},
		})
	}
	return out
}

// argsJSON renders tool-call arguments as compact JSON with HTML characters
// unescaped, matching Python's json.dumps(..., ensure_ascii=False).
func argsJSON(args map[string]any) string {
	if args == nil {
		return "{}"
	}
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(args); err != nil {
		return "{}"
	}
	return strings.TrimSuffix(buf.String(), "\n")
}
