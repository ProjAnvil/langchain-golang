package decisions

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

// This file converts caller-supplied state into the Decisions API `input`
// field, mirroring langchain_openai.decisions._input (PR #41125):
//
//   - strings pass through unchanged;
//   - a single HumanMessage or a sequence of HumanMessages becomes native
//     user messages (text and images as input parts);
//   - anything else — conversations with system/AI/tool messages, or JSON
//     structures containing messages — is serialized to JSON text in a
//     single user message, with base64 images lifted out of the JSON and
//     kept in place as input_image parts.
//
// Divergence: Go map iteration order is randomized, so object keys inside
// the JSON text are sorted (encoding/json behavior) rather than following
// the caller's construction order (Python dict insertion order). See
// DIVERGENCES.md #8.

// toDecisionInput converts state to a value suitable for the Decisions API
// input field (Python to_decision_input, _input.py:21).
func toDecisionInput(input any) (any, error) {
	if s, ok := input.(string); ok {
		return s, nil
	}
	if msg, ok := input.(messages.Message); ok && msg.Role == messages.RoleHuman {
		return humanToUserMessages([]messages.Message{msg})
	}
	if msgs, ok := input.([]messages.Message); ok && len(msgs) > 0 && allHuman(msgs) {
		return humanToUserMessages(msgs)
	}
	if anys, ok := input.([]any); ok && len(anys) > 0 && allHumanValues(anys) {
		return humanAnyMessagesToUserMessages(anys)
	}
	return toJSONUserMessage(input)
}

func allHuman(msgs []messages.Message) bool {
	for _, msg := range msgs {
		if msg.Role != messages.RoleHuman {
			return false
		}
	}
	return true
}

func allHumanValues(values []any) bool {
	for _, value := range values {
		msg, ok := value.(messages.Message)
		if !ok || msg.Role != messages.RoleHuman {
			return false
		}
	}
	return true
}

// humanToUserMessages converts human messages into native user messages
// (Python _to_user_message, _input.py:51): plain-text content stays a bare
// string; block content becomes input parts via messageToInputParts.
func humanToUserMessages(msgs []messages.Message) ([]any, error) {
	out := make([]any, 0, len(msgs))
	for _, msg := range msgs {
		user, err := humanToUserMessage(msg)
		if err != nil {
			return nil, err
		}
		out = append(out, user)
	}
	return out, nil
}

func humanAnyMessagesToUserMessages(values []any) ([]any, error) {
	out := make([]any, 0, len(values))
	for _, value := range values {
		msg, _ := value.(messages.Message)
		user, err := humanToUserMessage(msg)
		if err != nil {
			return nil, err
		}
		out = append(out, user)
	}
	return out, nil
}

func humanToUserMessage(msg messages.Message) (map[string]any, error) {
	if len(msg.ContentBlocks) == 0 {
		return map[string]any{"role": "user", "content": msg.Content}, nil
	}
	parts, err := messageToInputParts(msg)
	if err != nil {
		return nil, err
	}
	content := make([]any, len(parts))
	for i, part := range parts {
		content[i] = part
	}
	return map[string]any{"role": "user", "content": content}, nil
}

// toJSONUserMessage serializes state the API cannot accept natively into a
// single-element list holding one user message (Python returns
// [_to_json_user_message(state)], _input.py:48). Base64 images in any
// message are replaced by lc-image-<uuid>-<n> tokens before marshaling,
// then re-expanded as input_image parts at their original positions so the
// wire format stays the same whether or not the state contains images.
func toJSONUserMessage(state any) ([]any, error) {
	var images []string
	token := "lc-image-" + randomToken()
	value, err := toJSONValue(state, &images, token)
	if err != nil {
		return nil, err
	}
	text, err := marshalJSONText(value)
	if err != nil {
		return nil, fmt.Errorf("%w: could not serialize OpenAI Decisions state: %v", lcerrors.ErrInvalidInput, err)
	}
	content := []any{}
	position := 0
	pattern := regexp.MustCompile(regexp.QuoteMeta(token) + `-(\d+)`)
	for _, match := range pattern.FindAllStringSubmatchIndex(text, -1) {
		content = append(content, map[string]any{"type": "input_text", "text": text[position:match[0]]})
		index, err := strconv.Atoi(text[match[2]:match[3]])
		if err != nil || index >= len(images) {
			return nil, fmt.Errorf("%w: invalid image token in serialized OpenAI Decisions state", lcerrors.ErrInvalidInput)
		}
		content = append(content, map[string]any{"type": "input_image", "image_url": images[index]})
		position = match[1]
	}
	content = append(content, map[string]any{"type": "input_text", "text": text[position:]})
	return []any{map[string]any{"role": "user", "content": content}}, nil
}

// toJSONValue recursively converts state to JSON-safe values, replacing
// base64 image blocks with tokens along the way (Python _to_json_value and
// _with_image_tokens, _input.py:98-128). Messages become one-element arrays
// holding their OpenAI role/content dict; maps require string keys; scalars
// pass through; anything else is unsupported.
func toJSONValue(value any, images *[]string, token string) (any, error) {
	switch v := value.(type) {
	case nil:
		return nil, nil
	case string:
		return v, nil
	case messages.Message:
		return messageWithImageTokens(v, images, token)
	}
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Bool, reflect.String,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return value, nil
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return nil, fmt.Errorf(
				"%w: OpenAI Decisions state object keys must be strings",
				lcerrors.ErrInvalidInput,
			)
		}
		out := make(map[string]any, rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			converted, err := toJSONValue(iter.Value().Interface(), images, token)
			if err != nil {
				return nil, err
			}
			out[iter.Key().String()] = converted
		}
		return out, nil
	case reflect.Slice, reflect.Array:
		out := make([]any, rv.Len())
		for i := range rv.Len() {
			converted, err := toJSONValue(rv.Index(i).Interface(), images, token)
			if err != nil {
				return nil, err
			}
			out[i] = converted
		}
		return out, nil
	default:
		return nil, fmt.Errorf(
			"%w: Unsupported OpenAI Decisions state value: %T",
			lcerrors.ErrInvalidInput, value,
		)
	}
}

// messageWithImageTokens converts one message into a one-element array
// holding its OpenAI dict, with base64 image blocks replaced by tokens
// (Python returns convert_to_openai_messages(...) — a list — from
// _to_json_value, _input.py:100).
func messageWithImageTokens(msg messages.Message, images *[]string, token string) ([]any, error) {
	dict, err := messageToOpenAIDict(msg, images, token)
	if err != nil {
		return nil, err
	}
	return []any{dict}, nil
}

// messageToOpenAIDict builds the OpenAI {"role","content"} dict for a
// message. Base64 images become token text blocks; hosted-URL images stay
// in the JSON as image_url blocks (matching Python's
// convert_to_openai_messages output); audio/video/file blocks are rejected
// with ErrInvalidInput (parity wave 3 plan: the JSON path errors on media
// the API does not accept).
func messageToOpenAIDict(msg messages.Message, images *[]string, token string) (map[string]any, error) {
	role, err := openAIRole(msg.Role)
	if err != nil {
		return nil, err
	}
	dict := map[string]any{"role": role}
	if msg.Name != "" {
		dict["name"] = msg.Name
	}
	if msg.ToolCallID != "" {
		dict["tool_call_id"] = msg.ToolCallID
	}
	if msg.Role == messages.RoleAI && len(msg.ToolCalls) > 0 {
		dict["tool_calls"] = openAIToolCallDicts(msg.ToolCalls)
	}
	if len(msg.ContentBlocks) == 0 {
		dict["content"] = msg.Content
		return dict, nil
	}
	content := make([]any, 0, len(msg.ContentBlocks))
	for _, block := range msg.ContentBlocks {
		switch b := block.(type) {
		case messages.TextBlock:
			content = append(content, map[string]any{"type": "text", "text": b.Text})
		case messages.ImageBlock:
			switch {
			case IsBase64ImageBlock(block):
				url, err := imageSourceURL(b)
				if err != nil {
					return nil, err
				}
				content = append(content, map[string]any{
					"type": "text",
					"text": fmt.Sprintf("%s-%d", token, len(*images)),
				})
				*images = append(*images, url)
			case b.URL != "":
				content = append(content, map[string]any{
					"type":      "image_url",
					"image_url": map[string]any{"url": b.URL},
				})
			default:
				return nil, unsupportedBlockError("image")
			}
		case messages.AudioBlock, messages.VideoBlock, messages.FileBlock:
			return nil, unsupportedBlockError(block.BlockType())
		default:
			content = append(content, messages.BlockToMap(block))
		}
	}
	dict["content"] = content
	return dict, nil
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
		return "", fmt.Errorf("%w: unknown message role %q", lcerrors.ErrInvalidInput, role)
	}
}

// openAIToolCallDicts renders tool calls in the OpenAI wire shape
// ({"type":"function","id",...,"function":{"name","arguments":<json>}}),
// matching core/messages' unexported openAIToolCalls.
func openAIToolCallDicts(calls []messages.ToolCall) []map[string]any {
	out := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		out = append(out, map[string]any{
			"type": "function",
			"id":   call.ID,
			"function": map[string]any{
				"name":      call.Name,
				"arguments": toolArgsJSON(call.Args),
			},
		})
	}
	return out
}

func toolArgsJSON(args map[string]any) string {
	if args == nil {
		return "{}"
	}
	data, err := marshalJSONText(args)
	if err != nil {
		return "{}"
	}
	return data
}

// marshalJSONText marshals with HTML escaping disabled, matching Python's
// json.dumps(..., ensure_ascii=False).
func marshalJSONText(value any) (string, error) {
	var buf strings.Builder
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return "", err
	}
	return strings.TrimSuffix(buf.String(), "\n"), nil
}

// randomToken returns a 32-hex-char random token (Python uuid4().hex).
func randomToken() string {
	b := make([]byte, 16)
	if n, err := rand.Read(b); err != nil || n != len(b) {
		// crypto/rand never fails on supported platforms; fall back to a
		// time-derived token rather than panicking inside serialization.
		return strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return hex.EncodeToString(b)
}
