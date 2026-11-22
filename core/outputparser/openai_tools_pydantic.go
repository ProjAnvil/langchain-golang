package outputparser

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// ParseToolCallOptions configures ParseToolCall and ParseToolCalls. It mirrors
// the keyword options of Python's parse_tool_call / parse_tool_calls.
type ParseToolCallOptions struct {
	// Partial parses best-effort partial JSON via ParsePartialJSON.
	Partial bool
	// Strict rejects non-JSON-compliant literal control characters inside
	// strings. When false, such characters are tolerated (Python's default).
	Strict bool
	// ReturnID copies the raw "id" onto the parsed tool call.
	ReturnID bool
}

// ParseToolCall parses a single raw OpenAI-style tool call. It returns (nil,
// nil) when the call has no "function" payload or when partial parsing cannot
// recover an object. Invalid JSON in non-partial mode returns an
// *lcerrors.OutputParserException. It mirrors Python's parse_tool_call.
func ParseToolCall(rawToolCall map[string]any, opts ParseToolCallOptions) (*ParsedToolCall, error) {
	function, ok := rawToolCall["function"].(map[string]any)
	if !ok {
		return nil, nil
	}
	name, _ := function["name"].(string)
	arguments, _ := function["arguments"].(string)

	var args map[string]any
	switch {
	case opts.Partial:
		parsed, ok, err := ParsePartialJSON(arguments)
		if err != nil || !ok {
			return nil, nil
		}
		object, ok := parsed.(map[string]any)
		if !ok {
			return nil, nil
		}
		args = object
	case arguments == "":
		// Parameter-less tools: Python maps empty/None arguments to {}.
		args = map[string]any{}
	default:
		decoded, err := decodeToolArguments(arguments, opts.Strict)
		if err != nil {
			msg := fmt.Sprintf(
				"Function %s arguments:\n\n%s\n\nare not valid JSON. Received error: %v",
				name, arguments, err,
			)
			return nil, &lcerrors.OutputParserException{
				Message:   msg,
				LLMOutput: arguments,
				Err:       err,
			}
		}
		args = decoded
	}

	call := &ParsedToolCall{Type: name, Args: args}
	if opts.ReturnID {
		call.ID, _ = rawToolCall["id"].(string)
	}
	return call, nil
}

// MakeInvalidToolCall builds an InvalidToolCallBlock from a raw tool call,
// mirroring Python's make_invalid_tool_call. An empty errorMsg maps to no
// error (the zero Error string).
func MakeInvalidToolCall(rawToolCall map[string]any, errorMsg string) messages.InvalidToolCallBlock {
	function, _ := rawToolCall["function"].(map[string]any)
	name, _ := function["name"].(string)
	args, _ := function["arguments"].(string)
	id, _ := rawToolCall["id"].(string)
	return messages.InvalidToolCallBlock{
		Name:  name,
		Args:  args,
		ID:    id,
		Error: errorMsg,
	}
}

// ParseToolCalls parses a list of raw tool calls, aggregating the messages of
// every invalid call into a single OutputParserException (joined by a blank
// line). It mirrors Python's parse_tool_calls.
func ParseToolCalls(rawToolCalls []map[string]any, opts ParseToolCallOptions) ([]ParsedToolCall, error) {
	final := make([]ParsedToolCall, 0, len(rawToolCalls))
	exceptions := []string{}
	for _, raw := range rawToolCalls {
		call, err := ParseToolCall(raw, opts)
		if err != nil {
			exceptions = append(exceptions, err.Error())
			continue
		}
		if call != nil {
			final = append(final, *call)
		}
	}
	if len(exceptions) > 0 {
		return nil, &lcerrors.OutputParserException{
			Message: strings.Join(exceptions, "\n\n"),
		}
	}
	return final, nil
}

// decodeToolArguments decodes a tool-call argument string into an object.
// Non-object JSON values map to an empty object (matching Python's
// "args or {}" for falsy values); only genuine decode failures are errors.
func decodeToolArguments(arguments string, strict bool) (map[string]any, error) {
	text := arguments
	if !strict {
		text = relaxJSONControlChars(text)
	}
	var raw any
	if err := json.Unmarshal([]byte(text), &raw); err != nil {
		return nil, err
	}
	object, ok := raw.(map[string]any)
	if !ok {
		return map[string]any{}, nil
	}
	return object, nil
}

// relaxJSONControlChars escapes literal control characters (0x00-0x1F) that
// appear inside JSON string literals, which is what json.loads(strict=False)
// tolerates. Control characters outside strings are left untouched (they are
// either legal whitespace or still invalid JSON).
func relaxJSONControlChars(text string) string {
	var b strings.Builder
	inString := false
	escaped := false
	for i := 0; i < len(text); i++ {
		c := text[i]
		if inString {
			if escaped {
				b.WriteByte(c)
				escaped = false
				continue
			}
			switch c {
			case '\\':
				b.WriteByte(c)
				escaped = true
			case '"':
				b.WriteByte(c)
				inString = false
			default:
				if c < 0x20 {
					b.WriteString(fmt.Sprintf("\\u%04x", c))
				} else {
					b.WriteByte(c)
				}
			}
			continue
		}
		b.WriteByte(c)
		if c == '"' {
			inString = true
		}
	}
	return b.String()
}

// PydanticToolsParser validates parsed OpenAI tool calls against a set of
// named JSON schemas and returns the validated objects. It mirrors Python's
// PydanticToolsParser, using toolCallsFromMessage for extraction and
// validateSchema for validation.
type PydanticToolsParser struct {
	// Tools maps tool names to the JSON schema their arguments must satisfy.
	Tools map[string]schema.Schema
	// ReturnID includes tool call IDs in extraction.
	ReturnID bool
	// FirstToolOnly returns a single object (or nil) instead of a slice.
	FirstToolOnly bool
	// Partial tolerates incomplete or not-yet-valid arguments by skipping them.
	Partial bool
}

// ParseMessage parses and validates all tool calls on a message.
func (p PydanticToolsParser) ParseMessage(message messages.Message) (any, error) {
	if err := RaiseIfRefused(message); err != nil {
		return nil, err
	}
	calls, err := toolCallsFromMessage(message, toolParseOptions{
		ReturnID: p.ReturnID,
		Partial:  p.Partial,
	})
	if err != nil {
		return nil, err
	}
	objects := make([]map[string]any, 0, len(calls))
	for _, call := range calls {
		spec, ok := p.Tools[call.Type]
		if !ok {
			return nil, &lcerrors.OutputParserException{
				Message: fmt.Sprintf(
					"Unknown tool type: %q. Available tools: %s",
					call.Type, availableToolNames(p.Tools),
				),
			}
		}
		var raw any = call.Args
		if err := validateSchema(raw, spec, "$"); err != nil {
			if p.Partial {
				continue
			}
			return nil, &lcerrors.OutputParserException{
				Message: fmt.Sprintf("Tool %q arguments failed validation: %v", call.Type, err),
				Err:     err,
			}
		}
		objects = append(objects, cloneArgs(call.Args))
	}
	if p.FirstToolOnly {
		if len(objects) == 0 {
			return nil, nil
		}
		return objects[0], nil
	}
	return objects, nil
}

func availableToolNames(tools map[string]schema.Schema) string {
	names := make([]string, 0, len(tools))
	for name := range tools {
		names = append(names, name)
	}
	sort.Strings(names)
	if len(names) == 0 {
		return "<no_tools>"
	}
	return strings.Join(names, ", ")
}
