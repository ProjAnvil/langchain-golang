package utils

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// ParseGoogleDocstring extracts the function description and the per-argument
// descriptions from a Google-style docstring. It is a direct port of Python's
// langchain_core.utils.function_calling._parse_google_docstring: docstring
// blocks are split on blank lines, the "Args:" block is located, and argument
// lines are parsed with indentation-aware continuation handling.
//
// The returned args map contains every argument described in the "Args:"
// section (keyed by argument name). err is reserved for malformed input to
// keep the signature consistent with the Python helper's validation mode; the
// exported wrapper never enables strict validation, so err is always nil.
func ParseGoogleDocstring(docstring string) (description string, args map[string]string, err error) {
	return largeUtilParseGoogleDocstring(docstring, nil, false)
}

// largeUtilParseGoogleDocstring is the shared implementation. argNames and
// errorOnInvalid mirror the Python helper's optional strict-validation mode:
// when enabled, a docstring that documents args but has no "Args:" block is
// rejected. argNames is only used for that validation (the returned map always
// describes every argument found in the docstring).
func largeUtilParseGoogleDocstring(docstring string, argNames []string, errorOnInvalid bool) (string, map[string]string, error) {
	argDescriptions := map[string]string{}
	var description string
	var argsBlock string

	if docstring != "" {
		blocks := strings.Split(docstring, "\n\n")
		if errorOnInvalid {
			reserved := map[string]bool{"run_manager": true, "callbacks": true, "runtime": true, "return": true}
			filtered := 0
			for _, name := range argNames {
				if !reserved[name] {
					filtered++
				}
			}
			hasArgsBlock := false
			for _, block := range blocks[1:] {
				if strings.HasPrefix(block, "Args:") {
					hasArgsBlock = true
					break
				}
			}
			const minDocstringBlocks = 2
			if filtered > 0 && (len(blocks) < minDocstringBlocks || !hasArgsBlock) {
				return "", nil, fmt.Errorf("found invalid Google-Style docstring")
			}
		}

		descriptors := []string{}
		pastDescriptors := false
		for _, block := range blocks {
			if strings.HasPrefix(block, "Args:") {
				argsBlock = block
				break
			}
			if strings.HasPrefix(block, "Returns:") || strings.HasPrefix(block, "Example:") {
				pastDescriptors = true
			} else if !pastDescriptors {
				descriptors = append(descriptors, block)
			} else {
				continue
			}
		}
		description = strings.TrimSpace(strings.Join(descriptors, " "))
	} else if errorOnInvalid {
		return "", nil, fmt.Errorf("found invalid Google-Style docstring")
	}

	if argsBlock != "" {
		lines := strings.Split(argsBlock, "\n")
		if len(lines) > 1 {
			lines = lines[1:]
		} else {
			lines = nil
		}
		var arg string
		haveArg := false
		var argIndent *int
		for _, line := range lines {
			if strings.TrimSpace(line) == "" {
				continue
			}
			currentIndent := len(line) - len(strings.TrimLeftFunc(line, unicode.IsSpace))
			if argIndent == nil && strings.Contains(line, ":") {
				ind := currentIndent
				argIndent = &ind
			}
			isContinuation := argIndent != nil && currentIndent > *argIndent
			if haveArg && isContinuation {
				argDescriptions[arg] += " " + strings.TrimSpace(line)
			} else if strings.Contains(line, ":") {
				parts := strings.SplitN(line, ":", 2)
				name := strings.TrimSpace(parts[0])
				desc := strings.TrimSpace(parts[1])
				if idx := strings.Index(name, " "); idx >= 0 {
					argName := name[:idx]
					annotations := name[idx+1:]
					if strings.HasPrefix(annotations, "(") && strings.HasSuffix(annotations, ")") {
						name = argName
					}
				}
				arg = name
				haveArg = true
				argDescriptions[arg] = desc
			} else if arg != "" {
				argDescriptions[arg] += " " + strings.TrimSpace(line)
			}
		}
	}
	return description, argDescriptions, nil
}

// RemoveTitles recursively removes "title" fields from a JSON-schema-shaped
// map, mirroring Python's _rm_titles. A "title" whose value is a nested object
// and that sits directly under a "properties" key is preserved (and recursed
// into); all other "title" keys are dropped. Both map[string]any and
// schema.Schema values are handled so schemas built with the repo helpers work.
func RemoveTitles(input map[string]any) map[string]any {
	return largeUtilRemoveTitles(input, "")
}

func largeUtilRemoveTitles(kv map[string]any, prevKey string) map[string]any {
	out := make(map[string]any, len(kv))
	for k, v := range kv {
		if k == "title" {
			if child, ok := largeUtilToStringMap(v); ok && prevKey == "properties" {
				out[k] = largeUtilRemoveTitles(child, k)
			}
			continue
		}
		if child, ok := largeUtilToStringMap(v); ok {
			out[k] = largeUtilRemoveTitles(child, k)
		} else {
			out[k] = v
		}
	}
	return out
}

// largeUtilToStringMap coerces the map-shaped values used by JSON schemas into
// a plain map[string]any so the recursion works for both dynamic types.
func largeUtilToStringMap(v any) (map[string]any, bool) {
	switch m := v.(type) {
	case map[string]any:
		return m, true
	case schema.Schema:
		return map[string]any(m), true
	default:
		return nil, false
	}
}

// ToolCallExample describes a single tool call in an example. Name stands in
// for the Pydantic model class name used by Python's tool_example_to_messages,
// and Arguments is serialized to the JSON string placed in the tool call.
type ToolCallExample struct {
	Name      string
	Arguments any
}

// ToolExampleOptions holds the optional parts of ToolExampleToMessages.
type ToolExampleOptions struct {
	// ToolOutputs, when non-empty, must have one entry per tool call.
	ToolOutputs []string
	// AIResponse, when non-empty, appends a final AI message.
	AIResponse string
}

// ToolExampleToMessages converts a single example into the message sequence
// expected by tool-calling chat models: a HumanMessage with the input, an
// AIMessage carrying the tool calls (in AdditionalKwargs), one ToolMessage per
// call, and an optional final AIMessage. It is the Go port of Python's
// tool_example_to_messages; tool call ids are generated with UUID4.
func ToolExampleToMessages(input string, toolCalls []ToolCallExample, opts ...ToolExampleOptions) ([]messages.Message, error) {
	var options ToolExampleOptions
	if len(opts) > 0 {
		options = opts[0]
	}

	out := []messages.Message{messages.Human(input)}

	ids := make([]string, len(toolCalls))
	calls := make([]map[string]any, len(toolCalls))
	for i, tc := range toolCalls {
		id, err := UUID4()
		if err != nil {
			return nil, fmt.Errorf("tool example: generate tool call id: %w", err)
		}
		arguments, err := largeUtilMarshalArguments(tc.Arguments)
		if err != nil {
			return nil, err
		}
		ids[i] = id
		calls[i] = map[string]any{
			"id":   id,
			"type": "function",
			"function": map[string]any{
				"name":      tc.Name,
				"arguments": arguments,
			},
		}
	}

	ai := messages.AI("")
	ai.AdditionalKwargs = map[string]any{"tool_calls": calls}
	out = append(out, ai)

	outputs := options.ToolOutputs
	if len(outputs) > 0 && len(outputs) != len(calls) {
		return nil, fmt.Errorf(
			"the number of tool_outputs (%d) must match the number of tool_calls (%d). Got %d output(s) for %d tool call(s)",
			len(outputs), len(calls), len(outputs), len(calls),
		)
	}
	if len(outputs) == 0 {
		outputs = make([]string, len(calls))
		for i := range outputs {
			outputs[i] = "You have correctly called this tool."
		}
	}
	for i, output := range outputs {
		out = append(out, messages.Tool(ids[i], output))
	}

	if options.AIResponse != "" {
		out = append(out, messages.AI(options.AIResponse))
	}
	return out, nil
}

// largeUtilMarshalArguments renders tool call arguments as a JSON string. A
// string is used verbatim (it is already JSON); other values are JSON-encoded,
// and a nil value becomes an empty object.
func largeUtilMarshalArguments(v any) (string, error) {
	if s, ok := v.(string); ok {
		return s, nil
	}
	if v == nil {
		return "{}", nil
	}
	data, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("tool example: marshal arguments: %w", err)
	}
	return string(data), nil
}
