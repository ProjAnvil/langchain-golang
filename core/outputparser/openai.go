package outputparser

import (
	"encoding/json"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
)

// FunctionCall is an OpenAI-style function_call payload.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// OutputFunctionsParser extracts the raw function_call payload from a message.
type OutputFunctionsParser struct {
	ArgsOnly bool
}

// ParseMessage parses a function_call from message additional kwargs.
func (p OutputFunctionsParser) ParseMessage(message messages.Message) (any, error) {
	call, err := functionCallFromMessage(message)
	if err != nil {
		return nil, err
	}
	if p.ArgsOnly {
		return call.Arguments, nil
	}
	return call, nil
}

// JSONOutputFunctionsParser parses function_call.arguments as JSON.
type JSONOutputFunctionsParser struct {
	ArgsOnly bool
	Partial  bool
}

// ParseMessage parses function_call arguments as JSON.
func (p JSONOutputFunctionsParser) ParseMessage(message messages.Message) (any, error) {
	call, err := functionCallFromMessage(message)
	if err != nil {
		return nil, err
	}
	var args any
	if p.Partial {
		parsed, ok, err := ParsePartialJSON(call.Arguments)
		if err != nil {
			return nil, fmt.Errorf("parse partial function call arguments: %w", err)
		}
		if !ok {
			return nil, nil
		}
		args = parsed
	} else if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		return nil, fmt.Errorf("parse function call arguments: %w", err)
	}
	if p.ArgsOnly {
		return args, nil
	}
	return map[string]any{
		"name":      call.Name,
		"arguments": args,
	}, nil
}

// JSONKeyOutputFunctionsParser returns one key from JSON function arguments.
type JSONKeyOutputFunctionsParser struct {
	KeyName string
	Partial bool
}

// ParseMessage returns the configured key from function_call arguments.
func (p JSONKeyOutputFunctionsParser) ParseMessage(message messages.Message) (any, error) {
	value, err := (JSONOutputFunctionsParser{ArgsOnly: true, Partial: p.Partial}).ParseMessage(message)
	if err != nil {
		return nil, err
	}
	if p.Partial && value == nil {
		return nil, nil
	}
	object, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("function arguments are not a JSON object")
	}
	out, ok := object[p.KeyName]
	if !ok {
		if p.Partial {
			return nil, nil
		}
		return nil, fmt.Errorf("function arguments missing key %q", p.KeyName)
	}
	return out, nil
}

// ParsedToolCall is the OpenAI tools parser output shape. Type mirrors Python's
// backwards-compatible "type" field for the tool name.
type ParsedToolCall struct {
	ID   string         `json:"id,omitempty"`
	Type string         `json:"type"`
	Args map[string]any `json:"args"`
}

// JSONOutputToolsParser parses OpenAI-style tool calls from a message.
type JSONOutputToolsParser struct {
	ReturnID      bool
	FirstToolOnly bool
	Partial       bool
}

// ParseMessage parses all tool calls from message.ToolCalls or
// additional_kwargs["tool_calls"].
func (p JSONOutputToolsParser) ParseMessage(message messages.Message) (any, error) {
	calls, err := toolCallsFromMessage(message, toolParseOptions{ReturnID: p.ReturnID, Partial: p.Partial})
	if err != nil {
		return nil, err
	}
	if p.FirstToolOnly {
		if len(calls) == 0 {
			return nil, nil
		}
		return calls[0], nil
	}
	return calls, nil
}

// JSONOutputKeyToolsParser filters parsed tool calls by tool name.
type JSONOutputKeyToolsParser struct {
	KeyName       string
	ReturnID      bool
	FirstToolOnly bool
	Partial       bool
}

// ParseMessage parses and filters tool calls.
func (p JSONOutputKeyToolsParser) ParseMessage(message messages.Message) (any, error) {
	calls, err := toolCallsFromMessage(message, toolParseOptions{ReturnID: p.ReturnID, Partial: p.Partial})
	if err != nil {
		return nil, err
	}
	matched := make([]ParsedToolCall, 0, len(calls))
	for _, call := range calls {
		if call.Type == p.KeyName {
			matched = append(matched, call)
		}
	}
	if p.FirstToolOnly {
		if len(matched) == 0 {
			return nil, nil
		}
		if p.ReturnID {
			return matched[0], nil
		}
		return matched[0].Args, nil
	}
	if p.ReturnID {
		return matched, nil
	}
	args := make([]map[string]any, len(matched))
	for i, call := range matched {
		args[i] = call.Args
	}
	return args, nil
}

func functionCallFromMessage(message messages.Message) (FunctionCall, error) {
	raw, ok := message.AdditionalKwargs["function_call"]
	if !ok {
		return FunctionCall{}, fmt.Errorf("could not parse function call: missing function_call")
	}
	switch value := raw.(type) {
	case FunctionCall:
		return value, nil
	case map[string]any:
		args, _ := value["arguments"].(string)
		name, _ := value["name"].(string)
		if args == "" {
			return FunctionCall{}, fmt.Errorf("function_call missing arguments")
		}
		return FunctionCall{Name: name, Arguments: args}, nil
	default:
		data, err := json.Marshal(value)
		if err != nil {
			return FunctionCall{}, err
		}
		var call FunctionCall
		if err := json.Unmarshal(data, &call); err != nil {
			return FunctionCall{}, err
		}
		if call.Arguments == "" {
			return FunctionCall{}, fmt.Errorf("function_call missing arguments")
		}
		return call, nil
	}
}

type toolParseOptions struct {
	ReturnID bool
	Partial  bool
}

func toolCallsFromMessage(message messages.Message, opts toolParseOptions) ([]ParsedToolCall, error) {
	if len(message.ToolCalls) > 0 {
		out := make([]ParsedToolCall, len(message.ToolCalls))
		for i, call := range message.ToolCalls {
			out[i] = ParsedToolCall{Type: call.Name, Args: cloneArgs(call.Args)}
			if opts.ReturnID {
				out[i].ID = call.ID
			}
		}
		return out, nil
	}

	raw, ok := message.AdditionalKwargs["tool_calls"]
	if !ok {
		return nil, nil
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}
	var calls []rawToolCall
	if err := json.Unmarshal(data, &calls); err != nil {
		return nil, fmt.Errorf("parse raw tool calls: %w", err)
	}
	out := make([]ParsedToolCall, 0, len(calls))
	for _, call := range calls {
		if call.Function == nil {
			continue
		}
		args := map[string]any{}
		if call.Function.Arguments != "" {
			if opts.Partial {
				parsed, ok, err := ParsePartialJSON(call.Function.Arguments)
				if err != nil {
					return nil, fmt.Errorf("function %s partial arguments are not valid JSON: %w", call.Function.Name, err)
				}
				if !ok {
					continue
				}
				parsedArgs, ok := parsed.(map[string]any)
				if !ok {
					continue
				}
				args = parsedArgs
			} else if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
				return nil, fmt.Errorf("function %s arguments are not valid JSON: %w", call.Function.Name, err)
			}
		}
		parsed := ParsedToolCall{Type: call.Function.Name, Args: args}
		if opts.ReturnID {
			parsed.ID = call.ID
		}
		out = append(out, parsed)
	}
	return out, nil
}

type rawToolCall struct {
	ID       string           `json:"id,omitempty"`
	Function *rawToolFunction `json:"function,omitempty"`
}

type rawToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func cloneArgs(args map[string]any) map[string]any {
	if args == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(args))
	for key, value := range args {
		out[key] = value
	}
	return out
}
