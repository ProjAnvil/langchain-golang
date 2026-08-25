package openai

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// CustomTool is an OpenAI Responses-API custom tool: a tool with a freeform
// string input, optionally constrained by a context-free grammar, mirroring
// Python's @custom_tool decorator (langchain_openai/tools/custom_tool.py).
// Bound on a ChatModel it serializes as
// {"type":"custom","name":...,"description":...,"format":{...}?}; the model's
// call arrives as a ToolCall with Args {"__arg1": input}
// (chat_models/base.py:4883-4891).
//
// Divergence: Python's decorator wraps the invoke result into a
// custom_tool_call_output content block; Go's tools.Result has no content
// blocks, so Invoke returns the plain string as Result.Content.
type CustomTool struct {
	name        string
	description string
	format      map[string]any
	fn          func(context.Context, string) (string, error)
}

// Compile-time guard: CustomTool satisfies the shared tool contract.
var _ tools.Tool = CustomTool{}

// CustomToolOption configures a CustomTool.
type CustomToolOption func(*CustomTool)

// WithCustomToolFormat constrains the tool input with a context-free grammar,
// e.g. {"type": "grammar", "syntax": "lark", "definition": "..."}.
func WithCustomToolFormat(format map[string]any) CustomToolOption {
	return func(t *CustomTool) { t.format = format }
}

// NewCustomTool creates an OpenAI custom tool.
func NewCustomTool(
	name, description string,
	fn func(context.Context, string) (string, error),
	opts ...CustomToolOption,
) (CustomTool, error) {
	if name == "" {
		return CustomTool{}, fmt.Errorf("tool name is required")
	}
	if fn == nil {
		return CustomTool{}, fmt.Errorf("tool function is required")
	}
	tool := CustomTool{name: name, description: description, fn: fn}
	for _, opt := range opts {
		opt(&tool)
	}
	return tool, nil
}

// Name returns the tool name.
func (t CustomTool) Name() string { return t.name }

// Description returns the tool description.
func (t CustomTool) Description() string { return t.description }

// ArgsSchema reports nil: custom tools take a freeform string and send no
// JSON parameters schema to OpenAI.
func (t CustomTool) ArgsSchema() schema.Schema { return nil }

// Invoke runs the tool with the freeform string from input["__arg1"] (the key
// custom_tool_call parsing produces), mirroring Python's single-argument
// custom tool invocation.
func (t CustomTool) Invoke(ctx context.Context, input map[string]any) (tools.Result, error) {
	text, ok := input["__arg1"].(string)
	if !ok {
		return tools.Result{}, fmt.Errorf("custom tool %s expects a string input under __arg1", t.name)
	}
	out, err := t.fn(ctx, text)
	if err != nil {
		return tools.Result{}, err
	}
	return tools.Result{Content: out}, nil
}
