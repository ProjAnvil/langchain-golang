package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/utils"
)

// Result is returned by a tool invocation.
type Result struct {
	Content  string         `json:"content,omitempty"`
	Artifact any            `json:"artifact,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Tool is the shared executable tool contract.
type Tool interface {
	Name() string
	Description() string
	ArgsSchema() schema.Schema
	Invoke(ctx context.Context, input map[string]any) (Result, error)
}

// Simple is a single-input tool matching Python's Tool behavior. It accepts
// exactly one string input and exposes the backwards-compatible tool_input
// argument schema.
type Simple struct {
	name        string
	description string
	fn          func(context.Context, string) (Result, error)
}

// Func is a tool backed by a Go function and an explicit schema.
type Func struct {
	name        string
	description string
	argsSchema  schema.Schema
	fn          func(context.Context, map[string]any) (Result, error)
}

// NewSimple creates a single-input tool.
func NewSimple(
	name string,
	description string,
	fn func(context.Context, string) (Result, error),
) (Simple, error) {
	if name == "" {
		return Simple{}, fmt.Errorf("tool name is required")
	}
	if fn == nil {
		return Simple{}, fmt.Errorf("tool function is required")
	}
	return Simple{name: name, description: description, fn: fn}, nil
}

// NewFunc creates a tool from an explicit schema and function.
func NewFunc(
	name string,
	description string,
	argsSchema schema.Schema,
	fn func(context.Context, map[string]any) (Result, error),
) (Func, error) {
	if name == "" {
		return Func{}, fmt.Errorf("tool name is required")
	}
	if fn == nil {
		return Func{}, fmt.Errorf("tool function is required")
	}
	if err := ValidateArgsSchema(argsSchema); err != nil {
		return Func{}, err
	}

	return Func{
		name:        name,
		description: description,
		argsSchema:  argsSchema,
		fn:          fn,
	}, nil
}

// NewStructuredFunc creates a multi-input structured tool. It is equivalent to
// NewFunc and exists to mirror Python's StructuredTool.from_function boundary.
func NewStructuredFunc(
	name string,
	description string,
	argsSchema schema.Schema,
	fn func(context.Context, map[string]any) (Result, error),
) (Func, error) {
	return NewFunc(name, description, argsSchema, fn)
}

// Name returns the tool name.
func (t Simple) Name() string {
	return t.name
}

// Description returns the tool description.
func (t Simple) Description() string {
	return t.description
}

// ArgsSchema returns the single-input tool schema.
func (t Simple) ArgsSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"tool_input": schema.String("tool input"),
	}, "tool_input")
}

// InvokeString executes the tool with a direct string input.
func (t Simple) InvokeString(ctx context.Context, input string) (Result, error) {
	if t.fn == nil {
		return Result{}, fmt.Errorf("tool function is required")
	}
	return t.fn(ctx, input)
}

// Invoke executes the single-input tool from a map. The map must contain
// exactly one string value. Prefer key "tool_input"; a single alternate key is
// accepted for compatibility with structured dispatchers.
func (t Simple) Invoke(ctx context.Context, input map[string]any) (Result, error) {
	if len(input) != 1 {
		values := make([]any, 0, len(input))
		for _, value := range input {
			values = append(values, value)
		}
		return Result{}, fmt.Errorf("too many arguments to single-input tool %s: %v", t.name, values)
	}
	value, ok := input["tool_input"]
	if !ok {
		for _, candidate := range input {
			value = candidate
		}
	}
	text, ok := value.(string)
	if !ok {
		return Result{}, fmt.Errorf("single-input tool %s expects a string input", t.name)
	}
	return t.InvokeString(ctx, text)
}

// Name returns the tool name.
func (t Func) Name() string {
	return t.name
}

// Description returns the tool description.
func (t Func) Description() string {
	return t.description
}

// ArgsSchema returns the tool argument schema.
func (t Func) ArgsSchema() schema.Schema {
	return t.argsSchema
}

// Invoke executes the tool.
func (t Func) Invoke(ctx context.Context, input map[string]any) (Result, error) {
	if t.fn == nil {
		return Result{}, fmt.Errorf("tool function is required")
	}
	return t.fn(ctx, input)
}

// ValidateArgsSchema checks the minimum JSON-schema shape expected by tool
// calling providers.
func ValidateArgsSchema(argsSchema schema.Schema) error {
	if argsSchema == nil {
		return nil
	}
	if typ, ok := argsSchema["type"].(string); ok && typ != "object" {
		return fmt.Errorf("tool args schema must be an object")
	}
	if props, ok := argsSchema["properties"]; ok {
		if _, ok := props.(map[string]any); !ok {
			return fmt.Errorf("tool args schema properties must be an object")
		}
	}
	return nil
}

// RenderTextDescription renders tool names and descriptions as plain text.
func RenderTextDescription(tools []Tool) string {
	lines := make([]string, 0, len(tools))
	for _, tool := range tools {
		lines = append(lines, fmt.Sprintf("%s - %s", tool.Name(), tool.Description()))
	}
	return strings.Join(lines, "\n")
}

// RenderTextDescriptionAndArgs renders tool names, descriptions, and JSON args.
func RenderTextDescriptionAndArgs(tools []Tool) string {
	lines := make([]string, 0, len(tools))
	for _, tool := range tools {
		args, err := json.Marshal(tool.ArgsSchema())
		if err != nil {
			args = []byte("{}")
		}
		lines = append(lines, fmt.Sprintf("%s - %s, args: %s", tool.Name(), tool.Description(), args))
	}
	return strings.Join(lines, "\n")
}

// ToFunctionSpec converts a Tool to a provider-neutral function-calling spec.
func ToFunctionSpec(tool Tool) (utils.FunctionSpec, error) {
	if tool == nil {
		return utils.FunctionSpec{}, fmt.Errorf("tool is required")
	}
	return utils.NewFunctionSpec(tool.Name(), tool.Description(), tool.ArgsSchema())
}

// ToOpenAIToolSpec converts a Tool to an OpenAI-style function tool object.
func ToOpenAIToolSpec(tool Tool) (utils.ToolSpec, error) {
	spec, err := ToFunctionSpec(tool)
	if err != nil {
		return utils.ToolSpec{}, err
	}
	return utils.ConvertToOpenAITool(spec), nil
}

// RetrieverToolOptions configures CreateRetrieverTool.
type RetrieverToolOptions struct {
	DocumentSeparator string
	IncludeArtifact   bool
	FormatDocument    func(documents.Document) string
}

// CreateRetrieverTool adapts a retriever into a structured query tool.
func CreateRetrieverTool(
	retriever retrievers.Retriever,
	name string,
	description string,
	opts RetrieverToolOptions,
) (Func, error) {
	if retriever == nil {
		return Func{}, fmt.Errorf("retriever is required")
	}
	separator := opts.DocumentSeparator
	if separator == "" {
		separator = "\n\n"
	}
	formatDocument := opts.FormatDocument
	if formatDocument == nil {
		formatDocument = func(doc documents.Document) string { return doc.PageContent }
	}
	return NewFunc(
		name,
		description,
		schema.Object(map[string]schema.Schema{
			"query": schema.String("query to look up in retriever"),
		}, "query"),
		func(ctx context.Context, input map[string]any) (Result, error) {
			query, ok := input["query"].(string)
			if !ok || query == "" {
				return Result{}, fmt.Errorf("query must be a non-empty string")
			}
			docs, err := retriever.GetRelevantDocuments(ctx, query)
			if err != nil {
				return Result{}, err
			}
			parts := make([]string, len(docs))
			for i, doc := range docs {
				parts[i] = formatDocument(doc)
			}
			result := Result{Content: strings.Join(parts, separator)}
			if opts.IncludeArtifact {
				result.Artifact = docs
			}
			return result, nil
		},
	)
}
