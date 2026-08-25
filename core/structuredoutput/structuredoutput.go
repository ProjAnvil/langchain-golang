package structuredoutput

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/outputparser"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// JSONSchema configures provider-native structured output using a JSON schema.
type JSONSchema struct {
	Name   string
	Schema schema.Schema
	Strict bool
}

// NewJSONSchema creates a structured output schema binding.
func NewJSONSchema(name string, outputSchema schema.Schema, strict bool) JSONSchema {
	if name == "" {
		name = "structured_output"
	}
	return JSONSchema{
		Name:   name,
		Schema: outputSchema,
		Strict: strict,
	}
}

// JSONSchemaModel is a chat model that can enable provider-native JSON-schema
// structured output and return a configured copy of itself.
type JSONSchemaModel[M language.ChatModel] interface {
	language.ChatModel
	WithStructuredOutput(name string, outputSchema schema.Schema, strict bool) M
}

// BindJSON configures provider-native JSON-schema output and parses the model's
// final text into T.
func BindJSON[M JSONSchemaModel[M], T any](
	model M,
	name string,
	outputSchema schema.Schema,
	strict bool,
) (runnables.Runnable[[]messages.Message, T], error) {
	if !model.Capabilities().StructuredOutput {
		return nil, fmt.Errorf("structured output is not supported")
	}

	configured := model.WithStructuredOutput(name, outputSchema, strict)
	parser := outputparser.NewPydanticParser[T](outputSchema)
	return runnables.NewFunc(
		func(ctx context.Context, input []messages.Message, opts ...runnables.Option) (T, error) {
			message, err := configured.Invoke(ctx, input, opts...)
			if err != nil {
				var zero T
				return zero, err
			}
			return parser.Parse(ctx, message.Content)
		},
		configured.InputSchema(),
		outputSchema,
	), nil
}

// Method selects how the model is steered toward the schema, mirroring
// Python's with_structured_output(method=...). (json_mode is a partner-level
// concern and intentionally not part of this core type.)
type Method string

const (
	// MethodJSONSchema uses provider-native JSON-schema response formatting.
	MethodJSONSchema Method = "json_schema"
	// MethodFunctionCalling binds the schema as a tool and parses the first
	// matching tool call's arguments (Python chat_models.py:2512-2527).
	MethodFunctionCalling Method = "function_calling"
)

// StructuredResult is the include_raw envelope, mirroring Python's
// {"raw", "parsed", "parsing_error"} dict: Raw is the model's message, Parsed
// is the structured value (zero value of T when ParsingError is non-nil),
// and ParsingError carries an output-parsing failure instead of raising.
type StructuredResult[T any] struct {
	Raw          messages.Message
	Parsed       T
	ParsingError error
}

// Options configures BindOptions/BindOptionsWithRaw. An empty Method is
// MethodJSONSchema. An empty Name falls back to the schema's "title", then
// "structured_output".
type Options struct {
	Name   string
	Schema schema.Schema
	Strict bool
	Method Method
}

// parseError marks output-parsing failures so BindOptionsWithRaw can capture
// them in StructuredResult.ParsingError while model/invocation errors
// propagate (mirroring Python's parser_with_fallback scoping).
type parseError struct{ err error }

func (e parseError) Error() string { return e.err.Error() }
func (e parseError) Unwrap() error { return e.err }

// BindOptions steers the model toward opts.Schema using opts.Method and
// parses the result into T. Parsing failures are returned as errors,
// mirroring Python with_structured_output(include_raw=False).
func BindOptions[M JSONSchemaModel[M], T any](
	model M,
	opts Options,
) (runnables.Runnable[[]messages.Message, T], error) {
	if err := validateMethod(model, opts); err != nil {
		return nil, err
	}
	return runnables.NewFunc(
		func(ctx context.Context, input []messages.Message, runOpts ...runnables.Option) (T, error) {
			_, parsed, err := structuredInvoke[M, T](model, opts, ctx, input, runOpts)
			return parsed, err
		},
		model.InputSchema(),
		opts.Schema,
	), nil
}

// BindOptionsWithRaw mirrors Python with_structured_output(include_raw=True):
// the runnable returns a StructuredResult carrying the raw model message, the
// parsed value (zero on failure), and the parsing error instead of raising.
// Model invocation errors still propagate as errors.
func BindOptionsWithRaw[M JSONSchemaModel[M], T any](
	model M,
	opts Options,
) (runnables.Runnable[[]messages.Message, StructuredResult[T]], error) {
	if err := validateMethod(model, opts); err != nil {
		return nil, err
	}
	return runnables.NewFunc(
		func(ctx context.Context, input []messages.Message, runOpts ...runnables.Option) (StructuredResult[T], error) {
			raw, parsed, err := structuredInvoke[M, T](model, opts, ctx, input, runOpts)
			var pErr parseError
			switch {
			case err == nil:
				return StructuredResult[T]{Raw: raw, Parsed: parsed}, nil
			case errors.As(err, &pErr):
				return StructuredResult[T]{Raw: raw, ParsingError: pErr.err}, nil
			default:
				return StructuredResult[T]{}, err
			}
		},
		model.InputSchema(),
		opts.Schema,
	), nil
}

func validateMethod[M JSONSchemaModel[M]](model M, opts Options) error {
	switch opts.Method {
	case "", MethodJSONSchema:
		if !model.Capabilities().StructuredOutput {
			return fmt.Errorf("structured output is not supported")
		}
	case MethodFunctionCalling:
		if !model.Capabilities().ToolCalling {
			return fmt.Errorf("function calling structured output requires a model with tool calling support")
		}
	default:
		return fmt.Errorf("unsupported structured output method %q", opts.Method)
	}
	return nil
}

// structuredInvoke runs one structured-output call. Parse-stage failures are
// wrapped in parseError; everything else is returned as a plain error.
func structuredInvoke[M JSONSchemaModel[M], T any](
	model M,
	opts Options,
	ctx context.Context,
	input []messages.Message,
	runOpts []runnables.Option,
) (messages.Message, T, error) {
	var zero T
	name := opts.Name
	if name == "" {
		if title, ok := opts.Schema["title"].(string); ok && title != "" {
			name = title
		} else {
			name = "structured_output"
		}
	}
	parser := outputparser.NewPydanticParser[T](opts.Schema)

	if opts.Method == MethodFunctionCalling {
		tool, err := tools.NewFunc(
			name,
			"Structured output conforming to the requested schema",
			opts.Schema,
			func(context.Context, map[string]any) (tools.Result, error) {
				return tools.Result{}, fmt.Errorf("structured output schema tool %q is not invocable", name)
			},
		)
		if err != nil {
			return messages.Message{}, zero, err
		}
		bound, err := model.BindTools([]tools.Tool{tool})
		if err != nil {
			return messages.Message{}, zero, err
		}
		message, err := bound.Invoke(ctx, input, runOpts...)
		if err != nil {
			return messages.Message{}, zero, err
		}
		args, err := firstToolCallArgs(message, name)
		if err != nil {
			return message, zero, parseError{err}
		}
		data, err := json.Marshal(args)
		if err != nil {
			return message, zero, parseError{err}
		}
		parsed, err := parser.Parse(ctx, string(data))
		if err != nil {
			return message, zero, parseError{err}
		}
		return message, parsed, nil
	}

	configured := model.WithStructuredOutput(name, opts.Schema, opts.Strict)
	message, err := configured.Invoke(ctx, input, runOpts...)
	if err != nil {
		return messages.Message{}, zero, err
	}
	parsed, err := parser.Parse(ctx, message.Content)
	if err != nil {
		return message, zero, parseError{err}
	}
	return message, parsed, nil
}

// firstToolCallArgs returns the args of the first tool call matching name.
// Divergence from Python: JsonOutputKeyToolsParser returns None when no tool
// call matches (parsed=None without an error); Go surfaces it as a parse
// error so a silent zero value is never mistaken for a valid parse.
func firstToolCallArgs(message messages.Message, name string) (map[string]any, error) {
	for _, call := range message.ToolCalls {
		if call.Name == name {
			if call.Args == nil {
				return map[string]any{}, nil
			}
			return call.Args, nil
		}
	}
	return nil, fmt.Errorf("model did not return a tool call named %q", name)
}
