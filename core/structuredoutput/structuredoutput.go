package structuredoutput

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/outputparser"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
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
