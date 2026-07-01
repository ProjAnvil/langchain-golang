package prompts

import (
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// StructuredPrompt is a chat prompt paired with an expected structured output
// schema and provider-specific structured-output options.
type StructuredPrompt struct {
	ChatPromptTemplate
	Schema                 schema.Schema
	StructuredOutputKwargs map[string]any
}

// NewStructuredPrompt creates a structured prompt.
func NewStructuredPrompt(prompt ChatPromptTemplate, outputSchema schema.Schema, kwargs map[string]any) (StructuredPrompt, error) {
	if len(outputSchema) == 0 {
		return StructuredPrompt{}, fmt.Errorf("structured output schema is required")
	}
	return StructuredPrompt{
		ChatPromptTemplate:     prompt,
		Schema:                 cloneSchema(outputSchema),
		StructuredOutputKwargs: cloneMapAny(kwargs),
	}, nil
}

// NewStructuredPromptFromParts creates a structured prompt from chat prompt
// parts and a schema.
func NewStructuredPromptFromParts(outputSchema schema.Schema, kwargs map[string]any, parts ...ChatPromptPart) (StructuredPrompt, error) {
	return NewStructuredPrompt(NewChatPromptTemplateFromParts(parts...), outputSchema, kwargs)
}

// OutputSchema returns a defensive copy of the structured output schema.
func (p StructuredPrompt) OutputSchema() schema.Schema {
	return cloneSchema(p.Schema)
}

// FormatMessages renders the underlying chat prompt.
func (p StructuredPrompt) FormatMessages(values map[string]any) ([]messages.Message, error) {
	return p.ChatPromptTemplate.FormatMessages(values)
}

func cloneSchema(input schema.Schema) schema.Schema {
	if input == nil {
		return nil
	}
	out := make(schema.Schema, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}
