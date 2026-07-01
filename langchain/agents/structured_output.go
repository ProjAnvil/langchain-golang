package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

type SchemaKind string

const (
	SchemaKindJSONSchema SchemaKind = "json_schema"
)

type MultipleStructuredOutputsError struct {
	ToolNames []string
	AIMessage messages.Message
}

func NewMultipleStructuredOutputsError(toolNames []string, aiMessage messages.Message) *MultipleStructuredOutputsError {
	return &MultipleStructuredOutputsError{
		ToolNames: append([]string(nil), toolNames...),
		AIMessage: aiMessage,
	}
}

func (e *MultipleStructuredOutputsError) Error() string {
	return fmt.Sprintf(
		"Model incorrectly returned multiple structured responses (%s) when only one is expected.",
		strings.Join(e.ToolNames, ", "),
	)
}

type StructuredOutputValidationError struct {
	ToolName  string
	Source    error
	AIMessage messages.Message
}

func NewStructuredOutputValidationError(toolName string, source error, aiMessage messages.Message) *StructuredOutputValidationError {
	return &StructuredOutputValidationError{
		ToolName:  toolName,
		Source:    source,
		AIMessage: aiMessage,
	}
}

func (e *StructuredOutputValidationError) Error() string {
	return fmt.Sprintf("Failed to parse structured output for tool '%s': %s.", e.ToolName, e.Source)
}

func (e *StructuredOutputValidationError) Unwrap() error {
	return e.Source
}

type SchemaSpec struct {
	Schema      schema.Schema
	Name        string
	Description string
	SchemaKind  SchemaKind
	JSONSchema  schema.Schema
	Strict      *bool
}

func NewSchemaSpec(jsonSchema schema.Schema, opts ...SchemaSpecOption) SchemaSpec {
	spec := SchemaSpec{
		Schema:     jsonSchema,
		Name:       schemaName(jsonSchema),
		SchemaKind: SchemaKindJSONSchema,
		JSONSchema: jsonSchema,
	}
	if description, ok := jsonSchema["description"].(string); ok {
		spec.Description = description
	}
	for _, opt := range opts {
		opt(&spec)
	}
	return spec
}

type SchemaSpecOption func(*SchemaSpec)

func WithSchemaName(name string) SchemaSpecOption {
	return func(spec *SchemaSpec) {
		spec.Name = name
	}
}

func WithSchemaDescription(description string) SchemaSpecOption {
	return func(spec *SchemaSpec) {
		spec.Description = description
	}
}

type ToolStrategy struct {
	Schema             schema.Schema
	SchemaSpecs        []SchemaSpec
	ToolMessageContent string
	HandleErrors       any
}

type ToolStrategyOption func(*ToolStrategy)

func WithToolMessageContent(content string) ToolStrategyOption {
	return func(strategy *ToolStrategy) {
		strategy.ToolMessageContent = content
	}
}

func WithHandleErrors(handleErrors any) ToolStrategyOption {
	return func(strategy *ToolStrategy) {
		strategy.HandleErrors = handleErrors
	}
}

func NewToolStrategy(jsonSchema schema.Schema, opts ...ToolStrategyOption) ToolStrategy {
	strategy := ToolStrategy{
		Schema:       jsonSchema,
		SchemaSpecs:  schemaSpecsFromSchema(jsonSchema),
		HandleErrors: true,
	}
	for _, opt := range opts {
		opt(&strategy)
	}
	return strategy
}

type ProviderStrategy struct {
	Schema     schema.Schema
	SchemaSpec SchemaSpec
}

type ProviderStrategyOption func(*ProviderStrategy)

func WithStrict(strict bool) ProviderStrategyOption {
	return func(strategy *ProviderStrategy) {
		strategy.SchemaSpec.Strict = &strict
	}
}

func NewProviderStrategy(jsonSchema schema.Schema, opts ...ProviderStrategyOption) ProviderStrategy {
	strategy := ProviderStrategy{
		Schema:     jsonSchema,
		SchemaSpec: NewSchemaSpec(jsonSchema),
	}
	for _, opt := range opts {
		opt(&strategy)
	}
	return strategy
}

func (s ProviderStrategy) ToModelKwargs() map[string]any {
	jsonSchema := map[string]any{
		"name":   s.SchemaSpec.Name,
		"schema": s.SchemaSpec.JSONSchema,
	}
	if s.SchemaSpec.Strict != nil && *s.SchemaSpec.Strict {
		jsonSchema["strict"] = true
	}
	return map[string]any{
		"response_format": map[string]any{
			"type":        "json_schema",
			"json_schema": jsonSchema,
		},
	}
}

type OutputToolBinding struct {
	Schema     schema.Schema
	SchemaKind SchemaKind
	Tool       tools.Func
}

func OutputToolBindingFromSchemaSpec(spec SchemaSpec) (OutputToolBinding, error) {
	tool, err := tools.NewFunc(
		spec.Name,
		spec.Description,
		spec.JSONSchema,
		func(context.Context, map[string]any) (tools.Result, error) {
			return tools.Result{}, nil
		},
	)
	if err != nil {
		return OutputToolBinding{}, err
	}
	return OutputToolBinding{
		Schema:     spec.Schema,
		SchemaKind: spec.SchemaKind,
		Tool:       tool,
	}, nil
}

func (b OutputToolBinding) Parse(toolArgs map[string]any) (map[string]any, error) {
	return toolArgs, nil
}

type ProviderStrategyBinding struct {
	Schema     schema.Schema
	SchemaKind SchemaKind
	Name       string
}

func ProviderStrategyBindingFromSchemaSpec(spec SchemaSpec) ProviderStrategyBinding {
	return ProviderStrategyBinding{
		Schema:     spec.Schema,
		SchemaKind: spec.SchemaKind,
		Name:       spec.Name,
	}
}

func (b ProviderStrategyBinding) Parse(response messages.Message) (map[string]any, error) {
	rawText := extractTextContent(response)
	var data map[string]any
	if err := json.Unmarshal([]byte(rawText), &data); err != nil {
		return nil, fmt.Errorf("Native structured output expected valid JSON for %s, but parsing failed: %w", b.Name, err)
	}
	return data, nil
}

type AutoStrategy struct {
	Schema schema.Schema
}

func NewAutoStrategy(jsonSchema schema.Schema) AutoStrategy {
	return AutoStrategy{Schema: jsonSchema}
}

func schemaSpecsFromSchema(jsonSchema schema.Schema) []SchemaSpec {
	if variants, ok := jsonSchema["oneOf"].([]any); ok {
		specs := make([]SchemaSpec, 0, len(variants))
		for _, variant := range variants {
			if variantSchema, ok := variant.(schema.Schema); ok {
				specs = append(specs, NewSchemaSpec(variantSchema))
				continue
			}
			if variantMap, ok := variant.(map[string]any); ok {
				specs = append(specs, NewSchemaSpec(schema.Schema(variantMap)))
			}
		}
		return specs
	}
	return []SchemaSpec{NewSchemaSpec(jsonSchema)}
}

func schemaName(jsonSchema schema.Schema) string {
	if title, ok := jsonSchema["title"].(string); ok && title != "" {
		return title
	}
	return "response_format"
}

func extractTextContent(message messages.Message) string {
	if message.Content != "" {
		return message.Content
	}
	var parts []string
	for _, block := range message.ContentBlocks {
		if text, ok := block["text"].(string); ok && block["type"] == "text" {
			parts = append(parts, text)
			continue
		}
		if content, ok := block["content"].(string); ok {
			parts = append(parts, content)
		}
	}
	return strings.Join(parts, "")
}
