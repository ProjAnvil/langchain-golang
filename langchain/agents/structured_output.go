package agents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/language"
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

// structuredOutputErrorTemplate mirrors Python's STRUCTURED_OUTPUT_ERROR_TEMPLATE
// (factory.py:113): the retry ToolMessage content shown to the model when a
// ToolStrategy error is retried via HandleErrors.
const structuredOutputErrorTemplate = "Error: %s\n Please fix your mistakes."

// handleStructuredOutputError mirrors Python's `_handle_structured_output_error`
// (factory.py:596-624): it decides whether a structured-output failure is
// retried by injecting an error ToolMessage back into the loop (shouldRetry
// true, with the message the model sees) or raised to the caller. Only a
// ToolStrategy participates (a nil strategy means no retry, matching Python's
// early return for non-ToolStrategy response formats).
//
// HandleErrors forms and their meaning:
//   - nil, true: retry every error with the default template. nil is treated
//     as true to match NewToolStrategy's default (Python's constructor
//     default is handle_errors=True).
//   - false: never retry; the error propagates.
//   - string: retry every error with this custom message.
//   - func(error) string: retry with the message the function returns.
//   - []string: retry only when the error's Unwrap chain contains an error
//     whose Go type name (base name, after the final dot) is listed — the Go
//     stand-in for Python's type[Exception] / tuple-of-types matching.
//   - any other type: no retry (Python cannot reach this case; an invalid Go
//     value raises rather than guessing).
func handleStructuredOutputError(strategy *ToolStrategy, exc error) (shouldRetry bool, message string) {
	if strategy == nil {
		return false, ""
	}
	switch handle := strategy.HandleErrors.(type) {
	case nil:
		return true, fmt.Sprintf(structuredOutputErrorTemplate, exc.Error())
	case bool:
		if !handle {
			return false, ""
		}
		return true, fmt.Sprintf(structuredOutputErrorTemplate, exc.Error())
	case string:
		return true, handle
	case func(error) string:
		return true, handle(exc)
	case []string:
		for _, typeName := range handle {
			if errorChainHasTypeName(exc, typeName) {
				return true, fmt.Sprintf(structuredOutputErrorTemplate, exc.Error())
			}
		}
		return false, ""
	default:
		return false, ""
	}
}

// errorChainHasTypeName reports whether exc or any error in its Unwrap chain
// has a Go type whose base name (after the final package dot) equals typeName.
func errorChainHasTypeName(exc error, typeName string) bool {
	for e := exc; e != nil; e = errors.Unwrap(e) {
		t := fmt.Sprintf("%T", e)
		if idx := strings.LastIndexByte(t, '.'); idx >= 0 {
			t = t[idx+1:]
		}
		if t == typeName {
			return true
		}
	}
	return false
}

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

// StructuredOutputUnsupportedError is returned by AutoStrategy.Resolve when the
// agent's model supports neither tool calling nor native structured output, so
// no concrete strategy can be selected. It is a typed error so callers (and
// CreateAgent's option-validation path) can distinguish "model can't do
// structured output" from a generic configuration error via errors.As.
type StructuredOutputUnsupportedError struct {
	Model language.ChatModel
}

// NewStructuredOutputUnsupportedError constructs a *StructuredOutputUnsupportedError
// for model. model may be nil when Resolve is called without a model.
func NewStructuredOutputUnsupportedError(model language.ChatModel) *StructuredOutputUnsupportedError {
	return &StructuredOutputUnsupportedError{Model: model}
}

func (e *StructuredOutputUnsupportedError) Error() string {
	if e == nil || e.Model == nil {
		return "agents: model supports neither tool calling nor structured output; cannot resolve AutoStrategy"
	}
	return fmt.Sprintf("agents: model %T supports neither tool calling nor structured output; cannot resolve AutoStrategy", e.Model)
}

type AutoStrategy struct {
	Schema schema.Schema
}

func NewAutoStrategy(jsonSchema schema.Schema) AutoStrategy {
	return AutoStrategy{Schema: jsonSchema}
}

// Resolve selects a concrete structured-output strategy based on the agent
// model's declared capabilities, mirroring Python's auto-mode selection in
// `_select_response_format`: ToolStrategy is selected when the model supports
// tool calling (the more capable path, since it parses a structured tool-call
// rather than free-form text); otherwise ProviderStrategy when the model
// supports native structured output; otherwise a *StructuredOutputUnsupportedError.
//
// The returned value is always a non-nil ToolStrategy or ProviderStrategy value
// (value, not pointer) of the same shape WithAgentResponseFormat accepts, so
// the result re-enters CreateAgent's existing dispatch unchanged. Resolution is
// deterministic and side-effect-free; CreateAgent calls it eagerly at build time
// against the agent's bound model (see resolveResponseFormat).
//
// ToolCalling is preferred over StructuredOutput when both are present, matching
// Python's preference order.
func (s AutoStrategy) Resolve(model language.ChatModel) (any, error) {
	if model == nil {
		return nil, NewStructuredOutputUnsupportedError(nil)
	}
	caps := model.Capabilities()
	if caps.ToolCalling {
		return NewToolStrategy(s.Schema), nil
	}
	if caps.StructuredOutput {
		return NewProviderStrategy(s.Schema), nil
	}
	return nil, NewStructuredOutputUnsupportedError(model)
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
		m := messages.BlockToMap(block)
		if text, ok := m["text"].(string); ok && m["type"] == "text" {
			parts = append(parts, text)
			continue
		}
		if content, ok := m["content"].(string); ok {
			parts = append(parts, content)
		}
	}
	return strings.Join(parts, "")
}
