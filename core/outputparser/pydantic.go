package outputparser

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strings"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/schema"
)

const pydanticFormatInstructions = `The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
` + "```" + `
%s
` + "```"

// PydanticParser is the Go equivalent of Python's PydanticOutputParser: it
// parses JSON, validates it against a JSON-schema-shaped contract, and decodes
// it into a typed Go value.
type PydanticParser[T any] struct {
	OutputSchema schema.Schema
	Instructions PydanticInstructionOptions
}

// PydanticInstructionStyle selects the prompt instructions emitted by a
// PydanticParser.
type PydanticInstructionStyle string

const (
	// PydanticInstructionDefault matches Python's PydanticOutputParser wording.
	PydanticInstructionDefault PydanticInstructionStyle = ""
	// PydanticInstructionJSONMode is concise and works well with provider JSON
	// mode APIs that still rely on prompt-level schema guidance.
	PydanticInstructionJSONMode PydanticInstructionStyle = "json_mode"
	// PydanticInstructionProviderNative is for provider-native structured output
	// bindings where the schema is supplied out of band.
	PydanticInstructionProviderNative PydanticInstructionStyle = "provider_native"
)

// PydanticInstructionOptions configures format-instruction rendering.
type PydanticInstructionOptions struct {
	Style                   PydanticInstructionStyle
	Name                    string
	Strict                  bool
	IncludeSchema           bool
	RemoveTopLevelTypeTitle bool
	IndentSchema            bool
}

// PydanticParserOption configures a PydanticParser.
type PydanticParserOption func(*PydanticInstructionOptions)

// NewPydanticParser creates a typed JSON parser with schema validation.
func NewPydanticParser[T any](outputSchema schema.Schema) PydanticParser[T] {
	return NewPydanticParserWithOptions[T](outputSchema)
}

// NewPydanticParserWithOptions creates a typed JSON parser with configurable
// format instructions.
func NewPydanticParserWithOptions[T any](outputSchema schema.Schema, opts ...PydanticParserOption) PydanticParser[T] {
	instructions := defaultPydanticInstructionOptions()
	for _, opt := range opts {
		opt(&instructions)
	}
	return PydanticParser[T]{
		OutputSchema: outputSchema,
		Instructions: instructions,
	}
}

// WithPydanticInstructionStyle selects the emitted format-instruction style.
func WithPydanticInstructionStyle(style PydanticInstructionStyle) PydanticParserOption {
	return func(opts *PydanticInstructionOptions) {
		opts.Style = style
	}
}

// WithPydanticInstructionName sets the provider-native structured output name.
func WithPydanticInstructionName(name string) PydanticParserOption {
	return func(opts *PydanticInstructionOptions) {
		opts.Name = name
	}
}

// WithPydanticInstructionStrict marks provider-native instructions as strict.
func WithPydanticInstructionStrict(strict bool) PydanticParserOption {
	return func(opts *PydanticInstructionOptions) {
		opts.Strict = strict
	}
}

// WithPydanticInstructionSchema controls whether the JSON schema is included.
func WithPydanticInstructionSchema(include bool) PydanticParserOption {
	return func(opts *PydanticInstructionOptions) {
		opts.IncludeSchema = include
	}
}

// WithPydanticInstructionIndentedSchema renders the schema with indentation.
func WithPydanticInstructionIndentedSchema(indent bool) PydanticParserOption {
	return func(opts *PydanticInstructionOptions) {
		opts.IndentSchema = indent
	}
}

// Parse validates the model output and decodes it into T.
func (p PydanticParser[T]) Parse(_ context.Context, text string) (T, error) {
	var zero T
	var raw any
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	if err := decoder.Decode(&raw); err != nil {
		return zero, fmt.Errorf("parse pydantic json output: %w", err)
	}
	if err := validateSchema(raw, p.OutputSchema, "$"); err != nil {
		data, _ := json.Marshal(raw)
		return zero, fmt.Errorf("%w: failed to parse output from completion %s: %v", lcerrors.ErrSchemaValidation, data, err)
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return zero, err
	}
	var out T
	if err := json.Unmarshal(data, &out); err != nil {
		return zero, fmt.Errorf("%w: decode typed pydantic output: %v", lcerrors.ErrSchemaValidation, err)
	}
	return out, nil
}

// FormatInstructions returns JSON schema format instructions matching Python's
// PydanticOutputParser wording.
func (p PydanticParser[T]) FormatInstructions() string {
	options := p.Instructions.withDefaults()
	schemaText, err := p.instructionSchemaText(options)
	if err != nil {
		return "Return valid JSON matching the provided schema."
	}
	switch options.Style {
	case PydanticInstructionJSONMode:
		if options.IncludeSchema {
			return fmt.Sprintf("Return only a valid JSON object that conforms to this JSON schema. Do not include markdown, prose, or code fences.\n\n%s", schemaText)
		}
		return "Return only a valid JSON object. Do not include markdown, prose, or code fences."
	case PydanticInstructionProviderNative:
		name := options.Name
		if name == "" {
			name = "structured_output"
		}
		mode := "provider-native structured output"
		if options.Strict {
			mode += " in strict mode"
		}
		if options.IncludeSchema {
			return fmt.Sprintf("Use %s named %q and return data that conforms to this JSON schema. Do not include explanatory text.\n\n%s", mode, name, schemaText)
		}
		return fmt.Sprintf("Use %s named %q. Return only the structured output value and do not include explanatory text.", mode, name)
	default:
		return fmt.Sprintf(pydanticFormatInstructions, schemaText)
	}
}

func reduceInstructionSchema(input schema.Schema) map[string]any {
	out := make(map[string]any, len(input))
	for key, value := range input {
		if key == "title" || key == "type" {
			continue
		}
		out[key] = value
	}
	return out
}

func defaultPydanticInstructionOptions() PydanticInstructionOptions {
	return PydanticInstructionOptions{
		IncludeSchema:           true,
		RemoveTopLevelTypeTitle: true,
	}
}

func (o PydanticInstructionOptions) withDefaults() PydanticInstructionOptions {
	if o == (PydanticInstructionOptions{}) {
		return defaultPydanticInstructionOptions()
	}
	if o.Style == PydanticInstructionDefault {
		o.RemoveTopLevelTypeTitle = true
	}
	return o
}

func (p PydanticParser[T]) instructionSchemaText(options PydanticInstructionOptions) (string, error) {
	if !options.IncludeSchema {
		return "", nil
	}
	var value any = p.OutputSchema
	if options.RemoveTopLevelTypeTitle {
		value = reduceInstructionSchema(p.OutputSchema)
	}
	var data []byte
	var err error
	if options.IndentSchema {
		data, err = json.MarshalIndent(value, "", "  ")
	} else {
		data, err = json.Marshal(value)
	}
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func validateSchema(value any, spec schema.Schema, path string) error {
	if len(spec) == 0 {
		return nil
	}
	if err := validateCombinators(value, spec, path); err != nil {
		return err
	}
	if err := validateEnumConst(value, spec, path); err != nil {
		return err
	}
	kinds := schemaTypes(spec["type"])
	if len(kinds) == 0 {
		kinds = []string{"any"}
	}
	var lastErr error
	for _, kind := range kinds {
		if kind == "null" {
			if value == nil {
				return nil
			}
			lastErr = fmt.Errorf("%s: expected null, got %T", path, value)
			continue
		}
		if err := validateSchemaType(value, spec, path, kind); err != nil {
			lastErr = err
			continue
		}
		return nil
	}
	return lastErr
}

func validateSchemaType(value any, spec schema.Schema, path string, kind string) error {
	switch kind {
	case "", "any":
		return nil
	case "object":
		return validateObject(value, spec, path)
	case "array":
		return validateArray(value, spec, path)
	case "string":
		text, ok := value.(string)
		if !ok {
			return fmt.Errorf("%s: expected string, got %T", path, value)
		}
		return validateString(text, spec, path)
	case "integer":
		if !isInteger(value) {
			return fmt.Errorf("%s: expected integer, got %T", path, value)
		}
		return validateNumber(value, spec, path)
	case "number":
		if !isNumber(value) {
			return fmt.Errorf("%s: expected number, got %T", path, value)
		}
		return validateNumber(value, spec, path)
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("%s: expected boolean, got %T", path, value)
		}
	default:
		return fmt.Errorf("%s: unsupported schema type %q", path, kind)
	}
	return nil
}

func validateObject(value any, spec schema.Schema, path string) error {
	obj, ok := value.(map[string]any)
	if !ok {
		return fmt.Errorf("%s: expected object, got %T", path, value)
	}
	for _, key := range requiredKeys(spec["required"]) {
		if _, ok := obj[key]; !ok {
			return fmt.Errorf("%s.%s: required field missing", path, key)
		}
	}
	properties, ok := spec["properties"].(map[string]any)
	if !ok {
		properties = map[string]any{}
	}
	for key, propertySpec := range properties {
		child, ok := obj[key]
		if !ok {
			continue
		}
		childSchema, ok := asSchema(propertySpec)
		if !ok {
			continue
		}
		if err := validateSchema(child, childSchema, path+"."+key); err != nil {
			return err
		}
	}
	if additional, ok := spec["additionalProperties"]; ok {
		for key, child := range obj {
			if _, defined := properties[key]; defined {
				continue
			}
			switch typed := additional.(type) {
			case bool:
				if !typed {
					return fmt.Errorf("%s.%s: additional property not allowed", path, key)
				}
			default:
				childSchema, ok := asSchema(typed)
				if !ok {
					continue
				}
				if err := validateSchema(child, childSchema, path+"."+key); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func validateArray(value any, spec schema.Schema, path string) error {
	arr, ok := value.([]any)
	if !ok {
		return fmt.Errorf("%s: expected array, got %T", path, value)
	}
	if min, ok := numericConstraint(spec["minItems"]); ok && float64(len(arr)) < min {
		return fmt.Errorf("%s: expected at least %v items, got %d", path, min, len(arr))
	}
	if max, ok := numericConstraint(spec["maxItems"]); ok && float64(len(arr)) > max {
		return fmt.Errorf("%s: expected at most %v items, got %d", path, max, len(arr))
	}
	itemSchema, ok := asSchema(spec["items"])
	if !ok {
		return nil
	}
	for i, item := range arr {
		if err := validateSchema(item, itemSchema, fmt.Sprintf("%s[%d]", path, i)); err != nil {
			return err
		}
	}
	return nil
}

func validateCombinators(value any, spec schema.Schema, path string) error {
	if schemas := schemaList(spec["allOf"]); len(schemas) > 0 {
		for _, child := range schemas {
			if err := validateSchema(value, child, path); err != nil {
				return fmt.Errorf("%s: allOf failed: %w", path, err)
			}
		}
	}
	if schemas := schemaList(spec["anyOf"]); len(schemas) > 0 {
		var lastErr error
		for _, child := range schemas {
			if err := validateSchema(value, child, path); err == nil {
				return nil
			} else {
				lastErr = err
			}
		}
		return fmt.Errorf("%s: anyOf failed: %v", path, lastErr)
	}
	if schemas := schemaList(spec["oneOf"]); len(schemas) > 0 {
		matches := 0
		for _, child := range schemas {
			if err := validateSchema(value, child, path); err == nil {
				matches++
			}
		}
		if matches != 1 {
			return fmt.Errorf("%s: oneOf expected exactly one match, got %d", path, matches)
		}
	}
	return nil
}

func validateEnumConst(value any, spec schema.Schema, path string) error {
	if expected, ok := spec["const"]; ok && !jsonEqual(value, expected) {
		return fmt.Errorf("%s: expected const %v, got %v", path, expected, value)
	}
	if values, ok := spec["enum"]; ok {
		for _, candidate := range anySlice(values) {
			if jsonEqual(value, candidate) {
				return nil
			}
		}
		return fmt.Errorf("%s: value %v is not in enum %v", path, value, values)
	}
	return nil
}

func validateString(value string, spec schema.Schema, path string) error {
	if min, ok := numericConstraint(spec["minLength"]); ok && float64(len(value)) < min {
		return fmt.Errorf("%s: expected string length at least %v, got %d", path, min, len(value))
	}
	if max, ok := numericConstraint(spec["maxLength"]); ok && float64(len(value)) > max {
		return fmt.Errorf("%s: expected string length at most %v, got %d", path, max, len(value))
	}
	if pattern, ok := spec["pattern"].(string); ok && pattern != "" {
		re, err := regexp.Compile(pattern)
		if err != nil {
			return fmt.Errorf("%s: invalid pattern %q: %w", path, pattern, err)
		}
		if !re.MatchString(value) {
			return fmt.Errorf("%s: string %q does not match pattern %q", path, value, pattern)
		}
	}
	return nil
}

func validateNumber(value any, spec schema.Schema, path string) error {
	number, ok := numberFloat(value)
	if !ok {
		return fmt.Errorf("%s: expected number, got %T", path, value)
	}
	if min, ok := numericConstraint(spec["minimum"]); ok && number < min {
		return fmt.Errorf("%s: expected minimum %v, got %v", path, min, number)
	}
	if min, ok := numericConstraint(spec["exclusiveMinimum"]); ok && number <= min {
		return fmt.Errorf("%s: expected greater than %v, got %v", path, min, number)
	}
	if max, ok := numericConstraint(spec["maximum"]); ok && number > max {
		return fmt.Errorf("%s: expected maximum %v, got %v", path, max, number)
	}
	if max, ok := numericConstraint(spec["exclusiveMaximum"]); ok && number >= max {
		return fmt.Errorf("%s: expected less than %v, got %v", path, max, number)
	}
	return nil
}

func requiredKeys(value any) []string {
	switch typed := value.(type) {
	case []string:
		return append([]string(nil), typed...)
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if key, ok := item.(string); ok {
				out = append(out, key)
			}
		}
		return out
	default:
		return nil
	}
}

func asSchema(value any) (schema.Schema, bool) {
	switch typed := value.(type) {
	case schema.Schema:
		return typed, true
	case map[string]any:
		return schema.Schema(typed), true
	default:
		return nil, false
	}
}

func schemaTypes(value any) []string {
	switch typed := value.(type) {
	case string:
		return []string{typed}
	case []string:
		return append([]string(nil), typed...)
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if kind, ok := item.(string); ok {
				out = append(out, kind)
			}
		}
		return out
	default:
		return nil
	}
}

func schemaList(value any) []schema.Schema {
	items := anySlice(value)
	out := make([]schema.Schema, 0, len(items))
	for _, item := range items {
		if child, ok := asSchema(item); ok {
			out = append(out, child)
		}
	}
	return out
}

func anySlice(value any) []any {
	switch typed := value.(type) {
	case []any:
		return typed
	case []string:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = item
		}
		return out
	case []schema.Schema:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = item
		}
		return out
	default:
		return nil
	}
}

func numericConstraint(value any) (float64, bool) {
	switch typed := value.(type) {
	case json.Number:
		out, err := typed.Float64()
		return out, err == nil
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int8:
		return float64(typed), true
	case int16:
		return float64(typed), true
	case int32:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case uint:
		return float64(typed), true
	case uint8:
		return float64(typed), true
	case uint16:
		return float64(typed), true
	case uint32:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	default:
		return 0, false
	}
}

func numberFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case json.Number:
		out, err := typed.Float64()
		return out, err == nil
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int8:
		return float64(typed), true
	case int16:
		return float64(typed), true
	case int32:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case uint:
		return float64(typed), true
	case uint8:
		return float64(typed), true
	case uint16:
		return float64(typed), true
	case uint32:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	default:
		return 0, false
	}
}

func jsonEqual(left any, right any) bool {
	leftData, leftErr := json.Marshal(left)
	rightData, rightErr := json.Marshal(right)
	if leftErr != nil || rightErr != nil {
		return fmt.Sprint(left) == fmt.Sprint(right)
	}
	var normalizedLeft any
	var normalizedRight any
	leftDecoder := json.NewDecoder(strings.NewReader(string(leftData)))
	leftDecoder.UseNumber()
	rightDecoder := json.NewDecoder(strings.NewReader(string(rightData)))
	rightDecoder.UseNumber()
	if leftDecoder.Decode(&normalizedLeft) != nil || rightDecoder.Decode(&normalizedRight) != nil {
		return string(leftData) == string(rightData)
	}
	return reflect.DeepEqual(normalizedLeft, normalizedRight)
}

func isInteger(value any) bool {
	switch typed := value.(type) {
	case json.Number:
		if _, err := typed.Int64(); err == nil {
			return true
		}
		f, err := typed.Float64()
		return err == nil && math.Trunc(f) == f
	case float64:
		return math.Trunc(typed) == typed
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	default:
		return false
	}
}

func isNumber(value any) bool {
	switch typed := value.(type) {
	case json.Number:
		_, err := typed.Float64()
		return err == nil
	case float64, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	default:
		return false
	}
}
