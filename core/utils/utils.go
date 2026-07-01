package utils

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"html"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/schema"
)

// Iterator is a context-aware pull iterator.
type Iterator[T any] interface {
	Next(ctx context.Context) (T, bool, error)
	Close() error
}

// SliceIterator iterates over an in-memory slice.
type SliceIterator[T any] struct {
	values []T
	index  int
}

// NewSliceIterator creates an iterator over a defensive copy of values.
func NewSliceIterator[T any](values []T) *SliceIterator[T] {
	copied := append([]T(nil), values...)
	return &SliceIterator[T]{values: copied}
}

// Next returns the next item.
func (i *SliceIterator[T]) Next(ctx context.Context) (T, bool, error) {
	var zero T
	if err := ctx.Err(); err != nil {
		return zero, false, err
	}
	if i.index >= len(i.values) {
		return zero, false, nil
	}
	value := i.values[i.index]
	i.index++
	return value, true, nil
}

// Close releases iterator resources.
func (i *SliceIterator[T]) Close() error {
	i.index = len(i.values)
	return nil
}

// CollectIterator consumes and closes an iterator.
func CollectIterator[T any](ctx context.Context, iter Iterator[T]) ([]T, error) {
	if iter == nil {
		return nil, fmt.Errorf("iterator is required")
	}
	defer iter.Close()
	out := []T{}
	for {
		value, ok, err := iter.Next(ctx)
		if err != nil {
			return nil, err
		}
		if !ok {
			return out, nil
		}
		out = append(out, value)
	}
}

// IteratorToChannel consumes an iterator in a goroutine and sends values on a
// channel until the iterator ends, errors, or the context is canceled.
func IteratorToChannel[T any](ctx context.Context, iter Iterator[T], buffer int) (<-chan T, <-chan error) {
	if buffer < 0 {
		buffer = 0
	}
	values := make(chan T, buffer)
	errs := make(chan error, 1)
	go func() {
		defer close(values)
		defer close(errs)
		if iter == nil {
			errs <- fmt.Errorf("iterator is required")
			return
		}
		defer iter.Close()
		for {
			value, ok, err := iter.Next(ctx)
			if err != nil {
				errs <- err
				return
			}
			if !ok {
				return
			}
			select {
			case values <- value:
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			}
		}
	}()
	return values, errs
}

// MergeMaps recursively merges maps from left to right.
func MergeMaps(base map[string]any, overlays ...map[string]any) map[string]any {
	out := cloneMap(base)
	for _, overlay := range overlays {
		for key, value := range overlay {
			if existing, ok := out[key].(map[string]any); ok {
				if next, ok := value.(map[string]any); ok {
					out[key] = MergeMaps(existing, next)
					continue
				}
			}
			out[key] = value
		}
	}
	return out
}

// GetFromEnv returns the first non-empty environment variable or a default.
func GetFromEnv(keys []string, defaultValue string) string {
	for _, key := range keys {
		if value := os.Getenv(key); value != "" {
			return value
		}
	}
	return defaultValue
}

// MustGetFromEnv returns the first non-empty environment variable or an error.
func MustGetFromEnv(keys ...string) (string, error) {
	if value := GetFromEnv(keys, ""); value != "" {
		return value, nil
	}
	return "", fmt.Errorf("missing required environment variable: %s", strings.Join(keys, ", "))
}

// ToJSONString renders deterministic JSON for schema-shaped payloads.
func ToJSONString(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// StringifyValue formats a value for prompt helpers.
func StringifyValue(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprint(v)
	}
}

// EnsureID returns id when non-empty, otherwise generates an lc_ prefixed UUID.
func EnsureID(id string) (string, error) {
	if id != "" {
		return id, nil
	}
	uuid, err := UUID4()
	if err != nil {
		return "", err
	}
	return "lc_" + uuid, nil
}

// UUID4 returns a random RFC 4122 version 4 UUID string.
func UUID4() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4],
		b[4:6],
		b[6:8],
		b[8:10],
		b[10:16],
	), nil
}

// FunctionSpec is the provider-neutral function-calling shape.
type FunctionSpec struct {
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Parameters  schema.Schema `json:"parameters"`
}

// ToolSpec is an OpenAI-style tool wrapper around a function spec.
type ToolSpec struct {
	Type     string       `json:"type"`
	Function FunctionSpec `json:"function"`
}

// NewFunctionSpec validates and returns a function-calling spec.
func NewFunctionSpec(name string, description string, parameters schema.Schema) (FunctionSpec, error) {
	if !validFunctionName(name) {
		return FunctionSpec{}, fmt.Errorf("invalid function name %q", name)
	}
	if parameters == nil {
		parameters = schema.Object(nil)
	}
	if typ, ok := parameters["type"].(string); ok && typ != "object" {
		return FunctionSpec{}, fmt.Errorf("function parameters must be an object schema")
	}
	return FunctionSpec{Name: name, Description: description, Parameters: CloneSchema(parameters)}, nil
}

// ConvertToOpenAITool wraps a function spec in an OpenAI-style tool object.
func ConvertToOpenAITool(fn FunctionSpec) ToolSpec {
	return ToolSpec{Type: "function", Function: fn}
}

// CloneSchema defensively copies a top-level schema map.
func CloneSchema(input schema.Schema) schema.Schema {
	if input == nil {
		return nil
	}
	out := make(schema.Schema, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

// RemoveJSONSchemaDefinitions removes local definition blocks that provider
// function-calling APIs commonly reject.
func RemoveJSONSchemaDefinitions(input schema.Schema) schema.Schema {
	out := CloneSchema(input)
	delete(out, "$defs")
	delete(out, "definitions")
	return out
}

// IsObjectSchema reports whether a JSON schema has object shape. It is the Go
// schema-oriented analogue of checking whether a Pydantic model exposes fields.
func IsObjectSchema(input schema.Schema) bool {
	typ, _ := input["type"].(string)
	return typ == "" || typ == "object"
}

// SchemaProperties returns a defensive copy of an object schema's properties.
func SchemaProperties(input schema.Schema) map[string]schema.Schema {
	raw, ok := input["properties"]
	if !ok || raw == nil {
		return map[string]schema.Schema{}
	}
	out := map[string]schema.Schema{}
	switch props := raw.(type) {
	case map[string]schema.Schema:
		for key, value := range props {
			out[key] = CloneSchema(value)
		}
	case map[string]any:
		for key, value := range props {
			switch typed := value.(type) {
			case schema.Schema:
				out[key] = CloneSchema(typed)
			case map[string]any:
				out[key] = CloneSchema(schema.Schema(typed))
			}
		}
	}
	return out
}

// SchemaRequired returns required field names in schema order.
func SchemaRequired(input schema.Schema) []string {
	raw, ok := input["required"]
	if !ok || raw == nil {
		return nil
	}
	switch required := raw.(type) {
	case []string:
		return append([]string(nil), required...)
	case []any:
		out := make([]string, 0, len(required))
		for _, value := range required {
			if text, ok := value.(string); ok {
				out = append(out, text)
			}
		}
		return out
	default:
		return nil
	}
}

// CreateSubsetSchema creates an object schema containing only selected fields,
// preserving required markers and optional field descriptions.
func CreateSubsetSchema(name string, input schema.Schema, fieldNames []string, descriptions map[string]string, fnDescription string) (schema.Schema, error) {
	if !IsObjectSchema(input) {
		return nil, fmt.Errorf("expected object schema")
	}
	props := SchemaProperties(input)
	requiredSet := map[string]bool{}
	for _, field := range SchemaRequired(input) {
		requiredSet[field] = true
	}
	outProps := map[string]schema.Schema{}
	outRequired := []string{}
	for _, field := range fieldNames {
		prop, ok := props[field]
		if !ok {
			return nil, fmt.Errorf("schema field %q not found", field)
		}
		prop = CloneSchema(prop)
		if descriptions != nil {
			if description, ok := descriptions[field]; ok {
				prop["description"] = description
			}
		}
		outProps[field] = prop
		if requiredSet[field] {
			outRequired = append(outRequired, field)
		}
	}
	out := schema.Object(outProps, outRequired...)
	if name != "" {
		out["title"] = name
	}
	if fnDescription != "" {
		out["description"] = fnDescription
	} else if description, ok := input["description"].(string); ok && description != "" {
		out["description"] = description
	}
	return out, nil
}

// MustacheTemplateVariables returns sorted top-level variable names used by a
// simple Mustache template. Dotted paths such as {{user.name}} return "user",
// matching LangChain prompt-variable behavior.
func MustacheTemplateVariables(template string) []string {
	variables := map[string]struct{}{}
	for _, tag := range mustacheTags(template) {
		name, ok := normalizeMustacheVariableTag(tag)
		if !ok {
			continue
		}
		if idx := strings.Index(name, "."); idx >= 0 {
			name = name[:idx]
		}
		if name != "" {
			variables[name] = struct{}{}
		}
	}
	out := make([]string, 0, len(variables))
	for variable := range variables {
		out = append(out, variable)
	}
	sort.Strings(out)
	return out
}

// RenderSimpleMustache renders variable substitutions for the Mustache subset
// commonly used by prompt templates. It supports {{name}}, {{user.name}},
// {{{name}}}, and {{& name}}. Sections, inverted sections, partials, delimiter
// changes, and lambdas are intentionally unsupported and left unchanged.
func RenderSimpleMustache(template string, data map[string]any) string {
	return mustacheTagRegexp.ReplaceAllStringFunc(template, func(match string) string {
		tag := strings.TrimSpace(strings.TrimPrefix(strings.TrimSuffix(match, "}}"), "{{"))
		unescaped := false
		if strings.HasPrefix(match, "{{{") && strings.HasSuffix(match, "}}}") {
			tag = strings.TrimSpace(strings.TrimPrefix(strings.TrimSuffix(match, "}}}"), "{{{"))
			unescaped = true
		}
		if strings.HasPrefix(tag, "&") {
			tag = strings.TrimSpace(strings.TrimPrefix(tag, "&"))
			unescaped = true
		}
		name, ok := normalizeMustacheVariableTag(tag)
		if !ok {
			if strings.HasPrefix(tag, "!") {
				return ""
			}
			return match
		}
		value, ok := lookupDottedValue(data, name)
		if !ok || value == nil {
			return ""
		}
		rendered := StringifyValue(value)
		if unescaped {
			return rendered
		}
		return html.EscapeString(rendered)
	})
}

func validFunctionName(name string) bool {
	if name == "" || len(name) > 64 {
		return false
	}
	return regexp.MustCompile(`^[A-Za-z0-9_-]+$`).MatchString(name)
}

var mustacheTagRegexp = regexp.MustCompile(`(?s)\{\{\{.*?\}\}\}|\{\{.*?\}\}`)

func mustacheTags(template string) []string {
	matches := mustacheTagRegexp.FindAllString(template, -1)
	out := make([]string, 0, len(matches))
	for _, match := range matches {
		if strings.HasPrefix(match, "{{{") {
			out = append(out, strings.TrimSpace(strings.TrimPrefix(strings.TrimSuffix(match, "}}}"), "{{{")))
			continue
		}
		out = append(out, strings.TrimSpace(strings.TrimPrefix(strings.TrimSuffix(match, "}}"), "{{")))
	}
	return out
}

func normalizeMustacheVariableTag(tag string) (string, bool) {
	tag = strings.TrimSpace(tag)
	if tag == "" {
		return "", false
	}
	if strings.HasPrefix(tag, "&") {
		tag = strings.TrimSpace(strings.TrimPrefix(tag, "&"))
	}
	switch tag[0] {
	case '#', '^', '/', '!', '>', '<', '=':
		return "", false
	}
	return tag, true
}

func lookupDottedValue(data map[string]any, path string) (any, bool) {
	parts := strings.Split(path, ".")
	var current any = data
	for _, part := range parts {
		values, ok := current.(map[string]any)
		if !ok {
			return nil, false
		}
		current, ok = values[part]
		if !ok {
			return nil, false
		}
	}
	return current, true
}

func cloneMap(values map[string]any) map[string]any {
	if values == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}
