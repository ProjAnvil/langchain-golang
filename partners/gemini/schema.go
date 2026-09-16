package gemini

import (
	"fmt"

	"github.com/projanvil/langchain-golang/core/schema"
	"google.golang.org/genai"
)

// toGenAISchema converts a core JSON-schema map into the genai SDK's typed
// Schema. Python never needs this conversion: it passes dict schemas straight
// into google-genai, whose process_schema() handles the typing and $ref
// resolution internally. The Go SDK's equivalent escape hatches
// (FunctionDeclaration.ParametersJsonSchema / GenerateContentConfig
// .ResponseJsonSchema) are used where available; this converter serves the
// places that require the typed Schema — tool function declarations'
// Parameters (the field Google's own Go examples use).
//
// The genai Schema is a subset of OpenAPI 3.0 with uppercase type enums, so
// the conversion is a recursive walk that lower-case JSON-schema types onto
// genai.Type values and copies the constraints the Gemini API understands.
func toGenAISchema(s schema.Schema) (*genai.Schema, error) {
	return mapToGenAISchema(map[string]any(s))
}

func mapToGenAISchema(m map[string]any) (*genai.Schema, error) {
	out := &genai.Schema{}
	if err := applySchemaType(out, m["type"]); err != nil {
		return nil, err
	}
	if v, ok := m["description"].(string); ok {
		out.Description = v
	}
	if v, ok := m["format"].(string); ok {
		out.Format = v
	}
	if enum, err := schemaStringSlice(m["enum"]); err == nil && len(enum) > 0 {
		out.Enum = enum
	}
	if v, ok := m["nullable"].(bool); ok {
		out.Nullable = &v
	}
	out.Default = m["default"]

	if items, ok := asSchemaMap(m["items"]); ok {
		converted, err := mapToGenAISchema(items)
		if err != nil {
			return nil, err
		}
		out.Items = converted
	}
	if properties, ok := m["properties"].(map[string]any); ok && len(properties) > 0 {
		out.Properties = make(map[string]*genai.Schema, len(properties))
		for name, sub := range properties {
			subMap, ok := asSchemaMap(sub)
			if !ok {
				return nil, fmt.Errorf("gemini: schema property %q is not an object", name)
			}
			converted, err := mapToGenAISchema(subMap)
			if err != nil {
				return nil, fmt.Errorf("gemini: schema property %q: %w", name, err)
			}
			out.Properties[name] = converted
		}
	}
	if required, err := schemaStringSlice(m["required"]); err == nil && len(required) > 0 {
		out.Required = required
	}
	if anyOf, ok := m["anyOf"].([]any); ok && len(anyOf) > 0 {
		out.AnyOf = make([]*genai.Schema, 0, len(anyOf))
		for i, sub := range anyOf {
			subMap, ok := asSchemaMap(sub)
			if !ok {
				return nil, fmt.Errorf("gemini: anyOf[%d] is not an object", i)
			}
			converted, err := mapToGenAISchema(subMap)
			if err != nil {
				return nil, fmt.Errorf("gemini: anyOf[%d]: %w", i, err)
			}
			out.AnyOf = append(out.AnyOf, converted)
		}
	}

	if v, ok := numericValue(m["minimum"]); ok {
		out.Minimum = &v
	}
	if v, ok := numericValue(m["maximum"]); ok {
		out.Maximum = &v
	}
	if v, ok := intValue(m["minItems"]); ok {
		out.MinItems = &v
	}
	if v, ok := intValue(m["maxItems"]); ok {
		out.MaxItems = &v
	}
	if v, ok := intValue(m["minLength"]); ok {
		out.MinLength = &v
	}
	if v, ok := intValue(m["maxLength"]); ok {
		out.MaxLength = &v
	}
	return out, nil
}

// asSchemaMap reads a nested schema node that is either a plain
// map[string]any (decoded JSON) or a schema.Schema value (the named map type
// the core helpers produce — its dynamic type does not match a bare
// map[string]any assertion).
func asSchemaMap(value any) (map[string]any, bool) {
	switch v := value.(type) {
	case map[string]any:
		return v, true
	case schema.Schema:
		return v, true
	default:
		return nil, false
	}
}

// applySchemaType maps the JSON-schema "type" value (a string or a list of
// strings) onto the genai typed enum. A list takes the first non-"null"
// entry, mirroring how Python's nullable-type unions reduce.
func applySchemaType(out *genai.Schema, value any) error {
	switch t := value.(type) {
	case nil:
		// No type key: the Gemini API infers from the shape (Python relies on
		// the same leniency for partial schemas).
		return nil
	case string:
		typ, err := genaiType(t)
		if err != nil {
			return err
		}
		out.Type = typ
		return nil
	case []any:
		for _, entry := range t {
			name, ok := entry.(string)
			if !ok || name == "null" {
				continue
			}
			typ, err := genaiType(name)
			if err != nil {
				return err
			}
			out.Type = typ
			return nil
		}
		return nil
	default:
		return fmt.Errorf("gemini: unsupported schema type value %#v", value)
	}
}

func genaiType(name string) (genai.Type, error) {
	switch name {
	case "string":
		return genai.TypeString, nil
	case "number":
		return genai.TypeNumber, nil
	case "integer":
		return genai.TypeInteger, nil
	case "boolean":
		return genai.TypeBoolean, nil
	case "array":
		return genai.TypeArray, nil
	case "object":
		return genai.TypeObject, nil
	case "null":
		return genai.TypeNULL, nil
	default:
		return "", fmt.Errorf("gemini: unsupported schema type %q", name)
	}
}

// schemaStringSlice accepts []string (the repo's schema.Object convention) or
// a []any of strings (generic JSON) and normalizes it to []string.
func schemaStringSlice(value any) ([]string, error) {
	switch v := value.(type) {
	case []string:
		return v, nil
	case []any:
		out := make([]string, 0, len(v))
		for _, entry := range v {
			s, ok := entry.(string)
			if !ok {
				return nil, fmt.Errorf("gemini: expected string entries, got %#v", entry)
			}
			out = append(out, s)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("gemini: expected a string list, got %#v", value)
	}
}

func numericValue(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	default:
		return 0, false
	}
}

func intValue(value any) (int64, bool) {
	switch v := value.(type) {
	case float64:
		return int64(v), true
	case int:
		return int64(v), true
	case int64:
		return v, true
	default:
		return 0, false
	}
}
