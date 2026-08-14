package utils

import (
	"fmt"

	"github.com/projanvil/langchain-golang/core/schema"
)

// toolLike is the minimal structural contract shared by core/tools.Func,
// core/tools.Simple, and the core/tools.Tool interface. core/utils cannot
// import core/tools (that package imports core/utils), so the tool arm of
// ConvertToOpenAIFunction matches tools structurally rather than by name.
type toolLike interface {
	Name() string
	Description() string
	ArgsSchema() schema.Schema
}

// ConvertToOpenAIFunction converts a function-like value into an OpenAI
// function-calling dict of shape {"name", "description", "parameters"}.
//
// It accepts, mirroring Python's convert_to_openai_function:
//
//   - a map[string]any already in OpenAI function form (carrying name,
//     description, parameters, and optionally strict); unknown keys are
//     dropped;
//   - a schema.Schema (treated like a dict);
//   - a FunctionSpec (or *FunctionSpec) built by NewFunctionSpec;
//   - a tool matching the structural core/tools.Tool contract (Name,
//     Description, and ArgsSchema).
//
// For dict inputs a missing "parameters" key is normalized to an empty
// object schema ({"type": "object", "properties": {}}), and a "strict" key,
// when present, is carried through unchanged.
func ConvertToOpenAIFunction(spec any) (map[string]any, error) {
	switch value := spec.(type) {
	case map[string]any:
		return convertDictToOpenAIFunction(value)
	case schema.Schema:
		return convertDictToOpenAIFunction(map[string]any(value))
	case FunctionSpec:
		return functionSpecToOpenAIFunction(value), nil
	case *FunctionSpec:
		if value == nil {
			return nil, fmt.Errorf("ConvertToOpenAIFunction: nil *FunctionSpec")
		}
		return functionSpecToOpenAIFunction(*value), nil
	case toolLike:
		return toolToOpenAIFunction(value), nil
	default:
		return nil, fmt.Errorf("ConvertToOpenAIFunction: unsupported function type %T", spec)
	}
}

// convertDictToOpenAIFunction normalizes an OpenAI-function dict, keeping
// only name, description, parameters, and strict. name is required.
func convertDictToOpenAIFunction(input map[string]any) (map[string]any, error) {
	nameValue, ok := input["name"]
	if !ok {
		return nil, fmt.Errorf("ConvertToOpenAIFunction: dict must contain a 'name' key")
	}
	name, ok := nameValue.(string)
	if !ok {
		return nil, fmt.Errorf("ConvertToOpenAIFunction: 'name' must be a string, got %T", nameValue)
	}

	description := ""
	if raw, ok := input["description"]; ok {
		if text, ok := raw.(string); ok {
			description = text
		}
	}

	params := emptyObjectSchema()
	if raw, ok := input["parameters"]; ok {
		schemaMap, ok := asSchemaMap(raw)
		if !ok {
			return nil, fmt.Errorf("ConvertToOpenAIFunction: 'parameters' must be an object, got %T", raw)
		}
		params = schemaMap
	}

	out := map[string]any{
		"name":        name,
		"description": description,
		"parameters":  params,
	}
	if strict, ok := input["strict"]; ok {
		out["strict"] = strict
	}
	return out, nil
}

// toolToOpenAIFunction derives an OpenAI function dict from a structural tool.
func toolToOpenAIFunction(tool toolLike) map[string]any {
	params := map[string]any(tool.ArgsSchema())
	if params == nil {
		params = emptyObjectSchema()
	}
	return map[string]any{
		"name":        tool.Name(),
		"description": tool.Description(),
		"parameters":  params,
	}
}

// functionSpecToOpenAIFunction derives an OpenAI function dict from a
// provider-neutral FunctionSpec.
func functionSpecToOpenAIFunction(spec FunctionSpec) map[string]any {
	params := map[string]any(spec.Parameters)
	if params == nil {
		params = emptyObjectSchema()
	}
	return map[string]any{
		"name":        spec.Name,
		"description": spec.Description,
		"parameters":  params,
	}
}

// asSchemaMap coerces a schema-valued any into a plain map[string]any.
func asSchemaMap(value any) (map[string]any, bool) {
	switch v := value.(type) {
	case map[string]any:
		return v, true
	case schema.Schema:
		return map[string]any(v), true
	default:
		return nil, false
	}
}

// emptyObjectSchema returns a JSON object schema with no properties.
func emptyObjectSchema() map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}
}
