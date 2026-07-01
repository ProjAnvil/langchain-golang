package runnables

import (
	"sort"

	"github.com/projanvil/langchain-golang/core/schema"
)

type configSchemaProvider interface {
	ConfigSchema() schema.Schema
}

// GetConfigSchema returns the runtime configuration schema exposed by a
// runnable. Runnables without explicit configurable fields return an empty
// object schema.
func GetConfigSchema(runnable any) schema.Schema {
	if provider, ok := runnable.(configSchemaProvider); ok {
		return cloneSchema(provider.ConfigSchema())
	}
	return emptyConfigSchema()
}

func emptyConfigSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func configurableConfigSchema(properties map[string]schema.Schema, required ...string) schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"configurable": schema.Object(properties, required...),
	})
}

func mergeConfigSchemas(runnables ...any) schema.Schema {
	properties := map[string]schema.Schema{}
	requiredSet := map[string]bool{}
	for _, runnable := range runnables {
		cfg := GetConfigSchema(runnable)
		configurable, ok := configurableSchema(cfg)
		if !ok {
			continue
		}
		for key, prop := range schemaProperties(configurable) {
			properties[key] = prop
		}
		for _, key := range schemaRequired(configurable) {
			requiredSet[key] = true
		}
	}
	if len(properties) == 0 && len(requiredSet) == 0 {
		return emptyConfigSchema()
	}
	required := make([]string, 0, len(requiredSet))
	for key := range requiredSet {
		required = append(required, key)
	}
	sort.Strings(required)
	return configurableConfigSchema(properties, required...)
}

func configurableSchema(cfg schema.Schema) (schema.Schema, bool) {
	props := schemaProperties(cfg)
	value, ok := props["configurable"]
	return value, ok
}

func schemaProperties(cfg schema.Schema) map[string]schema.Schema {
	rawProps, ok := cfg["properties"]
	if !ok {
		return nil
	}
	if typed, ok := rawProps.(schema.Schema); ok {
		out := make(map[string]schema.Schema, len(typed))
		for key, value := range typed {
			switch child := value.(type) {
			case schema.Schema:
				out[key] = cloneSchema(child)
			case map[string]any:
				out[key] = cloneSchema(schema.Schema(child))
			}
		}
		return out
	}
	props, ok := rawProps.(map[string]any)
	if !ok {
		if typed, ok := rawProps.(map[string]schema.Schema); ok {
			return cloneSchemaMap(typed)
		}
		return nil
	}
	out := make(map[string]schema.Schema, len(props))
	for key, value := range props {
		switch typed := value.(type) {
		case schema.Schema:
			out[key] = cloneSchema(typed)
		case map[string]any:
			out[key] = cloneSchema(schema.Schema(typed))
		}
	}
	return out
}

func schemaRequired(cfg schema.Schema) []string {
	raw, ok := cfg["required"]
	if !ok {
		return nil
	}
	switch typed := raw.(type) {
	case []string:
		return append([]string(nil), typed...)
	case []any:
		out := make([]string, 0, len(typed))
		for _, value := range typed {
			if s, ok := value.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func cloneSchema(input schema.Schema) schema.Schema {
	if input == nil {
		return schema.Schema{}
	}
	out := make(schema.Schema, len(input))
	for key, value := range input {
		switch typed := value.(type) {
		case schema.Schema:
			out[key] = cloneSchema(typed)
		case map[string]schema.Schema:
			out[key] = cloneSchemaMap(typed)
		case map[string]any:
			out[key] = cloneSchema(schema.Schema(typed))
		case []string:
			out[key] = append([]string(nil), typed...)
		case []any:
			out[key] = append([]any(nil), typed...)
		default:
			out[key] = value
		}
	}
	return out
}

func cloneSchemaMap(input map[string]schema.Schema) map[string]schema.Schema {
	out := make(map[string]schema.Schema, len(input))
	for key, value := range input {
		out[key] = cloneSchema(value)
	}
	return out
}
