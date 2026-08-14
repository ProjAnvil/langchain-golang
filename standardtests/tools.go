package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// ToolFactory creates a fresh tool for standard tests.
type ToolFactory func(t testing.TB) tools.Tool

// RunToolBasics verifies behavior expected from every tool. It covers the same
// ground as Python's ToolsUnitTests (test_has_name, test_has_input_schema) and
// ToolsIntegrationTests (test_invoke_matches_output_schema).
func RunToolBasics(t *testing.T, factory ToolFactory) {
	t.Helper()

	t.Run("has name", func(t *testing.T) {
		tool := factory(t)
		if tool.Name() == "" {
			t.Fatal("expected non-empty tool name")
		}
	})

	t.Run("has input schema", func(t *testing.T) {
		tool := factory(t)
		if len(tool.ArgsSchema()) == 0 {
			t.Fatal("expected non-empty input schema")
		}
	})

	t.Run("invoke returns content", func(t *testing.T) {
		tool := factory(t)
		input := minimalToolInput(tool.ArgsSchema())
		result, err := tool.Invoke(context.Background(), input)
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if result.Content == "" {
			t.Fatal("expected non-empty content from invoke")
		}
	})
}

// minimalToolInput derives a plausible input map from a tool's argument schema,
// supplying a zero value for each declared property so the runner can invoke a
// tool without knowing its concrete argument shape.
func minimalToolInput(argsSchema schema.Schema) map[string]any {
	input := make(map[string]any)
	props, ok := argsSchema["properties"].(map[string]any)
	if !ok {
		return input
	}
	for name, raw := range props {
		prop := asSchema(raw)
		if prop == nil {
			continue
		}
		input[name] = zeroValueForSchema(prop)
	}
	return input
}

// asSchema coerces a schema property into schema.Schema, tolerating both the
// typed form and a plain map form.
func asSchema(raw any) schema.Schema {
	switch v := raw.(type) {
	case schema.Schema:
		return v
	case map[string]any:
		return schema.Schema(v)
	default:
		return nil
	}
}

// zeroValueForSchema returns a JSON-schema-valid zero value for the declared
// property type.
func zeroValueForSchema(prop schema.Schema) any {
	switch typ, _ := prop["type"].(string); typ {
	case "string":
		return ""
	case "integer", "number":
		return 0
	case "boolean":
		return false
	case "array":
		return []any{}
	case "object":
		return map[string]any{}
	default:
		return nil
	}
}
