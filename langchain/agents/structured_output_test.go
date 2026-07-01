package agents

import (
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

func weatherSchema() schema.Schema {
	return schema.Schema{
		"type":        "object",
		"title":       "weather_schema",
		"description": "Weather response.",
		"properties": map[string]any{
			"temperature": map[string]any{"type": "number", "description": "Temperature in fahrenheit"},
			"condition":   map[string]any{"type": "string", "description": "Weather condition"},
		},
		"required": []string{"temperature", "condition"},
	}
}

func locationSchema() schema.Schema {
	return schema.Schema{
		"type":  "object",
		"title": "location_schema",
		"properties": map[string]any{
			"city":    map[string]any{"type": "string"},
			"country": map[string]any{"type": "string"},
		},
		"required": []string{"city", "country"},
	}
}

func TestToolStrategyJSONSchema(t *testing.T) {
	strategy := NewToolStrategy(weatherSchema())
	if strategy.ToolMessageContent != "" {
		t.Fatalf("expected empty tool message content, got %q", strategy.ToolMessageContent)
	}
	if len(strategy.SchemaSpecs) != 1 {
		t.Fatalf("expected one schema spec, got %d", len(strategy.SchemaSpecs))
	}
	spec := strategy.SchemaSpecs[0]
	if spec.Name != "weather_schema" {
		t.Fatalf("schema name mismatch: got %q", spec.Name)
	}
	if spec.Description != "Weather response." {
		t.Fatalf("schema description mismatch: got %q", spec.Description)
	}
	if spec.SchemaKind != SchemaKindJSONSchema {
		t.Fatalf("schema kind mismatch: got %q", spec.SchemaKind)
	}
}

func TestToolStrategyWithToolMessageContent(t *testing.T) {
	strategy := NewToolStrategy(weatherSchema(), WithToolMessageContent("custom message"))
	if strategy.Schema == nil {
		t.Fatal("expected schema to be retained")
	}
	if strategy.ToolMessageContent != "custom message" {
		t.Fatalf("tool message content mismatch: got %q", strategy.ToolMessageContent)
	}
	if len(strategy.SchemaSpecs) != 1 {
		t.Fatalf("expected one schema spec, got %d", len(strategy.SchemaSpecs))
	}
}

func TestToolStrategyOneOfJSONSchemas(t *testing.T) {
	strategy := NewToolStrategy(schema.Schema{
		"oneOf": []any{weatherSchema(), locationSchema()},
	})
	if len(strategy.SchemaSpecs) != 2 {
		t.Fatalf("expected two schema specs, got %d", len(strategy.SchemaSpecs))
	}
	if strategy.SchemaSpecs[0].Name != "weather_schema" {
		t.Fatalf("first schema mismatch: got %q", strategy.SchemaSpecs[0].Name)
	}
	if strategy.SchemaSpecs[1].Name != "location_schema" {
		t.Fatalf("second schema mismatch: got %q", strategy.SchemaSpecs[1].Name)
	}
}

func TestSchemaSpecCustomNameAndDescription(t *testing.T) {
	spec := NewSchemaSpec(
		weatherSchema(),
		WithSchemaName("custom_tool_name"),
		WithSchemaDescription("Custom tool description"),
	)
	if spec.Name != "custom_tool_name" {
		t.Fatalf("name mismatch: got %q", spec.Name)
	}
	if spec.Description != "Custom tool description" {
		t.Fatalf("description mismatch: got %q", spec.Description)
	}
}

func TestProviderStrategyToModelKwargs(t *testing.T) {
	strategy := NewProviderStrategy(weatherSchema())
	got := strategy.ToModelKwargs()
	want := map[string]any{
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "weather_schema",
				"schema": weatherSchema(),
			},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("kwargs mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestProviderStrategyCreation(t *testing.T) {
	strategy := NewProviderStrategy(weatherSchema())
	if !reflect.DeepEqual(strategy.Schema, weatherSchema()) {
		t.Fatalf("schema mismatch: got %#v", strategy.Schema)
	}
	if strategy.SchemaSpec.Strict != nil {
		t.Fatalf("expected nil strict, got %#v", *strategy.SchemaSpec.Strict)
	}
}

func TestProviderStrategyToModelKwargsStrict(t *testing.T) {
	strategy := NewProviderStrategy(weatherSchema(), WithStrict(true))
	got := strategy.ToModelKwargs()

	responseFormat := got["response_format"].(map[string]any)
	jsonSchema := responseFormat["json_schema"].(map[string]any)
	if jsonSchema["strict"] != true {
		t.Fatalf("expected strict=true, got %#v", jsonSchema["strict"])
	}
}

func TestAutoStrategy(t *testing.T) {
	strategy := NewAutoStrategy(weatherSchema())
	if !reflect.DeepEqual(strategy.Schema, weatherSchema()) {
		t.Fatalf("schema mismatch: got %#v", strategy.Schema)
	}
}

func TestOutputToolBindingFromSchemaSpec(t *testing.T) {
	binding, err := OutputToolBindingFromSchemaSpec(NewSchemaSpec(weatherSchema()))
	if err != nil {
		t.Fatalf("binding from schema spec: %v", err)
	}
	if binding.SchemaKind != SchemaKindJSONSchema {
		t.Fatalf("schema kind mismatch: got %q", binding.SchemaKind)
	}
	if binding.Tool.Name() != "weather_schema" {
		t.Fatalf("tool name mismatch: got %q", binding.Tool.Name())
	}
	if binding.Tool.Description() != "Weather response." {
		t.Fatalf("tool description mismatch: got %q", binding.Tool.Description())
	}

	data := map[string]any{"temperature": 75.0, "condition": "sunny"}
	parsed, err := binding.Parse(data)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !reflect.DeepEqual(parsed, data) {
		t.Fatalf("parsed mismatch: got %#v want %#v", parsed, data)
	}
}

func TestMultipleStructuredOutputsError(t *testing.T) {
	aiMessage := messages.AI("tool calls")
	err := NewMultipleStructuredOutputsError([]string{"Weather", "Location"}, aiMessage)

	if !reflect.DeepEqual(err.ToolNames, []string{"Weather", "Location"}) {
		t.Fatalf("tool names mismatch: %#v", err.ToolNames)
	}
	if err.AIMessage.Content != "tool calls" {
		t.Fatalf("AI message mismatch: %#v", err.AIMessage)
	}
	if !strings.Contains(err.Error(), "Weather, Location") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestStructuredOutputValidationError(t *testing.T) {
	aiMessage := messages.AI("bad args")
	source := assertErr("missing required field")
	err := NewStructuredOutputValidationError("Weather", source, aiMessage)

	if err.ToolName != "Weather" {
		t.Fatalf("tool name mismatch: %q", err.ToolName)
	}
	if err.Source != source {
		t.Fatalf("source mismatch: %#v", err.Source)
	}
	if err.AIMessage.Content != "bad args" {
		t.Fatalf("AI message mismatch: %#v", err.AIMessage)
	}
	if !strings.Contains(err.Error(), "Failed to parse structured output for tool 'Weather': missing required field.") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestProviderStrategyBindingParseJSONSchema(t *testing.T) {
	binding := ProviderStrategyBindingFromSchemaSpec(NewSchemaSpec(weatherSchema()))
	parsed, err := binding.Parse(messages.AI(`{"temperature":75,"condition":"sunny"}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := map[string]any{"temperature": float64(75), "condition": "sunny"}
	if !reflect.DeepEqual(parsed, want) {
		t.Fatalf("parsed mismatch: got %#v want %#v", parsed, want)
	}
}

type assertErr string

func (e assertErr) Error() string {
	return string(e)
}

func TestProviderStrategyBindingParseInvalidJSON(t *testing.T) {
	binding := ProviderStrategyBindingFromSchemaSpec(NewSchemaSpec(weatherSchema()))
	_, err := binding.Parse(messages.AI("invalid json"))
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "Native structured output expected valid JSON for weather_schema") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestProviderStrategyBindingParseContentBlocks(t *testing.T) {
	binding := ProviderStrategyBindingFromSchemaSpec(NewSchemaSpec(weatherSchema()))
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		{"content": `{"temperature":`},
		{"type": "text", "text": `75,`},
		{"type": "text", "text": `"condition":"sunny"}`},
	}

	parsed, err := binding.Parse(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := map[string]any{"temperature": float64(75), "condition": "sunny"}
	if !reflect.DeepEqual(parsed, want) {
		t.Fatalf("parsed mismatch: got %#v want %#v", parsed, want)
	}
}
