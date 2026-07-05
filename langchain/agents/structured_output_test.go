package agents

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
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

// TestAutoStrategy_SelectsTool covers AutoStrategy.Resolve's capability-based
// dispatch: ToolCalling models select ToolStrategy, StructuredOutput-only
// models select ProviderStrategy, and models supporting neither surface a typed
// *StructuredOutputUnsupportedError. Mirrors the brief's required test name.
func TestAutoStrategy_SelectsTool(t *testing.T) {
	schemaSpec := weatherSchema()

	t.Run("tool_calling_model_selects_tool_strategy", func(t *testing.T) {
		auto := NewAutoStrategy(schemaSpec)
		model := language.NewFakeChatModel(language.WithCapabilities(language.ChatModelCapabilities{
			ToolCalling:      true,
			StructuredOutput: true,
		}))
		resolved, err := auto.Resolve(model)
		if err != nil {
			t.Fatalf("Resolve: unexpected error: %v", err)
		}
		toolStrategy, ok := resolved.(ToolStrategy)
		if !ok {
			t.Fatalf("expected ToolStrategy, got %T", resolved)
		}
		if !reflect.DeepEqual(toolStrategy.Schema, schemaSpec) {
			t.Fatalf("ToolStrategy schema mismatch: got %#v", toolStrategy.Schema)
		}
	})

	t.Run("tool_calling_takes_precedence_over_structured_output", func(t *testing.T) {
		auto := NewAutoStrategy(schemaSpec)
		model := language.NewFakeChatModel(language.WithCapabilities(language.ChatModelCapabilities{
			ToolCalling:      true,
			StructuredOutput: true,
		}))
		resolved, err := auto.Resolve(model)
		if err != nil {
			t.Fatalf("Resolve: unexpected error: %v", err)
		}
		if _, ok := resolved.(ToolStrategy); !ok {
			t.Fatalf("expected ToolStrategy to win when both capabilities are set, got %T", resolved)
		}
	})

	t.Run("structured_output_only_model_selects_provider_strategy", func(t *testing.T) {
		auto := NewAutoStrategy(schemaSpec)
		model := language.NewFakeChatModel(language.WithCapabilities(language.ChatModelCapabilities{
			ToolCalling:      false,
			StructuredOutput: true,
		}))
		resolved, err := auto.Resolve(model)
		if err != nil {
			t.Fatalf("Resolve: unexpected error: %v", err)
		}
		providerStrategy, ok := resolved.(ProviderStrategy)
		if !ok {
			t.Fatalf("expected ProviderStrategy, got %T", resolved)
		}
		if !reflect.DeepEqual(providerStrategy.Schema, schemaSpec) {
			t.Fatalf("ProviderStrategy schema mismatch: got %#v", providerStrategy.Schema)
		}
	})

	t.Run("neither_capability_returns_typed_error", func(t *testing.T) {
		auto := NewAutoStrategy(schemaSpec)
		model := language.NewFakeChatModel(language.WithCapabilities(language.ChatModelCapabilities{
			ToolCalling:      false,
			StructuredOutput: false,
		}))
		resolved, err := auto.Resolve(model)
		if err == nil {
			t.Fatalf("expected error, got resolved value %T", resolved)
		}
		var unsupported *StructuredOutputUnsupportedError
		if !errors.As(err, &unsupported) {
			t.Fatalf("expected *StructuredOutputUnsupportedError, got %T (%v)", err, err)
		}
		if resolved != nil {
			t.Fatalf("expected nil resolved strategy on error, got %T", resolved)
		}
	})

	t.Run("nil_model_returns_typed_error", func(t *testing.T) {
		auto := NewAutoStrategy(schemaSpec)
		resolved, err := auto.Resolve(nil)
		if err == nil {
			t.Fatalf("expected error, got resolved value %T", resolved)
		}
		var unsupported *StructuredOutputUnsupportedError
		if !errors.As(err, &unsupported) {
			t.Fatalf("expected *StructuredOutputUnsupportedError, got %T (%v)", err, err)
		}
	})
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

// nativeStructuredSequenceModel embeds *sequenceModel to satisfy
// language.ChatModel while also implementing language.StructuredCaller. It
// records whether InvokeStructured was called (the native path) and shares the
// response queue + invocations log with the embedded sequenceModel so the test
// can detect whether plain Invoke was called instead (the fallback path).
type nativeStructuredSequenceModel struct {
	*sequenceModel
	nativeCalled bool
	nativeSchema schema.Schema
}

// InvokeStructured records the native call and dequeues the next response from
// the embedded sequenceModel's queue. It deliberately does NOT call the
// embedded Invoke, so len(sequenceModel.invocations) stays 0 when the native
// path is used — the test asserts this to prove ProviderStrategy routes
// through InvokeStructured instead of Invoke.
func (m *nativeStructuredSequenceModel) InvokeStructured(
	ctx context.Context,
	input []messages.Message,
	sch schema.Schema,
) (messages.Message, error) {
	m.nativeCalled = true
	m.nativeSchema = sch
	m.sequenceModel.mu.Lock()
	defer m.sequenceModel.mu.Unlock()
	if m.sequenceModel.idx >= len(m.sequenceModel.responses) {
		return messages.Message{}, fmt.Errorf("nativeStructuredSequenceModel: no more responses (call %d)", m.sequenceModel.idx+1)
	}
	resp := m.sequenceModel.responses[m.sequenceModel.idx]
	m.sequenceModel.idx++
	return resp, nil
}

// BindTools forwards to the embedded sequenceModel but returns the wrapper so
// the bound model still implements StructuredCaller (the bound value is what
// invokeModel inspects against the StructuredCaller interface — see the brief's
// "BindTools must run before the StructuredCaller check" constraint).
func (m *nativeStructuredSequenceModel) BindTools(boundTools []coretools.Tool) (language.ChatModel, error) {
	if _, err := m.sequenceModel.BindTools(boundTools); err != nil {
		return nil, err
	}
	return m, nil
}

// TestProviderStrategyUsesStructuredCaller proves Task 4.3: when the agent's
// ResponseFormat is a ProviderStrategy and the bound model implements
// language.StructuredCaller, the model call routes through
// language.InvokeStructured (the native path) instead of plain model.Invoke.
// The post-hoc detectStructuredOutput parse still extracts structured_response.
func TestProviderStrategyUsesStructuredCaller(t *testing.T) {
	strategy := NewProviderStrategy(weatherSchema())

	model := &nativeStructuredSequenceModel{
		sequenceModel: &sequenceModel{responses: []messages.Message{
			messages.AI(`{"temperature":72,"condition":"sunny"}`),
		}},
	}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("weather in Tokyo?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if !model.nativeCalled {
		t.Fatal("expected InvokeStructured (native path) to be called for ProviderStrategy + StructuredCaller model")
	}
	if len(model.sequenceModel.invocations) != 0 {
		t.Fatalf("expected zero plain Invoke calls on the fallback path; got %d", len(model.sequenceModel.invocations))
	}
	if !reflect.DeepEqual(model.nativeSchema, weatherSchema()) {
		t.Fatalf("native InvokeStructured schema mismatch: got %#v want %#v", model.nativeSchema, weatherSchema())
	}
	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["condition"] != "sunny" {
		t.Fatalf("expected structured_response.condition=sunny, got %#v", state["structured_response"])
	}
}

// TestProviderStrategyFallsBackWithoutStructuredCaller proves the fallback
// path: when the bound model does NOT implement StructuredCaller, the existing
// Invoke + post-hoc JSON-decode path still extracts structured_response. This
// preserves backward compatibility for non-native models.
func TestProviderStrategyFallsBackWithoutStructuredCaller(t *testing.T) {
	strategy := NewProviderStrategy(weatherSchema())

	// Plain sequenceModel does NOT implement StructuredCaller.
	model := &sequenceModel{responses: []messages.Message{
		messages.AI(`{"temperature":55,"condition":"rainy"}`),
	}}

	agent, err := CreateAgent(model, nil, WithAgentResponseFormat(strategy))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("weather in London?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(model.invocations) != 1 {
		t.Fatalf("expected exactly one plain Invoke call (fallback path); got %d", len(model.invocations))
	}
	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["condition"] != "rainy" {
		t.Fatalf("expected structured_response.condition=rainy, got %#v", state["structured_response"])
	}
}
