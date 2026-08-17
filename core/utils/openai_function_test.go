package utils_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/core/utils"
)

func TestConvertToOpenAIFunctionDictPassthrough(t *testing.T) {
	input := map[string]any{
		"name":        "search",
		"description": "searches things",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{"query": map[string]any{"type": "string"}},
			"required":   []string{"query"},
		},
		"extra": "should be dropped",
	}
	got, err := utils.ConvertToOpenAIFunction(input)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"name":        "search",
		"description": "searches things",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{"query": map[string]any{"type": "string"}},
			"required":   []string{"query"},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ConvertToOpenAIFunction:\n got %#v\nwant %#v", got, want)
	}
}

func TestConvertToOpenAIFunctionDictDefaultParameters(t *testing.T) {
	input := map[string]any{"name": "search", "description": "searches things"}
	got, err := utils.ConvertToOpenAIFunction(input)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"name":        "search",
		"description": "searches things",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ConvertToOpenAIFunction:\n got %#v\nwant %#v", got, want)
	}
}

func TestConvertToOpenAIFunctionFromFunc(t *testing.T) {
	fn, err := tools.NewFunc(
		"search",
		"searches things",
		schema.Object(map[string]schema.Schema{"query": schema.String("query text")}, "query"),
		func(context.Context, map[string]any) (tools.Result, error) { return tools.Result{}, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	got, err := utils.ConvertToOpenAIFunction(fn)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"name":        "search",
		"description": "searches things",
		"parameters":  map[string]any(fn.ArgsSchema()),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ConvertToOpenAIFunction:\n got %#v\nwant %#v", got, want)
	}
}

func TestConvertToOpenAIFunctionPreservesStrict(t *testing.T) {
	input := map[string]any{
		"name":        "search",
		"description": "searches things",
		"parameters":  map[string]any{"type": "object", "properties": map[string]any{}},
		"strict":      true,
	}
	got, err := utils.ConvertToOpenAIFunction(input)
	if err != nil {
		t.Fatal(err)
	}
	if strict, ok := got["strict"].(bool); !ok || !strict {
		t.Fatalf("strict not preserved: %#v", got)
	}
	if got["name"] != "search" || got["description"] != "searches things" {
		t.Fatalf("unexpected function: %#v", got)
	}
}

func TestConvertToOpenAIFunctionFromFunctionSpec(t *testing.T) {
	spec, err := utils.NewFunctionSpec(
		"search",
		"searches things",
		schema.Object(map[string]schema.Schema{"query": schema.String("query text")}, "query"),
	)
	if err != nil {
		t.Fatal(err)
	}
	got, err := utils.ConvertToOpenAIFunction(spec)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"name":        "search",
		"description": "searches things",
		"parameters":  map[string]any(spec.Parameters),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ConvertToOpenAIFunction:\n got %#v\nwant %#v", got, want)
	}
}

func TestConvertToOpenAIFunctionUnsupported(t *testing.T) {
	if _, err := utils.ConvertToOpenAIFunction("nope"); err == nil {
		t.Fatal("expected error for unsupported input")
	}
	if _, err := utils.ConvertToOpenAIFunction(map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}); err == nil {
		t.Fatal("expected error for dict without name")
	}
}

// nilSchemaTool satisfies the structural tool contract with a nil ArgsSchema.
type nilSchemaTool struct{}

func (nilSchemaTool) Name() string              { return "nil_schema_tool" }
func (nilSchemaTool) Description() string       { return "tool with nil schema" }
func (nilSchemaTool) ArgsSchema() schema.Schema { return nil }

func TestConvertToOpenAIFunctionFromSchema(t *testing.T) {
	input := schema.Schema{
		"name":       "search",
		"parameters": schema.Object(map[string]schema.Schema{"query": schema.String("query")}, "query"),
	}
	got, err := utils.ConvertToOpenAIFunction(input)
	if err != nil {
		t.Fatal(err)
	}
	if got["name"] != "search" || got["description"] != "" {
		t.Fatalf("unexpected function: %#v", got)
	}
	params, ok := got["parameters"].(map[string]any)
	if !ok || params["type"] != "object" {
		t.Fatalf("parameters: %#v", got["parameters"])
	}
}

func TestConvertToOpenAIFunctionFromFunctionSpecPointer(t *testing.T) {
	spec, err := utils.NewFunctionSpec("search", "searches things", nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := utils.ConvertToOpenAIFunction(&spec)
	if err != nil {
		t.Fatal(err)
	}
	if got["name"] != "search" || got["description"] != "searches things" {
		t.Fatalf("unexpected function: %#v", got)
	}
	if _, err := utils.ConvertToOpenAIFunction((*utils.FunctionSpec)(nil)); err == nil {
		t.Fatal("expected error for nil *FunctionSpec")
	}
}

func TestConvertToOpenAIFunctionFunctionSpecNilParameters(t *testing.T) {
	got, err := utils.ConvertToOpenAIFunction(utils.FunctionSpec{Name: "bare"})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"name":        "bare",
		"description": "",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ConvertToOpenAIFunction:\n got %#v\nwant %#v", got, want)
	}
}

func TestConvertToOpenAIFunctionToolNilSchema(t *testing.T) {
	got, err := utils.ConvertToOpenAIFunction(nilSchemaTool{})
	if err != nil {
		t.Fatal(err)
	}
	if got["name"] != "nil_schema_tool" || got["description"] != "tool with nil schema" {
		t.Fatalf("unexpected function: %#v", got)
	}
	params, ok := got["parameters"].(map[string]any)
	if !ok || params["type"] != "object" {
		t.Fatalf("nil schema should default to empty object schema: %#v", got["parameters"])
	}
}

func TestConvertToOpenAIFunctionDictSchemaParameters(t *testing.T) {
	input := map[string]any{
		"name":       "search",
		"parameters": schema.Object(map[string]schema.Schema{"query": schema.String("query")}),
	}
	got, err := utils.ConvertToOpenAIFunction(input)
	if err != nil {
		t.Fatal(err)
	}
	params, ok := got["parameters"].(map[string]any)
	if !ok || params["type"] != "object" {
		t.Fatalf("parameters: %#v", got["parameters"])
	}
}

func TestConvertToOpenAIFunctionDictErrors(t *testing.T) {
	if _, err := utils.ConvertToOpenAIFunction(map[string]any{"name": 42}); err == nil {
		t.Fatal("expected error for non-string name")
	}
	if _, err := utils.ConvertToOpenAIFunction(map[string]any{
		"name":       "search",
		"parameters": "not an object",
	}); err == nil {
		t.Fatal("expected error for non-object parameters")
	}
	// A non-string description is ignored rather than an error.
	got, err := utils.ConvertToOpenAIFunction(map[string]any{"name": "search", "description": 7})
	if err != nil {
		t.Fatal(err)
	}
	if got["description"] != "" {
		t.Fatalf("non-string description should be dropped: %#v", got)
	}
}
