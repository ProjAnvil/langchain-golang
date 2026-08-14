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
