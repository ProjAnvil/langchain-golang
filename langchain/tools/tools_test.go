package tools

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

func TestV1ToolsExportCore(t *testing.T) {
	tool, err := NewFunc("echo", "", schema.Object(nil), func(ctx context.Context, input map[string]any) (Result, error) {
		return Result{Content: input["x"].(string)}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := tool.Invoke(context.Background(), map[string]any{"x": "ok"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Content != "ok" {
		t.Fatalf("Content = %q", got.Content)
	}
}

func TestV1ToolsExportSimpleAndStructured(t *testing.T) {
	simple, err := NewSimple("echo", "", func(context.Context, string) (Result, error) {
		return Result{Content: "echo"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if simple.Name() != "echo" {
		t.Fatalf("simple name = %q", simple.Name())
	}

	structured, err := NewStructuredFunc("structured", "", schema.Object(nil), func(context.Context, map[string]any) (Result, error) {
		return Result{Content: "ok"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := structured.Invoke(context.Background(), map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Content != "ok" {
		t.Fatalf("Content = %q", got.Content)
	}
}
