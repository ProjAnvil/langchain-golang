package runnables

import (
	"context"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

// TestConfigurableFieldsOverridesAndDefaults covers the core observable
// behavior of ConfigurableFields: an explicit Configurable override is passed
// through to the inner runnable, and an unset field falls back to its default.
func TestConfigurableFieldsOverridesAndDefaults(t *testing.T) {
	var seen []any
	inner := NewFunc(func(_ context.Context, input string, opts ...Option) (string, error) {
		cfg := NewConfig(opts...)
		seen = append(seen, cfg.Configurable["temperature"])
		return input, nil
	}, schema.String(""), schema.String(""))

	wrapped, err := ConfigurableFields[string, string](inner, ConfigurableField{
		ID:          "temperature",
		Name:        "LLM Temperature",
		Description: "The temperature of the model",
		Default:     0.7,
	})
	if err != nil {
		t.Fatalf("configurable fields: %v", err)
	}

	// (b) Explicit override is passed through to the inner runnable.
	if _, err := wrapped.Invoke(context.Background(), "input", WithConfigurable("temperature", 0.9)); err != nil {
		t.Fatalf("invoke with override: %v", err)
	}
	if len(seen) != 1 || seen[0] != 0.9 {
		t.Fatalf("override seen=%#v", seen)
	}

	// (c) Unset key falls back to the field default.
	if _, err := wrapped.Invoke(context.Background(), "input"); err != nil {
		t.Fatalf("invoke with default: %v", err)
	}
	if len(seen) != 2 || seen[1] != 0.7 {
		t.Fatalf("default seen=%#v", seen)
	}
}

func TestConfigurableFieldsConstruction(t *testing.T) {
	inner := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))

	if _, err := ConfigurableFields[string, string](nil); err == nil {
		t.Fatal("expected error for nil runnable")
	}
	if _, err := ConfigurableFields[string, string](inner, ConfigurableField{ID: ""}); err == nil {
		t.Fatal("expected error for empty field id")
	}
	if _, err := ConfigurableFields[string, string](inner, ConfigurableField{ID: "temperature"}); err != nil {
		t.Fatalf("valid construction: %v", err)
	}
}

func TestConfigurableFieldsConfigSchema(t *testing.T) {
	inner := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))

	wrapped, err := ConfigurableFields[string, string](inner, ConfigurableField{
		ID:          "temperature",
		Description: "model temperature",
		Default:     0.7,
	})
	if err != nil {
		t.Fatalf("configurable fields: %v", err)
	}

	cfg := wrapped.ConfigSchema()
	configurable, ok := configurableSchema(cfg)
	if !ok {
		t.Fatalf("config schema missing configurable: %#v", cfg)
	}
	props := schemaProperties(configurable)
	if props["temperature"]["default"] != 0.7 {
		t.Fatalf("temperature default: %#v", props["temperature"])
	}
}

func TestConfigurableFieldsBatchStreamAndSchemas(t *testing.T) {
	var mu sync.Mutex
	var seen []Config
	inner := NewFunc(func(_ context.Context, input string, opts ...Option) (string, error) {
		mu.Lock()
		seen = append(seen, NewConfig(opts...))
		mu.Unlock()
		return input + "!", nil
	}, schema.String("in"), schema.String("out"))

	wrapped, err := ConfigurableFields[string, string](inner, ConfigurableField{
		ID:      "temperature",
		Default: 0.7,
	})
	if err != nil {
		t.Fatalf("configurable fields: %v", err)
	}

	got, err := wrapped.Batch(context.Background(), []string{"a", "b"}, WithRunID("root"))
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if got[0] != "a!" || got[1] != "b!" {
		t.Fatalf("batch got %#v", got)
	}
	// The child run carries the parent run ID, an empty run ID, the wrapper
	// name, and the applied field default.
	child := seen[0]
	if child.Name != "configurable_fields" || child.ParentID != "root" || child.RunID != "" {
		t.Fatalf("child identity: %#v", child)
	}
	if child.Configurable["temperature"] != 0.7 {
		t.Fatalf("child configurable: %#v", child.Configurable)
	}

	stream, err := wrapped.Stream(context.Background(), "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk != "x!" {
		t.Fatalf("chunk=%q ok=%v err=%v", chunk, ok, err)
	}

	if wrapped.InputSchema()["description"] != "in" {
		t.Fatalf("input schema: %#v", wrapped.InputSchema())
	}
	if wrapped.OutputSchema()["description"] != "out" {
		t.Fatalf("output schema: %#v", wrapped.OutputSchema())
	}
}

func TestConfigurableFieldsConfigSchemaWithoutAnnotation(t *testing.T) {
	inner := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))
	wrapped, err := ConfigurableFields[string, string](inner, ConfigurableField{ID: "flag"})
	if err != nil {
		t.Fatalf("configurable fields: %v", err)
	}
	prop := configurableProperty(t, wrapped.ConfigSchema(), "flag")
	if prop["type"] != "any" {
		t.Fatalf("flag property: %#v", prop)
	}
}
