package runnables

import (
	"context"
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
