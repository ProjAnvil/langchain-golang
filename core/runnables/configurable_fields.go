package runnables

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/schema"
)

// RunnableConfigurableFields wraps a runnable and makes a set of its config
// fields configurable at invoke time via Config.Configurable overrides. It is
// the Go equivalent of Python's RunnableConfigurableFields, typically created
// through ConfigurableFields.
type RunnableConfigurableFields[I any, O any] struct {
	Runnable Runnable[I, O]
	Fields   []ConfigurableField
}

// ConfigurableFields wraps runnable with the given configurable fields. When
// the wrapped runnable is invoked, each field ID set in Config.Configurable is
// passed through untouched; unset fields fall back to their configured default.
func ConfigurableFields[I any, O any](
	r Runnable[I, O],
	fields ...ConfigurableField,
) (RunnableConfigurableFields[I, O], error) {
	if r == nil {
		return RunnableConfigurableFields[I, O]{}, fmt.Errorf("runnable is required")
	}
	copied := make([]ConfigurableField, len(fields))
	for i, field := range fields {
		if field.ID == "" {
			return RunnableConfigurableFields[I, O]{}, fmt.Errorf("configurable field id is required")
		}
		copied[i] = field
	}
	return RunnableConfigurableFields[I, O]{
		Runnable: r,
		Fields:   copied,
	}, nil
}

// Invoke applies field defaults to the config before invoking the inner
// runnable.
func (r RunnableConfigurableFields[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	return r.Runnable.Invoke(ctx, input, r.childConfig(opts...)...)
}

// Batch applies field defaults once for the whole batch.
func (r RunnableConfigurableFields[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	return r.Runnable.Batch(ctx, inputs, r.childConfig(opts...)...)
}

// Stream applies field defaults before streaming from the inner runnable.
func (r RunnableConfigurableFields[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	return r.Runnable.Stream(ctx, input, r.childConfig(opts...)...)
}

// InputSchema returns the wrapped runnable input schema.
func (r RunnableConfigurableFields[I, O]) InputSchema() schema.Schema {
	return r.Runnable.InputSchema()
}

// OutputSchema returns the wrapped runnable output schema.
func (r RunnableConfigurableFields[I, O]) OutputSchema() schema.Schema {
	return r.Runnable.OutputSchema()
}

// ConfigSchema returns the wrapped runnable config schema extended with the
// configurable fields declared on this wrapper.
func (r RunnableConfigurableFields[I, O]) ConfigSchema() schema.Schema {
	cfg := GetConfigSchema(r.Runnable)
	configurable, _ := configurableSchema(cfg)
	props := schemaProperties(configurable)
	if props == nil {
		props = map[string]schema.Schema{}
	}
	for _, field := range r.Fields {
		kind := field.Annotation
		if kind == "" {
			kind = "any"
		}
		props[field.ID] = schema.Schema{
			"type":        kind,
			"description": field.Description,
			"default":     field.Default,
		}
	}
	return configurableConfigSchema(props)
}

// childConfig builds the config passed to the inner runnable: it clones the
// incoming config, applies each unset field's default, and marks the child run.
func (r RunnableConfigurableFields[I, O]) childConfig(opts ...Option) []Option {
	cfg := NewConfig(opts...)
	for _, field := range r.Fields {
		if _, ok := cfg.Configurable[field.ID]; !ok {
			cfg.Configurable[field.ID] = field.Default
		}
	}
	if cfg.RunID != "" {
		cfg.ParentID = cfg.RunID
		cfg.RunID = ""
	}
	cfg.Name = "configurable_fields"
	return []Option{configOption(cfg)}
}
