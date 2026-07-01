package runnables

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/chathistory"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestConfigurableAlternativesConfigSchema(t *testing.T) {
	base := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))
	alt := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))
	runnable, err := NewConfigurableAlternatives[string, string](
		"model",
		"default",
		base,
		map[string]Runnable[string, string]{"alt": alt},
	)
	if err != nil {
		t.Fatalf("new alternatives: %v", err)
	}

	cfg := GetConfigSchema(runnable)
	prop := configurableProperty(t, cfg, "model")
	if prop["type"] != "string" {
		t.Fatalf("type: %#v", prop)
	}
	if prop["default"] != "default" {
		t.Fatalf("default: %#v", prop)
	}
	if !reflect.DeepEqual(prop["enum"], []string{"alt", "default"}) {
		t.Fatalf("enum: %#v", prop["enum"])
	}
}

func TestCompositionConfigSchemaMergesChildren(t *testing.T) {
	first := configSchemaOnlyRunnable{schema: configurableConfigSchema(map[string]schema.Schema{
		"first": schema.String("first field"),
	}, "first")}
	second := configSchemaOnlyRunnable{schema: configurableConfigSchema(map[string]schema.Schema{
		"second": schema.Boolean("second field"),
	})}
	seq, err := NewSequence[any, any, any](first, second)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}

	cfg := GetConfigSchema(seq)
	if configurableProperty(t, cfg, "first")["type"] != "string" {
		t.Fatalf("first property: %#v", cfg)
	}
	if configurableProperty(t, cfg, "second")["type"] != "boolean" {
		t.Fatalf("second property: %#v", cfg)
	}
	required := schemaRequired(mustConfigurableSchema(t, cfg))
	if !reflect.DeepEqual(required, []string{"first"}) {
		t.Fatalf("required: %#v", required)
	}
}

func TestRunnableWithMessageHistoryConfigSchemaRequiresFactoryKeys(t *testing.T) {
	base := configSchemaOnlyRunnable{schema: configurableConfigSchema(map[string]schema.Schema{
		"tenant": schema.String("tenant"),
	}, "tenant")}
	wrapped, err := NewRunnableWithMessageHistory(
		base,
		func(context.Context, map[string]any) (chathistory.History, error) {
			return chathistory.NewInMemoryChatMessageHistory(messages.AI("hi")), nil
		},
		"user_id",
		"conversation_id",
	)
	if err != nil {
		t.Fatalf("new history wrapper: %v", err)
	}

	cfg := GetConfigSchema(wrapped)
	if configurableProperty(t, cfg, "tenant")["type"] != "string" {
		t.Fatalf("tenant property: %#v", cfg)
	}
	if configurableProperty(t, cfg, "user_id")["type"] != "string" {
		t.Fatalf("user_id property: %#v", cfg)
	}
	required := schemaRequired(mustConfigurableSchema(t, cfg))
	if !reflect.DeepEqual(required, []string{"conversation_id", "tenant", "user_id"}) {
		t.Fatalf("required: %#v", required)
	}
}

type configSchemaOnlyRunnable struct {
	schema schema.Schema
}

func (r configSchemaOnlyRunnable) Invoke(context.Context, any, ...Option) (any, error) {
	return nil, nil
}

func (r configSchemaOnlyRunnable) Batch(context.Context, []any, ...Option) ([]any, error) {
	return nil, nil
}

func (r configSchemaOnlyRunnable) Stream(context.Context, any, ...Option) (Stream[any], error) {
	return NewSliceStream([]any{nil}), nil
}

func (r configSchemaOnlyRunnable) InputSchema() schema.Schema  { return schema.Schema{} }
func (r configSchemaOnlyRunnable) OutputSchema() schema.Schema { return schema.Schema{} }
func (r configSchemaOnlyRunnable) ConfigSchema() schema.Schema { return r.schema }

func configurableProperty(t *testing.T, cfg schema.Schema, name string) schema.Schema {
	t.Helper()
	configurable := mustConfigurableSchema(t, cfg)
	prop, ok := schemaProperties(configurable)[name]
	if !ok {
		t.Fatalf("missing configurable property %q in %#v", name, cfg)
	}
	return prop
}

func mustConfigurableSchema(t *testing.T, cfg schema.Schema) schema.Schema {
	t.Helper()
	configurable, ok := configurableSchema(cfg)
	if !ok {
		t.Fatalf("missing configurable schema in %#v", cfg)
	}
	return configurable
}
