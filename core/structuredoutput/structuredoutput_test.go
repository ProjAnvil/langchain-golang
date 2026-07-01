package structuredoutput

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

func TestNewJSONSchemaDefaultsName(t *testing.T) {
	cfg := NewJSONSchema("", schema.Object(map[string]schema.Schema{
		"name": schema.String("name"),
	}, "name"), true)

	if cfg.Name != "structured_output" {
		t.Fatalf("name: got %q", cfg.Name)
	}
	if !cfg.Strict {
		t.Fatal("expected strict mode")
	}
	if cfg.Schema["type"] != "object" {
		t.Fatalf("schema: %+v", cfg.Schema)
	}
}

func TestBindJSONParsesTypedOutput(t *testing.T) {
	model := testStructuredModel{
		response: messages.AI(`{"name":"Ada","age":37}`),
		capabilities: language.ChatModelCapabilities{
			StructuredOutput: true,
		},
	}

	runnable, err := BindJSON[testStructuredModel, person](
		model,
		"person",
		schema.Object(map[string]schema.Schema{
			"name": schema.String("person name"),
			"age":  schema.Integer("person age"),
		}, "name", "age"),
		true,
	)
	if err != nil {
		t.Fatalf("bind json: %v", err)
	}

	got, err := runnable.Invoke(context.Background(), []messages.Message{
		messages.Human("extract person"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if got.Name != "Ada" || got.Age != 37 {
		t.Fatalf("parsed output: %+v", got)
	}
	if runnable.OutputSchema()["type"] != "object" {
		t.Fatalf("output schema: %+v", runnable.OutputSchema())
	}
}

func TestBindJSONRejectsUnsupportedModel(t *testing.T) {
	_, err := BindJSON[testStructuredModel, person](
		testStructuredModel{},
		"person",
		schema.Object(map[string]schema.Schema{
			"name": schema.String("person name"),
		}, "name"),
		true,
	)
	if err == nil {
		t.Fatal("expected unsupported error")
	}
	if err.Error() != "structured output is not supported" {
		t.Fatalf("error: got %q", err.Error())
	}
}

type person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

type testStructuredModel struct {
	response     messages.Message
	capabilities language.ChatModelCapabilities
}

func (m testStructuredModel) Invoke(
	_ context.Context,
	_ []messages.Message,
	_ ...runnables.Option,
) (messages.Message, error) {
	return m.response, nil
}

func (m testStructuredModel) Batch(
	ctx context.Context,
	inputs [][]messages.Message,
	opts ...runnables.Option,
) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, input := range inputs {
		response, err := m.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = response
	}
	return out, nil
}

func (m testStructuredModel) Stream(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	response, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{response}), nil
}

func (m testStructuredModel) InputSchema() schema.Schema {
	return schema.Schema{"type": "array"}
}

func (m testStructuredModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"content": schema.String("message content"),
	})
}

func (m testStructuredModel) BindTools([]tools.Tool) (language.ChatModel, error) {
	return m, nil
}

func (m testStructuredModel) Capabilities() language.ChatModelCapabilities {
	return m.capabilities
}

func (m testStructuredModel) WithStructuredOutput(
	_ string,
	_ schema.Schema,
	_ bool,
) testStructuredModel {
	return m
}
