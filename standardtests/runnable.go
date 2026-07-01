package standardtests

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
)

// RunnableFactory creates a fresh runnable for standard contract tests.
type RunnableFactory[I any, O any] func(t testing.TB) runnables.Runnable[I, O]

// RunRunnableSchemaBasics verifies stable input, output, and config schema
// surfaces for a runnable implementation.
func RunRunnableSchemaBasics[I any, O any](
	t *testing.T,
	factory RunnableFactory[I, O],
	wantInput schema.Schema,
	wantOutput schema.Schema,
	wantConfig schema.Schema,
) {
	t.Helper()

	t.Run("input schema", func(t *testing.T) {
		runnable := factory(t)
		if got := runnable.InputSchema(); !reflect.DeepEqual(got, wantInput) {
			t.Fatalf("input schema got %#v want %#v", got, wantInput)
		}
	})

	t.Run("output schema", func(t *testing.T) {
		runnable := factory(t)
		if got := runnable.OutputSchema(); !reflect.DeepEqual(got, wantOutput) {
			t.Fatalf("output schema got %#v want %#v", got, wantOutput)
		}
	})

	t.Run("config schema", func(t *testing.T) {
		runnable := factory(t)
		if got := runnables.GetConfigSchema(runnable); !jsonSchemaEqual(got, wantConfig) {
			t.Fatalf("config schema got %#v want %#v", got, wantConfig)
		}
	})
}

// RunRunnableConfigPropagation verifies that runtime configurable values are
// visible to the runnable during invoke, batch, and stream.
func RunRunnableConfigPropagation[I any, O any](
	t *testing.T,
	factory RunnableFactory[I, O],
	input I,
	configurableKey string,
	configurableValue any,
) {
	t.Helper()

	t.Run("invoke config", func(t *testing.T) {
		runnable := factory(t)
		_, err := runnable.Invoke(
			context.Background(),
			input,
			runnables.WithConfigurable(configurableKey, configurableValue),
		)
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
	})

	t.Run("batch config", func(t *testing.T) {
		runnable := factory(t)
		_, err := runnable.Batch(
			context.Background(),
			[]I{input},
			runnables.WithConfigurable(configurableKey, configurableValue),
		)
		if err != nil {
			t.Fatalf("batch: %v", err)
		}
	})

	t.Run("stream config", func(t *testing.T) {
		runnable := factory(t)
		stream, err := runnable.Stream(
			context.Background(),
			input,
			runnables.WithConfigurable(configurableKey, configurableValue),
		)
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		defer stream.Close()
		if _, _, err := stream.Next(context.Background()); err != nil {
			t.Fatalf("next: %v", err)
		}
	})
}

// RunRunnableGraphExport verifies graph export surfaces shared by runnables.
func RunRunnableGraphExport[I any, O any](t *testing.T, factory RunnableFactory[I, O]) {
	t.Helper()
	t.Run("graph export", func(t *testing.T) {
		runnable := factory(t)
		graph := runnables.GetGraph(runnable)
		if len(graph.Nodes) == 0 {
			t.Fatal("expected graph nodes")
		}
		if !strings.Contains(graph.DrawASCII(), "graph:") {
			t.Fatalf("unexpected ASCII graph: %q", graph.DrawASCII())
		}
		if !strings.Contains(graph.DrawMermaid(), "graph TD;") {
			t.Fatalf("unexpected Mermaid graph: %q", graph.DrawMermaid())
		}
		if _, err := graph.MarshalJSONStable(); err != nil {
			t.Fatalf("marshal graph: %v", err)
		}
	})
}

func jsonSchemaEqual(left schema.Schema, right schema.Schema) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && reflect.DeepEqual(leftJSON, rightJSON)
}
