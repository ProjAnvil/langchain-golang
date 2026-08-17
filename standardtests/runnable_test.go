package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestRunRunnableSchemaBasics(t *testing.T) {
	inputSchema := schema.String("input")
	outputSchema := schema.String("output")
	configSchema := schema.Object(map[string]schema.Schema{
		"configurable": schema.Object(map[string]schema.Schema{
			"mode": schema.String("mode"),
		}),
	})

	RunRunnableSchemaBasics(
		t,
		func(testing.TB) runnables.Runnable[string, string] {
			return standardRunnable{
				inputSchema:  inputSchema,
				outputSchema: outputSchema,
				configSchema: configSchema,
			}
		},
		inputSchema,
		outputSchema,
		configSchema,
	)
}

func TestRunRunnableConfigPropagation(t *testing.T) {
	RunRunnableConfigPropagation(
		t,
		func(testing.TB) runnables.Runnable[string, string] {
			return standardRunnable{wantConfigKey: "mode", wantConfigValue: "fast"}
		},
		"input",
		"mode",
		"fast",
	)
}

func TestRunRunnableGraphExport(t *testing.T) {
	RunRunnableGraphExport(
		t,
		func(testing.TB) runnables.Runnable[string, string] {
			return standardRunnable{}
		},
	)
}

type standardRunnable struct {
	inputSchema     schema.Schema
	outputSchema    schema.Schema
	configSchema    schema.Schema
	wantConfigKey   string
	wantConfigValue any
}

func (r standardRunnable) Invoke(_ context.Context, input string, opts ...runnables.Option) (string, error) {
	if err := r.assertConfig(opts...); err != nil {
		return "", err
	}
	return input, nil
}

func (r standardRunnable) Batch(ctx context.Context, inputs []string, opts ...runnables.Option) ([]string, error) {
	out := make([]string, len(inputs))
	for i, input := range inputs {
		value, err := r.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = value
	}
	return out, nil
}

func (r standardRunnable) Stream(ctx context.Context, input string, opts ...runnables.Option) (runnables.Stream[string], error) {
	value, err := r.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]string{value}), nil
}

func (r standardRunnable) InputSchema() schema.Schema {
	if r.inputSchema == nil {
		return schema.Schema{}
	}
	return r.inputSchema
}

func (r standardRunnable) OutputSchema() schema.Schema {
	if r.outputSchema == nil {
		return schema.Schema{}
	}
	return r.outputSchema
}

func (r standardRunnable) ConfigSchema() schema.Schema {
	if r.configSchema == nil {
		return runnables.GetConfigSchema(nil)
	}
	return r.configSchema
}

func (r standardRunnable) assertConfig(opts ...runnables.Option) error {
	if r.wantConfigKey == "" {
		return nil
	}
	cfg := runnables.NewConfig(opts...)
	if cfg.Configurable[r.wantConfigKey] != r.wantConfigValue {
		return errMissingConfig
	}
	return nil
}

type standardRunnableError string

func (e standardRunnableError) Error() string { return string(e) }

const errMissingConfig = standardRunnableError("missing configurable value")

func TestRunRunnableSchemaBasicsMismatch(t *testing.T) {
	inputSchema := schema.String("input")
	outputSchema := schema.String("output")
	configSchema := schema.Object(map[string]schema.Schema{
		"configurable": schema.Object(map[string]schema.Schema{
			"mode": schema.String("mode"),
		}),
	})

	expectConformanceFailure(t, "schema mismatches are reported", func(t *testing.T) {
		RunRunnableSchemaBasics(
			t,
			func(testing.TB) runnables.Runnable[string, string] {
				return standardRunnable{
					inputSchema:  inputSchema,
					outputSchema: outputSchema,
					configSchema: configSchema,
				}
			},
			schema.String("wrong input"),
			schema.String("wrong output"),
			schema.String("wrong config"),
		)
	})
}

func TestRunRunnableConfigPropagationMismatch(t *testing.T) {
	expectConformanceFailure(t, "missing configurable values are reported", func(t *testing.T) {
		RunRunnableConfigPropagation(
			t,
			func(testing.TB) runnables.Runnable[string, string] {
				return standardRunnable{wantConfigKey: "mode", wantConfigValue: "fast"}
			},
			"input",
			"mode",
			"slow",
		)
	})
}

// errNextRunnable streams values whose iteration always fails.
type errNextRunnable struct {
	standardRunnable
}

func (r errNextRunnable) Stream(
	ctx context.Context,
	input string,
	opts ...runnables.Option,
) (runnables.Stream[string], error) {
	if _, err := r.Invoke(ctx, input, opts...); err != nil {
		return nil, err
	}
	return errNextStream{}, nil
}

type errNextStream struct{}

func (errNextStream) Next(context.Context) (string, bool, error) {
	return "", false, errMissingConfig
}

func (errNextStream) Close() error { return nil }

func TestRunRunnableConfigPropagationStreamError(t *testing.T) {
	expectConformanceFailure(t, "stream iteration errors are reported", func(t *testing.T) {
		RunRunnableConfigPropagation(
			t,
			func(testing.TB) runnables.Runnable[string, string] {
				return errNextRunnable{}
			},
			"input",
			"mode",
			"fast",
		)
	})
}

// badMetadataGraphRunnable exports a graph whose node metadata cannot be
// marshaled to JSON.
type badMetadataGraphRunnable struct {
	standardRunnable
}

func (badMetadataGraphRunnable) Graph() runnables.Graph {
	return runnables.Graph{Nodes: []runnables.GraphNode{{
		ID:       "node",
		Name:     "node",
		Type:     "node",
		Metadata: map[string]any{"unmarshalable": func() {}},
	}}}
}

func TestRunRunnableGraphExportMarshalError(t *testing.T) {
	expectConformanceFailure(t, "graph marshal errors are reported", func(t *testing.T) {
		RunRunnableGraphExport(
			t,
			func(testing.TB) runnables.Runnable[string, string] {
				return badMetadataGraphRunnable{}
			},
		)
	})
}
