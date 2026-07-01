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
