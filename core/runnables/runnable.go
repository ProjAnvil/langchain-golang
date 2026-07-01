package runnables

import (
	"context"
	"errors"
	"sync"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/schema"
)

// Option configures runnable execution.
type Option func(*Config)

// Config carries execution metadata shared by runnables, models, tools, and
// callbacks.
type Config struct {
	Name         string
	RunID        string
	ParentID     string
	Tags         []string
	Metadata     map[string]any
	Configurable map[string]any
	Callbacks    callbacks.Manager
}

// NewConfig applies options to an execution config.
func NewConfig(opts ...Option) Config {
	cfg := Config{
		Metadata:     map[string]any{},
		Configurable: map[string]any{},
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Clone returns a copy of the config with independent slice and map fields.
func (cfg Config) Clone() Config {
	out := cfg
	out.Tags = append([]string(nil), cfg.Tags...)
	out.Metadata = cloneConfigMap(cfg.Metadata)
	out.Configurable = cloneConfigMap(cfg.Configurable)
	return out
}

// WithName sets the runnable name for tracing.
func WithName(name string) Option {
	return func(cfg *Config) {
		cfg.Name = name
	}
}

// WithRunID sets the run ID for tracing.
func WithRunID(runID string) Option {
	return func(cfg *Config) {
		cfg.RunID = runID
	}
}

// WithParentID sets the parent run ID for tracing.
func WithParentID(parentID string) Option {
	return func(cfg *Config) {
		cfg.ParentID = parentID
	}
}

// WithTags appends tracing tags.
func WithTags(tags ...string) Option {
	return func(cfg *Config) {
		cfg.Tags = append(cfg.Tags, tags...)
	}
}

// WithMetadata sets one tracing metadata key.
func WithMetadata(name string, value any) Option {
	return func(cfg *Config) {
		if cfg.Metadata == nil {
			cfg.Metadata = map[string]any{}
		}
		cfg.Metadata[name] = value
	}
}

// WithConfigurable sets one configurable runtime field.
func WithConfigurable(name string, value any) Option {
	return func(cfg *Config) {
		if cfg.Configurable == nil {
			cfg.Configurable = map[string]any{}
		}
		cfg.Configurable[name] = value
	}
}

// WithCallbacks sets callback handlers for tracing.
func WithCallbacks(manager callbacks.Manager) Option {
	return func(cfg *Config) {
		cfg.Callbacks = manager
	}
}

func configOption(cfg Config) Option {
	return func(out *Config) {
		*out = cfg.Clone()
		if out.Metadata == nil {
			out.Metadata = map[string]any{}
		}
		if out.Configurable == nil {
			out.Configurable = map[string]any{}
		}
	}
}

func childOptions(name string, opts ...Option) []Option {
	cfg := NewConfig(opts...).Clone()
	if cfg.RunID != "" {
		cfg.ParentID = cfg.RunID
		cfg.RunID = ""
	}
	cfg.Name = name
	return []Option{configOption(cfg)}
}

func cloneConfigMap(input map[string]any) map[string]any {
	if input == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

// Runnable is the Go equivalent of the LangChain Runnable contract.
type Runnable[I any, O any] interface {
	Invoke(ctx context.Context, input I, opts ...Option) (O, error)
	Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error)
	Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error)
	InputSchema() schema.Schema
	OutputSchema() schema.Schema
}

// Stream is a pull-based stream of runnable output chunks.
type Stream[O any] interface {
	Next(ctx context.Context) (O, bool, error)
	Close() error
}

// Func wraps a function as a Runnable.
type Func[I any, O any] struct {
	fn           func(context.Context, I, ...Option) (O, error)
	inputSchema  schema.Schema
	outputSchema schema.Schema
}

// NewFunc creates a Runnable from a function.
func NewFunc[I any, O any](
	fn func(context.Context, I, ...Option) (O, error),
	inputSchema schema.Schema,
	outputSchema schema.Schema,
) Func[I, O] {
	return Func[I, O]{
		fn:           fn,
		inputSchema:  inputSchema,
		outputSchema: outputSchema,
	}
}

// Invoke executes the runnable for one input.
func (r Func[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	return r.fn(ctx, input, opts...)
}

// Batch executes the runnable for all inputs concurrently while preserving
// output order.
func (r Func[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	outputs := make([]O, len(inputs))
	errs := make([]error, len(inputs))

	var wg sync.WaitGroup
	for i, input := range inputs {
		wg.Add(1)
		go func(i int, input I) {
			defer wg.Done()
			outputs[i], errs[i] = r.Invoke(ctx, input, opts...)
		}(i, input)
	}
	wg.Wait()

	return outputs, errors.Join(errs...)
}

// Stream returns a single-value stream by default.
func (r Func[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	output, err := r.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return NewSliceStream([]O{output}), nil
}

// InputSchema returns the runnable input schema.
func (r Func[I, O]) InputSchema() schema.Schema {
	return r.inputSchema
}

// OutputSchema returns the runnable output schema.
func (r Func[I, O]) OutputSchema() schema.Schema {
	return r.outputSchema
}

// SliceStream streams values from a slice.
type SliceStream[O any] struct {
	values []O
	index  int
}

// NewSliceStream creates a stream from a finite slice.
func NewSliceStream[O any](values []O) *SliceStream[O] {
	return &SliceStream[O]{values: values}
}

// Next returns the next value.
func (s *SliceStream[O]) Next(_ context.Context) (O, bool, error) {
	var zero O
	if s.index >= len(s.values) {
		return zero, false, nil
	}
	value := s.values[s.index]
	s.index++
	return value, true, nil
}

// Close releases stream resources.
func (s *SliceStream[O]) Close() error {
	s.index = len(s.values)
	return nil
}
