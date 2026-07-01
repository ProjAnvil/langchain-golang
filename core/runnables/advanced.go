package runnables

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/projanvil/langchain-golang/core/retry"
	"github.com/projanvil/langchain-golang/core/schema"
)

// Retry wraps a runnable and retries Invoke/Stream failures according to a
// retry policy. Batch retries each input independently.
type Retry[I any, O any] struct {
	Runnable          Runnable[I, O]
	MaxAttempts       int
	Delay             time.Duration
	BackoffMultiplier float64
	MaxDelay          time.Duration
	ShouldRetry       func(error) bool
}

// NewRetry creates a retrying runnable.
func NewRetry[I any, O any](runnable Runnable[I, O], maxAttempts int) (Retry[I, O], error) {
	if runnable == nil {
		return Retry[I, O]{}, fmt.Errorf("runnable is required")
	}
	if maxAttempts <= 0 {
		maxAttempts = 3
	}
	return Retry[I, O]{
		Runnable:    runnable,
		MaxAttempts: maxAttempts,
		ShouldRetry: func(error) bool {
			return true
		},
	}, nil
}

// Invoke invokes the wrapped runnable with retry.
func (r Retry[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	var output O
	err := retry.Do(ctx, retry.Policy{
		MaxAttempts:       r.MaxAttempts,
		Delay:             r.Delay,
		BackoffMultiplier: r.BackoffMultiplier,
		MaxDelay:          r.MaxDelay,
		ShouldRetry:       r.ShouldRetry,
	}, func() error {
		var err error
		output, err = r.Runnable.Invoke(ctx, input, opts...)
		return err
	})
	return output, err
}

// Batch retries each input independently.
func (r Retry[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	outputs := make([]O, len(inputs))
	errs := make([]error, len(inputs))
	for i, input := range inputs {
		outputs[i], errs[i] = r.Invoke(ctx, input, opts...)
	}
	return outputs, errors.Join(errs...)
}

// Stream retries stream construction. Errors emitted after a stream is returned
// belong to the stream consumer and are not replayed.
func (r Retry[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	var stream Stream[O]
	err := retry.Do(ctx, retry.Policy{
		MaxAttempts:       r.MaxAttempts,
		Delay:             r.Delay,
		BackoffMultiplier: r.BackoffMultiplier,
		MaxDelay:          r.MaxDelay,
		ShouldRetry:       r.ShouldRetry,
	}, func() error {
		var err error
		stream, err = r.Runnable.Stream(ctx, input, opts...)
		return err
	})
	return stream, err
}

// InputSchema returns the wrapped runnable input schema.
func (r Retry[I, O]) InputSchema() schema.Schema { return r.Runnable.InputSchema() }

// OutputSchema returns the wrapped runnable output schema.
func (r Retry[I, O]) OutputSchema() schema.Schema { return r.Runnable.OutputSchema() }

// ConfigSchema returns the wrapped runnable config schema.
func (r Retry[I, O]) ConfigSchema() schema.Schema { return GetConfigSchema(r.Runnable) }

// RouterInput is the input accepted by Router.
type RouterInput[I any] struct {
	Key   string
	Input I
}

// Router routes input to a runnable selected by key.
type Router[I any, O any] struct {
	Runnables map[string]Runnable[I, O]
}

// NewRouter creates a router runnable.
func NewRouter[I any, O any](runnables map[string]Runnable[I, O]) Router[I, O] {
	copied := make(map[string]Runnable[I, O], len(runnables))
	for key, runnable := range runnables {
		copied[key] = runnable
	}
	return Router[I, O]{Runnables: copied}
}

// Invoke routes to the selected runnable.
func (r Router[I, O]) Invoke(ctx context.Context, input RouterInput[I], opts ...Option) (O, error) {
	runnable, ok := r.Runnables[input.Key]
	if !ok {
		var zero O
		return zero, fmt.Errorf("no runnable associated with key %q", input.Key)
	}
	return runnable.Invoke(ctx, input.Input, childOptions("route:"+input.Key, opts...)...)
}

// Batch routes each input independently while preserving order.
func (r Router[I, O]) Batch(ctx context.Context, inputs []RouterInput[I], opts ...Option) ([]O, error) {
	outputs := make([]O, len(inputs))
	errs := make([]error, len(inputs))
	for i, input := range inputs {
		outputs[i], errs[i] = r.Invoke(ctx, input, opts...)
	}
	return outputs, errors.Join(errs...)
}

// Stream routes stream construction to the selected runnable.
func (r Router[I, O]) Stream(ctx context.Context, input RouterInput[I], opts ...Option) (Stream[O], error) {
	runnable, ok := r.Runnables[input.Key]
	if !ok {
		return nil, fmt.Errorf("no runnable associated with key %q", input.Key)
	}
	return runnable.Stream(ctx, input.Input, childOptions("route:"+input.Key, opts...)...)
}

// InputSchema returns a generic router input schema.
func (r Router[I, O]) InputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"key":   schema.String("route key"),
		"input": schema.Schema{},
	}, "key", "input")
}

// OutputSchema returns the first child output schema when available.
func (r Router[I, O]) OutputSchema() schema.Schema {
	for _, runnable := range r.Runnables {
		return runnable.OutputSchema()
	}
	return schema.Schema{}
}

// ConfigSchema returns the union of route config schemas.
func (r Router[I, O]) ConfigSchema() schema.Schema {
	children := make([]any, 0, len(r.Runnables))
	for _, runnable := range r.Runnables {
		children = append(children, runnable)
	}
	return mergeConfigSchemas(children...)
}

// ConfigurableAlternatives selects a runnable from Config.Configurable at
// invocation time, matching Python's configurable_alternatives behavior.
type ConfigurableAlternatives[I any, O any] struct {
	Field      string
	DefaultKey string
	Default    Runnable[I, O]
	Choices    map[string]Runnable[I, O]
}

// NewConfigurableAlternatives creates a runtime-selectable runnable wrapper.
func NewConfigurableAlternatives[I any, O any](
	field string,
	defaultKey string,
	defaultRunnable Runnable[I, O],
	choices map[string]Runnable[I, O],
) (ConfigurableAlternatives[I, O], error) {
	if field == "" {
		return ConfigurableAlternatives[I, O]{}, fmt.Errorf("configurable field is required")
	}
	if defaultRunnable == nil {
		return ConfigurableAlternatives[I, O]{}, fmt.Errorf("default runnable is required")
	}
	copied := make(map[string]Runnable[I, O], len(choices))
	for key, runnable := range choices {
		if key == "" {
			return ConfigurableAlternatives[I, O]{}, fmt.Errorf("configurable alternative key cannot be empty")
		}
		if runnable == nil {
			return ConfigurableAlternatives[I, O]{}, fmt.Errorf("configurable alternative %q is nil", key)
		}
		copied[key] = runnable
	}
	return ConfigurableAlternatives[I, O]{
		Field:      field,
		DefaultKey: defaultKey,
		Default:    defaultRunnable,
		Choices:    copied,
	}, nil
}

// Invoke selects the configured alternative and invokes it.
func (r ConfigurableAlternatives[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	runnable, key, err := r.selected(opts...)
	if err != nil {
		var zero O
		return zero, err
	}
	return runnable.Invoke(ctx, input, childOptions("configurable:"+key, opts...)...)
}

// Batch selects one configured alternative for the whole batch.
func (r ConfigurableAlternatives[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	runnable, key, err := r.selected(opts...)
	if err != nil {
		return nil, err
	}
	return runnable.Batch(ctx, inputs, childOptions("configurable:"+key, opts...)...)
}

// Stream selects the configured alternative and streams from it.
func (r ConfigurableAlternatives[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	runnable, key, err := r.selected(opts...)
	if err != nil {
		return nil, err
	}
	return runnable.Stream(ctx, input, childOptions("configurable:"+key, opts...)...)
}

// InputSchema returns the default runnable input schema.
func (r ConfigurableAlternatives[I, O]) InputSchema() schema.Schema {
	if r.Default == nil {
		return schema.Schema{}
	}
	return r.Default.InputSchema()
}

// OutputSchema returns the default runnable output schema.
func (r ConfigurableAlternatives[I, O]) OutputSchema() schema.Schema {
	if r.Default == nil {
		return schema.Schema{}
	}
	return r.Default.OutputSchema()
}

// ConfigSchema returns an enum schema for the selector field plus the selected
// alternatives' own configurable fields.
func (r ConfigurableAlternatives[I, O]) ConfigSchema() schema.Schema {
	children := make([]any, 0, len(r.Choices)+1)
	children = append(children, r.Default)
	for _, key := range sortedRunnableKeys(r.Choices) {
		children = append(children, r.Choices[key])
	}
	cfg := mergeConfigSchemas(children...)
	configurable, _ := configurableSchema(cfg)
	props := schemaProperties(configurable)
	if props == nil {
		props = map[string]schema.Schema{}
	}
	props[r.Field] = schema.Schema{
		"type":        "string",
		"description": "configurable alternative selector",
		"enum":        r.availableKeys(),
		"default":     r.defaultChildKey(),
	}
	return configurableConfigSchema(props)
}

func (r ConfigurableAlternatives[I, O]) selected(opts ...Option) (Runnable[I, O], string, error) {
	cfg := NewConfig(opts...)
	value, ok := cfg.Configurable[r.Field]
	if !ok || value == nil {
		return r.Default, r.defaultChildKey(), nil
	}
	key, ok := value.(string)
	if !ok {
		key = fmt.Sprint(value)
	}
	if key == "" || key == r.DefaultKey {
		return r.Default, r.defaultChildKey(), nil
	}
	runnable, ok := r.Choices[key]
	if !ok {
		return nil, "", fmt.Errorf("unknown configurable alternative %q for field %q; available: %s", key, r.Field, stringsList(r.availableKeys()))
	}
	return runnable, key, nil
}

func (r ConfigurableAlternatives[I, O]) defaultChildKey() string {
	if r.DefaultKey == "" {
		return "default"
	}
	return r.DefaultKey
}

func (r ConfigurableAlternatives[I, O]) availableKeys() []string {
	keys := make([]string, 0, len(r.Choices)+1)
	if r.DefaultKey != "" {
		keys = append(keys, r.DefaultKey)
	}
	for key := range r.Choices {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func stringsList(values []string) string {
	if len(values) == 0 {
		return "[]"
	}
	return fmt.Sprintf("%q", values)
}

// Sequence composes two runnables: first output becomes second input.
type Sequence[I any, M any, O any] struct {
	First  Runnable[I, M]
	Second Runnable[M, O]
}

// NewSequence creates a two-step sequence.
func NewSequence[I any, M any, O any](first Runnable[I, M], second Runnable[M, O]) (Sequence[I, M, O], error) {
	if first == nil || second == nil {
		return Sequence[I, M, O]{}, fmt.Errorf("sequence requires both runnables")
	}
	return Sequence[I, M, O]{First: first, Second: second}, nil
}

// Invoke runs first then second.
func (r Sequence[I, M, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	mid, err := r.First.Invoke(ctx, input, childOptions("seq:step:1", opts...)...)
	if err != nil {
		var zero O
		return zero, err
	}
	return r.Second.Invoke(ctx, mid, childOptions("seq:step:2", opts...)...)
}

// Batch runs the sequence for each input.
func (r Sequence[I, M, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	mids, err := r.First.Batch(ctx, inputs, childOptions("seq:step:1", opts...)...)
	if err != nil {
		var zero []O
		return zero, err
	}
	return r.Second.Batch(ctx, mids, childOptions("seq:step:2", opts...)...)
}

// Stream streams the first runnable into the second runnable and flattens
// second-stage chunks in order.
func (r Sequence[I, M, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	first, err := r.First.Stream(ctx, input, childOptions("seq:step:1", opts...)...)
	if err != nil {
		return nil, err
	}
	return &sequenceStream[M, O]{
		first: first,
		next: func(ctx context.Context, value M) (Stream[O], error) {
			return r.Second.Stream(ctx, value, childOptions("seq:step:2", opts...)...)
		},
	}, nil
}

// InputSchema returns first input schema.
func (r Sequence[I, M, O]) InputSchema() schema.Schema { return r.First.InputSchema() }

// OutputSchema returns second output schema.
func (r Sequence[I, M, O]) OutputSchema() schema.Schema { return r.Second.OutputSchema() }

// ConfigSchema returns the union of both step config schemas.
func (r Sequence[I, M, O]) ConfigSchema() schema.Schema {
	return mergeConfigSchemas(r.First, r.Second)
}

// Parallel invokes multiple runnables with the same input and returns a map of
// keyed outputs.
type Parallel[I any] struct {
	Steps map[string]Runnable[I, any]
}

// NewParallel creates a parallel runnable.
func NewParallel[I any](steps map[string]Runnable[I, any]) Parallel[I] {
	copied := make(map[string]Runnable[I, any], len(steps))
	for key, step := range steps {
		copied[key] = step
	}
	return Parallel[I]{Steps: copied}
}

// Invoke invokes all steps and collects keyed outputs.
func (r Parallel[I]) Invoke(ctx context.Context, input I, opts ...Option) (map[string]any, error) {
	out := make(map[string]any, len(r.Steps))
	var errs []error
	for key, step := range r.Steps {
		value, err := step.Invoke(ctx, input, childOptions("map:key:"+key, opts...)...)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", key, err))
			continue
		}
		out[key] = value
	}
	return out, errors.Join(errs...)
}

// Batch invokes parallel for each input.
func (r Parallel[I]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]map[string]any, error) {
	outputs := make([]map[string]any, len(inputs))
	errs := make([]error, len(inputs))
	for i, input := range inputs {
		outputs[i], errs[i] = r.Invoke(ctx, input, opts...)
	}
	return outputs, errors.Join(errs...)
}

// Stream streams all steps and emits keyed chunks as each step produces values.
func (r Parallel[I]) Stream(ctx context.Context, input I, opts ...Option) (Stream[map[string]any], error) {
	keys := sortedRunnableKeys(r.Steps)
	items := make([]parallelStreamItem, 0, len(keys))
	for _, key := range keys {
		stream, err := r.Steps[key].Stream(ctx, input, childOptions("map:key:"+key, opts...)...)
		if err != nil {
			closeParallelStreams(items)
			return nil, fmt.Errorf("%s: %w", key, err)
		}
		items = append(items, parallelStreamItem{key: key, stream: stream})
	}
	return &parallelStream{items: items}, nil
}

// InputSchema returns the first step input schema when available.
func (r Parallel[I]) InputSchema() schema.Schema {
	for _, step := range r.Steps {
		return step.InputSchema()
	}
	return schema.Schema{}
}

// OutputSchema returns an object schema keyed by step names.
func (r Parallel[I]) OutputSchema() schema.Schema {
	props := make(map[string]schema.Schema, len(r.Steps))
	for key, step := range r.Steps {
		props[key] = step.OutputSchema()
	}
	return schema.Object(props)
}

// ConfigSchema returns the union of all step config schemas.
func (r Parallel[I]) ConfigSchema() schema.Schema {
	children := make([]any, 0, len(r.Steps))
	for _, step := range r.Steps {
		children = append(children, step)
	}
	return mergeConfigSchemas(children...)
}

type sequenceStream[M any, O any] struct {
	first   Stream[M]
	current Stream[O]
	next    func(context.Context, M) (Stream[O], error)
}

func (s *sequenceStream[M, O]) Next(ctx context.Context) (O, bool, error) {
	var zero O
	for {
		if s.current != nil {
			value, ok, err := s.current.Next(ctx)
			if err != nil {
				return zero, false, err
			}
			if ok {
				return value, true, nil
			}
			if err := s.current.Close(); err != nil {
				return zero, false, err
			}
			s.current = nil
		}
		mid, ok, err := s.first.Next(ctx)
		if err != nil {
			return zero, false, err
		}
		if !ok {
			return zero, false, nil
		}
		s.current, err = s.next(ctx, mid)
		if err != nil {
			return zero, false, err
		}
	}
}

func (s *sequenceStream[M, O]) Close() error {
	var errs []error
	if s.current != nil {
		errs = append(errs, s.current.Close())
		s.current = nil
	}
	if s.first != nil {
		errs = append(errs, s.first.Close())
		s.first = nil
	}
	return errors.Join(errs...)
}

type parallelStreamItem struct {
	key    string
	stream Stream[any]
}

type parallelStream struct {
	items []parallelStreamItem
	index int
}

func (s *parallelStream) Next(ctx context.Context) (map[string]any, bool, error) {
	for len(s.items) > 0 {
		if s.index >= len(s.items) {
			s.index = 0
		}
		item := s.items[s.index]
		value, ok, err := item.stream.Next(ctx)
		if err != nil {
			return nil, false, err
		}
		if ok {
			s.index = (s.index + 1) % len(s.items)
			return map[string]any{item.key: value}, true, nil
		}
		if err := item.stream.Close(); err != nil {
			return nil, false, err
		}
		s.items = append(s.items[:s.index], s.items[s.index+1:]...)
	}
	return nil, false, nil
}

func (s *parallelStream) Close() error {
	err := closeParallelStreams(s.items)
	s.items = nil
	return err
}

func closeParallelStreams(items []parallelStreamItem) error {
	var errs []error
	for _, item := range items {
		if item.stream != nil {
			errs = append(errs, item.stream.Close())
		}
	}
	return errors.Join(errs...)
}
