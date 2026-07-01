package runnables

import (
	"context"
	"errors"
	"fmt"

	"github.com/projanvil/langchain-golang/core/schema"
)

// Passthrough returns inputs unchanged and optionally runs a side-effect hook.
type Passthrough[T any] struct {
	OnInvoke func(context.Context, T, ...Option) error
	schema   schema.Schema
}

// NewPassthrough creates an identity runnable.
func NewPassthrough[T any](inputSchema schema.Schema) Passthrough[T] {
	return Passthrough[T]{schema: inputSchema}
}

// Invoke returns input unchanged.
func (r Passthrough[T]) Invoke(ctx context.Context, input T, opts ...Option) (T, error) {
	if r.OnInvoke != nil {
		if err := r.OnInvoke(ctx, input, opts...); err != nil {
			var zero T
			return zero, err
		}
	}
	return input, nil
}

// Batch invokes passthrough for each input.
func (r Passthrough[T]) Batch(ctx context.Context, inputs []T, opts ...Option) ([]T, error) {
	out := make([]T, len(inputs))
	copy(out, inputs)
	if r.OnInvoke == nil {
		return out, nil
	}
	var errs []error
	for _, input := range inputs {
		if err := r.OnInvoke(ctx, input, opts...); err != nil {
			errs = append(errs, err)
		}
	}
	return out, errors.Join(errs...)
}

// Stream streams the input unchanged.
func (r Passthrough[T]) Stream(ctx context.Context, input T, opts ...Option) (Stream[T], error) {
	output, err := r.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return NewSliceStream([]T{output}), nil
}

// InputSchema returns the passthrough input schema.
func (r Passthrough[T]) InputSchema() schema.Schema { return r.schema }

// OutputSchema returns the passthrough output schema.
func (r Passthrough[T]) OutputSchema() schema.Schema { return r.schema }

// ConfigSchema returns an empty config schema.
func (r Passthrough[T]) ConfigSchema() schema.Schema { return emptyConfigSchema() }

// Assign merges computed fields into a map input, like Python
// RunnablePassthrough.assign.
type Assign struct {
	Steps map[string]Runnable[map[string]any, any]
}

// NewAssign creates an assign runnable.
func NewAssign(steps map[string]Runnable[map[string]any, any]) Assign {
	copied := make(map[string]Runnable[map[string]any, any], len(steps))
	for key, step := range steps {
		copied[key] = step
	}
	return Assign{Steps: copied}
}

// Invoke returns input merged with all computed fields.
func (r Assign) Invoke(ctx context.Context, input map[string]any, opts ...Option) (map[string]any, error) {
	out := cloneMap(input)
	for key, step := range r.Steps {
		value, err := step.Invoke(ctx, cloneMap(out), childOptions("assign:key:"+key, opts...)...)
		if err != nil {
			return nil, err
		}
		out[key] = value
	}
	return out, nil
}

// Batch invokes assign for all inputs.
func (r Assign) Batch(ctx context.Context, inputs []map[string]any, opts ...Option) ([]map[string]any, error) {
	outputs := make([]map[string]any, len(inputs))
	errs := make([]error, len(inputs))
	for i, input := range inputs {
		outputs[i], errs[i] = r.Invoke(ctx, input, opts...)
	}
	return outputs, errors.Join(errs...)
}

// Stream returns a single-value stream.
func (r Assign) Stream(ctx context.Context, input map[string]any, opts ...Option) (Stream[map[string]any], error) {
	output, err := r.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return NewSliceStream([]map[string]any{output}), nil
}

// InputSchema returns a generic object schema.
func (r Assign) InputSchema() schema.Schema { return schema.Schema{"type": "object"} }

// OutputSchema returns a generic object schema.
func (r Assign) OutputSchema() schema.Schema { return schema.Schema{"type": "object"} }

// ConfigSchema returns the union of assign step config schemas.
func (r Assign) ConfigSchema() schema.Schema {
	children := make([]any, 0, len(r.Steps))
	for _, step := range r.Steps {
		children = append(children, step)
	}
	return mergeConfigSchemas(children...)
}

// BranchCase is one condition/runnable pair for Branch.
type BranchCase[I any, O any] struct {
	Condition Runnable[I, bool]
	Runnable  Runnable[I, O]
}

// Branch selects the first runnable whose condition returns true; otherwise it
// invokes Default.
type Branch[I any, O any] struct {
	Cases   []BranchCase[I, O]
	Default Runnable[I, O]
}

// NewBranch creates a Branch runnable.
func NewBranch[I any, O any](cases []BranchCase[I, O], def Runnable[I, O]) (Branch[I, O], error) {
	if len(cases) == 0 {
		return Branch[I, O]{}, fmt.Errorf("branch requires at least one conditional case")
	}
	if def == nil {
		return Branch[I, O]{}, fmt.Errorf("branch default runnable is required")
	}
	return Branch[I, O]{
		Cases:   append([]BranchCase[I, O](nil), cases...),
		Default: def,
	}, nil
}

// Invoke selects and invokes a branch.
func (r Branch[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	for i, item := range r.Cases {
		ok, err := item.Condition.Invoke(ctx, input, childOptions(fmt.Sprintf("condition:%d", i+1), opts...)...)
		if err != nil {
			var zero O
			return zero, err
		}
		if ok {
			return item.Runnable.Invoke(ctx, input, childOptions(fmt.Sprintf("branch:%d", i+1), opts...)...)
		}
	}
	return r.Default.Invoke(ctx, input, childOptions("branch:default", opts...)...)
}

// Batch invokes the branch for all inputs.
func (r Branch[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	outputs := make([]O, len(inputs))
	errs := make([]error, len(inputs))
	for i, input := range inputs {
		outputs[i], errs[i] = r.Invoke(ctx, input, opts...)
	}
	return outputs, errors.Join(errs...)
}

// Stream streams the selected runnable.
func (r Branch[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	for i, item := range r.Cases {
		ok, err := item.Condition.Invoke(ctx, input, childOptions(fmt.Sprintf("condition:%d", i+1), opts...)...)
		if err != nil {
			return nil, err
		}
		if ok {
			return item.Runnable.Stream(ctx, input, childOptions(fmt.Sprintf("branch:%d", i+1), opts...)...)
		}
	}
	return r.Default.Stream(ctx, input, childOptions("branch:default", opts...)...)
}

// InputSchema returns the default runnable input schema.
func (r Branch[I, O]) InputSchema() schema.Schema { return r.Default.InputSchema() }

// OutputSchema returns the default runnable output schema.
func (r Branch[I, O]) OutputSchema() schema.Schema { return r.Default.OutputSchema() }

// ConfigSchema returns the union of condition, branch, and default config
// schemas.
func (r Branch[I, O]) ConfigSchema() schema.Schema {
	children := make([]any, 0, len(r.Cases)*2+1)
	for _, item := range r.Cases {
		children = append(children, item.Condition, item.Runnable)
	}
	children = append(children, r.Default)
	return mergeConfigSchemas(children...)
}

// WithFallbacks invokes Runnable first, then each fallback until one succeeds.
type WithFallbacks[I any, O any] struct {
	Runnable  Runnable[I, O]
	Fallbacks []Runnable[I, O]
}

// NewWithFallbacks creates a fallback runnable.
func NewWithFallbacks[I any, O any](runnable Runnable[I, O], fallbacks ...Runnable[I, O]) (WithFallbacks[I, O], error) {
	if runnable == nil {
		return WithFallbacks[I, O]{}, fmt.Errorf("primary runnable is required")
	}
	return WithFallbacks[I, O]{
		Runnable:  runnable,
		Fallbacks: append([]Runnable[I, O](nil), fallbacks...),
	}, nil
}

// Invoke tries the primary runnable and fallbacks in order.
func (r WithFallbacks[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	var firstErr error
	runnables := append([]Runnable[I, O]{r.Runnable}, r.Fallbacks...)
	for i, runnable := range runnables {
		output, err := runnable.Invoke(ctx, input, childOptions(fallbackChildName(i), opts...)...)
		if err == nil {
			return output, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	var zero O
	return zero, firstErr
}

// Batch invokes fallback behavior for each input.
func (r WithFallbacks[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	outputs := make([]O, len(inputs))
	errs := make([]error, len(inputs))
	for i, input := range inputs {
		outputs[i], errs[i] = r.Invoke(ctx, input, opts...)
	}
	return outputs, errors.Join(errs...)
}

// Stream tries each runnable's stream in order.
func (r WithFallbacks[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	var firstErr error
	runnables := append([]Runnable[I, O]{r.Runnable}, r.Fallbacks...)
	for i, runnable := range runnables {
		stream, err := runnable.Stream(ctx, input, childOptions(fallbackChildName(i), opts...)...)
		if err == nil {
			return stream, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	return nil, firstErr
}

// InputSchema returns the primary runnable input schema.
func (r WithFallbacks[I, O]) InputSchema() schema.Schema { return r.Runnable.InputSchema() }

// OutputSchema returns the primary runnable output schema.
func (r WithFallbacks[I, O]) OutputSchema() schema.Schema { return r.Runnable.OutputSchema() }

// ConfigSchema returns the union of primary and fallback config schemas.
func (r WithFallbacks[I, O]) ConfigSchema() schema.Schema {
	children := make([]any, 0, len(r.Fallbacks)+1)
	children = append(children, r.Runnable)
	for _, runnable := range r.Fallbacks {
		children = append(children, runnable)
	}
	return mergeConfigSchemas(children...)
}

func cloneMap(input map[string]any) map[string]any {
	if input == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func fallbackChildName(index int) string {
	if index == 0 {
		return "fallback:primary"
	}
	return fmt.Sprintf("fallback:%d", index)
}
