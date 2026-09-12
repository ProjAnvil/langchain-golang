// This file adds the remaining core LCEL combinators: Bind (Python
// Runnable.bind -> RunnableBinding), Pick (Python Runnable.pick ->
// RunnablePick), and Each (Python Runnable.map -> RunnableEach).
package runnables

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/schema"
)

// RunnableBinding fixes Config settings on a runnable (Python base.py:1839
// Runnable.bind -> RunnableBinding). Like Python, the binding is a decorator:
// Invoke/Batch/Stream delegate to the wrapped runnable unchanged, with the
// bound options merged into every call.
//
// Precedence matches Python: the call site wins over the binding, because
// RunnableBindingBase._merge_configs is merge_configs(self.config, *configs)
// (later configs win, base.py:6003) and invoke passes **{**self.kwargs,
// **kwargs} (call kwargs last, base.py:6010). Here options are applied in
// order — bound options first, call-site options after — so a call-site
// WithMaxConcurrency overrides a bound one, exactly as it would in Python.
// Options that accumulate (WithTags appends) still merge both sides. Go
// having no **kwargs, the binding carries Options (Config settings such as
// callbacks or max concurrency), the only parameter channel a Go Runnable
// has; Python's stop=["-"]-style kwargs have no Go equivalent.
type RunnableBinding[I any, O any] struct {
	Bound   Runnable[I, O]
	Options []Option
}

// Bind returns a runnable that applies opts on top of every Invoke, Batch,
// and Stream call to r (Python: runnable.bind(**kwargs)). Call-site options
// take precedence over bound ones; see RunnableBinding for merge semantics.
func Bind[I any, O any](r Runnable[I, O], opts ...Option) (RunnableBinding[I, O], error) {
	if r == nil {
		return RunnableBinding[I, O]{}, fmt.Errorf("runnable is required")
	}
	return RunnableBinding[I, O]{
		Bound:   r,
		Options: append([]Option(nil), opts...),
	}, nil
}

// Bind returns a new binding with opts layered on top of the existing bound
// options, so later bindings win (Python: RunnableBinding.bind merges kwargs
// with the new kwargs last). The receiver is left unchanged.
func (b RunnableBinding[I, O]) Bind(opts ...Option) RunnableBinding[I, O] {
	merged := make([]Option, 0, len(b.Options)+len(opts))
	merged = append(merged, b.Options...)
	merged = append(merged, opts...)
	return RunnableBinding[I, O]{Bound: b.Bound, Options: merged}
}

// merged returns bound options followed by call-site options, so call-site
// settings win on conflict (Python parity; see RunnableBinding).
func (b RunnableBinding[I, O]) merged(opts []Option) []Option {
	merged := make([]Option, 0, len(b.Options)+len(opts))
	merged = append(merged, b.Options...)
	merged = append(merged, opts...)
	return merged
}

// Invoke runs the bound runnable with the merged config.
func (b RunnableBinding[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	return b.Bound.Invoke(ctx, input, b.merged(opts)...)
}

// Batch runs the bound runnable with the merged config.
func (b RunnableBinding[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	return b.Bound.Batch(ctx, inputs, b.merged(opts)...)
}

// Stream runs the bound runnable with the merged config.
func (b RunnableBinding[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	return b.Bound.Stream(ctx, input, b.merged(opts)...)
}

// InputSchema returns the bound runnable input schema.
func (b RunnableBinding[I, O]) InputSchema() schema.Schema { return b.Bound.InputSchema() }

// OutputSchema returns the bound runnable output schema.
func (b RunnableBinding[I, O]) OutputSchema() schema.Schema { return b.Bound.OutputSchema() }

// ConfigSchema returns the bound runnable config schema.
func (b RunnableBinding[I, O]) ConfigSchema() schema.Schema { return GetConfigSchema(b.Bound) }

// Pick projects the map output of a runnable onto selected keys (Python
// base.py:761 Runnable.pick -> RunnablePick). The wrapped runnable's output
// type must be map[string]any.
//
// Key semantics mirror Python: with exactly one key the output is that key's
// value (pick("k") returns d["k"]); with several keys the output is the
// subset map; with no keys the map passes through unchanged.
//
// Missing keys return an error. This matches older Python (which raised
// KeyError) and Go's no-silent-nil convention; current Python master silently
// returns None for missing keys, which would surface here as mysterious zero
// values.
//
// O selects the static output type: map[string]any for zero or several keys,
// or the picked value's type for a single key (use any when it is not
// statically known). A mismatched O fails the type assertion at runtime.
type Pick[I any, O any] struct {
	Runnable Runnable[I, map[string]any]
	Keys     []string
}

// NewPick wraps r with an output projection. keys follows the semantics
// documented on Pick; see Pick for choosing O.
func NewPick[I any, O any](r Runnable[I, map[string]any], keys ...string) (Pick[I, O], error) {
	if r == nil {
		return Pick[I, O]{}, fmt.Errorf("runnable is required")
	}
	return Pick[I, O]{
		Runnable: r,
		Keys:     append([]string(nil), keys...),
	}, nil
}

// Invoke projects the wrapped runnable's output onto the picked keys. With
// callbacks active Pick is its own chain run; the wrapped runnable runs as a
// child (name "pick") and carries the projected output on Pick's chain_end.
func (r Pick[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	ctx, run := startChainRun(ctx, opts, "pick", input)
	output, err := r.Runnable.Invoke(ctx, input, run.child("pick", opts...)...)
	if err != nil {
		var zero O
		run.fail(ctx, err)
		return zero, err
	}
	picked, err := r.project(output)
	if err != nil {
		var zero O
		run.fail(ctx, err)
		return zero, err
	}
	run.end(ctx, picked)
	return picked, nil
}

// Batch projects every wrapped runnable output onto the picked keys.
func (r Pick[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	outputs, err := r.Runnable.Batch(ctx, inputs, childOptions("pick", opts...)...)
	if err != nil {
		return nil, err
	}
	picked := make([]O, len(outputs))
	for i, output := range outputs {
		picked[i], err = r.project(output)
		if err != nil {
			return nil, err
		}
	}
	return picked, nil
}

// Stream returns a single-value stream with the projected output.
func (r Pick[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	output, err := r.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return NewSliceStream([]O{output}), nil
}

// project reduces one output map according to Keys.
func (r Pick[I, O]) project(output map[string]any) (O, error) {
	var picked any
	switch len(r.Keys) {
	case 0:
		picked = output
	case 1:
		value, ok := output[r.Keys[0]]
		if !ok {
			var zero O
			return zero, fmt.Errorf("pick: key %q missing from output", r.Keys[0])
		}
		picked = value
	default:
		subset := make(map[string]any, len(r.Keys))
		for _, key := range r.Keys {
			value, ok := output[key]
			if !ok {
				var zero O
				return zero, fmt.Errorf("pick: key %q missing from output", key)
			}
			subset[key] = value
		}
		picked = subset
	}
	out, ok := picked.(O)
	if !ok {
		var zero O
		return zero, fmt.Errorf("pick: output %T not assignable to output type", picked)
	}
	return out, nil
}

// InputSchema returns the wrapped runnable input schema.
func (r Pick[I, O]) InputSchema() schema.Schema { return r.Runnable.InputSchema() }

// OutputSchema projects the wrapped output schema onto the picked keys. A
// single-key pick returns an untyped schema because the value's shape is not
// statically known.
func (r Pick[I, O]) OutputSchema() schema.Schema {
	if len(r.Keys) == 0 {
		return r.Runnable.OutputSchema()
	}
	if len(r.Keys) == 1 {
		return schema.Schema{}
	}
	props := schemaProperties(r.Runnable.OutputSchema())
	picked := make(map[string]schema.Schema, len(r.Keys))
	for _, key := range r.Keys {
		if prop, ok := props[key]; ok {
			picked[key] = prop
		}
	}
	if len(picked) == 0 {
		// The wrapped output schema carries no per-key properties to project.
		return schema.Schema{"type": "object"}
	}
	return schema.Object(picked, r.Keys...)
}

// ConfigSchema returns the wrapped runnable config schema.
func (r Pick[I, O]) ConfigSchema() schema.Schema { return GetConfigSchema(r.Runnable) }

// Each applies a runnable to every element of a slice input (Python
// base.py:2153 Runnable.map -> RunnableEach). Each[I, O] implements
// Runnable[[]I, []O]: Invoke fans the elements out through the wrapped
// runnable's Batch, so concurrency is bounded by Config.MaxConcurrency
// (default DefaultParallelism) and outputs stay in input order, exactly like
// a direct Batch call.
type Each[I any, O any] struct {
	Runnable Runnable[I, O]
}

// NewEach creates a runnable that maps r over every element of its slice
// input (Python: runnable.map()).
func NewEach[I any, O any](r Runnable[I, O]) (Each[I, O], error) {
	if r == nil {
		return Each[I, O]{}, fmt.Errorf("runnable is required")
	}
	return Each[I, O]{Runnable: r}, nil
}

// Invoke applies the wrapped runnable to every element via its Batch, so a
// runnable with native batch support benefits from it and MaxConcurrency
// bounds the fan-out. With callbacks active Each is one chain run around the
// whole fan-out and every element is a distinct child run (Func-based
// runnables mint per-element IDs in their Batch; interleaved child events
// pair by RunID, not by global nesting order).
func (e Each[I, O]) Invoke(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	ctx, run := startChainRun(ctx, opts, "each", inputs)
	outputs, err := e.Runnable.Batch(ctx, inputs, run.child("each", opts...)...)
	if err != nil {
		run.fail(ctx, err)
		return nil, err
	}
	run.end(ctx, outputs)
	return outputs, nil
}

// Batch maps over each slice input (Python: RunnableEach.batch over a list of
// lists), fanning out across inputs with the same MaxConcurrency bound.
func (e Each[I, O]) Batch(ctx context.Context, inputs [][]I, opts ...Option) ([][]O, error) {
	cfg := NewConfig(opts...)
	return ParallelMap(ctx, cfg, inputs, func(ctx context.Context, input []I) ([]O, error) {
		return e.Invoke(ctx, input, opts...)
	})
}

// Stream returns a single-value stream with all element outputs.
func (e Each[I, O]) Stream(ctx context.Context, input []I, opts ...Option) (Stream[[]O], error) {
	output, err := e.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return NewSliceStream([][]O{output}), nil
}

// InputSchema returns an array schema of the wrapped input schema.
func (e Each[I, O]) InputSchema() schema.Schema {
	return schema.Schema{"type": "array", "items": e.Runnable.InputSchema()}
}

// OutputSchema returns an array schema of the wrapped output schema.
func (e Each[I, O]) OutputSchema() schema.Schema {
	return schema.Schema{"type": "array", "items": e.Runnable.OutputSchema()}
}

// ConfigSchema returns the wrapped runnable config schema.
func (e Each[I, O]) ConfigSchema() schema.Schema { return GetConfigSchema(e.Runnable) }

// Compile-time assertions that the combinators satisfy Runnable.
var (
	_ Runnable[any, any]     = RunnableBinding[any, any]{}
	_ Runnable[any, any]     = Pick[any, any]{}
	_ Runnable[[]any, []any] = Each[any, any]{}
)
