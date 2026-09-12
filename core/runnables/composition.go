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

// Invoke returns input unchanged, wrapped in a chain run (start/end, or
// chain_error when the OnInvoke hook fails) when callbacks are configured.
func (r Passthrough[T]) Invoke(ctx context.Context, input T, opts ...Option) (T, error) {
	ctx, run := startChainRun(ctx, opts, "passthrough", input)
	if r.OnInvoke != nil {
		if err := r.OnInvoke(ctx, input, opts...); err != nil {
			var zero T
			run.fail(ctx, err)
			return zero, err
		}
	}
	run.end(ctx, input)
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

// Invoke returns input merged with all computed fields. With callbacks
// active Assign is one chain run and every mapper step runs as a child named
// "assign:key:K". Map iteration order is Go-random, so with several keys the
// child runs interleave nondeterministically.
func (r Assign) Invoke(ctx context.Context, input map[string]any, opts ...Option) (map[string]any, error) {
	ctx, run := startChainRun(ctx, opts, "assign", input)
	out := cloneMap(input)
	for key, step := range r.Steps {
		value, err := step.Invoke(ctx, cloneMap(out), run.child("assign:key:"+key, opts...)...)
		if err != nil {
			run.fail(ctx, err)
			return nil, err
		}
		out[key] = value
	}
	run.end(ctx, out)
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

// Invoke selects and invokes a branch. With callbacks active Branch is its
// own chain run; each evaluated condition and the selected branch runnable
// run as children named "condition:N" / "branch:N" (or "branch:default").
func (r Branch[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	ctx, run := startChainRun(ctx, opts, "branch", input)
	for i, item := range r.Cases {
		ok, err := item.Condition.Invoke(ctx, input, run.child(fmt.Sprintf("condition:%d", i+1), opts...)...)
		if err != nil {
			var zero O
			run.fail(ctx, err)
			return zero, err
		}
		if ok {
			output, err := item.Runnable.Invoke(ctx, input, run.child(fmt.Sprintf("branch:%d", i+1), opts...)...)
			if err != nil {
				var zero O
				run.fail(ctx, err)
				return zero, err
			}
			run.end(ctx, output)
			return output, nil
		}
	}
	output, err := r.Default.Invoke(ctx, input, run.child("branch:default", opts...)...)
	if err != nil {
		var zero O
		run.fail(ctx, err)
		return zero, err
	}
	run.end(ctx, output)
	return output, nil
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

// Stream streams the selected runnable, with the same chain-run shape as
// Invoke (Branch's own run wraps the selected stream: chain_stream per chunk,
// chain_end/chain_error on drain).
func (r Branch[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	ctx, run := startChainRun(ctx, opts, "branch", input)
	for i, item := range r.Cases {
		ok, err := item.Condition.Invoke(ctx, input, run.child(fmt.Sprintf("condition:%d", i+1), opts...)...)
		if err != nil {
			run.fail(ctx, err)
			return nil, err
		}
		if ok {
			stream, err := item.Runnable.Stream(ctx, input, run.child(fmt.Sprintf("branch:%d", i+1), opts...)...)
			if err != nil {
				run.fail(ctx, err)
				return nil, err
			}
			return wrapChainStream(stream, run), nil
		}
	}
	stream, err := r.Default.Stream(ctx, input, run.child("branch:default", opts...)...)
	if err != nil {
		run.fail(ctx, err)
		return nil, err
	}
	return wrapChainStream(stream, run), nil
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
	// ExceptionsToHandle gates when fallbacks activate (Python base.py:2007
	// RunnableWithFallbacks.exceptions_to_handle). An error from the primary
	// runnable or a fallback triggers the next fallback only when at least one
	// matcher returns true; any other error is returned immediately without
	// trying further fallbacks. nil means every error activates the fallback
	// chain, matching Python's default of (Exception,).
	//
	// Build matchers with MatchErrors for sentinel errors, or supply an
	// errors.As closure for typed errors — the Go counterpart of Python's
	// isinstance check.
	ExceptionsToHandle []func(error) bool
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

// Invoke tries the primary runnable and fallbacks in order. Only errors
// matched by ExceptionsToHandle move on to the next fallback; unmatched
// errors propagate immediately. With callbacks active WithFallbacks is one
// chain run and every attempt is a child run named "fallback:primary" /
// "fallback:N"; a failed attempt closes its child run with chain_error
// (emitted by the attempt's own instrumentation) before the next one starts.
func (r WithFallbacks[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	ctx, run := startChainRun(ctx, opts, "with_fallbacks", input)
	var firstErr error
	runnables := append([]Runnable[I, O]{r.Runnable}, r.Fallbacks...)
	for i, runnable := range runnables {
		output, err := runnable.Invoke(ctx, input, run.child(fallbackChildName(i), opts...)...)
		if err == nil {
			run.end(ctx, output)
			return output, nil
		}
		if !r.handles(err) {
			var zero O
			run.fail(ctx, err)
			return zero, err
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	var zero O
	run.fail(ctx, firstErr)
	return zero, firstErr
}

// handles reports whether err should activate the fallback chain.
func (r WithFallbacks[I, O]) handles(err error) bool {
	if len(r.ExceptionsToHandle) == 0 {
		return true
	}
	for _, match := range r.ExceptionsToHandle {
		if match(err) {
			return true
		}
	}
	return false
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

// Stream tries each runnable's stream in order, gated by ExceptionsToHandle
// exactly like Invoke. With callbacks active the surviving stream is wrapped
// in the WithFallbacks run (chain_stream per chunk; chain_error if the
// consumer's pull fails — mid-stream errors never trigger another fallback).
func (r WithFallbacks[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	ctx, run := startChainRun(ctx, opts, "with_fallbacks", input)
	var firstErr error
	runnables := append([]Runnable[I, O]{r.Runnable}, r.Fallbacks...)
	for i, runnable := range runnables {
		stream, err := runnable.Stream(ctx, input, run.child(fallbackChildName(i), opts...)...)
		if err == nil {
			return wrapChainStream(stream, run), nil
		}
		if !r.handles(err) {
			run.fail(ctx, err)
			return nil, err
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	run.fail(ctx, firstErr)
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

// MatchErrors builds an ExceptionsToHandle matcher for WithFallbacks that
// reports true when err matches any target sentinel via errors.Is (the Go
// counterpart of Python's isinstance check against exceptions_to_handle). It
// also matches wrapped errors, unlike a bare == comparison.
//
// For typed errors, supply an errors.As closure directly:
//
//	fb.ExceptionsToHandle = []func(error) bool{
//	    runnables.MatchErrors(errRateLimited),
//	    func(err error) bool {
//	        var timeout *net.DNSError
//	        return errors.As(err, &timeout)
//	    },
//	}
func MatchErrors(targets ...error) func(error) bool {
	return func(err error) bool {
		for _, target := range targets {
			if errors.Is(err, target) {
				return true
			}
		}
		return false
	}
}
