// This file implements LCEL-style sequential composition: Pipe, Pipe3, Pipe4,
// Pipe5, Pipe6.
//
// Python LCEL chains runnables with the | operator:
//
//	chain = prompt | model | parser
//
// Go does not support operator overloading, so the idiomatic equivalent is:
//
//	chain := runnables.Pipe3(prompt, model, parser)
//
// Each Pipe* function checks the type chain (I -> A -> B -> ... -> O) at
// compile time at the call site and returns a SeqN[I,O], which satisfies
// Runnable[I,O] and therefore composes with every existing combinator —
// NewWithFallbacks, NewRetry, NewBranch, NewConfigurableAlternatives, ...
// For chains longer than six steps, nest Pipe: a SeqN is itself a Runnable,
// so Pipe(Pipe3(a, b, c), Pipe3(d, e, f)) works directly. Pipe is also
// the zero-error variant of NewSequence: a nil runnable is a programmer error
// and panics at construction rather than threading an error through the chain.
package runnables

import (
	"context"
	"errors"
	"fmt"

	"github.com/projanvil/langchain-golang/core/schema"
)

// Pipe composes two runnables so the first's output feeds the second,
// mirroring Python LCEL's `a | b`. Unlike NewSequence it returns no error:
// a nil runnable is a programmer error and panics at construction, so chains
// stay flat for readability. The result satisfies Runnable[I, O] and composes
// with every existing combinator (NewWithFallbacks, NewRetry, NewBranch, ...).
func Pipe[I, M, O any](first Runnable[I, M], second Runnable[M, O]) SeqN[I, O] {
	if first == nil || second == nil {
		panic("runnables.Pipe: nil runnable")
	}
	return SeqN[I, O]{
		steps: []Runnable[any, any]{
			erase[I, M](first),
			erase[M, O](second),
		},
		inSchema:  first.InputSchema(),
		outSchema: second.OutputSchema(),
	}
}

// Pipe3 composes three runnables left-to-right, mirroring `a | b | c`. The
// type chain (I -> A -> B -> O) is checked at compile time at the call site.
func Pipe3[I, A, B, O any](r1 Runnable[I, A], r2 Runnable[A, B], r3 Runnable[B, O]) SeqN[I, O] {
	if r1 == nil || r2 == nil || r3 == nil {
		panic("runnables.Pipe3: nil runnable")
	}
	return SeqN[I, O]{
		steps: []Runnable[any, any]{
			erase[I, A](r1),
			erase[A, B](r2),
			erase[B, O](r3),
		},
		inSchema:  r1.InputSchema(),
		outSchema: r3.OutputSchema(),
	}
}

// Pipe4 composes four runnables left-to-right, mirroring `a | b | c | d`.
func Pipe4[I, A, B, C, O any](r1 Runnable[I, A], r2 Runnable[A, B], r3 Runnable[B, C], r4 Runnable[C, O]) SeqN[I, O] {
	if r1 == nil || r2 == nil || r3 == nil || r4 == nil {
		panic("runnables.Pipe4: nil runnable")
	}
	return SeqN[I, O]{
		steps: []Runnable[any, any]{
			erase[I, A](r1),
			erase[A, B](r2),
			erase[B, C](r3),
			erase[C, O](r4),
		},
		inSchema:  r1.InputSchema(),
		outSchema: r4.OutputSchema(),
	}
}

// Pipe5 composes five runnables left-to-right.
func Pipe5[I, A, B, C, D, O any](
	r1 Runnable[I, A], r2 Runnable[A, B], r3 Runnable[B, C], r4 Runnable[C, D], r5 Runnable[D, O],
) SeqN[I, O] {
	if r1 == nil || r2 == nil || r3 == nil || r4 == nil || r5 == nil {
		panic("runnables.Pipe5: nil runnable")
	}
	return SeqN[I, O]{
		steps: []Runnable[any, any]{
			erase[I, A](r1),
			erase[A, B](r2),
			erase[B, C](r3),
			erase[C, D](r4),
			erase[D, O](r5),
		},
		inSchema:  r1.InputSchema(),
		outSchema: r5.OutputSchema(),
	}
}

// Pipe6 composes six runnables left-to-right. For longer chains, nest Pipe
// (Pipe(Pipe(a, b), Pipe3(c, d, e))...) — since SeqN satisfies Runnable[I, O],
// it composes with Pipe just like any other runnable.
func Pipe6[I, A, B, C, D, E, O any](
	r1 Runnable[I, A], r2 Runnable[A, B], r3 Runnable[B, C], r4 Runnable[C, D], r5 Runnable[D, E], r6 Runnable[E, O],
) SeqN[I, O] {
	if r1 == nil || r2 == nil || r3 == nil || r4 == nil || r5 == nil || r6 == nil {
		panic("runnables.Pipe6: nil runnable")
	}
	return SeqN[I, O]{
		steps: []Runnable[any, any]{
			erase[I, A](r1),
			erase[A, B](r2),
			erase[B, C](r3),
			erase[C, D](r4),
			erase[D, E](r5),
			erase[E, O](r6),
		},
		inSchema:  r1.InputSchema(),
		outSchema: r6.OutputSchema(),
	}
}

// SeqN is the N-step sequential composition returned by Pipe/Pipe3/....
// Internally each step is type-erased to Runnable[any, any] (mirroring Python's
// RunnableSequence, which also passes values dynamically between steps); the
// type chain is validated at compile time at the Pipe*/NewSequence call site,
// so the erasure is safe under correct use. SeqN[I, O] satisfies Runnable[I, O].
type SeqN[I, O any] struct {
	steps     []Runnable[any, any]
	inSchema  schema.Schema
	outSchema schema.Schema
}

// Invoke runs every step in order, threading each output into the next step.
func (s SeqN[I, O]) Invoke(ctx context.Context, input I, opts ...Option) (O, error) {
	var v any = input
	for i, step := range s.steps {
		next, err := step.Invoke(ctx, v, childOptions(fmt.Sprintf("seq:step:%d", i+1), opts...)...)
		if err != nil {
			var z O
			return z, err
		}
		v = next
	}
	out, ok := v.(O)
	if !ok {
		var z O
		return z, fmt.Errorf("runnables: sequence output %T not assignable to output type", v)
	}
	return out, nil
}

// Batch runs the whole sequence for each input, batching one step at a time so
// a step with native batch support (e.g. a chat model) still benefits from it.
func (s SeqN[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	values := make([]any, len(inputs))
	for i, in := range inputs {
		values[i] = in
	}
	for i, step := range s.steps {
		out, err := step.Batch(ctx, values, childOptions(fmt.Sprintf("seq:step:%d", i+1), opts...)...)
		if err != nil {
			return nil, err
		}
		values = out
	}
	result := make([]O, len(values))
	for i, v := range values {
		o, ok := v.(O)
		if !ok {
			return nil, fmt.Errorf("runnables: sequence output %d (%T) not assignable to output type", i, v)
		}
		result[i] = o
	}
	return result, nil
}

// Stream streams the sequence by piping each chunk of one stage into the next
// stage's Stream and flattening the results in order, mirroring Python
// RunnableSequence streaming. The final Stream[any] is narrowed to Stream[O]
// at the tail.
func (s SeqN[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	if len(s.steps) == 0 {
		return nil, fmt.Errorf("runnables.SeqN: empty sequence")
	}
	cur, err := s.steps[0].Stream(ctx, any(input), childOptions("seq:step:1", opts...)...)
	if err != nil {
		return nil, err
	}
	for i := 1; i < len(s.steps); i++ {
		stepIdx := i      // captured by makeNext below
		step := s.steps[i]
		makeNext := func(ctx context.Context, v any) (Stream[any], error) {
			return step.Stream(ctx, v, childOptions(fmt.Sprintf("seq:step:%d", stepIdx+1), opts...)...)
		}
		cur = &seqNStream{first: cur, makeNext: makeNext}
	}
	return &seqNTailStream[O]{inner: cur}, nil
}

// InputSchema returns the first step's input schema.
func (s SeqN[I, O]) InputSchema() schema.Schema { return s.inSchema }

// OutputSchema returns the last step's output schema.
func (s SeqN[I, O]) OutputSchema() schema.Schema { return s.outSchema }

// ConfigSchema returns the union of all step config schemas.
func (s SeqN[I, O]) ConfigSchema() schema.Schema {
	children := make([]any, 0, len(s.steps))
	for _, step := range s.steps {
		children = append(children, step)
	}
	return mergeConfigSchemas(children...)
}

// Compile-time assertion that SeqN[I, O] satisfies Runnable[I, O].
var _ Runnable[any, any] = SeqN[any, any]{}

// erase adapts a Runnable[A, B] to Runnable[any, any] by boxing/unboxing at
// the boundaries. Used internally by Pipe* to store heterogeneous steps in a
// single slice. Type assertions surface a clear error if a sequence is wired
// with mismatched types (which the Pipe* call sites prevent at compile time).
func erase[A, B any](r Runnable[A, B]) Runnable[any, any] {
	return &erasedRunnable[A, B]{inner: r}
}

type erasedRunnable[A, B any] struct {
	inner Runnable[A, B]
}

func (e *erasedRunnable[A, B]) Invoke(ctx context.Context, input any, opts ...Option) (any, error) {
	a, ok := input.(A)
	if !ok {
		return nil, fmt.Errorf("runnables: sequence step input %T not assignable to step input type", input)
	}
	return e.inner.Invoke(ctx, a, opts...)
}

func (e *erasedRunnable[A, B]) Batch(ctx context.Context, inputs []any, opts ...Option) ([]any, error) {
	as := make([]A, len(inputs))
	for i, in := range inputs {
		a, ok := in.(A)
		if !ok {
			return nil, fmt.Errorf("runnables: sequence step batch input %d (%T) not assignable to step input type", i, in)
		}
		as[i] = a
	}
	out, err := e.inner.Batch(ctx, as, opts...)
	if err != nil {
		return nil, err
	}
	result := make([]any, len(out))
	for i, v := range out {
		result[i] = v
	}
	return result, nil
}

func (e *erasedRunnable[A, B]) Stream(ctx context.Context, input any, opts ...Option) (Stream[any], error) {
	a, ok := input.(A)
	if !ok {
		return nil, fmt.Errorf("runnables: sequence step stream input %T not assignable to step input type", input)
	}
	s, err := e.inner.Stream(ctx, a, opts...)
	if err != nil {
		return nil, err
	}
	return &erasedStream[B]{inner: s}, nil
}

func (e *erasedRunnable[A, B]) InputSchema() schema.Schema  { return e.inner.InputSchema() }
func (e *erasedRunnable[A, B]) OutputSchema() schema.Schema { return e.inner.OutputSchema() }

// ConfigSchema forwards the wrapped runnable's config schema so SeqN's merged
// ConfigSchema sees each step's configurable fields. Without this, the type
// erasure would drop the configSchemaProvider interface and SeqN.ConfigSchema
// would return an empty schema. GetConfigSchema handles the absent-case.
func (e *erasedRunnable[A, B]) ConfigSchema() schema.Schema { return GetConfigSchema(e.inner) }

type erasedStream[B any] struct {
	inner Stream[B]
}

func (s *erasedStream[B]) Next(ctx context.Context) (any, bool, error) {
	v, ok, err := s.inner.Next(ctx)
	if err != nil {
		var z any
		return z, false, err
	}
	if !ok {
		var z any
		return z, false, nil
	}
	return v, true, nil
}

func (s *erasedStream[B]) Close() error { return s.inner.Close() }

// seqNStream flattens a staged stream: each value pulled from first is fed to
// makeNext, which returns the next stage's stream; that stream is drained
// before pulling another value from first. This yields depth-first order
// (a1,a2,b1,b2 for first=[a,b] with transform producing [x1,x2] each),
// matching Sequence's sequenceStream behavior generalized to N stages.
type seqNStream struct {
	first    Stream[any]
	current  Stream[any]
	makeNext func(context.Context, any) (Stream[any], error)
}

func (s *seqNStream) Next(ctx context.Context) (any, bool, error) {
	var z any
	for {
		if s.current != nil {
			v, ok, err := s.current.Next(ctx)
			if err != nil {
				return z, false, err
			}
			if ok {
				return v, true, nil
			}
			if err := s.current.Close(); err != nil {
				return z, false, err
			}
			s.current = nil
		}
		mid, ok, err := s.first.Next(ctx)
		if err != nil {
			return z, false, err
		}
		if !ok {
			return z, false, nil
		}
		s.current, err = s.makeNext(ctx, mid)
		if err != nil {
			return z, false, err
		}
	}
}

func (s *seqNStream) Close() error {
	var errs []error
	if s.current != nil {
		errs = append(errs, s.current.Close())
		s.current = nil
	}
	if s.first != nil {
		errs = append(errs, s.first.Close())
	}
	return errors.Join(errs...)
}

// seqNTailStream narrows the final Stream[any] to Stream[O], asserting each
// chunk's type. A mismatch means the sequence was wired with inconsistent
// types, which the Pipe* call sites prevent at compile time.
type seqNTailStream[O any] struct {
	inner Stream[any]
}

func (s *seqNTailStream[O]) Next(ctx context.Context) (O, bool, error) {
	var z O
	v, ok, err := s.inner.Next(ctx)
	if err != nil {
		return z, false, err
	}
	if !ok {
		return z, false, nil
	}
	o, isO := v.(O)
	if !isO {
		return z, false, fmt.Errorf("runnables: stream output %T not assignable to output type", v)
	}
	return o, true, nil
}

func (s *seqNTailStream[O]) Close() error { return s.inner.Close() }
