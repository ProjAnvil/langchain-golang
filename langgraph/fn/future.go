package fn

import (
	"context"

	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Future is a pending task result with exactly two outcome states: a value
// or an error (a task that interrupted carries a *types.GraphInterrupt
// instead — see Get).
type Future[O any] struct {
	// done is closed when the future resolves. val/err/gi are written
	// before the close, so the close orders them before any read in Get
	// (happens-before via the channel).
	done chan struct{}
	val  O
	err  error
	gi   *types.GraphInterrupt
}

// Get blocks until the task completes and returns its result. A task that
// called graph.Interrupt re-panics that *types.GraphInterrupt from Get, so
// an interrupt raised inside a task pauses the enclosing run exactly as if
// the entrypoint function itself had interrupted (Python parity: interrupts
// from call tasks are the parent's responsibility, `_algo.py:844-846`).
// Callers must not recover this panic. Get also honors ctx cancellation.
func (f *Future[O]) Get(ctx context.Context) (O, error) {
	select {
	case <-f.done:
		if f.gi != nil {
			panic(f.gi)
		}
		return f.val, f.err
	case <-ctx.Done():
		var zero O
		return zero, ctx.Err()
	}
}

// resolvedFuture returns an already-resolved Future, used for checkpoint
// replays and cache hits where the outcome is known without executing the
// task. (Go does not allow generic methods on non-generic types, so the
// resolver helpers are package-level generic functions.)
func resolvedFuture[O any](val O, err error, gi *types.GraphInterrupt) *Future[O] {
	f := &Future[O]{done: make(chan struct{}), val: val, err: err, gi: gi}
	close(f.done)
	return f
}

// AwaitAll waits for every future and returns their values in argument
// order; the first error (in argument order among failed futures) aborts
// with that error. A *types.GraphInterrupt carried by any future propagates
// as a panic (same rule as Future.Get).
func AwaitAll[T any](ctx context.Context, futs ...*Future[T]) ([]T, error) {
	out := make([]T, len(futs))
	for i, fut := range futs {
		v, err := fut.Get(ctx)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}
