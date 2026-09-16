// This file implements BatchAsCompleted, the completion-ordered batch
// variant (Python base.py:970 Runnable.batch_as_completed).
package runnables

import (
	"context"
	"fmt"
	"iter"
	"sync"
)

// BatchResult is the outcome of one input in a BatchAsCompleted run.
type BatchResult[O any] struct {
	// Output is the runnable's output; valid only when Err is nil.
	Output O
	// Err is the invocation error for this input, or context.Cause(ctx) for
	// inputs skipped after cancellation.
	Err error
}

// BatchAsCompleted invokes r for every input with bounded concurrency and
// yields (index, result) pairs in completion order (Python base.py:970
// Runnable.batch_as_completed). The fan-out is bounded by
// Config.MaxConcurrency (default DefaultParallelism), exactly like Batch.
//
// Go has no exceptions, so there is no return_exceptions flag: a failed input
// yields BatchResult.Err and iteration continues. Every input yields exactly
// one pair, including inputs skipped after ctx cancellation, which yield
// context.Cause(ctx).
//
// Stop ranging early to abandon the run: the iterator does NOT cancel ctx —
// cancellation only ever arrives from the caller cancelling the ctx passed
// in. The sequence function blocks until the in-flight invocations finish
// (leaking no goroutines), and inputs that never started are skipped.
func BatchAsCompleted[I any, O any](
	ctx context.Context,
	r Runnable[I, O],
	inputs []I,
	opts ...Option,
) iter.Seq2[int, BatchResult[O]] {
	return func(yield func(int, BatchResult[O]) bool) {
		if len(inputs) == 0 {
			return
		}
		cfg := NewConfig(opts...)
		limit := cfg.MaxConcurrency
		if limit <= 0 {
			limit = DefaultParallelism()
		}
		if limit > len(inputs) {
			limit = len(inputs)
		}

		type outcome struct {
			index  int
			result BatchResult[O]
		}
		indexes := make(chan int)
		results := make(chan outcome)
		done := make(chan struct{})

		// Fixed worker pool over an index channel: goroutine count is bounded
		// by limit, not by batch size, mirroring ParallelMap.
		var wg sync.WaitGroup
		wg.Add(limit)
		for range limit {
			go func() {
				defer wg.Done()
				for i := range indexes {
					var result BatchResult[O]
					if err := context.Cause(ctx); err != nil {
						result.Err = err
					} else {
						output, err := r.Invoke(ctx, inputs[i], childOptions(fmt.Sprintf("batch:%d", i+1), opts...)...)
						result.Output, result.Err = output, err
					}
					select {
					case results <- outcome{index: i, result: result}:
					case <-done:
						return
					}
				}
			}()
		}

		// Dispatch indexes to the workers; on cancellation, report the cause
		// for every remaining input so each index yields exactly one result.
		go func() {
			defer close(indexes)
			for i := range inputs {
				select {
				case indexes <- i:
				case <-done:
					return
				case <-ctx.Done():
					for ; i < len(inputs); i++ {
						select {
						case results <- outcome{index: i, result: BatchResult[O]{Err: context.Cause(ctx)}}:
						case <-done:
							return
						}
					}
					return
				}
			}
		}()

		// Closing done unblocks workers and the dispatcher when the consumer
		// stops early; wg.Wait keeps the seq call from returning while worker
		// goroutines still reference the channel graph.
		defer func() {
			close(done)
			wg.Wait()
		}()

		for received := 0; received < len(inputs); received++ {
			out := <-results
			if !yield(out.index, out.result) {
				return
			}
		}
	}
}
