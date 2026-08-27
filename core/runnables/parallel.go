package runnables

import (
	"context"
	"errors"
	"runtime"
	"sync"
)

// DefaultParallelism is the concurrency bound used when MaxConcurrency is
// unset: min(32, GOMAXPROCS+4), the same default as Python's global batch
// executor (concurrent.futures.ThreadPoolExecutor default max_workers). A
// bound matters because batch inputs map to provider API calls; unbounded
// fan-out gets a large batch rate-limited or banned rather than served
// faster.
func DefaultParallelism() int {
	workers := runtime.GOMAXPROCS(0) + 4
	if workers > 32 {
		workers = 32
	}
	if workers < 1 {
		workers = 1
	}
	return workers
}

// ParallelMap invokes fn for every input with bounded concurrency and
// returns the outputs in input order. The bound is cfg.MaxConcurrency when
// positive, otherwise DefaultParallelism. It is the shared engine behind
// Batch implementations (Python parity: RunnableConfig.max_concurrency
// gates every executor in Python's batch path), so all batch surfaces —
// runnables, chat models, LLMs, providers — fan out identically.
//
// Errors from fn are collected per input and joined in input order; fn keeps
// running for the remaining inputs unless ctx is canceled, in which case
// unstarted inputs are skipped and context.Cause(ctx) is joined onto the
// result. Each fn call receives ctx, so cancellation also reaches in-flight
// work through whatever the underlying implementation does with it.
func ParallelMap[I, O any](
	ctx context.Context,
	cfg Config,
	inputs []I,
	fn func(context.Context, I) (O, error),
) ([]O, error) {
	outputs := make([]O, len(inputs))
	if len(inputs) == 0 {
		return outputs, nil
	}

	limit := cfg.MaxConcurrency
	if limit <= 0 {
		limit = DefaultParallelism()
	}
	if limit == 1 || len(inputs) == 1 {
		errs := make([]error, len(inputs))
		for i, input := range inputs {
			if ctx.Err() != nil {
				errs[i] = context.Cause(ctx)
				continue
			}
			outputs[i], errs[i] = fn(ctx, input)
		}
		return outputs, errors.Join(errs...)
	}
	if limit > len(inputs) {
		limit = len(inputs)
	}

	errs := make([]error, len(inputs))
	// Fixed worker pool over an index channel: goroutine count is bounded by
	// limit, not by batch size. Distinct indices write distinct slice slots,
	// so no locking is needed on outputs/errs.
	indexes := make(chan int)
	var wg sync.WaitGroup
	wg.Add(limit)
	for w := 0; w < limit; w++ {
		go func() {
			defer wg.Done()
			for i := range indexes {
				if ctx.Err() != nil {
					errs[i] = context.Cause(ctx)
					continue
				}
				outputs[i], errs[i] = fn(ctx, inputs[i])
			}
		}()
	}
dispatch:
	for i := range inputs {
		select {
		case indexes <- i:
		case <-ctx.Done():
			// Drain: remaining workers exit when the channel closes, and
			// skipped inputs surface the cancellation cause below.
			for ; i < len(inputs); i++ {
				errs[i] = context.Cause(ctx)
			}
			break dispatch
		}
	}
	close(indexes)
	wg.Wait()

	return outputs, errors.Join(errs...)
}
