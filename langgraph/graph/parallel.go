package graph

import (
	"runtime"
	"sync"
)

// defaultSuperstepParallelism is the concurrency bound for one superstep
// when Options.MaxConcurrency is unset: min(32, GOMAXPROCS+4). This matches
// Python Pregel's global executor default (a ThreadPoolExecutor with
// default max_workers), which also caps parallel node execution when
// max_concurrency is not configured.
func defaultSuperstepParallelism() int {
	workers := runtime.GOMAXPROCS(0) + 4
	if workers > 32 {
		workers = 32
	}
	if workers < 1 {
		workers = 1
	}
	return workers
}

// superstepBound resolves the effective worker count for a superstep of
// size count given the caller's MaxConcurrency option.
func superstepBound(maxConcurrency, count int) int {
	limit := maxConcurrency
	if limit <= 0 {
		limit = defaultSuperstepParallelism()
	}
	if limit > count {
		limit = count
	}
	if limit < 1 {
		limit = 1
	}
	return limit
}

// runSuperstepBounded invokes fn(i, t) for every active[i] with at most
// limit concurrent invocations, via a fixed worker pool. fn must write its
// results only for its own index (workers own disjoint indices, so no
// locking is needed). Unlike the batch engine in core/runnables this helper
// never skips work on cancellation: a superstep's tasks all run to
// completion so checkpoint/interrupt bookkeeping sees every outcome;
// cancellation surfaces through the task bodies themselves.
func runSuperstepBounded(active []task, execute []bool, limit int, fn func(i int, t task)) {
	if limit <= 1 {
		for i, t := range active {
			if execute[i] {
				fn(i, t)
			}
		}
		return
	}
	indexes := make(chan int)
	var wg sync.WaitGroup
	wg.Add(limit)
	for w := 0; w < limit; w++ {
		go func() {
			defer wg.Done()
			for i := range indexes {
				fn(i, active[i])
			}
		}()
	}
	for i := range active {
		if execute[i] {
			indexes <- i
		}
	}
	close(indexes)
	wg.Wait()
}
