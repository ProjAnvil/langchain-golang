package runnables

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/schema"
)

func TestBatchAsCompletedYieldsInCompletionOrder(t *testing.T) {
	runnable := NewFunc(func(_ context.Context, i int, _ ...Option) (int, error) {
		// Input 0 is slowest, later inputs get progressively faster, so
		// completion order must be 5, 4, 3, 2, 1, 0.
		time.Sleep(time.Duration(20*(6-i)) * time.Millisecond)
		return i * 10, nil
	}, schema.Integer(""), schema.Integer(""))

	seen := map[int]BatchResult[int]{}
	var order []int
	for index, result := range BatchAsCompleted(t.Context(), runnable,
		[]int{0, 1, 2, 3, 4, 5}, WithMaxConcurrency(6)) {
		seen[index] = result
		order = append(order, index)
	}

	if len(order) != 6 {
		t.Fatalf("yielded %d results, want 6", len(order))
	}
	if order[0] != 5 || order[len(order)-1] != 0 {
		t.Fatalf("completion order = %v, want fastest-first with slow input 0 last", order)
	}
	for i, result := range seen {
		if result.Err != nil {
			t.Fatalf("result[%d].Err = %v", i, result.Err)
		}
		if result.Output != i*10 {
			t.Fatalf("result[%d].Output = %d, want %d", i, result.Output, i*10)
		}
	}
}

func TestBatchAsCompletedSequentialWithOneWorker(t *testing.T) {
	runnable := NewFunc(func(_ context.Context, i int, _ ...Option) (int, error) {
		// Staggered sleeps make completion order observable.
		time.Sleep(time.Duration(6-i) * time.Millisecond)
		return i, nil
	}, schema.Integer(""), schema.Integer(""))

	var order []int
	for index, result := range BatchAsCompleted(t.Context(), runnable,
		[]int{0, 1, 2, 3, 4, 5}, WithMaxConcurrency(1)) {
		if result.Err != nil {
			t.Fatalf("result[%d].Err = %v", index, result.Err)
		}
		order = append(order, index)
	}
	want := []int{0, 1, 2, 3, 4, 5}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order with one worker = %v, want %v", order, want)
		}
	}
}

func TestBatchAsCompletedHonorsMaxConcurrency(t *testing.T) {
	var inFlight, peak atomic.Int32
	var mu sync.Mutex
	runnable := NewFunc(func(_ context.Context, n int, _ ...Option) (int, error) {
		cur := inFlight.Add(1)
		mu.Lock()
		if cur > peak.Load() {
			peak.Store(cur)
		}
		mu.Unlock()
		time.Sleep(2 * time.Millisecond)
		inFlight.Add(-1)
		return n, nil
	}, schema.Integer(""), schema.Integer(""))

	inputs := make([]int, 24)
	for i := range inputs {
		inputs[i] = i
	}
	count := 0
	for index, result := range BatchAsCompleted(t.Context(), runnable, inputs, WithMaxConcurrency(3)) {
		if result.Err != nil {
			t.Fatalf("result[%d].Err = %v", index, result.Err)
		}
		count++
	}
	if count != len(inputs) {
		t.Fatalf("yielded %d results, want %d", count, len(inputs))
	}
	if peak.Load() > 3 {
		t.Fatalf("peak concurrency = %d, want <= 3", peak.Load())
	}
}

func TestBatchAsCompletedYieldsErrorsPerInput(t *testing.T) {
	runnable := NewFunc(func(_ context.Context, i int, _ ...Option) (int, error) {
		if i == 2 {
			return 0, errTestSentinel
		}
		return i, nil
	}, schema.Integer(""), schema.Integer(""))

	results := maps.Collect(BatchAsCompleted(t.Context(), runnable, []int{0, 1, 2, 3}))
	if len(results) != 4 {
		t.Fatalf("yielded %d results, want 4", len(results))
	}
	if err := results[2].Err; !errors.Is(err, errTestSentinel) {
		t.Fatalf("results[2].Err = %v, want %v", err, errTestSentinel)
	}
	for _, i := range []int{0, 1, 3} {
		if results[i].Err != nil || results[i].Output != i {
			t.Fatalf("results[%d] = %#v, want success with output %d", i, results[i], i)
		}
	}
}

func TestBatchAsCompletedEarlyBreakStopsYields(t *testing.T) {
	runnable := NewFunc(func(_ context.Context, i int, _ ...Option) (int, error) {
		return i, nil
	}, schema.Integer(""), schema.Integer(""))

	inputs := []int{0, 1, 2, 3}
	count := 0
	for _, result := range BatchAsCompleted(t.Context(), runnable, inputs) {
		if result.Err != nil {
			t.Fatalf("result.Err = %v", result.Err)
		}
		count++
		break // abandon the iteration after the first result
	}
	if count != 1 {
		t.Fatalf("consumed %d results after break, want 1", count)
	}
}

func TestBatchAsCompletedCanceledContextYieldsEveryIndex(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	runnable := NewFunc(func(_ context.Context, i int, _ ...Option) (int, error) {
		if i == 0 {
			cancel()
		}
		return i, nil
	}, schema.Integer(""), schema.Integer(""))

	results := maps.Collect(BatchAsCompleted(ctx, runnable, []int{0, 1, 2, 3}, WithMaxConcurrency(1)))
	if len(results) != 4 {
		t.Fatalf("yielded %d results, want one per input (4)", len(results))
	}
	if err := results[3].Err; !errors.Is(err, context.Canceled) {
		t.Fatalf("results[3].Err = %v, want %v", err, context.Canceled)
	}
}

func TestBatchAsCompletedEmptyInputs(t *testing.T) {
	runnable := NewFunc(func(_ context.Context, i int, _ ...Option) (int, error) {
		return i, nil
	}, schema.Integer(""), schema.Integer(""))

	for _, result := range BatchAsCompleted(t.Context(), runnable, nil) {
		t.Fatalf("unexpected yield: %#v", result)
	}
}

func TestBatchAsCompletedPreservesCancellationCause(t *testing.T) {
	for _, limit := range []int{1, 8} {
		t.Run(fmt.Sprintf("limit=%d", limit), func(t *testing.T) {
			cause := errors.New("batch canceled by caller")
			ctx, cancel := context.WithCancelCause(t.Context())
			cancel(cause)
			var calls atomic.Int32
			runnable := NewFunc(func(_ context.Context, i int, _ ...Option) (int, error) {
				calls.Add(1)
				return i, nil
			}, schema.Integer(""), schema.Integer(""))

			seen := make([]bool, 3)
			for index, result := range BatchAsCompleted(ctx, runnable, []int{1, 2, 3}, WithMaxConcurrency(limit)) {
				if index < 0 || index >= len(seen) || seen[index] {
					t.Fatalf("unexpected or duplicate result index %d", index)
				}
				seen[index] = true
				if !errors.Is(result.Err, cause) || result.Output != 0 {
					t.Fatalf("result[%d] = %#v, want zero output and cancellation cause %v", index, result, cause)
				}
			}
			for i, yielded := range seen {
				if !yielded {
					t.Fatalf("input %d did not yield a result", i)
				}
			}
			if got := calls.Load(); got != 0 {
				t.Fatalf("invoked %d inputs after cancellation, want 0", got)
			}
		})
	}
}
