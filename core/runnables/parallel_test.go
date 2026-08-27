package runnables

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestParallelMapPreservesOrderUnderConcurrency(t *testing.T) {
	inputs := make([]int, 200)
	for i := range inputs {
		inputs[i] = i
	}
	outputs, err := ParallelMap(context.Background(), Config{MaxConcurrency: 8}, inputs,
		func(_ context.Context, n int) (string, error) {
			// Stagger completion so earlier inputs finish late.
			time.Sleep(time.Duration(200-n) * time.Microsecond)
			return fmt.Sprintf("out-%d", n), nil
		})
	if err != nil {
		t.Fatalf("ParallelMap: %v", err)
	}
	for i, got := range outputs {
		if got != fmt.Sprintf("out-%d", i) {
			t.Fatalf("outputs[%d] = %q, want index-ordered value", i, got)
		}
	}
}

func TestParallelMapRespectsConcurrencyLimit(t *testing.T) {
	var inFlight, peak atomic.Int32
	var mu sync.Mutex // guards peak readback
	inputs := make([]int, 64)
	outputs, err := ParallelMap(context.Background(), Config{MaxConcurrency: 4}, inputs,
		func(_ context.Context, n int) (int, error) {
			cur := inFlight.Add(1)
			mu.Lock()
			if cur > peak.Load() {
				peak.Store(cur)
			}
			mu.Unlock()
			time.Sleep(2 * time.Millisecond)
			inFlight.Add(-1)
			return n, nil
		})
	if err != nil {
		t.Fatalf("ParallelMap: %v", err)
	}
	if len(outputs) != len(inputs) {
		t.Fatalf("len(outputs) = %d, want %d", len(outputs), len(inputs))
	}
	if got := peak.Load(); got > 4 {
		t.Fatalf("peak concurrency = %d, want <= 4", got)
	}
}

func TestParallelMapSequentialWhenLimitOne(t *testing.T) {
	var inFlight, peak atomic.Int32
	inputs := make([]int, 16)
	_, err := ParallelMap(context.Background(), Config{MaxConcurrency: 1}, inputs,
		func(_ context.Context, n int) (int, error) {
			cur := inFlight.Add(1)
			for {
				p := peak.Load()
				if cur <= p || peak.CompareAndSwap(p, cur) {
					break
				}
			}
			time.Sleep(time.Millisecond)
			inFlight.Add(-1)
			return n, nil
		})
	if err != nil {
		t.Fatalf("ParallelMap: %v", err)
	}
	if got := peak.Load(); got != 1 {
		t.Fatalf("peak concurrency = %d, want 1", got)
	}
}

func TestParallelMapDefaultBoundApplies(t *testing.T) {
	if DefaultParallelism() < 1 {
		t.Fatalf("DefaultParallelism = %d, want >= 1", DefaultParallelism())
	}
	var inFlight, peak atomic.Int32
	inputs := make([]int, DefaultParallelism()*4)
	_, err := ParallelMap(context.Background(), Config{}, inputs,
		func(_ context.Context, n int) (int, error) {
			cur := inFlight.Add(1)
			for {
				p := peak.Load()
				if cur <= p || peak.CompareAndSwap(p, cur) {
					break
				}
			}
			time.Sleep(time.Millisecond)
			inFlight.Add(-1)
			return n, nil
		})
	if err != nil {
		t.Fatalf("ParallelMap: %v", err)
	}
	if got := peak.Load(); got > int32(DefaultParallelism()) {
		t.Fatalf("peak concurrency = %d, want <= default %d", got, DefaultParallelism())
	}
}

func TestParallelMapJoinsErrorsInOrder(t *testing.T) {
	inputs := []int{0, 1, 2}
	_, err := ParallelMap(context.Background(), Config{}, inputs,
		func(_ context.Context, n int) (int, error) {
			if n == 1 {
				return 0, fmt.Errorf("boom-%d", n)
			}
			return n, nil
		})
	if err == nil {
		t.Fatal("expected joined error")
	}
	if got := err.Error(); got != "boom-1" {
		t.Fatalf("err = %q, want single boom-1", got)
	}

	_, err = ParallelMap(context.Background(), Config{}, []int{0, 1, 2},
		func(_ context.Context, n int) (int, error) {
			return 0, fmt.Errorf("e%d", n)
		})
	want := errors.Join(fmt.Errorf("e0"), fmt.Errorf("e1"), fmt.Errorf("e2")).Error()
	if err.Error() != want {
		t.Fatalf("joined err = %q, want %q", err, want)
	}
}

func TestParallelMapCancelSkipsUnstarted(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	inputs := make([]int, 500)
	_, err := ParallelMap(ctx, Config{MaxConcurrency: 2}, inputs,
		func(_ context.Context, n int) (int, error) {
			if n == 0 {
				cancel()
			}
			time.Sleep(time.Millisecond)
			return n, nil
		})
	if err == nil {
		t.Fatal("expected cancellation error joined onto result")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled inside", err)
	}
}

func TestFuncBatchHonorsMaxConcurrency(t *testing.T) {
	var inFlight, peak atomic.Int32
	fn := NewFunc(
		func(_ context.Context, n int, _ ...Option) (int, error) {
			cur := inFlight.Add(1)
			for {
				p := peak.Load()
				if cur <= p || peak.CompareAndSwap(p, cur) {
					break
				}
			}
			time.Sleep(time.Millisecond)
			inFlight.Add(-1)
			return n * 2, nil
		},
		nil,
		nil,
	)
	inputs := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	outputs, err := fn.Batch(context.Background(), inputs, WithMaxConcurrency(2))
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	for i, got := range outputs {
		if got != inputs[i]*2 {
			t.Fatalf("outputs[%d] = %d, want %d", i, got, inputs[i]*2)
		}
	}
	if got := peak.Load(); got > 2 {
		t.Fatalf("peak concurrency = %d, want <= 2", got)
	}
}
