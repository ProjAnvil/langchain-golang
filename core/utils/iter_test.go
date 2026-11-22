package utils

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// largeUtilCountingIter is a close-tracking slice iterator used by tee tests.
type largeUtilCountingIter struct {
	values []int
	index  int
	closed atomic.Bool
}

func (i *largeUtilCountingIter) Next(ctx context.Context) (int, bool, error) {
	if err := ctx.Err(); err != nil {
		return 0, false, err
	}
	if i.index >= len(i.values) {
		return 0, false, nil
	}
	v := i.values[i.index]
	i.index++
	return v, true, nil
}

func (i *largeUtilCountingIter) Close() error {
	i.closed.Store(true)
	return nil
}

// largeUtilBlockingSource only produces a value when one is sent on ch, and
// otherwise blocks until the context is canceled.
type largeUtilBlockingSource struct {
	ch chan int
}

func (b *largeUtilBlockingSource) Next(ctx context.Context) (int, bool, error) {
	select {
	case v := <-b.ch:
		return v, true, nil
	case <-ctx.Done():
		return 0, false, ctx.Err()
	}
}

func (b *largeUtilBlockingSource) Close() error { return nil }

func TestLargeUtilTeeSequentialConsumers(t *testing.T) {
	src := &largeUtilCountingIter{values: []int{1, 2, 3, 4, 5}}
	tee, err := NewTee(t.Context(), src, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tee.Close() }()

	peers := tee.Peers()
	if len(peers) != 2 || tee.Len() != 2 {
		t.Fatalf("expected 2 peers, got %d", len(peers))
	}
	if _, err := tee.At(0); err != nil {
		t.Fatal(err)
	}
	if _, err := tee.At(2); err == nil {
		t.Fatal("expected out-of-range error")
	}

	want := []int{1, 2, 3, 4, 5}
	got0, err := CollectIterator(t.Context(), peers[0])
	if err != nil {
		t.Fatal(err)
	}
	got1, err := CollectIterator(t.Context(), peers[1])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got0, want) || !reflect.DeepEqual(got1, want) {
		t.Fatalf("peers diverged: got0=%#v got1=%#v want=%#v", got0, got1, want)
	}
}

func TestLargeUtilTeeDifferentSpeedsWithBackpressure(t *testing.T) {
	values := make([]int, 40)
	for i := range values {
		values[i] = i
	}
	// A tiny buffer forces real backpressure: the hub cannot outrun the slowest
	// consumer by more than two items.
	tee, err := NewTee(t.Context(), NewSliceIterator(values), 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tee.Close() }()

	ctx := t.Context()
	peers := tee.Peers()
	results := make([][]int, len(peers))
	errs := make([]error, len(peers))
	var wg sync.WaitGroup
	for i := range peers {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			out := []int{}
			for {
				v, ok, err := peers[idx].Next(ctx)
				if err != nil {
					errs[idx] = err
					return
				}
				if !ok {
					break
				}
				out = append(out, v)
				if idx == 1 {
					time.Sleep(time.Millisecond) // deliberately slow consumer
				}
			}
			results[idx] = out
		}(i)
	}
	wg.Wait()

	for i := range peers {
		if errs[i] != nil {
			t.Fatalf("peer %d error: %v", i, errs[i])
		}
		if !reflect.DeepEqual(results[i], values) {
			t.Fatalf("peer %d got %d items, want %d identical items", i, len(results[i]), len(values))
		}
	}
}

func TestLargeUtilTeeContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	src := &largeUtilBlockingSource{ch: make(chan int)}
	tee, err := NewTee(ctx, src, 2)
	if err != nil {
		t.Fatal(err)
	}
	peers := tee.Peers()

	// A per-call canceled context returns promptly while the tee is still alive.
	perCtx, perCancel := context.WithCancel(context.Background())
	perCancel()
	if _, _, perr := peers[0].Next(perCtx); !errors.Is(perr, context.Canceled) {
		t.Fatalf("expected per-call context.Canceled, got %v", perr)
	}

	// Canceling the constructor context shuts the whole tee down cleanly.
	cancel()
	select {
	case <-tee.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("tee did not shut down after context cancellation")
	}
	if _, _, nerr := peers[1].Next(context.Background()); !errors.Is(nerr, context.Canceled) {
		t.Fatalf("expected terminal context.Canceled, got %v", nerr)
	}
	if err := tee.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLargeUtilTeeCloseReleasesSource(t *testing.T) {
	src := &largeUtilCountingIter{values: []int{1, 2, 3}}
	tee, err := NewTee(t.Context(), src, 2)
	if err != nil {
		t.Fatal(err)
	}
	peers := tee.Peers()
	for _, p := range peers {
		v, ok, err := p.Next(t.Context())
		if err != nil || !ok || v != 1 {
			t.Fatalf("first item = (%d, %v, %v)", v, ok, err)
		}
	}
	if err := tee.Close(); err != nil {
		t.Fatal(err)
	}
	if !src.closed.Load() {
		t.Fatal("source Close was not called on tee shutdown")
	}
	select {
	case <-tee.Done():
	default:
		t.Fatal("Done channel should be closed after Close")
	}
	if _, ok, _ := peers[0].Next(t.Context()); ok {
		t.Fatal("closed peer should be exhausted")
	}
	if err := tee.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestLargeUtilSafeteeAlias(t *testing.T) {
	var safetee *Safetee[int]
	tee, err := NewSafetee(t.Context(), NewSliceIterator([]int{1, 2}), 2)
	if err != nil {
		t.Fatal(err)
	}
	safetee = tee // *Tee[int] is assignable to *Safetee[int] (generic alias)
	if safetee.Len() != 2 {
		t.Fatalf("safetee Len = %d", safetee.Len())
	}
	got, err := CollectIterator(t.Context(), safetee.Peers()[0])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("safetee values = %#v", got)
	}
	_ = safetee.Close()
}

func TestLargeUtilBatchIterateFixedSize(t *testing.T) {
	it, err := BatchIterate(3, NewSliceIterator([]int{1, 2, 3, 4, 5, 6, 7}))
	if err != nil {
		t.Fatal(err)
	}
	batches, err := CollectIterator(t.Context(), it)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]int{{1, 2, 3}, {4, 5, 6}, {7}}
	if !reflect.DeepEqual(batches, want) {
		t.Fatalf("batches = %#v, want %#v", batches, want)
	}
}

func TestLargeUtilBatchIterateSingleBatch(t *testing.T) {
	it, err := BatchIterate(0, NewSliceIterator([]int{1, 2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	batches, err := CollectIterator(t.Context(), it)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]int{{1, 2, 3}}
	if !reflect.DeepEqual(batches, want) {
		t.Fatalf("single batch = %#v, want %#v", batches, want)
	}
}

func TestLargeUtilBatchIterateEdgeCases(t *testing.T) {
	if _, err := BatchIterate(-1, NewSliceIterator([]int{1})); err == nil {
		t.Fatal("expected error for negative size")
	}
	if _, err := BatchIterate[int](2, nil); err == nil {
		t.Fatal("expected error for nil source")
	}

	// Empty source yields no batches at all.
	empty, err := BatchIterate(2, NewSliceIterator([]int{}))
	if err != nil {
		t.Fatal(err)
	}
	batches, err := CollectIterator(t.Context(), empty)
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) != 0 {
		t.Fatalf("empty source produced batches: %#v", batches)
	}

	// Size larger than the source yields one partial batch.
	big, err := BatchIterate(10, NewSliceIterator([]int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	bigBatches, err := CollectIterator(t.Context(), big)
	if err != nil {
		t.Fatal(err)
	}
	if len(bigBatches) != 1 || !reflect.DeepEqual(bigBatches[0], []int{1, 2}) {
		t.Fatalf("oversized batch = %#v", bigBatches)
	}

	// Exact multiple yields no trailing partial batch.
	exact, err := BatchIterate(2, NewSliceIterator([]int{1, 2, 3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	exactBatches, err := CollectIterator(t.Context(), exact)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(exactBatches, [][]int{{1, 2}, {3, 4}}) {
		t.Fatalf("exact batches = %#v", exactBatches)
	}

	// A canceled context is observed by Next.
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	ctxIt, err := BatchIterate(2, NewSliceIterator([]int{1, 2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, err := ctxIt.Next(canceled); err == nil && ok {
		t.Fatal("expected canceled context error")
	}
	_ = ctxIt.Close()
}
