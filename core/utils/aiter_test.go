package utils

import (
	"context"
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestLargeUtilAsyncTeeFanOut(t *testing.T) {
	values := make([]int, 20)
	for i := range values {
		values[i] = i
	}
	// A small buffer with concurrent consumers exercises backpressure.
	atee, err := NewAsyncTee(t.Context(), NewSliceIterator(values), 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	channels := atee.Channels()
	if len(channels) != 3 || atee.Len() != 3 {
		t.Fatalf("expected 3 channels, got %d", len(channels))
	}
	if _, err := atee.At(0); err != nil {
		t.Fatal(err)
	}
	if _, err := atee.At(3); err == nil {
		t.Fatal("expected out-of-range error")
	}

	results := make([][]int, len(channels))
	var wg sync.WaitGroup
	for i := range channels {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			for v := range channels[idx] {
				results[idx] = append(results[idx], v)
			}
		}(i)
	}
	wg.Wait()

	for i := range channels {
		if !reflect.DeepEqual(results[i], values) {
			t.Fatalf("channel %d diverged: got %d items, want %d identical items", i, len(results[i]), len(values))
		}
	}
	if err := atee.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLargeUtilAsyncTeeFromChannel(t *testing.T) {
	src := make(chan int, 8)
	for i := 1; i <= 5; i++ {
		src <- i
	}
	close(src)

	atee, err := NewAsyncTeeFromChannel(t.Context(), src, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	ch0, err := atee.At(0)
	if err != nil {
		t.Fatal(err)
	}
	ch1 := atee.Channels()[1]

	got0 := []int{}
	for v := range ch0 {
		got0 = append(got0, v)
	}
	got1 := []int{}
	for v := range ch1 {
		got1 = append(got1, v)
	}
	want := []int{1, 2, 3, 4, 5}
	if !reflect.DeepEqual(got0, want) || !reflect.DeepEqual(got1, want) {
		t.Fatalf("channels diverged: got0=%#v got1=%#v want=%#v", got0, got1, want)
	}
	if err := atee.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLargeUtilAsyncTeeCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	src := make(chan int) // never fed, never closed
	atee, err := NewAsyncTeeFromChannel(ctx, src, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	ch := atee.Channels()[0]

	cancel()
	select {
	case _, ok := <-ch:
		if ok {
			t.Fatal("output channel delivered a value after cancellation")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("output channel did not close after cancellation")
	}
	select {
	case <-atee.Done():
	default:
		t.Fatal("Done channel should be closed after cancellation")
	}
	if atee.Err() == nil {
		t.Fatal("expected a terminal error after cancellation")
	}
	if err := atee.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLargeUtilAsyncTeeClose(t *testing.T) {
	src := make(chan int) // blocking source
	atee, err := NewAsyncTeeFromChannel(t.Context(), src, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := atee.Close(); err != nil {
		t.Fatal(err)
	}
	for i, ch := range atee.Channels() {
		if _, ok := <-ch; ok {
			t.Fatalf("channel %d was not closed by Close", i)
		}
	}
	// Close is safe to call again.
	if err := atee.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLargeUtilAsyncTeeValidation(t *testing.T) {
	if _, err := NewAsyncTee[int](t.Context(), nil, 2); err == nil {
		t.Fatal("expected nil iterator error")
	}
	if _, err := NewAsyncTeeFromChannel[int](t.Context(), nil, 2); err == nil {
		t.Fatal("expected nil channel error")
	}
	if _, err := NewAsyncTee(context.Background(), NewSliceIterator([]int{1}), 0); err == nil {
		t.Fatal("expected non-positive consumer count error")
	}
	if _, err := NewAsyncTee(context.Background(), NewSliceIterator([]int{1}), 2, -3); err == nil {
		t.Fatal("expected negative buffer size error")
	}
}
