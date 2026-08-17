package ratelimiters

import (
	"context"
	"testing"
	"time"
)

func TestInMemoryRateLimiterNonBlockingStartsEmpty(t *testing.T) {
	limiter := NewInMemory(1000, time.Millisecond, 1)
	ok, err := limiter.Acquire(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("first non-blocking acquire should not burst")
	}
	time.Sleep(2 * time.Millisecond)
	ok, err = limiter.Acquire(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected token after refill")
	}
}

func TestNewInMemoryDefaults(t *testing.T) {
	limiter := NewInMemory(0, 0, 0)
	if limiter.RequestsPerSecond != 1 {
		t.Errorf("RequestsPerSecond = %v; want 1", limiter.RequestsPerSecond)
	}
	if limiter.CheckEvery != 100*time.Millisecond {
		t.Errorf("CheckEvery = %v; want 100ms", limiter.CheckEvery)
	}
	if limiter.MaxBucketSize != 1 {
		t.Errorf("MaxBucketSize = %v; want 1", limiter.MaxBucketSize)
	}

	limiter = NewInMemory(-5, -time.Second, 0.5)
	if limiter.RequestsPerSecond != 1 {
		t.Errorf("negative RequestsPerSecond should default to 1, got %v", limiter.RequestsPerSecond)
	}
	if limiter.CheckEvery != 100*time.Millisecond {
		t.Errorf("negative CheckEvery should default to 100ms, got %v", limiter.CheckEvery)
	}
	if limiter.MaxBucketSize != 1 {
		t.Errorf("MaxBucketSize < 1 should default to 1, got %v", limiter.MaxBucketSize)
	}

	limiter = NewInMemory(10, 5*time.Millisecond, 3)
	if limiter.RequestsPerSecond != 10 || limiter.CheckEvery != 5*time.Millisecond || limiter.MaxBucketSize != 3 {
		t.Errorf("valid arguments should be preserved, got %+v", limiter)
	}
}

func TestInMemoryRateLimiterNonBlockingCanceledContext(t *testing.T) {
	limiter := NewInMemory(1000, time.Millisecond, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ok, err := limiter.Acquire(ctx, false)
	if err == nil {
		t.Fatal("expected context error on non-blocking acquire")
	}
	if ok {
		t.Fatal("first non-blocking acquire should not burst")
	}
}

func TestInMemoryRateLimiterBlockingRefill(t *testing.T) {
	limiter := NewInMemory(50, time.Millisecond, 1)
	// First call initializes the bucket without a token.
	if ok, err := limiter.Acquire(context.Background(), false); err != nil || ok {
		t.Fatalf("Acquire = %v, %v; want false, nil", ok, err)
	}
	// Blocking acquire should wait for the ticker/refill and succeed.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	ok, err := limiter.Acquire(ctx, true)
	if err != nil || !ok {
		t.Fatalf("blocking Acquire = %v, %v; want true, nil", ok, err)
	}
}

func TestInMemoryRateLimiterBlockingCancelWhileWaiting(t *testing.T) {
	limiter := NewInMemory(0.01, 5*time.Millisecond, 1)
	// Consume the initial state so the blocking loop must wait.
	if ok, err := limiter.Acquire(context.Background(), false); err != nil || ok {
		t.Fatalf("Acquire = %v, %v; want false, nil", ok, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	ok, err := limiter.Acquire(ctx, true)
	if err == nil || ok {
		t.Fatalf("Acquire = %v, %v; want canceled", ok, err)
	}
}

func TestInMemoryRateLimiterBucketCap(t *testing.T) {
	limiter := NewInMemory(1000, time.Millisecond, 1)
	// Initialize the bucket.
	if ok, err := limiter.Acquire(context.Background(), false); err != nil || ok {
		t.Fatalf("Acquire = %v, %v; want false, nil", ok, err)
	}
	// Let enough time pass to accumulate far more tokens than the bucket holds.
	time.Sleep(10 * time.Millisecond)
	if ok, err := limiter.Acquire(context.Background(), false); err != nil || !ok {
		t.Fatalf("expected capped token, got %v, %v", ok, err)
	}
	// The bucket must have been capped at MaxBucketSize=1, so no second token.
	if ok, err := limiter.Acquire(context.Background(), false); err != nil || ok {
		t.Fatalf("bucket should be capped at MaxBucketSize, got %v, %v", ok, err)
	}
}

func TestInMemoryRateLimiterContextCancel(t *testing.T) {
	limiter := NewInMemory(0.01, time.Millisecond, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ok, err := limiter.Acquire(ctx, true)
	if err == nil || ok {
		t.Fatalf("Acquire = %v, %v; want canceled", ok, err)
	}
}
