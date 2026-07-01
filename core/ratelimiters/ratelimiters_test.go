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

func TestInMemoryRateLimiterContextCancel(t *testing.T) {
	limiter := NewInMemory(0.01, time.Millisecond, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ok, err := limiter.Acquire(ctx, true)
	if err == nil || ok {
		t.Fatalf("Acquire = %v, %v; want canceled", ok, err)
	}
}
