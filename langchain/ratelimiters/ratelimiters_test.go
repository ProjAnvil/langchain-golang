package ratelimiters

import (
	"context"
	"testing"
	"time"
)

func TestV1RateLimiterExportCore(t *testing.T) {
	limiter := NewInMemory(1000, time.Millisecond, 1)
	_, _ = limiter.Acquire(context.Background(), false)
	time.Sleep(2 * time.Millisecond)
	ok, err := limiter.Acquire(context.Background(), false)
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
}
