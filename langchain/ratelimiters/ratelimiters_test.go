package ratelimiters

import (
	"testing"
	"time"
)

func TestV1RateLimiterExportCore(t *testing.T) {
	limiter := NewInMemory(1000, time.Millisecond, 1)
	_, _ = limiter.Acquire(t.Context(), false)
	time.Sleep(2 * time.Millisecond)
	ok, err := limiter.Acquire(t.Context(), false)
	if err != nil || !ok {
		t.Fatalf("Acquire = %v, %v", ok, err)
	}
}
