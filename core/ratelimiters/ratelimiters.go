package ratelimiters

import (
	"context"
	"sync"
	"time"
)

// RateLimiter gates requests.
type RateLimiter interface {
	Acquire(ctx context.Context, blocking bool) (bool, error)
}

// InMemoryRateLimiter is a thread-safe token bucket limiter.
type InMemoryRateLimiter struct {
	RequestsPerSecond float64
	CheckEvery        time.Duration
	MaxBucketSize     float64

	mu              sync.Mutex
	availableTokens float64
	last            time.Time
}

func NewInMemory(requestsPerSecond float64, checkEvery time.Duration, maxBucketSize float64) *InMemoryRateLimiter {
	if requestsPerSecond <= 0 {
		requestsPerSecond = 1
	}
	if checkEvery <= 0 {
		checkEvery = 100 * time.Millisecond
	}
	if maxBucketSize < 1 {
		maxBucketSize = 1
	}
	return &InMemoryRateLimiter{
		RequestsPerSecond: requestsPerSecond,
		CheckEvery:        checkEvery,
		MaxBucketSize:     maxBucketSize,
	}
}

func (l *InMemoryRateLimiter) Acquire(ctx context.Context, blocking bool) (bool, error) {
	if !blocking {
		return l.consume(), ctx.Err()
	}
	ticker := time.NewTicker(l.CheckEvery)
	defer ticker.Stop()
	for {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		if l.consume() {
			return true, nil
		}
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-ticker.C:
		}
	}
}

func (l *InMemoryRateLimiter) consume() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	if l.last.IsZero() {
		l.last = now
		return false
	}
	elapsed := now.Sub(l.last).Seconds()
	if elapsed*l.RequestsPerSecond >= 1 {
		l.availableTokens += elapsed * l.RequestsPerSecond
		l.last = now
	}
	if l.availableTokens > l.MaxBucketSize {
		l.availableTokens = l.MaxBucketSize
	}
	if l.availableTokens >= 1 {
		l.availableTokens--
		return true
	}
	return false
}
