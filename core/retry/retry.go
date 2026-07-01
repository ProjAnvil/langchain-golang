package retry

import (
	"context"
	"math"
	"time"
)

// Policy controls retry behavior.
type Policy struct {
	MaxAttempts       int
	Delay             time.Duration
	BackoffMultiplier float64
	MaxDelay          time.Duration
	ShouldRetry       func(error) bool
	Sleep             func(context.Context, time.Duration) error
}

// Do executes fn until it succeeds, the policy stops retrying, or the context is
// canceled.
func Do(ctx context.Context, policy Policy, fn func() error) error {
	maxAttempts := policy.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 1
	}

	var err error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		err = fn()
		if err == nil {
			return nil
		}
		if attempt == maxAttempts || policy.ShouldRetry == nil || !policy.ShouldRetry(err) {
			return err
		}
		if delay := policy.delay(attempt); delay > 0 {
			if err := policy.sleep(ctx, delay); err != nil {
				return err
			}
		}
	}
	return err
}

func (p Policy) delay(attempt int) time.Duration {
	if p.Delay <= 0 {
		return 0
	}
	delay := p.Delay
	if p.BackoffMultiplier > 1 && attempt > 1 {
		multiplier := math.Pow(p.BackoffMultiplier, float64(attempt-1))
		delay = time.Duration(float64(delay) * multiplier)
	}
	if p.MaxDelay > 0 && delay > p.MaxDelay {
		delay = p.MaxDelay
	}
	return delay
}

func (p Policy) sleep(ctx context.Context, delay time.Duration) error {
	if p.Sleep != nil {
		return p.Sleep(ctx, delay)
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
