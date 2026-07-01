package retry

import (
	"context"
	"errors"
	"testing"
	"time"
)

var errTransient = errors.New("transient")
var errPermanent = errors.New("permanent")

func TestDoRetriesUntilSuccess(t *testing.T) {
	attempts := 0
	err := Do(context.Background(), Policy{
		MaxAttempts: 3,
		ShouldRetry: func(err error) bool {
			return errors.Is(err, errTransient)
		},
	}, func() error {
		attempts++
		if attempts < 3 {
			return errTransient
		}
		return nil
	})
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	if attempts != 3 {
		t.Fatalf("attempts: got %d want 3", attempts)
	}
}

func TestDoStopsOnPermanentError(t *testing.T) {
	attempts := 0
	err := Do(context.Background(), Policy{
		MaxAttempts: 3,
		ShouldRetry: func(err error) bool {
			return errors.Is(err, errTransient)
		},
	}, func() error {
		attempts++
		return errPermanent
	})
	if !errors.Is(err, errPermanent) {
		t.Fatalf("error: got %v want permanent", err)
	}
	if attempts != 1 {
		t.Fatalf("attempts: got %d want 1", attempts)
	}
}

func TestDoRespectsContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := Do(ctx, Policy{MaxAttempts: 3}, func() error {
		t.Fatal("function should not run")
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error: got %v want context canceled", err)
	}
}

func TestDoUsesExponentialBackoff(t *testing.T) {
	attempts := 0
	var delays []time.Duration
	err := Do(context.Background(), Policy{
		MaxAttempts:       4,
		Delay:             10 * time.Millisecond,
		BackoffMultiplier: 2,
		MaxDelay:          25 * time.Millisecond,
		ShouldRetry: func(err error) bool {
			return errors.Is(err, errTransient)
		},
		Sleep: func(_ context.Context, delay time.Duration) error {
			delays = append(delays, delay)
			return nil
		},
	}, func() error {
		attempts++
		if attempts < 4 {
			return errTransient
		}
		return nil
	})
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	want := []time.Duration{
		10 * time.Millisecond,
		20 * time.Millisecond,
		25 * time.Millisecond,
	}
	if len(delays) != len(want) {
		t.Fatalf("delays: got %v want %v", delays, want)
	}
	for i := range want {
		if delays[i] != want[i] {
			t.Fatalf("delay[%d]: got %v want %v", i, delays[i], want[i])
		}
	}
}

func TestDoStopsWhenSleepReturnsContextError(t *testing.T) {
	err := Do(context.Background(), Policy{
		MaxAttempts: 3,
		Delay:       time.Second,
		ShouldRetry: func(err error) bool {
			return errors.Is(err, errTransient)
		},
		Sleep: func(ctx context.Context, delay time.Duration) error {
			return context.Canceled
		},
	}, func() error {
		return errTransient
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error: got %v want context canceled", err)
	}
}
