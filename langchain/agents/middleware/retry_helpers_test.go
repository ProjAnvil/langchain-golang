package middleware

import (
	"strings"
	"testing"
	"time"
)

func TestValidateRetryParams(t *testing.T) {
	cases := []struct {
		name          string
		maxRetries    int
		initialDelay  time.Duration
		maxDelay      time.Duration
		backoffFactor float64
		wantErr       string
	}{
		{"valid", 1, time.Second, time.Minute, 2, ""},
		{"negative retries", -1, 0, 0, 0, "max_retries"},
		{"negative initial delay", 0, -time.Second, 0, 0, "initial_delay"},
		{"negative max delay", 0, 0, -time.Second, 0, "max_delay"},
		{"negative backoff", 0, 0, 0, -1, "backoff_factor"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			err := validateRetryParams(tt.maxRetries, tt.initialDelay, tt.maxDelay, tt.backoffFactor)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected %q error, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestCalculateRetryDelay(t *testing.T) {
	// No backoff factor: constant initial delay.
	if got := calculateRetryDelay(3, 0, time.Second, time.Minute, false); got != time.Second {
		t.Fatalf("no-backoff delay mismatch: %v", got)
	}
	// Exponential backoff: initial * factor^retry.
	if got := calculateRetryDelay(2, 2, time.Second, time.Minute, false); got != 4*time.Second {
		t.Fatalf("backoff delay mismatch: %v", got)
	}
	// Capped at maxDelay.
	if got := calculateRetryDelay(10, 2, time.Second, 5*time.Second, false); got != 5*time.Second {
		t.Fatalf("max delay cap mismatch: %v", got)
	}
	// Jitter stays within ±25% of the base delay.
	for i := 0; i < 50; i++ {
		got := calculateRetryDelay(0, 0, 100*time.Millisecond, time.Minute, true)
		if got < 75*time.Millisecond || got > 125*time.Millisecond {
			t.Fatalf("jittered delay out of range: %v", got)
		}
	}
	// Zero delay with jitter stays non-negative.
	if got := calculateRetryDelay(0, 0, 0, 0, true); got != 0 {
		t.Fatalf("zero delay mismatch: %v", got)
	}
}

func TestPow(t *testing.T) {
	if got := pow(2, 0); got != 1 {
		t.Fatalf("pow(x, 0) should be 1, got %v", got)
	}
	if got := pow(3, 3); got != 27 {
		t.Fatalf("pow(3, 3) mismatch: %v", got)
	}
	if got := pow(2, -1); got != 1 {
		t.Fatalf("pow with negative exponent should be 1, got %v", got)
	}
}
