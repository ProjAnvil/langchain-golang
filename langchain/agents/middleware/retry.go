package middleware

import (
	"errors"
	"math/rand"
	"reflect"
	"time"
)

type RetryPredicate func(error) bool
type FailureFormatter func(error) string

// RetryOnErrorTypes builds a RetryPredicate that matches when an error (or any
// error in its unwrap chain, mirroring Python exception chaining) has the same
// dynamic type as one of the given target errors. This is the Go equivalent of
// Python's `retry_on: tuple[type[Exception], ...]` isinstance-based matching —
// pass zero-value instances of the desired error types, e.g.
// RetryOnErrorTypes(&os.PathError{}, io.EOF).
func RetryOnErrorTypes(targets ...error) RetryPredicate {
	targetTypes := make([]reflect.Type, 0, len(targets))
	for _, t := range targets {
		if t != nil {
			targetTypes = append(targetTypes, reflect.TypeOf(t))
		}
	}
	return func(err error) bool {
		for err != nil {
			errType := reflect.TypeOf(err)
			for _, t := range targetTypes {
				if errType == t {
					return true
				}
			}
			err = errors.Unwrap(err)
		}
		return false
	}
}

func validateRetryParams(maxRetries int, initialDelay, maxDelay time.Duration, backoffFactor float64) error {
	if maxRetries < 0 {
		return errors.New("max_retries must be >= 0")
	}
	if initialDelay < 0 {
		return errors.New("initial_delay must be >= 0")
	}
	if maxDelay < 0 {
		return errors.New("max_delay must be >= 0")
	}
	if backoffFactor < 0 {
		return errors.New("backoff_factor must be >= 0")
	}
	return nil
}

func calculateRetryDelay(retryNumber int, backoffFactor float64, initialDelay, maxDelay time.Duration, jitter bool) time.Duration {
	delay := initialDelay
	if backoffFactor != 0 {
		delay = time.Duration(float64(initialDelay) * pow(backoffFactor, retryNumber))
	}
	if delay > maxDelay {
		delay = maxDelay
	}
	if jitter && delay > 0 {
		jitterRange := float64(delay) * 0.25
		offset := (rand.Float64()*2 - 1) * jitterRange
		delay = time.Duration(float64(delay) + offset)
		if delay < 0 {
			return 0
		}
	}
	return delay
}

func pow(base float64, exp int) float64 {
	out := 1.0
	for i := 0; i < exp; i++ {
		out *= base
	}
	return out
}
