package middleware

import (
	"errors"
	"math/rand"
	"reflect"
	"slices"
	"time"
)

type RetryPredicate func(error) bool
type FailureFormatter func(error) string

// modelRetryable matches any error that declares model-retry semantics
// (*lcerrors.ProviderError, *lcerrors.ModelRefusalError, ...), mirroring
// Python's `isinstance(exc, ModelError)` check in default_retry_on: the
// error's own is_retryable decides, so a refusal (is_retryable = False) is
// never retried while a 429 still is.
type modelRetryable interface{ IsModelRetryable() bool }

// DefaultRetryOn mirrors Python langchain 1.4's default_retry_on (#39538):
// classified model errors retry only when their kind is retryable (rate
// limit, server failure, connection, timeout); unclassified errors retry.
// It is the default RetryOn for ModelRetryMiddleware and ToolRetryMiddleware.
func DefaultRetryOn(err error) bool {
	var r modelRetryable
	if errors.As(err, &r) {
		return r.IsModelRetryable()
	}
	return true
}

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
			if slices.Contains(targetTypes, errType) {
				return true
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
	for range exp {
		out *= base
	}
	return out
}
