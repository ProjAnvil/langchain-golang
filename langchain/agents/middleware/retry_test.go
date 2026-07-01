package middleware

import (
	"fmt"
	"os"
	"testing"
)

type customRetryError struct{ msg string }

func (e *customRetryError) Error() string { return e.msg }

type otherRetryError struct{ msg string }

func (e *otherRetryError) Error() string { return e.msg }

func TestRetryOnErrorTypesMatchesExactType(t *testing.T) {
	predicate := RetryOnErrorTypes(&customRetryError{}, os.ErrNotExist)

	if !predicate(&customRetryError{msg: "boom"}) {
		t.Fatal("expected predicate to match *customRetryError")
	}
	if !predicate(os.ErrNotExist) {
		t.Fatal("expected predicate to match os.ErrNotExist")
	}
	if predicate(&otherRetryError{msg: "unrelated"}) {
		t.Fatal("expected predicate to reject unrelated error type")
	}
}

func TestRetryOnErrorTypesWalksUnwrapChain(t *testing.T) {
	predicate := RetryOnErrorTypes(&customRetryError{})
	wrapped := fmt.Errorf("context: %w", &customRetryError{msg: "inner"})

	if !predicate(wrapped) {
		t.Fatal("expected predicate to match wrapped error via unwrap chain")
	}
}

func TestRetryOnErrorTypesNilIsNoMatch(t *testing.T) {
	predicate := RetryOnErrorTypes(&customRetryError{})
	if predicate(nil) {
		t.Fatal("expected predicate to reject nil error")
	}
}
