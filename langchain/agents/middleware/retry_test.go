package middleware

import (
	"fmt"
	"os"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
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

// DefaultRetryOn must consult IsModelRetryable on ANY error declaring it, not
// just *lcerrors.ProviderError — otherwise a ModelRefusalError (is_retryable
// = False) would be retried as an unclassified error (#41582).
func TestDefaultRetryOnHonorsIsModelRetryable(t *testing.T) {
	// Retryable classified provider error (429): still retried.
	if !DefaultRetryOn(lcerrors.NewProviderError("openai", "/responses", 429, "", 0)) {
		t.Fatal("ProviderError 429 must be retried")
	}
	// Non-retryable classified provider error (400): still not retried.
	if DefaultRetryOn(lcerrors.NewProviderError("openai", "/responses", 400, "bad request", 0)) {
		t.Fatal("ProviderError 400 must not be retried")
	}
	// ModelRefusalError declares IsModelRetryable() = false: never retried.
	if DefaultRetryOn(&lcerrors.ModelRefusalError{}) {
		t.Fatal("ModelRefusalError must not be retried")
	}
	// The interface check walks the unwrap chain (errors.As), so a wrapped
	// refusal is also exempt.
	if DefaultRetryOn(fmt.Errorf("agent step: %w", &lcerrors.ModelRefusalError{})) {
		t.Fatal("wrapped ModelRefusalError must not be retried")
	}
	// Unclassified errors keep retrying.
	if !DefaultRetryOn(fmt.Errorf("boom")) {
		t.Fatal("plain error must be retried")
	}
}
