package standardtests

import (
	"reflect"
	"testing"
)

// requireNoErr fails the test if err is non-nil. Conformance runners route
// checks through these helpers so subtests stay branch-free and their failure
// branches can be covered by the failure-injection tests.
func requireNoErr(t *testing.T, op string, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: %v", op, err)
	}
}

// requireEqual fails the test unless got == want.
func requireEqual[T comparable](t *testing.T, what string, got, want T) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %v want %v", what, got, want)
	}
}

// requireLen fails the test unless len(got) == want.
func requireLen[T any](t *testing.T, what string, got []T, want int) {
	t.Helper()
	if len(got) != want {
		t.Fatalf("%s: got %d want %d", what, len(got), want)
	}
}

// requireDeepEqual fails the test unless reflect.DeepEqual(got, want).
func requireDeepEqual(t *testing.T, what string, got, want any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("%s: got %#v want %#v", what, got, want)
	}
}
