package sqlite

import (
	"database/sql"
	"errors"
	"fmt"
	"testing"
)

type failingCleanup struct{ err error }

func (c failingCleanup) Close() error    { return c.err }
func (c failingCleanup) Rollback() error { return c.err }

func TestCloseWithErrorPreservesBothFailures(t *testing.T) {
	primary := errors.New("operation failed")
	closeErr := errors.New("close failed")
	for _, initial := range []error{nil, primary} {
		got := initial
		closeWithError(failingCleanup{err: closeErr}, &got)
		if !errors.Is(got, closeErr) || (initial != nil && !errors.Is(got, initial)) {
			t.Fatalf("cleanup error = %v, want both %v and %v", got, initial, closeErr)
		}
	}
	got := primary
	closeWithError(failingCleanup{}, &got)
	if got != primary {
		t.Fatalf("successful close changed operation error: %v", got)
	}
}

func TestRollbackWithErrorIgnoresFinishedTransactions(t *testing.T) {
	primary := errors.New("operation failed")
	rollbackErr := errors.New("rollback failed")
	for _, err := range []error{nil, sql.ErrTxDone, fmt.Errorf("wrapped: %w", sql.ErrTxDone)} {
		got := primary
		rollbackWithError(failingCleanup{err: err}, &got)
		if got != primary {
			t.Fatalf("expected rollback result %v changed operation error: %v", err, got)
		}
	}
	got := primary
	rollbackWithError(failingCleanup{err: rollbackErr}, &got)
	if !errors.Is(got, primary) || !errors.Is(got, rollbackErr) {
		t.Fatalf("rollback error = %v, want operation and rollback failures", got)
	}
}
