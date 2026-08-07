package channels

import (
	"errors"
	"fmt"
)

// Channel is a stateful container for one graph-state key, the Go analog of
// Python's `BaseChannel`. The executor feeds it one superstep's writes at a
// time (`Update`), reads it via `Get`, and snapshots it via `Checkpoint`.
type Channel interface {
	// Update applies a whole superstep's writes for this key (possibly
	// empty — an empty slice is the step-boundary notification some
	// channels use to expire themselves). Reports whether the value changed.
	Update(values []any) (changed bool, err error)
	// Get returns the current value or ErrEmptyChannel if never updated.
	Get() (any, error)
	// IsAvailable reports whether Get would succeed.
	IsAvailable() bool
	// Checkpoint returns the serializable snapshot; ok=false means "omit
	// from the checkpoint" (channel empty).
	Checkpoint() (value any, ok bool)
	// FromCheckpoint returns a fresh channel restored from a Checkpoint
	// value (nil when the channel was omitted). It never mutates the receiver.
	FromCheckpoint(value any) Channel
}

// ErrEmptyChannel is returned by `Channel.Get` on a channel that has no
// value, mirroring Python's `EmptyChannelError`.
var ErrEmptyChannel = errors.New("channels: channel is empty")

// InvalidUpdateError mirrors Python's `InvalidUpdateError`: the writes for
// one superstep violate the channel's contract (e.g. more than one write to
// a `LastValue` channel).
type InvalidUpdateError struct {
	Channel string
	Reason  string
}

func (e *InvalidUpdateError) Error() string {
	return fmt.Sprintf("channels: invalid update for %s channel: %s", e.Channel, e.Reason)
}
