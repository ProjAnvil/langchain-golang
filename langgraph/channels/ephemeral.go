package channels

import "fmt"

// ephemeral mirrors Python's `EphemeralValue` channel: a value lives only
// until the next superstep, expiring on the empty step-boundary `Update`.
type ephemeral struct {
	guard bool
	value any
	set   bool
}

// NewEphemeral returns a `Channel` equivalent to Python's `EphemeralValue`.
// With guard=true, more than one write in a single superstep is an
// `*InvalidUpdateError`; with guard=false the last write wins.
//
// Known divergence from Python: Python also clears ephemeral channels via
// `finish()` when the graph run ends (`pregel/_algo.py`); this Go `Channel`
// interface has no Finish hook, so a value written in the FINAL superstep
// lingers in the run's last checkpoint instead of being cleared.
func NewEphemeral(guard bool) Channel {
	return &ephemeral{guard: guard}
}

func (c *ephemeral) Update(values []any) (bool, error) {
	if len(values) == 0 {
		if !c.set {
			return false, nil
		}
		c.value, c.set = nil, false
		return true, nil
	}
	if len(values) > 1 {
		if c.guard {
			return false, &InvalidUpdateError{
				Channel: "EphemeralValue",
				Reason:  fmt.Sprintf("expected at most 1 value per superstep, got %d", len(values)),
			}
		}
		values = values[len(values)-1:]
	}
	c.value, c.set = values[0], true
	return true, nil
}

func (c *ephemeral) Get() (any, error) {
	if !c.set {
		return nil, ErrEmptyChannel
	}
	return c.value, nil
}

func (c *ephemeral) IsAvailable() bool {
	return c.set
}

func (c *ephemeral) Checkpoint() (any, bool) {
	if !c.set {
		return nil, false
	}
	return c.value, true
}

func (c *ephemeral) FromCheckpoint(value any) Channel {
	if value == nil {
		return &ephemeral{guard: c.guard}
	}
	return &ephemeral{guard: c.guard, value: value, set: true}
}
