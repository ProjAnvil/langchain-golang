package channels

import "fmt"

// lastValue mirrors Python's `LastValue` channel: it holds the most recent
// write and errors when more than one value is written in a single
// superstep.
type lastValue struct {
	value any
	set   bool
}

// NewLastValue returns a `Channel` equivalent to Python's `LastValue`.
func NewLastValue() Channel {
	return &lastValue{}
}

func (c *lastValue) Update(values []any) (bool, error) {
	if len(values) == 0 {
		return false, nil
	}
	if len(values) > 1 {
		return false, &InvalidUpdateError{
			Channel: "LastValue",
			Reason:  fmt.Sprintf("expected at most 1 value per superstep, got %d", len(values)),
		}
	}
	c.value, c.set = values[0], true
	return true, nil
}

func (c *lastValue) Get() (any, error) {
	if !c.set {
		return nil, ErrEmptyChannel
	}
	return c.value, nil
}

func (c *lastValue) IsAvailable() bool {
	return c.set
}

func (c *lastValue) Checkpoint() (any, bool) {
	if !c.set {
		return nil, false
	}
	return c.value, true
}

func (c *lastValue) FromCheckpoint(value any) Channel {
	if value == nil {
		return &lastValue{}
	}
	return &lastValue{value: value, set: true}
}
