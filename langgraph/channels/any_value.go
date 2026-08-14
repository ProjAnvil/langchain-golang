package channels

// anyValue mirrors Python's `AnyValue` channel: it stores the most recent
// write and assumes that, if multiple values are received in one superstep,
// they are all equal. Unlike `lastValue`, any number of writes is allowed
// (the last one wins) and an empty step-boundary update clears a stored
// value.
type anyValue struct {
	value any
	set   bool
}

// NewAnyValue returns a `Channel` equivalent to Python's `AnyValue`.
func NewAnyValue() Channel {
	return &anyValue{}
}

func (c *anyValue) Update(values []any) (bool, error) {
	if len(values) == 0 {
		if !c.set {
			return false, nil
		}
		c.value, c.set = nil, false
		return true, nil
	}
	c.value, c.set = values[len(values)-1], true
	return true, nil
}

func (c *anyValue) Get() (any, error) {
	if !c.set {
		return nil, ErrEmptyChannel
	}
	return c.value, nil
}

func (c *anyValue) IsAvailable() bool {
	return c.set
}

func (c *anyValue) Checkpoint() (any, bool) {
	if !c.set {
		return nil, false
	}
	return c.value, true
}

func (c *anyValue) FromCheckpoint(value any) Channel {
	if value == nil {
		return &anyValue{}
	}
	return &anyValue{value: value, set: true}
}
