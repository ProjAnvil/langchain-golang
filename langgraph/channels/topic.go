package channels

// topic mirrors Python's `Topic` channel: it accumulates the values written
// across a superstep (flattening `[]any` values one level), optionally
// retaining values from previous supersteps.
type topic struct {
	accumulate bool
	values     []any
}

// NewTopic returns a `Channel` equivalent to Python's `Topic`. With
// accumulate=false the channel clears its values at the start of every
// `Update` call (including empty step-boundary notifications), so a value is
// visible only within the superstep that produced it; with accumulate=true
// values persist across supersteps.
func NewTopic(accumulate bool) Channel {
	return &topic{accumulate: accumulate}
}

func (c *topic) Update(values []any) (bool, error) {
	changed := false
	if !c.accumulate {
		changed = len(c.values) > 0
		c.values = nil
	}
	for _, v := range values {
		if flat, ok := v.([]any); ok {
			c.values = append(c.values, flat...)
		} else {
			c.values = append(c.values, v)
		}
	}
	return changed || len(values) > 0, nil
}

func (c *topic) Get() (any, error) {
	if len(c.values) == 0 {
		return nil, ErrEmptyChannel
	}
	return c.values, nil
}

func (c *topic) IsAvailable() bool {
	return len(c.values) > 0
}

func (c *topic) Checkpoint() (any, bool) {
	if len(c.values) == 0 {
		return nil, false
	}
	return c.values, true
}

func (c *topic) FromCheckpoint(value any) Channel {
	if value == nil {
		return &topic{accumulate: c.accumulate}
	}
	values, _ := value.([]any)
	return &topic{accumulate: c.accumulate, values: values}
}
