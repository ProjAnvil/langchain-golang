package channels

// binaryOperator mirrors Python's `BinaryOperatorAggregate` channel: it
// folds each superstep's writes into the current value with a `Reducer`
// (e.g. `AppendSliceReducer` or `MessagesReducer`).
type binaryOperator struct {
	op    Reducer
	value any
	set   bool
}

// NewBinaryOperator returns a `Channel` equivalent to Python's
// `BinaryOperatorAggregate`, reducing writes left-to-right with op.
func NewBinaryOperator(op Reducer) Channel {
	return &binaryOperator{op: op}
}

func (c *binaryOperator) Update(values []any) (bool, error) {
	if len(values) == 0 {
		return false, nil
	}
	for _, v := range values {
		if !c.set {
			// The first value seeds the channel; op is not applied.
			c.value, c.set = v, true
			continue
		}
		next, err := c.op(c.value, v)
		if err != nil {
			return false, err
		}
		c.value = next
	}
	return true, nil
}

func (c *binaryOperator) Get() (any, error) {
	if !c.set {
		return nil, ErrEmptyChannel
	}
	return c.value, nil
}

func (c *binaryOperator) IsAvailable() bool {
	return c.set
}

func (c *binaryOperator) Checkpoint() (any, bool) {
	if !c.set {
		return nil, false
	}
	return c.value, true
}

func (c *binaryOperator) FromCheckpoint(value any) Channel {
	if value == nil {
		return &binaryOperator{op: c.op}
	}
	return &binaryOperator{op: c.op, value: value, set: true}
}
