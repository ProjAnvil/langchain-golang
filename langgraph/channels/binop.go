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
	seenOverwrite := false
	for _, v := range values {
		if !c.set {
			// The first value seeds the channel; op is not applied and
			// Overwrite detection does not run (mirrors Python: values[0]
			// is consumed before the Overwrite-detection loop).
			c.value, c.set = v, true
			continue
		}
		if ow, ok := AsOverwrite(v); ok {
			if seenOverwrite {
				return false, &InvalidUpdateError{
					Channel: "BinaryOperator",
					Reason:  "can receive only one Overwrite value per super-step",
				}
			}
			// The Overwrite REPLACES the entire accumulated value,
			// bypassing the reducer. Mirrors Python's binop.update: after
			// an Overwrite, subsequent non-Overwrite values in the same
			// super-step are skipped (the `if not seen_overwrite` guard).
			c.value = ow
			seenOverwrite = true
			continue
		}
		if !seenOverwrite {
			next, err := c.op(c.value, v)
			if err != nil {
				return false, err
			}
			c.value = next
		}
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
