package channels

// untrackedValue mirrors Python's `UntrackedValue` channel: it stores the
// most recent write but is never persisted to a checkpoint. With guard=true,
// more than one write in a single superstep is an `*InvalidUpdateError`; with
// guard=false the last write wins.
type untrackedValue struct {
	value any
	set   bool
	guard bool
}

// NewUntrackedValue returns a `Channel` equivalent to Python's
// `UntrackedValue`. With guard=true, more than one write in a single
// superstep is an `*InvalidUpdateError`; with guard=false the last write wins.
func NewUntrackedValue(guard bool) Channel {
	return &untrackedValue{guard: guard}
}

func (c *untrackedValue) Update(values []any) (bool, error) {
	if len(values) == 0 {
		return false, nil
	}
	if len(values) != 1 && c.guard {
		return false, &InvalidUpdateError{
			Channel: "UntrackedValue",
			Reason:  "guard=true can receive only one value per step",
		}
	}
	c.value, c.set = values[len(values)-1], true
	return true, nil
}

func (c *untrackedValue) Get() (any, error) {
	if !c.set {
		return nil, ErrEmptyChannel
	}
	return c.value, nil
}

func (c *untrackedValue) IsAvailable() bool {
	return c.set
}

// Checkpoint always omits the channel: an untracked value is never persisted,
// even when a value has been written.
func (c *untrackedValue) Checkpoint() (any, bool) {
	return nil, false
}

// FromCheckpoint never restores a value: an untracked channel always resumes
// empty, preserving only its guard setting.
func (c *untrackedValue) FromCheckpoint(value any) Channel {
	return &untrackedValue{guard: c.guard}
}
