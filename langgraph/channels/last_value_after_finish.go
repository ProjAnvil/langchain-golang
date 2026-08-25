package channels

// lastValueAfterFinish mirrors Python's `LastValueAfterFinish` channel
// (channels/last_value.py:81-151): it buffers the last value received but
// only makes it available after Finish; once consumed (Consume), the value
// is cleared. Unlike LastValue, any number of writes per step is allowed
// and the last one wins.
type lastValueAfterFinish struct {
	value    any
	set      bool
	finished bool
}

// LastValueAfterFinishState is the Checkpoint payload of a
// LastValueAfterFinish channel: the buffered value plus whether Finish has
// been called on it (Python checkpoints the tuple `(value, finished)`).
type LastValueAfterFinishState struct {
	Value    any
	Finished bool
}

// NewLastValueAfterFinish returns a `Channel` equivalent to Python's
// `LastValueAfterFinish`. The Channel interface has no Finish hook, so the
// concrete type exposes Finish/Consume as additional methods, reached the
// same way the executor reaches Barrier.Consume — via an
// `interface{ Finish() bool }` / `interface{ Consume() bool }` assertion.
//
// Known divergence (shared with EphemeralValue, see ephemeral.go:17-23): the
// Go executor's edge-driven loop has no finish broadcast, so nothing calls
// Finish automatically at run end; the channel is a building block for
// consumers that drive Finish/Consume explicitly.
func NewLastValueAfterFinish() Channel {
	return &lastValueAfterFinish{}
}

func (c *lastValueAfterFinish) Update(values []any) (bool, error) {
	if len(values) == 0 {
		return false, nil
	}
	// Python: self.finished = False; self.value = values[-1]
	c.finished = false
	c.value, c.set = values[len(values)-1], true
	return true, nil
}

func (c *lastValueAfterFinish) Get() (any, error) {
	if !c.set || !c.finished {
		return nil, ErrEmptyChannel
	}
	return c.value, nil
}

func (c *lastValueAfterFinish) IsAvailable() bool {
	return c.set && c.finished
}

func (c *lastValueAfterFinish) Checkpoint() (any, bool) {
	if !c.set {
		return nil, false
	}
	return LastValueAfterFinishState{Value: c.value, Finished: c.finished}, true
}

func (c *lastValueAfterFinish) FromCheckpoint(value any) Channel {
	st, ok := value.(LastValueAfterFinishState)
	if !ok {
		// nil (omitted channel) or a foreign payload: start empty, mirroring
		// Python's `if checkpoint is not MISSING` guard.
		return &lastValueAfterFinish{}
	}
	return &lastValueAfterFinish{value: st.Value, set: true, finished: st.Finished}
}

// Finish marks the buffered value available, mirroring Python's finish():
// true only on the unfinished-with-value transition.
func (c *lastValueAfterFinish) Finish() bool {
	if !c.finished && c.set {
		c.finished = true
		return true
	}
	return false
}

// Consume clears a finished value, mirroring Python's consume(): true only
// when the channel was finished.
func (c *lastValueAfterFinish) Consume() bool {
	if c.finished {
		c.finished = false
		c.value, c.set = nil, false
		return true
	}
	return false
}
