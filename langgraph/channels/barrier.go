package channels

import (
	"fmt"
	"maps"
	"sort"
)

// Barrier mirrors Python's `NamedBarrierValue` channel
// (langgraph/channels/named_barrier_value.py): it accumulates a fixed set of
// string names — one per parent node of a waiting edge — and becomes
// available only once every name has arrived. The graph executor writes each
// parent's name when that parent commits, triggers the join child once the
// barrier is full, and calls Consume after the child commits so a looping
// graph re-arms the barrier for the next round.
//
// The barrier is control-plane state: the executor hides join channels from
// every user-visible state view (snapshot, node input, stream chunks), while
// Checkpoint/FromCheckpoint still persist partial arrivals so an interrupted
// run resumes with the arrival set intact.
type Barrier struct {
	names map[string]bool
	seen  map[string]bool
}

// NewBarrier returns a `Channel` equivalent to Python's
// `NamedBarrierValue(str, set(names))`. It returns the concrete *Barrier (not
// the Channel interface) so the executor can type-assert both the join-key
// marker and the optional Consume hook.
func NewBarrier(names ...string) *Barrier {
	b := &Barrier{names: make(map[string]bool, len(names)), seen: map[string]bool{}}
	for _, name := range names {
		b.names[name] = true
	}
	return b
}

// Names returns the barrier's expected arrival names, sorted.
func (c *Barrier) Names() []string {
	out := make([]string, 0, len(c.names))
	for name := range c.names {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Update records arrivals. Each value must be one of the barrier's names
// (anything else is an *InvalidUpdateError, mirroring Python); repeat
// arrivals are idempotent no-ops. An empty update (the applyWrites
// step-boundary notification) never expires the barrier.
func (c *Barrier) Update(values []any) (bool, error) {
	changed := false
	for _, v := range values {
		name, ok := v.(string)
		if !ok || !c.names[name] {
			return false, &InvalidUpdateError{
				Channel: "NamedBarrierValue",
				Reason:  fmt.Sprintf("value %v is not one of the barrier names %v", v, c.Names()),
			}
		}
		if !c.seen[name] {
			c.seen[name] = true
			changed = true
		}
	}
	return changed, nil
}

// Get returns nil once all names have arrived (Python's get() returns None —
// the value carries no information, only availability matters) and
// ErrEmptyChannel before that.
func (c *Barrier) Get() (any, error) {
	if !c.IsAvailable() {
		return nil, ErrEmptyChannel
	}
	return nil, nil
}

// IsAvailable reports whether every name has arrived.
func (c *Barrier) IsAvailable() bool {
	return len(c.seen) == len(c.names)
}

// Checkpoint persists the partial arrival set as a sorted []string (the serde
// registry's "[]string" entry round-trips it); an empty barrier is omitted
// from the checkpoint, matching the other channels' empty-omit contract.
func (c *Barrier) Checkpoint() (any, bool) {
	if len(c.seen) == 0 {
		return nil, false
	}
	out := make([]string, 0, len(c.seen))
	for name := range c.seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out, true
}

// FromCheckpoint returns a fresh barrier with the receiver's name set and the
// checkpointed arrivals. The receiver (a registered prototype) is never
// mutated. []any is accepted defensively for JSON-decoded checkpoints that
// lost the []string registry typing.
func (c *Barrier) FromCheckpoint(value any) Channel {
	b := &Barrier{names: maps.Clone(c.names), seen: map[string]bool{}}
	switch v := value.(type) {
	case nil:
	case []string:
		for _, name := range v {
			b.seen[name] = true
		}
	case []any:
		for _, item := range v {
			if name, ok := item.(string); ok {
				b.seen[name] = true
			}
		}
	}
	return b
}

// Consume resets a full barrier for the next loop round, mirroring Python's
// `NamedBarrierValue.consume`. It is deliberately NOT part of the Channel
// interface — the executor reaches it via an `interface{ Consume() bool }`
// assertion, the same shape as Python's optional `BaseChannel.consume`.
func (c *Barrier) Consume() bool {
	if !c.IsAvailable() {
		return false
	}
	c.seen = map[string]bool{}
	return true
}
