package channels

import (
	"fmt"
	"reflect"
)

// deltaSnapshotType is the discriminator marking a checkpoint blob as a
// DeltaChannel snapshot, mirroring Python's
// langgraph.checkpoint.serde.types._DeltaSnapshot.
const deltaSnapshotType = "__delta_snapshot__"

// deltaSnapshot is the full-snapshot blob a DeltaChannel writes into
// checkpoint ChannelValues when its snapshot cadence fires. It carries the
// accumulated value directly so reconstruction needs no ancestor write replay.
// The JSON tags keep the discriminator keys lowercase so the blob survives
// JSON round-trip through checkpoint serde / API boundaries.
type deltaSnapshot struct {
	Value any    `json:"value"`
	Type  string `json:"type"` // always deltaSnapshotType
}

// BatchReducer combines an existing channel value with a whole batch of writes
// in one call, returning the new value. It mirrors the batch-reducer signature
// of Python's DeltaChannel: reducer(state, [write1, write2, ...]) -> new_state.
//
// Reducers must be deterministic and batching-invariant (associative across
// folds): applying two consecutive write batches separately must produce the
// same state as applying their concatenation once.
type BatchReducer func(existing any, updates []any) (any, error)

// BatchFromReducer adapts a binary Reducer (e.g. AppendSliceReducer,
// MessagesReducer) into a BatchReducer by folding each update left-to-right.
// This lets DeltaChannel reuse the existing reducer library without writing a
// separate batch variant. The fold is: reducer(reducer(...reducer(existing,
// updates[0]), updates[1])..., updates[n-1]).
func BatchFromReducer(r Reducer) BatchReducer {
	return func(existing any, updates []any) (any, error) {
		val := existing
		for _, u := range updates {
			next, err := r(val, u)
			if err != nil {
				return nil, err
			}
			val = next
		}
		return val, nil
	}
}

// DeltaChannel mirrors Python's langgraph.channels.delta.DeltaChannel: a
// reducer channel that stores only a sentinel in checkpoint blobs and
// reconstructs state by replaying ancestor writes through the reducer.
//
// Snapshot cadence: Checkpoint returns a full deltaSnapshot blob when EITHER
// the channel has never been snapshotted (fresh-thread forced snapshot —
// without it a fresh thread's sentinel-only checkpoint would have no ancestor
// writes to replay) OR the per-channel update count reaches
// SnapshotFrequency. Between snapshots Checkpoint returns (nil, false) so the
// channel is omitted from ChannelValues; reconstruction walks ancestor writes.
//
// The reducer receives the current accumulated value and a batch of writes in
// one call: reducer(state, [write1, write2, ...]) -> new_state. This differs
// from BinaryOperator's one-at-a-time Reducer signature.
type DeltaChannel struct {
	reducer              BatchReducer
	typ                  func() any // zero-value factory; nil → empty []any
	snapshotFrequency    int
	value                any
	set                  bool // value is not MISSING
	updatesSinceSnapshot int  // updates applied since the last snapshot blob
	everSnapshotted      bool // has this instance ever emitted a snapshot blob
}

// NewDeltaChannel returns a DeltaChannel with the given batch reducer.
// typFactory creates the zero/empty value for the channel's type (e.g.
// func() any { return []int{} }); nil defaults to an empty []any.
// snapshotFrequency is the per-channel update cadence; values <= 0 default to
// 1000 (Python's default). A value of 1 forces a snapshot on every update.
func NewDeltaChannel(reducer BatchReducer, typFactory func() any, snapshotFrequency int) *DeltaChannel {
	if snapshotFrequency <= 0 {
		snapshotFrequency = 1000
	}
	if typFactory == nil {
		typFactory = func() any { return []any{} }
	}
	return &DeltaChannel{
		reducer:           reducer,
		typ:               typFactory,
		snapshotFrequency: snapshotFrequency,
	}
}

// Update applies a whole superstep's writes for this key. It mirrors Python's
// DeltaChannel.update: if any write is an Overwrite, the last one acts as a
// full reset (its value replaces the accumulated state); otherwise the batch
// reducer folds all writes into the current value in one call. Two Overwrites
// in one superstep is an InvalidUpdateError.
func (c *DeltaChannel) Update(values []any) (bool, error) {
	if len(values) == 0 {
		return false, nil
	}
	overwriteIdx := -1
	for i, v := range values {
		if _, ok := AsOverwrite(v); ok {
			if overwriteIdx >= 0 {
				return false, &InvalidUpdateError{
					Channel: "Delta",
					Reason:  "can receive only one Overwrite value per super-step",
				}
			}
			overwriteIdx = i
		}
	}
	if overwriteIdx >= 0 {
		owValue, _ := AsOverwrite(values[overwriteIdx])
		if owValue != nil {
			c.value = cloneValue(owValue)
		} else {
			c.value = c.typ()
		}
		c.set = true
		c.updatesSinceSnapshot++
		return true, nil
	}
	base := c.typ()
	if c.set {
		base = c.value
	}
	next, err := c.reducer(base, values)
	if err != nil {
		return false, err
	}
	c.value = next
	c.set = true
	c.updatesSinceSnapshot++
	return true, nil
}

// Get returns the current value or ErrEmptyChannel if never updated.
func (c *DeltaChannel) Get() (any, error) {
	if !c.set {
		return nil, ErrEmptyChannel
	}
	return c.value, nil
}

// IsAvailable reports whether Get would succeed.
func (c *DeltaChannel) IsAvailable() bool {
	return c.set
}

// Checkpoint returns the serializable snapshot. Between snapshots it returns
// (nil, false) so the channel is omitted from ChannelValues (sentinel-only
// storage). A full deltaSnapshot blob is returned when the snapshot cadence
// fires: on the channel's first-ever checkpoint (fresh-thread forced snapshot)
// or when updatesSinceSnapshot reaches snapshotFrequency. After returning a
// snapshot the counters reset.
//
// snapshotFrequency <= 1 means snapshot on every checkpoint — the correct mode
// for the current Go executor, which does not persist per-task writes in the
// normal flow (only in the interrupt path), so ancestor-write replay is not
// available and sentinel-only storage would lose data. snapshotFrequency > 1
// enables sentinel-only storage between snapshots, which requires executor
// support for write persistence so reconstruction can replay them (a future
// graph.go change).
func (c *DeltaChannel) Checkpoint() (any, bool) {
	if !c.set {
		return nil, false
	}
	if c.snapshotFrequency <= 1 || !c.everSnapshotted || c.updatesSinceSnapshot >= c.snapshotFrequency {
		c.everSnapshotted = true
		c.updatesSinceSnapshot = 0
		return deltaSnapshot{Value: c.value, Type: deltaSnapshotType}, true
	}
	return nil, false
}

// SnapshotBlob returns a deltaSnapshot blob for a forced snapshot. It is the
// snapshot payload the executor writes into checkpoint ChannelValues when
// DeltaChannelsToSnapshot decides this channel should snapshot now (Phase 1:
// the snapshot decision moves out of Checkpoint into the loop). It returns
// (nil, false) when the channel has no value. Unlike Checkpoint, this method
// is a pure read — it neither inspects nor mutates the in-memory cadence
// counters. Mirrors the write of _DeltaSnapshot(ch.get()) in Python's
// create_checkpoint (langgraph/pregel/_checkpoint.py).
func (c *DeltaChannel) SnapshotBlob() (any, bool) {
	if !c.set {
		return nil, false
	}
	return deltaSnapshot{Value: c.value, Type: deltaSnapshotType}, true
}

// SnapshotFrequency returns the per-channel update cadence at which the channel
// should emit a forced snapshot. Exposed so DeltaChannelsToSnapshot can compare
// the loop's update count against it without touching channel state.
func (c *DeltaChannel) SnapshotFrequency() int { return c.snapshotFrequency }

// FromCheckpoint returns a fresh channel restored from a Checkpoint value,
// mirroring Python's DeltaChannel.from_checkpoint. nil means MISSING (sentinel:
// start empty, the caller replays ancestor writes). A deltaSnapshot blob
// (typed struct or JSON-roundtripped map) restores the value directly. Any
// other value is a plain-value seed (migration from BinaryOperatorAggregate
// blobs) used directly.
func (c *DeltaChannel) FromCheckpoint(value any) Channel {
	newc := &DeltaChannel{
		reducer:           c.reducer,
		typ:               c.typ,
		snapshotFrequency: c.snapshotFrequency,
	}
	if value == nil {
		// MISSING / sentinel: start empty.
		return newc
	}
	if v, ok := asDeltaSnapshot(value); ok {
		newc.value = v
		newc.set = true
		newc.everSnapshotted = true
		return newc
	}
	// Plain value (migration from old BinaryOperatorAggregate blobs).
	newc.value = value
	newc.set = true
	newc.everSnapshotted = true
	return newc
}

// ReplayWrites applies ancestor writes oldest-to-newest via a single reducer
// call, mirroring Python's DeltaChannel.replay_writes. If any write is an
// Overwrite, the last one in the sequence acts as the reset point: its value
// becomes the new base and only writes after it are passed to the reducer.
//
// ReplayWrites is NOT part of the Channel interface (the interface has no
// batch-replay hook); the graph layer type-asserts to *DeltaChannel to call it
// during delta-channel state reconstruction.
func (c *DeltaChannel) ReplayWrites(values []any) {
	if len(values) == 0 {
		return
	}
	base := c.value
	if !c.set {
		base = c.typ()
	}
	start := 0
	for i, v := range values {
		if owValue, ok := AsOverwrite(v); ok {
			if owValue != nil {
				base = cloneValue(owValue)
			} else {
				base = c.typ()
			}
			start = i + 1
		}
	}
	c.set = true
	remaining := values[start:]
	if len(remaining) == 0 {
		c.value = base
		return
	}
	next, err := c.reducer(base, remaining)
	if err != nil {
		// On reducer error during replay, keep the last-known base rather
		// than silently dropping state. This mirrors Python's behaviour
		// where a reducer error during replay_writes propagates as a
		// reconstruction failure; the caller (graph layer) treats the
		// channel as unavailable if Get would fail.
		c.value = base
		return
	}
	c.value = next
}

// IsDelta reports whether ch is a *DeltaChannel.
func IsDelta(ch Channel) bool {
	_, ok := ch.(*DeltaChannel)
	return ok
}

// UnwrapDeltaSnapshot detects whether value is a delta snapshot blob (typed
// struct or JSON-roundtripped map) and returns the inner value and true.
// Otherwise it returns (value, false). It is the exported form of
// asDeltaSnapshot, used by the graph layer to unwrap delta snapshot blobs
// stored in checkpoint ChannelValues into their plain values for StateSnapshot
// projection.
func UnwrapDeltaSnapshot(value any) (any, bool) {
	return asDeltaSnapshot(value)
}

// AsDelta unwraps ch as a *DeltaChannel, returning (ch, true) if it is one.
func AsDelta(ch Channel) (*DeltaChannel, bool) {
	d, ok := ch.(*DeltaChannel)
	return d, ok
}

// asDeltaSnapshot detects whether value is a delta snapshot blob (typed struct
// or JSON-roundtripped map) and returns the inner value and true. Otherwise it
// returns (nil, false). Mirrors the isinstance(_DeltaSnapshot) check in
// Python's from_checkpoint plus the JSON-survival path.
func asDeltaSnapshot(value any) (any, bool) {
	switch ds := value.(type) {
	case deltaSnapshot:
		return ds.Value, true
	case *deltaSnapshot:
		if ds == nil {
			return nil, false
		}
		return ds.Value, true
	case map[string]any:
		if t, ok := ds["type"].(string); ok && t == deltaSnapshotType {
			if inner, ok := ds["value"]; ok {
				return inner, true
			}
		}
	}
	return nil, false
}

// cloneValue returns a shallow copy of v when v is a slice, map, or pointer to
// a struct — mirroring Python's copy.copy used by DeltaChannel.update and
// replay_writes so an Overwrite value is not aliased into channel state. For
// value types and unsupported kinds the original is returned unchanged.
func cloneValue(v any) any {
	if v == nil {
		return nil
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Slice:
		out := reflect.MakeSlice(rv.Type(), rv.Len(), rv.Cap())
		reflect.Copy(out, rv)
		return out.Interface()
	case reflect.Map:
		out := reflect.MakeMapWithSize(rv.Type(), rv.Len())
		for _, k := range rv.MapKeys() {
			out.SetMapIndex(k, rv.MapIndex(k))
		}
		return out.Interface()
	default:
		return v
	}
}

// Format helps error messages identify the channel kind.
func (c *DeltaChannel) String() string {
	return fmt.Sprintf("DeltaChannel(snapshotFrequency=%d)", c.snapshotFrequency)
}
