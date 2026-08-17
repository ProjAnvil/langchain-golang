package channels

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

// deltaListReducer is a batch reducer that concatenates existing (a slice)
// with each update in the batch, mirroring operator.add for lists. It is the
// DeltaChannel analogue of AppendSliceReducer.
func deltaListReducer(existing any, updates []any) (any, error) {
	base, ok := existing.([]int)
	if !ok {
		return nil, &InvalidUpdateError{Channel: "Delta", Reason: "existing must be []int"}
	}
	out := make([]int, len(base))
	copy(out, base)
	for _, u := range updates {
		add, ok := u.([]int)
		if !ok {
			return nil, &InvalidUpdateError{Channel: "Delta", Reason: "update must be []int"}
		}
		out = append(out, add...)
	}
	return out, nil
}

func intSliceFactory() any { return []int{} }

func newTestDelta(freq int) *DeltaChannel {
	return NewDeltaChannel(deltaListReducer, intSliceFactory, freq)
}

func TestDeltaEmpty(t *testing.T) {
	ch := newTestDelta(1000)
	requireEmpty(t, ch)
}

// Baseline 1: Checkpoint is a pure sentinel; state is reconstructed by replay.
// DeltaChannel.Checkpoint always returns (nil, false) — the snapshot decision
// lives in the executor (create_checkpoint writes a SnapshotBlob). Here we
// verify the sentinel contract and that FromCheckpoint(sentinel)+ReplayWrites
// reconstructs the accumulated value.
func TestDeltaSentinelOnlyCheckpointAndReplay(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1, 2})

	// Checkpoint is always a pure sentinel, regardless of snapshotFrequency.
	if cp, ok := ch.Checkpoint(); ok {
		t.Fatalf("Checkpoint() = (%v, true), want (nil, false) (pure sentinel)", cp)
	}

	// SnapshotBlob is the executor's forced-snapshot primitive: it returns the
	// current value as a deltaSnapshot blob without touching any cadence state.
	blob, ok := ch.SnapshotBlob()
	if !ok {
		t.Fatal("SnapshotBlob() ok = false, want true (channel has a value)")
	}
	ds, isSnap := asDeltaSnapshot(blob)
	if !isSnap {
		t.Fatalf("SnapshotBlob() = %T, want deltaSnapshot", blob)
	}
	if !reflect.DeepEqual(ds, []int{1, 2}) {
		t.Fatalf("snapshot value = %v, want [1 2]", ds)
	}

	// Further updates keep Checkpoint a sentinel; SnapshotBlob tracks the value.
	update(t, ch, []int{3})
	update(t, ch, []int{4})
	if _, ok := ch.Checkpoint(); ok {
		t.Fatal("Checkpoint() after more updates ok = true, want false (pure sentinel)")
	}

	// Simulate replay: restore from sentinel (nil), then replay ancestor writes.
	restored := ch.FromCheckpoint(nil)
	if restored.IsAvailable() {
		t.Fatal("FromCheckpoint(nil) should start empty")
	}
	delta := restored.(*DeltaChannel)
	delta.ReplayWrites([]any{[]int{1, 2}, []int{3}, []int{4}})
	if got := get(t, delta); !reflect.DeepEqual(got, []int{1, 2, 3, 4}) {
		t.Fatalf("after replay Get() = %v, want [1 2 3 4]", got)
	}
}

// Baseline 2: SnapshotBlob always reflects the current accumulated value
// (snapshot cadence is decided externally by DeltaChannelsToSnapshot, not by
// the channel), and SnapshotFrequency reports the configured cadence. The
// per-channel update counter that used to drive Checkpoint no longer exists.
func TestDeltaSnapshotFrequency(t *testing.T) {
	ch := newTestDelta(3)
	if got := ch.SnapshotFrequency(); got != 3 {
		t.Fatalf("SnapshotFrequency() = %d, want 3", got)
	}
	update(t, ch, []int{1})

	// Checkpoint is a sentinel; SnapshotBlob carries the value.
	if _, ok := ch.Checkpoint(); ok {
		t.Fatal("Checkpoint() ok = true, want false (pure sentinel)")
	}
	cp1, ok1 := ch.SnapshotBlob()
	if !ok1 {
		t.Fatal("SnapshotBlob() ok = false, want true")
	}
	if _, isSnap := asDeltaSnapshot(cp1); !isSnap {
		t.Fatal("SnapshotBlob() is not a deltaSnapshot")
	}

	// More updates: still sentinel via Checkpoint, value tracked via SnapshotBlob.
	update(t, ch, []int{2})
	update(t, ch, []int{3})
	update(t, ch, []int{4})
	if _, ok := ch.Checkpoint(); ok {
		t.Fatal("Checkpoint() ok = true, want false (pure sentinel, cadence is external)")
	}
	blob, _ := ch.SnapshotBlob()
	val, _ := asDeltaSnapshot(blob)
	if !reflect.DeepEqual(val, []int{1, 2, 3, 4}) {
		t.Fatalf("snapshot value = %v, want [1 2 3 4]", val)
	}
}

// Baseline 3: Overwrite in a DeltaChannel; two Overwrites → InvalidUpdateError.
func TestDeltaOverwrite(t *testing.T) {
	t.Run("overwrite resets base", func(t *testing.T) {
		ch := newTestDelta(1000)
		update(t, ch, []int{1, 2})
		update(t, ch, NewOverwrite([]int{10}))
		if got := get(t, ch); !reflect.DeepEqual(got, []int{10}) {
			t.Fatalf("after Overwrite Get() = %v, want [10]", got)
		}
		// Later writes replay on top.
		update(t, ch, []int{20})
		if got := get(t, ch); !reflect.DeepEqual(got, []int{10, 20}) {
			t.Fatalf("after Overwrite+write Get() = %v, want [10 20]", got)
		}
	})

	t.Run("double overwrite errors", func(t *testing.T) {
		ch := newTestDelta(1000)
		update(t, ch, []int{1})
		_, err := ch.Update([]any{
			NewOverwrite([]int{10}),
			NewOverwrite([]int{20}),
		})
		var iu *InvalidUpdateError
		if !errors.As(err, &iu) {
			t.Fatalf("Update(two Overwrites) error = %v, want *InvalidUpdateError", err)
		}
	})
}

// Baseline 4: a fresh channel's first write is observable via SnapshotBlob —
// the primitive the executor's create_checkpoint uses for the fresh-thread
// forced snapshot. Checkpoint itself stays a pure sentinel.
func TestDeltaFreshThreadForcedSnapshot(t *testing.T) {
	ch := newTestDelta(1000) // high frequency so cadence would not fire
	update(t, ch, []int{42})

	if _, ok := ch.Checkpoint(); ok {
		t.Fatal("Checkpoint() ok = true, want false (pure sentinel even on first write)")
	}
	cp, ok := ch.SnapshotBlob()
	if !ok {
		t.Fatal("SnapshotBlob() ok = false, want true (fresh-thread forced snapshot value)")
	}
	val, isSnap := asDeltaSnapshot(cp)
	if !isSnap {
		t.Fatalf("SnapshotBlob() = %T, want deltaSnapshot", cp)
	}
	if !reflect.DeepEqual(val, []int{42}) {
		t.Fatalf("forced snapshot value = %v, want [42]", val)
	}
}

// Baseline 5: Overwrite survives JSON roundtrip.
func TestDeltaOverwriteJSONRoundtrip(t *testing.T) {
	original := NewOverwrite([]int{7})
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("json.Marshal(Overwrite) error = %v", err)
	}
	var generic any
	if err := json.Unmarshal(data, &generic); err != nil {
		t.Fatalf("json.Unmarshal error = %v", err)
	}

	ch := newTestDelta(1000)
	update(t, ch, []int{1, 2})
	if _, err := ch.Update([]any{generic}); err != nil {
		t.Fatalf("Update(json-roundtripped Overwrite) error = %v", err)
	}
	// JSON numbers deserialize as float64, so the value arrives as []any{7.0}.
	got := get(t, ch)
	slice, ok := got.([]any)
	if !ok || len(slice) != 1 {
		t.Fatalf("after round-tripped Overwrite Get() = %v, want 1-element slice", got)
	}
	if num, _ := slice[0].(float64); num != 7 {
		t.Fatalf("value[0] = %v, want 7", slice[0])
	}
}

// Baseline 7: FromCheckpoint restores the accumulated value and preserves the
// configured snapshot cadence (SnapshotFrequency). The per-channel cadence
// counter that used to live on the instance (everSnapshotted) is gone —
// cadence is tracked externally by the executor's per-channel counters — so a
// restored channel simply carries its value and frequency, and Checkpoint
// stays a pure sentinel.
func TestDeltaFromCheckpointPreservesCadence(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1})
	cp, _ := ch.SnapshotBlob() // executor-style forced snapshot blob

	restored := ch.FromCheckpoint(cp)
	delta := restored.(*DeltaChannel)
	if !delta.IsAvailable() {
		t.Fatal("FromCheckpoint(snapshot) should be available")
	}
	if got := delta.SnapshotFrequency(); got != 1000 {
		t.Fatalf("restored SnapshotFrequency() = %d, want 1000", got)
	}
	// The restored value is present; Checkpoint is still a pure sentinel.
	if got := get(t, delta); !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("restored Get() = %v, want [1]", got)
	}
	if _, ok := delta.Checkpoint(); ok {
		t.Fatal("restored Checkpoint() ok = true, want false (pure sentinel)")
	}
}

// Baseline 8 (#8526/#8535): plain-value seed delta-history traversal. When
// the channel is seeded with a plain (non-Overwrite, non-_DeltaSnapshot)
// value, from_checkpoint uses it directly and replay works on top.
func TestDeltaPlainValueSeed(t *testing.T) {
	ch := newTestDelta(1000)
	// Simulate migration from old BinaryOperatorAggregate blobs: a plain
	// value arrives as the checkpoint blob.
	restored := ch.FromCheckpoint([]int{100, 200})
	if !restored.IsAvailable() {
		t.Fatal("FromCheckpoint(plain value) should be available")
	}
	if got := get(t, restored); !reflect.DeepEqual(got, []int{100, 200}) {
		t.Fatalf("FromCheckpoint(plain value) Get() = %v, want [100 200]", got)
	}
	// Replay writes on top of the plain-value seed.
	delta := restored.(*DeltaChannel)
	delta.ReplayWrites([]any{[]int{300}})
	if got := get(t, delta); !reflect.DeepEqual(got, []int{100, 200, 300}) {
		t.Fatalf("after replay on plain seed Get() = %v, want [100 200 300]", got)
	}
}

// Baseline 8b: delta snapshot blob survives JSON roundtrip and is recognized
// by from_checkpoint.
func TestDeltaSnapshotBlobJSONRoundtrip(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1, 2})
	cp, ok := ch.SnapshotBlob()
	if !ok {
		t.Fatal("SnapshotBlob() ok = false")
	}
	ds := cp.(deltaSnapshot)

	// Serialize the snapshot blob through JSON (as a real backend would).
	data, err := json.Marshal(ds)
	if err != nil {
		t.Fatalf("json.Marshal(deltaSnapshot) error = %v", err)
	}
	var generic any
	if err := json.Unmarshal(data, &generic); err != nil {
		t.Fatalf("json.Unmarshal error = %v", err)
	}

	// from_checkpoint must recognize the JSON-roundtripped blob.
	restored := ch.FromCheckpoint(generic)
	if got := get(t, restored); !reflect.DeepEqual(got, []any{float64(1), float64(2)}) {
		t.Fatalf("after JSON roundtrip Get() = %v, want [1 2]", got)
	}
}

// Baseline 6 (exit-mode task_id alignment): verify ReplayWrites applies
// writes in the correct order (oldest-to-newest). This mirrors the
// observable contract: writes collected in reverse during ancestor walk are
// re-reversed before replay.
func TestDeltaReplayWritesOrder(t *testing.T) {
	ch := newTestDelta(1000)
	restored := ch.FromCheckpoint(nil)
	delta := restored.(*DeltaChannel)
	// Writes arrive oldest-to-newest; the reducer must see them in order.
	delta.ReplayWrites([]any{[]int{1}, []int{2}, []int{3}})
	if got := get(t, delta); !reflect.DeepEqual(got, []int{1, 2, 3}) {
		t.Fatalf("replay order Get() = %v, want [1 2 3]", got)
	}
}

// Baseline 6b: ReplayWrites with Overwrite — last Overwrite resets base,
// subsequent writes replay on top.
func TestDeltaReplayWritesWithOverwrite(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1})
	restored := ch.FromCheckpoint(nil)
	delta := restored.(*DeltaChannel)
	delta.ReplayWrites([]any{
		[]int{1},
		NewOverwrite([]int{10}),
		[]int{20},
	})
	if got := get(t, delta); !reflect.DeepEqual(got, []int{10, 20}) {
		t.Fatalf("replay with overwrite Get() = %v, want [10 20]", got)
	}
}

func TestDeltaReplayWritesEmpty(t *testing.T) {
	ch := newTestDelta(1000)
	restored := ch.FromCheckpoint(nil)
	delta := restored.(*DeltaChannel)
	delta.ReplayWrites(nil) // should be a no-op
	if delta.IsAvailable() {
		t.Fatal("ReplayWrites(nil) should leave channel unavailable")
	}
}

func TestDeltaGetEmptyErrors(t *testing.T) {
	ch := newTestDelta(1000)
	if _, err := ch.Get(); !errors.Is(err, ErrEmptyChannel) {
		t.Fatalf("Get() error = %v, want ErrEmptyChannel", err)
	}
}

func TestDeltaEmptyUpdateIsNoop(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1})
	if update(t, ch) {
		t.Fatal("Update([]) changed = true, want false")
	}
	if got := get(t, ch); !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("after empty Update Get() = %v, want [1]", got)
	}
}

func TestDeltaReducerErrorPropagates(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1})
	if _, err := ch.Update([]any{"not-an-int-slice"}); err == nil {
		t.Fatal("Update(type-mismatch) error = nil, want error")
	}
}

func TestDeltaFromCheckpointNilEmpty(t *testing.T) {
	ch := newTestDelta(1000)
	restored := ch.FromCheckpoint(nil)
	requireEmpty(t, restored)
}

func TestNewDeltaChannelDefaultFrequency(t *testing.T) {
	ch := NewDeltaChannel(deltaListReducer, intSliceFactory, 0)
	if ch.snapshotFrequency != 1000 {
		t.Fatalf("snapshotFrequency = %d, want 1000 (default)", ch.snapshotFrequency)
	}
	ch2 := NewDeltaChannel(deltaListReducer, intSliceFactory, -5)
	if ch2.snapshotFrequency != 1000 {
		t.Fatalf("snapshotFrequency = %d, want 1000 (default for negative)", ch2.snapshotFrequency)
	}
}

func TestIsDeltaAndAsDelta(t *testing.T) {
	ch := newTestDelta(1000)
	if !IsDelta(ch) {
		t.Fatal("IsDelta(*DeltaChannel) = false, want true")
	}
	if _, ok := AsDelta(ch); !ok {
		t.Fatal("AsDelta(*DeltaChannel) ok = false, want true")
	}
	lv := NewLastValue()
	if IsDelta(lv) {
		t.Fatal("IsDelta(LastValue) = true, want false")
	}
	if _, ok := AsDelta(lv); ok {
		t.Fatal("AsDelta(LastValue) ok = true, want false")
	}
}

func TestDeltaBatchReducerBatchingInvariant(t *testing.T) {
	// Two batches applied separately must equal their concatenation applied
	// once (the batching-invariant contract documented on BatchReducer).
	ch1 := newTestDelta(1000)
	update(t, ch1, []int{1})
	update(t, ch1, []int{2}, []int{3})

	ch2 := newTestDelta(1000)
	update(t, ch2, []int{1})
	update(t, ch2, []int{2, 3})

	if got1, got2 := get(t, ch1), get(t, ch2); !reflect.DeepEqual(got1, got2) {
		t.Fatalf("batching invariant violated: separate=%v vs concatenated=%v", got1, got2)
	}
}

// BatchFromReducer adapts a binary Reducer into a BatchReducer by folding
// left-to-right; reducer errors propagate.
func TestBatchFromReducer(t *testing.T) {
	batch := BatchFromReducer(AppendSliceReducer)

	got, err := batch([]int{1}, []any{[]int{2}, []int{3, 4}})
	if err != nil {
		t.Fatalf("BatchFromReducer fold error = %v", err)
	}
	if !reflect.DeepEqual(got, []int{1, 2, 3, 4}) {
		t.Fatalf("BatchFromReducer fold = %v, want [1 2 3 4]", got)
	}

	// nil existing is treated as the zero value by the underlying reducer.
	got, err = batch(nil, []any{[]int{7}})
	if err != nil {
		t.Fatalf("BatchFromReducer(nil existing) error = %v", err)
	}
	if !reflect.DeepEqual(got, []int{7}) {
		t.Fatalf("BatchFromReducer(nil existing) = %v, want [7]", got)
	}

	// Empty batch returns the existing value unchanged.
	got, err = batch([]int{9}, nil)
	if err != nil {
		t.Fatalf("BatchFromReducer(empty batch) error = %v", err)
	}
	if !reflect.DeepEqual(got, []int{9}) {
		t.Fatalf("BatchFromReducer(empty batch) = %v, want [9]", got)
	}

	// A reducer error mid-fold propagates.
	if _, err := batch([]int{1}, []any{[]int{2}, "not-a-slice"}); err == nil {
		t.Fatal("BatchFromReducer(type mismatch) error = nil, want error")
	}
}

// NewDeltaChannel with a nil typFactory defaults the zero value to []any{}.
func TestNewDeltaChannelNilFactory(t *testing.T) {
	ch := NewDeltaChannel(BatchFromReducer(AppendSliceReducer), nil, 1)
	if update(t, ch, []any{"a"}) != true {
		t.Fatal("Update() changed = false, want true")
	}
	if got := get(t, ch); !reflect.DeepEqual(got, []any{"a"}) {
		t.Fatalf("Get() = %v, want [a]", got)
	}

	// An Overwrite(nil) resets to the default zero value []any{}.
	if _, err := ch.Update([]any{NewOverwrite(nil)}); err != nil {
		t.Fatalf("Update(Overwrite(nil)) error = %v", err)
	}
	if got := get(t, ch); !reflect.DeepEqual(got, []any{}) {
		t.Fatalf("after Overwrite(nil) Get() = %v, want []", got)
	}
}

// An Overwrite carrying a nil value resets the channel to its zero value.
func TestDeltaOverwriteNilValueResetsToZero(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1, 2})
	if _, err := ch.Update([]any{NewOverwrite(nil)}); err != nil {
		t.Fatalf("Update(Overwrite(nil)) error = %v", err)
	}
	if got := get(t, ch); !reflect.DeepEqual(got, []int{}) {
		t.Fatalf("after Overwrite(nil) Get() = %v, want []", got)
	}
}

// ReplayWrites: a nil-valued Overwrite resets the base to the zero value and
// only writes after it are replayed.
func TestDeltaReplayWritesNilOverwrite(t *testing.T) {
	ch := newTestDelta(1000)
	restored := ch.FromCheckpoint(nil)
	delta := restored.(*DeltaChannel)
	delta.ReplayWrites([]any{
		[]int{1},
		NewOverwrite(nil),
		[]int{5},
	})
	if got := get(t, delta); !reflect.DeepEqual(got, []int{5}) {
		t.Fatalf("replay with nil Overwrite Get() = %v, want [5]", got)
	}
}

// ReplayWrites: on a reducer error the channel keeps the last-known base
// rather than dropping state.
func TestDeltaReplayWritesReducerError(t *testing.T) {
	ch := newTestDelta(1000)
	restored := ch.FromCheckpoint(nil)
	delta := restored.(*DeltaChannel)
	delta.ReplayWrites([]any{[]int{1}, "not-an-int-slice"})
	// The failing write is dropped; the base accumulated so far is kept.
	if got := get(t, delta); !reflect.DeepEqual(got, []int{}) {
		t.Fatalf("replay with reducer error Get() = %v, want [] (last-known base)", got)
	}
	if !delta.IsAvailable() {
		t.Fatal("after replay with reducer error IsAvailable() = false, want true")
	}
}

// UnwrapDeltaSnapshot recognizes typed struct, pointer, and JSON-map forms
// and passes non-snapshot values through with ok=false.
func TestUnwrapDeltaSnapshot(t *testing.T) {
	inner := []int{1, 2}

	if got, ok := UnwrapDeltaSnapshot(deltaSnapshot{Value: inner, Type: deltaSnapshotType}); !ok || !reflect.DeepEqual(got, inner) {
		t.Fatalf("UnwrapDeltaSnapshot(struct) = (%v, %v), want ([1 2], true)", got, ok)
	}
	if got, ok := UnwrapDeltaSnapshot(&deltaSnapshot{Value: inner, Type: deltaSnapshotType}); !ok || !reflect.DeepEqual(got, inner) {
		t.Fatalf("UnwrapDeltaSnapshot(pointer) = (%v, %v), want ([1 2], true)", got, ok)
	}
	m := map[string]any{"type": deltaSnapshotType, "value": inner}
	if got, ok := UnwrapDeltaSnapshot(m); !ok || !reflect.DeepEqual(got, inner) {
		t.Fatalf("UnwrapDeltaSnapshot(map) = (%v, %v), want ([1 2], true)", got, ok)
	}
	if _, ok := UnwrapDeltaSnapshot("plain"); ok {
		t.Fatal("UnwrapDeltaSnapshot(plain) ok = true, want false")
	}
}

// asDeltaSnapshot rejects the forms that only look similar to a snapshot blob.
func TestAsDeltaSnapshotRejectsLookalikes(t *testing.T) {
	cases := []any{
		(*deltaSnapshot)(nil),                       // nil pointer
		map[string]any{"type": deltaSnapshotType},   // missing "value" key
		map[string]any{"type": "other", "value": 1}, // wrong discriminator
		map[string]any{"value": 1},                  // missing discriminator
		42,                                          // not a snapshot at all
		nil,
	}
	for i, v := range cases {
		if _, ok := asDeltaSnapshot(v); ok {
			t.Fatalf("case %d: asDeltaSnapshot(%v) ok = true, want false", i, v)
		}
	}
}

// cloneValue shallow-copies slices and maps so Overwrite values are not
// aliased into channel state; other kinds pass through unchanged.
func TestCloneValue(t *testing.T) {
	if got := cloneValue(nil); got != nil {
		t.Fatalf("cloneValue(nil) = %v, want nil", got)
	}

	// Map: mutating the clone must not affect the original.
	src := map[string]int{"a": 1}
	cloned := cloneValue(src).(map[string]int)
	cloned["a"] = 99
	if src["a"] != 1 {
		t.Fatalf("cloneValue(map) aliases original: src = %v", src)
	}

	// Slice: mutating the clone must not affect the original.
	srcSlice := []int{1, 2}
	clonedSlice := cloneValue(srcSlice).([]int)
	clonedSlice[0] = 99
	if srcSlice[0] != 1 {
		t.Fatalf("cloneValue(slice) aliases original: src = %v", srcSlice)
	}

	// Value types are returned unchanged.
	if got := cloneValue(42); got != 42 {
		t.Fatalf("cloneValue(42) = %v, want 42", got)
	}
	if got := cloneValue("s"); got != "s" {
		t.Fatalf("cloneValue(%q) = %v, want %q", "s", got, "s")
	}
}

// Overwrite values are cloned, not aliased, into channel state.
func TestDeltaOverwriteDoesNotAliasValue(t *testing.T) {
	ch := newTestDelta(1000)
	ow := []int{1, 2}
	update(t, ch, NewOverwrite(ow))
	ow[0] = 99 // mutate the caller's slice after the write
	if got := get(t, ch); !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("Get() = %v, want [1 2] (Overwrite value must be cloned)", got)
	}
}

func TestDeltaString(t *testing.T) {
	ch := newTestDelta(7)
	got := ch.String()
	want := "DeltaChannel(snapshotFrequency=7)"
	if got != want {
		t.Fatalf("String() = %q, want %q", got, want)
	}
}

// ReplayWrites ending on an Overwrite: the overwrite value is the final state,
// with no trailing writes to fold.
func TestDeltaReplayWritesEndsWithOverwrite(t *testing.T) {
	ch := newTestDelta(1000)
	restored := ch.FromCheckpoint(nil)
	delta := restored.(*DeltaChannel)
	delta.ReplayWrites([]any{
		[]int{1},
		NewOverwrite([]int{9}),
	})
	if got := get(t, delta); !reflect.DeepEqual(got, []int{9}) {
		t.Fatalf("replay ending with Overwrite Get() = %v, want [9]", got)
	}
}
