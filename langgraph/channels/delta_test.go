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

// Baseline 1: Sentinel-only checkpoint; state reconstructed by replay.
func TestDeltaSentinelOnlyCheckpointAndReplay(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1, 2}) // first update; forces snapshot on first Checkpoint

	// First checkpoint forces a snapshot (everSnapshotted == false).
	cp, ok := ch.Checkpoint()
	if !ok {
		t.Fatal("first Checkpoint() ok = false, want true (forced snapshot)")
	}
	ds, isSnap := asDeltaSnapshot(cp)
	if !isSnap {
		t.Fatalf("first Checkpoint() = %T, want deltaSnapshot", cp)
	}
	if !reflect.DeepEqual(ds, []int{1, 2}) {
		t.Fatalf("snapshot value = %v, want [1 2]", ds)
	}

	// Subsequent updates within the snapshot frequency window.
	update(t, ch, []int{3})
	update(t, ch, []int{4})
	if _, ok2 := ch.Checkpoint(); ok2 {
		t.Fatal("second Checkpoint() ok = true, want false (sentinel-only between snapshots)")
	}

	// Simulate replay: restore from sentinel (nil), then replay writes.
	restored := ch.FromCheckpoint(nil)
	if restored.IsAvailable() {
		t.Fatal("FromCheckpoint(nil) should start empty")
	}
	// Replay ancestor writes that occurred since the last snapshot.
	delta := restored.(*DeltaChannel)
	delta.ReplayWrites([]any{[]int{1, 2}, []int{3}, []int{4}})
	if got := get(t, delta); !reflect.DeepEqual(got, []int{1, 2, 3, 4}) {
		t.Fatalf("after replay Get() = %v, want [1 2 3 4]", got)
	}
}

// Baseline 2: snapshot_frequency triggers full snapshot blob every N updates.
func TestDeltaSnapshotFrequency(t *testing.T) {
	ch := newTestDelta(3)
	update(t, ch, []int{1})

	// First checkpoint: forced snapshot (everSnapshotted == false).
	cp1, ok1 := ch.Checkpoint()
	if !ok1 {
		t.Fatal("first Checkpoint() ok = false, want true (forced)")
	}
	if _, isSnap := asDeltaSnapshot(cp1); !isSnap {
		t.Fatal("first Checkpoint() is not a deltaSnapshot")
	}

	// 2nd update: still within window → sentinel.
	update(t, ch, []int{2})
	if _, ok := ch.Checkpoint(); ok {
		t.Fatal("Checkpoint() after 1 update since snapshot ok = true, want false (sentinel)")
	}

	// 3rd update since snapshot: triggers snapshot.
	update(t, ch, []int{3})
	update(t, ch, []int{4})
	cp3, ok3 := ch.Checkpoint()
	if !ok3 {
		t.Fatal("Checkpoint() after 3 updates ok = false, want true (snapshot_frequency hit)")
	}
	val, _ := asDeltaSnapshot(cp3)
	if !reflect.DeepEqual(val, []int{1, 2, 3, 4}) {
		t.Fatalf("snapshot value = %v, want [1 2 3 4]", val)
	}

	// Counter resets: next checkpoint after 1 update is sentinel again.
	update(t, ch, []int{4})
	if _, ok := ch.Checkpoint(); ok {
		t.Fatal("Checkpoint() after 1 update since reset ok = true, want false (sentinel)")
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

// Baseline 4: Fresh thread (no checkpoint) forces a snapshot on first write.
func TestDeltaFreshThreadForcedSnapshot(t *testing.T) {
	ch := newTestDelta(1000) // high frequency so it wouldn't normally fire
	update(t, ch, []int{42})

	cp, ok := ch.Checkpoint()
	if !ok {
		t.Fatal("first Checkpoint() on fresh channel ok = false, want true (forced snapshot)")
	}
	val, isSnap := asDeltaSnapshot(cp)
	if !isSnap {
		t.Fatalf("first Checkpoint() = %T, want deltaSnapshot", cp)
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

// Baseline 7: updateState metadata/counters survive delta channel
// reconstruction. Here we verify that FromCheckpoint preserves the snapshot
// cadence (everSnapshotted is carried forward) so a restored channel does not
// re-trigger a forced snapshot.
func TestDeltaFromCheckpointPreservesCadence(t *testing.T) {
	ch := newTestDelta(1000)
	update(t, ch, []int{1})
	cp, _ := ch.Checkpoint() // forced snapshot

	restored := ch.FromCheckpoint(cp)
	delta := restored.(*DeltaChannel)
	if !delta.everSnapshotted {
		t.Fatal("FromCheckpoint(snapshot) everSnapshotted = false, want true")
	}
	// One update is within window → sentinel, not forced snapshot.
	update(t, delta, []int{2})
	if _, ok := delta.Checkpoint(); ok {
		t.Fatal("restored channel Checkpoint() after 1 update ok = true, want false (no forced snapshot)")
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
	cp, ok := ch.Checkpoint()
	if !ok {
		t.Fatal("Checkpoint() ok = false")
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
