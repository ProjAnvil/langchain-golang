package channels

import (
	"testing"
)

// batchAppendReducer is a BatchReducer that concatenates existing (a []any)
// with each update in the batch. It is the direct-lambda form mandated by C4:
// NewDeltaChannel takes a BatchReducer directly, NOT BatchFromReducer(Reducer).
func batchAppendReducer(existing any, updates []any) (any, error) {
	return append(existing.([]any), updates...), nil
}

func TestDeltaChannelsToSnapshot(t *testing.T) {
	ch := NewDeltaChannel(
		batchAppendReducer,
		func() any { return []any{} },
		3, // snapshotFrequency=3
	)
	// Mark as having a value so IsAvailable() is true.
	ch.Update([]any{"x"})

	channels := map[string]Channel{"msg": ch}

	tests := []struct {
		name     string
		counters map[string][2]int
		want     map[string]bool
	}{
		{
			name:     "below frequency — no snapshot",
			counters: map[string][2]int{"msg": {2, 1}}, // updates=2 < 3
			want:     map[string]bool{},
		},
		{
			name:     "at frequency — snapshot",
			counters: map[string][2]int{"msg": {3, 1}}, // updates=3 >= 3
			want:     map[string]bool{"msg": true},
		},
		{
			name:     "at max supersteps — snapshot even without writes",
			counters: map[string][2]int{"msg": {0, 5000}}, // supersteps=5000 >= 5000
			want:     map[string]bool{"msg": true},
		},
		{
			name:     "empty counters — no snapshot",
			counters: nil,
			want:     map[string]bool{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DeltaChannelsToSnapshot(channels, tt.counters)
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for k := range tt.want {
				if !got[k] {
					t.Errorf("expected %q in result, got %v", k, got)
				}
			}
		})
	}
}

func TestDeltaChannelsToSnapshotSkipsUnavailable(t *testing.T) {
	ch := NewDeltaChannel(
		batchAppendReducer,
		nil, 1,
	)
	// Not updated → IsAvailable() == false.
	channels := map[string]Channel{"msg": ch}
	counters := map[string][2]int{"msg": {10, 10}}
	got := DeltaChannelsToSnapshot(channels, counters)
	if len(got) != 0 {
		t.Fatalf("expected empty (unavailable channel skipped), got %v", got)
	}
}

// TestDeltaChannelsToSnapshotSkipsNonDelta ensures non-delta channels are
// skipped by the predicate (AsDelta returns false for them).
func TestDeltaChannelsToSnapshotSkipsNonDelta(t *testing.T) {
	channels := map[string]Channel{"msg": NewLastValue()}
	counters := map[string][2]int{"msg": {100, 10000}}
	got := DeltaChannelsToSnapshot(channels, counters)
	if len(got) != 0 {
		t.Fatalf("expected empty (non-delta channel skipped), got %v", got)
	}
}

// TestDeltaSnapshotBlobAndFrequency covers the additive accessor methods
// mandated by M1. SnapshotBlob returns a deltaSnapshot when the channel has a
// value, (nil,false) otherwise. SnapshotFrequency echoes the configured cadence.
func TestDeltaSnapshotBlobAndFrequency(t *testing.T) {
	ch := NewDeltaChannel(batchAppendReducer, func() any { return []any{} }, 7)
	if got := ch.SnapshotFrequency(); got != 7 {
		t.Fatalf("SnapshotFrequency() = %d, want 7", got)
	}
	// Before any update: no value → (nil, false).
	if blob, ok := ch.SnapshotBlob(); ok {
		t.Fatalf("SnapshotBlob() before update = (%v, true), want (nil, false)", blob)
	}
	// After an update: returns a deltaSnapshot blob carrying the value.
	ch.Update([]any{"x", "y"})
	blob, ok := ch.SnapshotBlob()
	if !ok {
		t.Fatal("SnapshotBlob() after update ok = false, want true")
	}
	ds, isSnap := asDeltaSnapshot(blob)
	if !isSnap {
		t.Fatalf("SnapshotBlob() = %T, want deltaSnapshot", blob)
	}
	if got, ok := ds.([]any); !ok || len(got) != 2 || got[0] != "x" || got[1] != "y" {
		t.Fatalf("SnapshotBlob() value = %v, want [x y]", ds)
	}
}
