package channels

import (
	"reflect"
	"testing"
)

// TestBinaryOperatorFromCheckpointUnwrapsDeltaSnapshot: a thread written
// while the field was a DeltaChannel stores snapshot blobs in checkpoint
// values; restoring one must unwrap the blob to its inner value instead of
// keeping the blob struct as the channel's value (#9302).
func TestBinaryOperatorFromCheckpointUnwrapsDeltaSnapshot(t *testing.T) {
	base := NewBinaryOperator(AppendSliceReducer)

	cases := []struct {
		name  string
		blob  any
		inner any
	}{
		{
			name:  "JSON-map blob (SnapshotBlob form)",
			blob:  map[string]any{"type": deltaSnapshotType, "value": []int{1, 2}},
			inner: []int{1, 2},
		},
		{
			name:  "plain value passes through",
			blob:  []int{3},
			inner: []int{3},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ch := base.FromCheckpoint(tc.blob)
			got, err := ch.Get()
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.inner) {
				t.Fatalf("Get() = %v (%T), want %v", got, got, tc.inner)
			}
		})
	}

	// nil stays MISSING: the channel is empty and Update seeds it.
	ch := base.FromCheckpoint(nil)
	if ch.IsAvailable() {
		t.Fatal("FromCheckpoint(nil) restored a value, want an empty channel")
	}
}

// TestIsBinaryOperator pins the walk-inclusion discriminator (#9302): a
// reducer-registered key is a binaryOperator; DeltaChannel and LastValue are
// not.
func TestIsBinaryOperator(t *testing.T) {
	if !IsBinaryOperator(NewBinaryOperator(AppendSliceReducer)) {
		t.Fatal("NewBinaryOperator: IsBinaryOperator = false, want true")
	}
	if IsBinaryOperator(NewDeltaChannel(BatchFromReducer(AppendSliceReducer), nil, 1)) {
		t.Fatal("DeltaChannel: IsBinaryOperator = true, want false")
	}
	if IsBinaryOperator(NewLastValue()) {
		t.Fatal("LastValue: IsBinaryOperator = true, want false")
	}
}

// TestBinaryOperatorSeedsOverwriteInnerValue pins the seed path of the
// Overwrite protocol: an Overwrite as a channel's FIRST-ever write must seed
// the channel with the INNER value, never the marker itself. Overwrite means
// "write Value directly, bypassing the reducer" (overwrite.go), so storing
// the wrapper would leak the marker struct into Get/Checkpoint and into
// every later merge's type assertions. Python's binop.update consumes
// values[0] raw before its Overwrite-detection loop and would store the
// marker; the Go unwrap here is a deliberate, strictly-safer deviation
// (documented in DIVERGENCES.md, "langgraph binop zero-seed parity").
func TestBinaryOperatorSeedsOverwriteInnerValue(t *testing.T) {
	seed := NewOverwrite([]int{1, 2})
	ch := NewBinaryOperator(AppendSliceReducer)

	if changed, err := ch.Update([]any{seed}); err != nil || !changed {
		t.Fatalf("seed Update(Overwrite) = (%v, %v), want (true, nil)", changed, err)
	}
	got, err := ch.Get()
	if err != nil {
		t.Fatalf("Get after Overwrite seed: %v", err)
	}
	if _, isOverwrite := AsOverwrite(got); isOverwrite || !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("seeded value = %v (%T), want the inner []int{1,2} (marker must not leak)", got, got)
	}

	// A later superstep's Overwrite still replaces the whole value.
	if _, err := ch.Update([]any{NewOverwrite([]int{9})}); err != nil {
		t.Fatalf("second Overwrite Update: %v", err)
	}
	got, err = ch.Get()
	if err != nil {
		t.Fatalf("Get after second Overwrite: %v", err)
	}
	if !reflect.DeepEqual(got, []int{9}) {
		t.Fatalf("replaced value = %v (%T), want []int{9}", got, got)
	}
}
