package checkpoint

import (
	"context"
	"testing"
)

// TestGet covers the package-level Get helper, the Go analogue of Python's
// BaseCheckpointSaver.get (base/__init__.py:227-238): the checkpoint value
// without metadata/parent/writes, nil when no checkpoint exists.
func TestGet(t *testing.T) {
	ctx := context.Background()
	s := NewMemorySaver()

	if got, err := Get(ctx, s, Config{ThreadID: "no-such-thread"}); err != nil || got != nil {
		t.Fatalf("Get unknown thread = %v, %v; want nil, nil", got, err)
	}

	cp := Checkpoint{V: 1, ID: NewID(1), ChannelValues: map[string]any{"a": 1}}
	cfg, err := s.Put(ctx, Config{ThreadID: "t1"}, cp, Metadata{Source: "input", Step: -1}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	latest, err := Get(ctx, s, Config{ThreadID: "t1"})
	if err != nil || latest == nil {
		t.Fatalf("Get latest = %v, %v", latest, err)
	}
	if latest.ID != cp.ID || latest.ChannelValues["a"] != 1 {
		t.Fatalf("Get latest = %+v, want checkpoint %q", latest, cp.ID)
	}

	byID, err := Get(ctx, s, cfg)
	if err != nil || byID == nil || byID.ID != cp.ID {
		t.Fatalf("Get by ID = %v, %v; want checkpoint %q", byID, err, cp.ID)
	}
}

// TestMetadataMatchesFilterRunID covers the run_id key added to the filter
// projection (Python's free-form metadata filter accepts run_id).
func TestMetadataMatchesFilterRunID(t *testing.T) {
	md := Metadata{Source: "loop", Step: 1, RunID: "r1"}
	if !MetadataMatchesFilter(md, map[string]any{"run_id": "r1"}) {
		t.Fatal("filter {run_id: r1} must match")
	}
	if MetadataMatchesFilter(md, map[string]any{"run_id": "r2"}) {
		t.Fatal("filter {run_id: r2} must not match")
	}
	// An unstamped checkpoint does not match a run_id filter, and an empty
	// run_id never appears in the projection.
	if MetadataMatchesFilter(Metadata{Source: "loop"}, map[string]any{"run_id": ""}) {
		t.Fatal("empty RunID must not be matchable")
	}
}
