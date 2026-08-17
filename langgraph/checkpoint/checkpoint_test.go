package checkpoint

import (
	"context"
	"maps"
	"testing"
	"time"
)

// TestListFilter verifies ListOptions.Filter metadata filtering, mirroring
// the query_1..query_4 semantics of Python's `test_sync.py:214-260` with the
// Go Metadata keys source/step.
func TestListFilter(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	mds := []Metadata{
		{Source: "input", Step: -1},
		{Source: "loop", Step: 0},
		{Source: "loop", Step: 1},
	}
	ids := make([]string, len(mds))
	cfg := Config{ThreadID: "t"}
	for i, md := range mds {
		cp := Checkpoint{V: 1, ID: NewID(i), TS: time.Now()}
		var err error
		cfg, err = saver.Put(ctx, cfg, cp, md, nil)
		if err != nil {
			t.Fatalf("Put() error = %v", err)
		}
		ids[i] = cp.ID
	}

	list := func(filter map[string]any) []Tuple {
		t.Helper()
		tups, err := saver.List(ctx, Config{ThreadID: "t"}, ListOptions{Filter: filter})
		if err != nil {
			t.Fatalf("List(Filter=%v) error = %v", filter, err)
		}
		return tups
	}

	// {"source": "input"}: exactly the first checkpoint.
	if tups := list(map[string]any{"source": "input"}); len(tups) != 1 || tups[0].Checkpoint.ID != ids[0] {
		t.Fatalf("List(source=input) = %+v, want [%s]", tups, ids[0])
	}
	// {"source": "loop"}: both loop checkpoints, newest first.
	if tups := list(map[string]any{"source": "loop"}); len(tups) != 2 || tups[0].Checkpoint.ID != ids[2] || tups[1].Checkpoint.ID != ids[1] {
		t.Fatalf("List(source=loop) = %+v, want [%s %s] (newest first)", tups, ids[2], ids[1])
	}
	// {"step": 1}: exactly the third checkpoint.
	if tups := list(map[string]any{"step": 1}); len(tups) != 1 || tups[0].Checkpoint.ID != ids[2] {
		t.Fatalf("List(step=1) = %+v, want [%s]", tups, ids[2])
	}
	// {"source": "update"}: no matches.
	if tups := list(map[string]any{"source": "update"}); len(tups) != 0 {
		t.Fatalf("List(source=update) = %+v, want empty", tups)
	}
	// Empty filter: everything.
	if tups := list(map[string]any{}); len(tups) != 3 {
		t.Fatalf("List(empty filter) = %d tuples, want 3", len(tups))
	}
}

// TestPutWritesTaskPathRoundTrip verifies PutWrites stamps each write with
// the given taskPath and that it round-trips through GetTuple.
func TestPutWritesTaskPathRoundTrip(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	cp := Checkpoint{V: 1, ID: NewID(0), TS: time.Now()}
	cfg, err := saver.Put(ctx, Config{ThreadID: "t"}, cp, Metadata{Source: "loop", Step: 0}, nil)
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	if err := saver.PutWrites(ctx, cfg, []Write{{Channel: "c", Value: "v"}}, "task-1", "path/a"); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}
	tup, err := saver.GetTuple(ctx, Config{ThreadID: "t", CheckpointID: cp.ID})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if len(tup.PendingWrites) != 1 || tup.PendingWrites[0].TaskPath != "path/a" {
		t.Fatalf("PendingWrites = %+v, want one write with TaskPath %q", tup.PendingWrites, "path/a")
	}

	// The default "" argument stamps an empty TaskPath.
	if err := saver.PutWrites(ctx, cfg, []Write{{Channel: "c2", Value: "v2"}}, "task-2", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}
	tup, err = saver.GetTuple(ctx, Config{ThreadID: "t", CheckpointID: cp.ID})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if len(tup.PendingWrites) != 2 || tup.PendingWrites[1].TaskPath != "" {
		t.Fatalf("PendingWrites = %+v, want second write with empty TaskPath", tup.PendingWrites)
	}
}

func TestNewIDMonotonic(t *testing.T) {
	prev := ""
	for i := 0; i < 1000; i++ {
		id := NewID(i)
		if id <= prev {
			t.Fatalf("NewID(%d) = %q not greater than previous %q", i, id, prev)
		}
		prev = id
	}
}

func TestCopyOnRead(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	cp := Checkpoint{
		V:               1,
		ID:              NewID(0),
		TS:              time.Now(),
		ChannelValues:   map[string]any{"x": 1},
		ChannelVersions: map[string]int64{"x": 1},
		VersionsSeen:    map[string]map[string]int64{"task": {"x": 1}},
		Next:            []PlannedTask{{Node: "n"}},
	}
	md := Metadata{Source: "loop", Parents: map[string]string{"": "parent-id"}}
	if _, err := saver.Put(ctx, Config{ThreadID: "t"}, cp, md, nil); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	tup, err := saver.GetTuple(ctx, Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	// Mutate every map/slice handed back to the caller.
	tup.Checkpoint.ChannelValues["x"] = 999
	tup.Checkpoint.ChannelVersions["x"] = 999
	tup.Checkpoint.VersionsSeen["task"]["x"] = 999
	tup.Checkpoint.Next[0].Node = "mutated"
	tup.Metadata.Parents[""] = "mutated"

	fresh, err := saver.GetTuple(ctx, Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if fresh.Checkpoint.ChannelValues["x"] != 1 {
		t.Fatalf("store corrupted via ChannelValues: %+v", fresh.Checkpoint.ChannelValues)
	}
	if fresh.Checkpoint.ChannelVersions["x"] != 1 {
		t.Fatalf("store corrupted via ChannelVersions: %+v", fresh.Checkpoint.ChannelVersions)
	}
	if fresh.Checkpoint.VersionsSeen["task"]["x"] != 1 {
		t.Fatalf("store corrupted via VersionsSeen: %+v", fresh.Checkpoint.VersionsSeen)
	}
	if fresh.Checkpoint.Next[0].Node != "n" {
		t.Fatalf("store corrupted via Next: %+v", fresh.Checkpoint.Next)
	}
	if fresh.Metadata.Parents[""] != "parent-id" {
		t.Fatalf("store corrupted via Metadata.Parents: %+v", fresh.Metadata.Parents)
	}
}

func TestMemorySaverZeroValue(t *testing.T) {
	ctx := context.Background()
	var saver MemorySaver

	if tup, err := saver.GetTuple(ctx, Config{ThreadID: "x"}); err != nil || tup != nil {
		t.Fatalf("expected zero-value MemorySaver to report no checkpoint, got %+v err=%v", tup, err)
	}
	cp := Checkpoint{V: 1, ID: NewID(0), TS: time.Now(), Next: []PlannedTask{{Node: "n"}}}
	if _, err := saver.Put(ctx, Config{ThreadID: "x"}, cp, Metadata{Source: "loop"}, nil); err != nil {
		t.Fatalf("Put() on zero-value MemorySaver error = %v", err)
	}
	tup, err := saver.GetTuple(ctx, Config{ThreadID: "x"})
	if err != nil || tup == nil || tup.Checkpoint.Next[0].Node != "n" {
		t.Fatalf("expected checkpoint on zero-value MemorySaver, got %+v err=%v", tup, err)
	}
	if err := saver.DeleteThread(ctx, "x"); err != nil {
		t.Fatalf("DeleteThread() on zero-value MemorySaver error = %v", err)
	}
}

func TestNamespacesAreIndependent(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	for i, ns := range []string{"", "sub"} {
		cp := Checkpoint{V: 1, ID: NewID(i), TS: time.Now(), ChannelValues: map[string]any{"ns": ns}}
		if _, err := saver.Put(ctx, Config{ThreadID: "t", CheckpointNS: ns}, cp, Metadata{Source: "loop"}, nil); err != nil {
			t.Fatalf("Put() error = %v", err)
		}
	}

	root, err := saver.GetTuple(ctx, Config{ThreadID: "t"})
	if err != nil || root == nil || root.Checkpoint.ChannelValues["ns"] != "" {
		t.Fatalf("root namespace tuple = %+v err=%v", root, err)
	}
	sub, err := saver.GetTuple(ctx, Config{ThreadID: "t", CheckpointNS: "sub"})
	if err != nil || sub == nil || sub.Checkpoint.ChannelValues["ns"] != "sub" {
		t.Fatalf("sub namespace tuple = %+v err=%v", sub, err)
	}
}

// TestListBeforeAndLimit verifies ListOptions.Before excludes the named
// checkpoint and everything newer (strictly-older semantics) and that
// ListOptions.Limit truncates the newest-first result after filtering.
func TestListBeforeAndLimit(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	ids := make([]string, 4)
	cfg := Config{ThreadID: "t"}
	for i := range ids {
		cp := Checkpoint{V: 1, ID: NewID(i), TS: time.Now()}
		var err error
		cfg, err = saver.Put(ctx, cfg, cp, Metadata{Source: "loop", Step: i}, nil)
		if err != nil {
			t.Fatalf("Put() error = %v", err)
		}
		ids[i] = cp.ID
	}

	// Before ids[2]: only ids[1] and ids[0] remain, newest first.
	tups, err := saver.List(ctx, Config{ThreadID: "t"}, ListOptions{Before: &Config{CheckpointID: ids[2]}})
	if err != nil {
		t.Fatalf("List(Before) error = %v", err)
	}
	if len(tups) != 2 || tups[0].Checkpoint.ID != ids[1] || tups[1].Checkpoint.ID != ids[0] {
		t.Fatalf("List(Before=%s) = %+v, want [%s %s]", ids[2], tups, ids[1], ids[0])
	}

	// A Before config with an empty CheckpointID imposes no restriction.
	tups, err = saver.List(ctx, Config{ThreadID: "t"}, ListOptions{Before: &Config{}})
	if err != nil {
		t.Fatalf("List(Before with empty ID) error = %v", err)
	}
	if len(tups) != 4 {
		t.Fatalf("List(Before with empty ID) = %d tuples, want 4", len(tups))
	}

	// Limit caps the newest-first ordering.
	tups, err = saver.List(ctx, Config{ThreadID: "t"}, ListOptions{Limit: 2})
	if err != nil {
		t.Fatalf("List(Limit) error = %v", err)
	}
	if len(tups) != 2 || tups[0].Checkpoint.ID != ids[3] || tups[1].Checkpoint.ID != ids[2] {
		t.Fatalf("List(Limit=2) = %+v, want [%s %s]", tups, ids[3], ids[2])
	}
}

// TestPutNewVersions verifies Put merges newVersions into the stored
// ChannelVersions, creating the map when the checkpoint has none and
// merging into an existing one otherwise.
func TestPutNewVersions(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	// Checkpoint without ChannelVersions: newVersions creates the map.
	cp1 := Checkpoint{V: 1, ID: NewID(0), TS: time.Now()}
	cfg, err := saver.Put(ctx, Config{ThreadID: "t"}, cp1, Metadata{Source: "loop"}, map[string]int64{"a": 2})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	tup, err := saver.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %+v err=%v", tup, err)
	}
	if tup.Checkpoint.ChannelVersions["a"] != 2 {
		t.Fatalf("ChannelVersions = %+v, want a:2 merged into a fresh map", tup.Checkpoint.ChannelVersions)
	}

	// Checkpoint with existing ChannelVersions: newVersions merge in.
	cp2 := Checkpoint{V: 1, ID: NewID(1), TS: time.Now(), ChannelVersions: map[string]int64{"a": 3, "b": 1}}
	cfg, err = saver.Put(ctx, cfg, cp2, Metadata{Source: "loop"}, map[string]int64{"b": 5, "c": 7})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	tup, err = saver.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %+v err=%v", tup, err)
	}
	want := map[string]int64{"a": 3, "b": 5, "c": 7}
	if !maps.Equal(tup.Checkpoint.ChannelVersions, want) {
		t.Fatalf("ChannelVersions = %+v, want %+v", tup.Checkpoint.ChannelVersions, want)
	}
}

// TestPutWritesMissingCheckpoint verifies PutWrites against a checkpoint
// that does not exist errors instead of silently dropping the writes.
func TestPutWritesMissingCheckpoint(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	err := saver.PutWrites(ctx, Config{ThreadID: "nope", CheckpointID: "missing"}, []Write{{Channel: "c", Value: 1}}, "task", "")
	if err == nil {
		t.Fatalf("PutWrites() on missing checkpoint = nil error, want an error")
	}
}

// TestPutWritesSlotRules pins the write-slot dedup rules: a regular channel
// keeps the first write to an occupied (taskID, idx) slot, while a reserved
// channel's fixed negative slot is replaced in place on rewrite.
func TestPutWritesSlotRules(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	cp := Checkpoint{V: 1, ID: NewID(0), TS: time.Now()}
	cfg, err := saver.Put(ctx, Config{ThreadID: "t"}, cp, Metadata{Source: "loop"}, nil)
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	writes := func() []Write {
		t.Helper()
		tup, err := saver.GetTuple(ctx, cfg)
		if err != nil || tup == nil {
			t.Fatalf("GetTuple() = %+v err=%v", tup, err)
		}
		return tup.PendingWrites
	}

	// Regular channel, same task and positional idx: first write wins.
	if err := saver.PutWrites(ctx, cfg, []Write{{Channel: "c", Value: "first"}}, "task-1", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}
	if err := saver.PutWrites(ctx, cfg, []Write{{Channel: "c", Value: "second"}}, "task-1", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}
	if got := writes(); len(got) != 1 || got[0].Value != "first" {
		t.Fatalf("PendingWrites after regular rewrite = %+v, want the first write kept", got)
	}

	// A different task ID occupies a different slot even at the same idx.
	if err := saver.PutWrites(ctx, cfg, []Write{{Channel: "c", Value: "other task"}}, "task-2", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}
	if got := writes(); len(got) != 2 {
		t.Fatalf("PendingWrites = %+v, want a separate slot per task ID", got)
	}

	// Reserved channel: the rewrite replaces the stored value in place.
	if err := saver.PutWrites(ctx, cfg, []Write{{Channel: ReservedError, Value: "err-1"}}, "task-1", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}
	if err := saver.PutWrites(ctx, cfg, []Write{{Channel: ReservedError, Value: "err-2"}}, "task-1", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}
	got := writes()
	if len(got) != 3 {
		t.Fatalf("PendingWrites after reserved rewrite = %+v, want in-place replace (no new slot)", got)
	}
	if last := got[len(got)-1]; last.Channel != ReservedError || last.Value != "err-2" {
		t.Fatalf("reserved write = %+v, want {%s err-2}", last, ReservedError)
	}
}
