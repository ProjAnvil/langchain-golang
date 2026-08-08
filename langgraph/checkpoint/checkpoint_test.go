package checkpoint

import (
	"context"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/types"
)

func TestPutGetRoundTrip(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	// Absent thread: (nil, nil).
	tup, err := saver.GetTuple(ctx, Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if tup != nil {
		t.Fatalf("expected nil tuple for unknown thread, got %+v", tup)
	}

	cp1 := Checkpoint{
		V:               1,
		ID:              NewID(0),
		TS:              time.Now(),
		ChannelValues:   map[string]any{"x": 1},
		ChannelVersions: map[string]int64{"x": 1},
	}
	cfg1, err := saver.Put(ctx, Config{ThreadID: "t"}, cp1, Metadata{Source: "input", Step: -1}, nil)
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if cfg1.CheckpointID != cp1.ID || cfg1.ThreadID != "t" {
		t.Fatalf("Put() returned config %+v, want thread t id %s", cfg1, cp1.ID)
	}

	// Get latest.
	latest, err := saver.GetTuple(ctx, Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if latest == nil || latest.Checkpoint.ID != cp1.ID {
		t.Fatalf("latest tuple = %+v, want checkpoint %s", latest, cp1.ID)
	}
	if latest.Config.CheckpointID != cp1.ID {
		t.Fatalf("tuple config = %+v, want CheckpointID %s", latest.Config, cp1.ID)
	}
	if latest.Metadata.Source != "input" || latest.Metadata.Step != -1 {
		t.Fatalf("metadata = %+v, want Source=input Step=-1", latest.Metadata)
	}
	if latest.Checkpoint.ChannelValues["x"] != 1 {
		t.Fatalf("channel values = %+v, want x=1", latest.Checkpoint.ChannelValues)
	}
	// A thread's first checkpoint has no parent.
	if latest.ParentConfig != nil {
		t.Fatalf("ParentConfig = %+v, want nil for first checkpoint", latest.ParentConfig)
	}

	// Put a second checkpoint on top of the first.
	cp2 := Checkpoint{
		V:             1,
		ID:            NewID(1),
		TS:            time.Now(),
		ChannelValues: map[string]any{"x": 2},
	}
	cfg2, err := saver.Put(ctx, cfg1, cp2, Metadata{Source: "loop", Step: 0}, map[string]int64{"x": 2})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	latest, err = saver.GetTuple(ctx, Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if latest == nil || latest.Checkpoint.ID != cp2.ID {
		t.Fatalf("latest tuple = %+v, want checkpoint %s", latest, cp2.ID)
	}
	if latest.ParentConfig == nil || latest.ParentConfig.CheckpointID != cp1.ID {
		t.Fatalf("ParentConfig = %+v, want parent id %s", latest.ParentConfig, cp1.ID)
	}

	// Get by ID returns the older checkpoint.
	byID, err := saver.GetTuple(ctx, Config{ThreadID: "t", CheckpointID: cp1.ID})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if byID == nil || byID.Checkpoint.ID != cp1.ID || byID.Checkpoint.ChannelValues["x"] != 1 {
		t.Fatalf("get-by-id tuple = %+v, want checkpoint %s with x=1", byID, cp1.ID)
	}
	_ = cfg2
}

func TestListNewestFirstWithBeforeAndLimit(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	ids := make([]string, 3)
	cfg := Config{ThreadID: "t"}
	for i := range ids {
		cp := Checkpoint{V: 1, ID: NewID(i), TS: time.Now(), ChannelValues: map[string]any{"step": i}}
		var err error
		cfg, err = saver.Put(ctx, cfg, cp, Metadata{Source: "loop", Step: i - 1}, nil)
		if err != nil {
			t.Fatalf("Put() error = %v", err)
		}
		ids[i] = cp.ID
	}

	all, err := saver.List(ctx, Config{ThreadID: "t"}, ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("List() returned %d tuples, want 3", len(all))
	}
	for i, tup := range all {
		if want := ids[len(ids)-1-i]; tup.Checkpoint.ID != want {
			t.Fatalf("List()[%d].ID = %s, want %s (newest first)", i, tup.Checkpoint.ID, want)
		}
	}

	// Before: strictly before the newest -> the two older checkpoints.
	before, err := saver.List(ctx, Config{ThreadID: "t"}, ListOptions{Before: &Config{ThreadID: "t", CheckpointID: ids[2]}})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(before) != 2 || before[0].Checkpoint.ID != ids[1] || before[1].Checkpoint.ID != ids[0] {
		t.Fatalf("List(Before=%s) = %+v, want [%s %s]", ids[2], before, ids[1], ids[0])
	}

	// Limit caps the result.
	limited, err := saver.List(ctx, Config{ThreadID: "t"}, ListOptions{Limit: 1})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(limited) != 1 || limited[0].Checkpoint.ID != ids[2] {
		t.Fatalf("List(Limit=1) = %+v, want [%s]", limited, ids[2])
	}

	// Before + Limit compose.
	both, err := saver.List(ctx, Config{ThreadID: "t"}, ListOptions{Before: &Config{ThreadID: "t", CheckpointID: ids[2]}, Limit: 1})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(both) != 1 || both[0].Checkpoint.ID != ids[1] {
		t.Fatalf("List(Before+Limit=1) = %+v, want [%s]", both, ids[1])
	}
}

func TestPutWritesVisibleOnGetTuple(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	cp := Checkpoint{V: 1, ID: NewID(0), TS: time.Now(), ChannelValues: map[string]any{"x": 1}}
	cfg, err := saver.Put(ctx, Config{ThreadID: "t"}, cp, Metadata{Source: "loop", Step: 0}, nil)
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	writes := []Write{
		{Channel: ReservedInterrupt, Value: types.Interrupt{Value: "confirm?", ID: "int-1"}},
		{Channel: "x", Value: 2},
	}
	if err := saver.PutWrites(ctx, cfg, writes, "task-1", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}

	tup, err := saver.GetTuple(ctx, Config{ThreadID: "t", CheckpointID: cp.ID})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if len(tup.PendingWrites) != 2 {
		t.Fatalf("PendingWrites = %+v, want 2 writes", tup.PendingWrites)
	}
	for i, w := range tup.PendingWrites {
		if w.TaskID != "task-1" {
			t.Fatalf("PendingWrites[%d].TaskID = %q, want task-1", i, w.TaskID)
		}
	}
	intr, ok := tup.PendingWrites[0].Value.(types.Interrupt)
	if !ok || intr.ID != "int-1" {
		t.Fatalf("PendingWrites[0].Value = %+v, want interrupt int-1", tup.PendingWrites[0].Value)
	}
}

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

func TestDeleteThread(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	for _, thread := range []string{"a", "b"} {
		cp := Checkpoint{V: 1, ID: NewID(0), TS: time.Now()}
		if _, err := saver.Put(ctx, Config{ThreadID: thread}, cp, Metadata{Source: "loop"}, nil); err != nil {
			t.Fatalf("Put() error = %v", err)
		}
	}

	if err := saver.DeleteThread(ctx, "a"); err != nil {
		t.Fatalf("DeleteThread() error = %v", err)
	}
	if tup, _ := saver.GetTuple(ctx, Config{ThreadID: "a"}); tup != nil {
		t.Fatalf("expected thread a to be gone, got %+v", tup)
	}
	if list, _ := saver.List(ctx, Config{ThreadID: "a"}, ListOptions{}); len(list) != 0 {
		t.Fatalf("expected empty history for thread a, got %+v", list)
	}
	if tup, _ := saver.GetTuple(ctx, Config{ThreadID: "b"}); tup == nil {
		t.Fatal("expected thread b to survive DeleteThread(a)")
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

// TestParentConfigReflectsPutTimeConfig verifies D3: the parent link is the
// cfg passed to Put, not the thread's latest checkpoint — including after a
// fork, where a new checkpoint is built on an older checkpoint ID.
func TestParentConfigReflectsPutTimeConfig(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	putCp := func(cfg Config, seq int) Config {
		cp := Checkpoint{V: 1, ID: NewID(seq), TS: time.Now()}
		out, err := saver.Put(ctx, cfg, cp, Metadata{Source: "loop"}, nil)
		if err != nil {
			t.Fatalf("Put() error = %v", err)
		}
		return out
	}

	cfgA := putCp(Config{ThreadID: "t"}, 0)
	cfgB := putCp(cfgA, 1)
	// Fork: build cfgC on cfgA (an older checkpoint), not on the latest (cfgB).
	cfgC := putCp(cfgA, 2)

	tupC, err := saver.GetTuple(ctx, Config{ThreadID: "t", CheckpointID: cfgC.CheckpointID})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if tupC.ParentConfig == nil || tupC.ParentConfig.CheckpointID != cfgA.CheckpointID {
		t.Fatalf("fork ParentConfig = %+v, want parent id %s (the put-time cfg, not the latest %s)",
			tupC.ParentConfig, cfgA.CheckpointID, cfgB.CheckpointID)
	}

	tupB, err := saver.GetTuple(ctx, Config{ThreadID: "t", CheckpointID: cfgB.CheckpointID})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if tupB.ParentConfig == nil || tupB.ParentConfig.CheckpointID != cfgA.CheckpointID {
		t.Fatalf("ParentConfig = %+v, want parent id %s", tupB.ParentConfig, cfgA.CheckpointID)
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
