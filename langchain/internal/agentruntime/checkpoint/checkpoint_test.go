package checkpoint

import (
	"context"
	"testing"
	"time"

	lccheckpoint "github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// TestTypeAliasesAreIdentical verifies the declared aliases remain true
// aliases of the langgraph/checkpoint types: values constructed through the
// shim must be assignable to the langgraph/checkpoint types with no
// conversion. This fails to compile if an alias ever drifts.
func TestTypeAliasesAreIdentical(t *testing.T) {
	cfg := Config{ThreadID: "t1", CheckpointNS: "ns", CheckpointID: "cp-1"}
	var _ lccheckpoint.Config = cfg

	cp := Checkpoint{
		V:               1,
		ID:              "cp-1",
		TS:              time.Now(),
		ChannelValues:   map[string]any{"k": "v"},
		ChannelVersions: map[string]int64{"k": 1},
		VersionsSeen:    map[string]map[string]int64{"node": {"k": 1}},
		Next:            []PlannedTask{{ID: "task-1", Node: "n", Arg: map[string]any{"a": 1}}},
	}
	var _ lccheckpoint.Checkpoint = cp
	var _ lccheckpoint.PlannedTask = cp.Next[0]

	md := Metadata{Source: "loop", Step: 0, Parents: map[string]string{"": "cp-0"}}
	var _ lccheckpoint.Metadata = md

	w := Write{TaskID: "task-1", TaskPath: "p", Channel: "k", Value: "v"}
	var _ lccheckpoint.Write = w

	tup := Tuple{Config: cfg, Checkpoint: cp, Metadata: md, PendingWrites: []Write{w}}
	var _ lccheckpoint.Tuple = tup

	opts := ListOptions{Before: &cfg, Filter: map[string]any{"source": "loop"}, Limit: 1}
	var _ lccheckpoint.ListOptions = opts
}

// TestMemorySaverImplementsSaver verifies NewMemorySaver delegates to the
// langgraph/checkpoint constructor and the result satisfies the Saver
// interface through both the shim and upstream spellings.
func TestMemorySaverImplementsSaver(t *testing.T) {
	saver := NewMemorySaver()
	if saver == nil {
		t.Fatal("NewMemorySaver() returned nil")
	}
	var _ Saver = saver
	var _ lccheckpoint.Saver = saver
}

// TestPutGetRoundTrip exercises the versioned Saver contract through the
// shim: Put returns a Config identifying the stored checkpoint, GetTuple
// retrieves it by ID and as "latest", and the parent link is taken from the
// caller's current position.
func TestPutGetRoundTrip(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	first := Checkpoint{V: 1, ID: "cp-1", TS: time.Now(), ChannelValues: map[string]any{"step": 1}}
	md1 := Metadata{Source: "input", Step: -1}
	cfg1, err := saver.Put(ctx, Config{ThreadID: "t1"}, first, md1, map[string]int64{"step": 1})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if cfg1.CheckpointID != "cp-1" || cfg1.ThreadID != "t1" {
		t.Fatalf("Put() returned Config = %+v, want CheckpointID cp-1 on thread t1", cfg1)
	}

	second := Checkpoint{V: 1, ID: "cp-2", TS: time.Now(), ChannelValues: map[string]any{"step": 2}}
	md2 := Metadata{Source: "loop", Step: 0, Parents: map[string]string{"": "cp-1"}}
	if _, err := saver.Put(ctx, cfg1, second, md2, nil); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	// Latest (empty CheckpointID) is the most recent Put.
	latest, err := saver.GetTuple(ctx, Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if latest == nil || latest.Checkpoint.ID != "cp-2" {
		t.Fatalf("GetTuple(latest) = %+v, want cp-2", latest)
	}
	if latest.ParentConfig == nil || latest.ParentConfig.CheckpointID != "cp-1" {
		t.Fatalf("latest.ParentConfig = %+v, want parent cp-1", latest.ParentConfig)
	}
	if latest.Metadata.Source != "loop" || latest.Metadata.Step != 0 {
		t.Fatalf("latest.Metadata = %+v, want source=loop step=0", latest.Metadata)
	}

	// Addressable by ID; the first checkpoint has no parent.
	tup1, err := saver.GetTuple(ctx, cfg1)
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if tup1 == nil || tup1.Checkpoint.ID != "cp-1" || tup1.Checkpoint.ChannelValues["step"] != 1 {
		t.Fatalf("GetTuple(cp-1) = %+v", tup1)
	}
	if tup1.ParentConfig != nil {
		t.Fatalf("tup1.ParentConfig = %+v, want nil for the thread's first checkpoint", tup1.ParentConfig)
	}

	// Unknown thread yields (nil, nil), not an error.
	missing, err := saver.GetTuple(ctx, Config{ThreadID: "nope"})
	if err != nil || missing != nil {
		t.Fatalf("GetTuple(unknown thread) = %+v, %v; want nil, nil", missing, err)
	}
}

// TestList verifies List returns the thread history newest first and honors
// the ListOptions Limit and Filter through the shim.
func TestList(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	for _, tc := range []struct {
		id     string
		source string
	}{
		{"cp-1", "input"},
		{"cp-2", "loop"},
		{"cp-3", "loop"},
	} {
		cp := Checkpoint{V: 1, ID: tc.id, TS: time.Now()}
		if _, err := saver.Put(ctx, Config{ThreadID: "t1"}, cp, Metadata{Source: tc.source}, nil); err != nil {
			t.Fatalf("Put(%s) error = %v", tc.id, err)
		}
	}

	all, err := saver.List(ctx, Config{ThreadID: "t1"}, ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(all) != 3 || all[0].Checkpoint.ID != "cp-3" || all[2].Checkpoint.ID != "cp-1" {
		t.Fatalf("List() = %+v, want [cp-3 cp-2 cp-1] newest first", checkpointIDs(all))
	}

	limited, err := saver.List(ctx, Config{ThreadID: "t1"}, ListOptions{Limit: 2})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(limited) != 2 {
		t.Fatalf("List(Limit:2) returned %d tuples, want 2", len(limited))
	}

	filtered, err := saver.List(ctx, Config{ThreadID: "t1"}, ListOptions{Filter: map[string]any{"source": "input"}})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(filtered) != 1 || filtered[0].Checkpoint.ID != "cp-1" {
		t.Fatalf("List(Filter source=input) = %+v, want only cp-1", checkpointIDs(filtered))
	}

	before, err := saver.List(ctx, Config{ThreadID: "t1"}, ListOptions{Before: &Config{ThreadID: "t1", CheckpointID: "cp-3"}})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(before) != 2 || before[0].Checkpoint.ID != "cp-2" {
		t.Fatalf("List(Before cp-3) = %+v, want [cp-2 cp-1]", checkpointIDs(before))
	}
}

// TestPutWrites verifies writes are recorded against a checkpoint and
// surfaced as PendingWrites, and that writing against an unknown checkpoint
// errors.
func TestPutWrites(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()

	cfg, err := saver.Put(ctx, Config{ThreadID: "t1"}, Checkpoint{V: 1, ID: "cp-1", TS: time.Now()}, Metadata{Source: "loop"}, nil)
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	writes := []Write{
		{Channel: "messages", Value: []string{"hi"}},
		{Channel: "count", Value: 1},
	}
	if err := saver.PutWrites(ctx, cfg, writes, "task-1", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}

	tup, err := saver.GetTuple(ctx, cfg)
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if len(tup.PendingWrites) != 2 {
		t.Fatalf("PendingWrites = %+v, want 2 writes", tup.PendingWrites)
	}
	if tup.PendingWrites[0].Channel != "messages" || tup.PendingWrites[0].TaskID != "task-1" {
		t.Errorf("PendingWrites[0] = %+v, want channel=messages taskID=task-1", tup.PendingWrites[0])
	}

	err = saver.PutWrites(ctx, Config{ThreadID: "t1", CheckpointID: "nope"}, writes, "task-2", "")
	if err == nil {
		t.Error("expected PutWrites against a missing checkpoint to error")
	}
}

// TestDeleteThread verifies DeleteThread removes a thread's checkpoints and
// leaves other threads untouched.
func TestDeleteThread(t *testing.T) {
	ctx := context.Background()
	saver := NewMemorySaver()
	put := func(threadID, cpID string) {
		t.Helper()
		if _, err := saver.Put(ctx, Config{ThreadID: threadID}, Checkpoint{V: 1, ID: cpID, TS: time.Now()}, Metadata{Source: "loop"}, nil); err != nil {
			t.Fatalf("Put(%s/%s) error = %v", threadID, cpID, err)
		}
	}
	put("t1", "cp-1")
	put("t2", "cp-9")

	if err := saver.DeleteThread(ctx, "t1"); err != nil {
		t.Fatalf("DeleteThread() error = %v", err)
	}
	if tup, err := saver.GetTuple(ctx, Config{ThreadID: "t1"}); err != nil || tup != nil {
		t.Errorf("GetTuple(deleted thread) = %+v, %v; want nil, nil", tup, err)
	}
	if tup, err := saver.GetTuple(ctx, Config{ThreadID: "t2"}); err != nil || tup == nil {
		t.Errorf("GetTuple(other thread) = %+v, %v; want the checkpoint intact", tup, err)
	}
}

func checkpointIDs(tups []Tuple) []string {
	ids := make([]string, len(tups))
	for i, tup := range tups {
		ids[i] = tup.Checkpoint.ID
	}
	return ids
}
