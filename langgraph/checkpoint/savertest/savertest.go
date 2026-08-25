package savertest

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Run executes the full contract suite as subtests of t. newSaver must
// return a Saver backed by EMPTY storage (factories wrapping a shared
// database must truncate all tables).
func Run(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	t.Run("put_get_round_trip", func(t *testing.T) { testPutGetRoundTrip(t, newSaver) })
	t.Run("int_values_survive_as_int", func(t *testing.T) { testIntValuesSurviveAsInt(t, newSaver) })
	t.Run("parent_links", func(t *testing.T) { testParentLinks(t, newSaver) })
	t.Run("list_order_before_limit", func(t *testing.T) { testListOrderBeforeLimit(t, newSaver) })
	t.Run("list_filter", func(t *testing.T) { testListFilter(t, newSaver) })
	t.Run("put_writes_round_trip", func(t *testing.T) { testPutWritesRoundTrip(t, newSaver) })
	t.Run("put_writes_batch_rule", func(t *testing.T) { testPutWritesBatchRule(t, newSaver) })
	t.Run("put_writes_tasks_all_survive", func(t *testing.T) { testPutWritesTasksAllSurvive(t, newSaver) })
	t.Run("put_writes_task_path", func(t *testing.T) { testPutWritesTaskPath(t, newSaver) })
	t.Run("put_writes_missing_checkpoint", func(t *testing.T) { testPutWritesMissingCheckpoint(t, newSaver) })
	t.Run("delete_thread", func(t *testing.T) { testDeleteThread(t, newSaver) })
	t.Run("delete_for_runs", func(t *testing.T) { testDeleteForRuns(t, newSaver) })
	t.Run("copy_thread", func(t *testing.T) { testCopyThread(t, newSaver) })
	t.Run("prune", func(t *testing.T) { testPrune(t, newSaver) })
	t.Run("concurrent_put", func(t *testing.T) { testConcurrentPut(t, newSaver) })
	t.Run("serde_round_trip", func(t *testing.T) { testSerdeRoundTrip(t, newSaver) })
}

// sampleCheckpoint returns a checkpoint exercising every projection field:
// channel values spanning plain JSON and registry types, version maps, a
// planned task with a typed arg, and a fixed timestamp for exact comparison.
func sampleCheckpoint(id string) checkpoint.Checkpoint {
	return checkpoint.Checkpoint{
		V:  1,
		ID: id,
		TS: time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
		ChannelValues: map[string]any{
			"msgs":      []messages.Message{messages.Human("hi"), messages.AI("hello")},
			"note":      "plain string",
			"count":     7,
			"tags":      []string{"a", "b"},
			"interrupt": types.Interrupt{Value: "why?", ID: "int-1"},
			"extra":     map[string]any{"k": "v"},
		},
		ChannelVersions: map[string]int64{"msgs": 3, "note": 1},
		VersionsSeen:    map[string]map[string]int64{"node1": {"msgs": 2}},
		Next: []checkpoint.PlannedTask{
			{ID: "task-1", Node: "node1", Arg: map[string]any{"limit": 3}},
		},
	}
}

// testPutGetRoundTrip: a checkpoint with messages, interrupts, typed slices,
// plain maps and ints in its channel values survives a Put/GetTuple cycle
// value-for-value, addressable both as "latest" and by ID; Put's newVersions
// merge into the stored ChannelVersions; an unknown thread reads as
// (nil, nil).
func testPutGetRoundTrip(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	cp := sampleCheckpoint(checkpoint.NewID(1))
	md := checkpoint.Metadata{Source: "loop", Step: 0, Parents: map[string]string{"": "grandparent"}}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, md, map[string]int64{"msgs": 4})
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if cfg.CheckpointID != cp.ID || cfg.ThreadID != "t1" {
		t.Fatalf("Put returned %+v, want checkpoint_id %q", cfg, cp.ID)
	}

	// newVersions merge into the stored ChannelVersions.
	want := cp
	want.ChannelVersions = map[string]int64{"msgs": 4, "note": 1}

	latest, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetTuple latest: %v", err)
	}
	if latest == nil {
		t.Fatalf("GetTuple latest: got nil")
	}
	if !reflect.DeepEqual(latest.Checkpoint, want) {
		t.Fatalf("checkpoint mismatch:\n got %+v\nwant %+v", latest.Checkpoint, want)
	}
	if !reflect.DeepEqual(latest.Metadata, md) {
		t.Fatalf("metadata mismatch: got %+v want %+v", latest.Metadata, md)
	}
	if latest.ParentConfig != nil {
		t.Fatalf("first checkpoint should have no parent, got %+v", latest.ParentConfig)
	}
	if latest.Config.CheckpointID != cp.ID {
		t.Fatalf("tuple config checkpoint_id = %q, want %q", latest.Config.CheckpointID, cp.ID)
	}

	byID, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: cp.ID})
	if err != nil || byID == nil {
		t.Fatalf("GetTuple by ID: tup=%v err=%v", byID, err)
	}
	if !reflect.DeepEqual(byID.Checkpoint, want) {
		t.Fatalf("by-ID checkpoint mismatch:\n got %+v\nwant %+v", byID.Checkpoint, want)
	}

	missing, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "no-such-thread"})
	if err != nil || missing != nil {
		t.Fatalf("GetTuple unknown thread: got tup=%v err=%v, want nil, nil", missing, err)
	}
}

// testIntValuesSurviveAsInt pins the serde contract that plain `int` and
// `int64` channel values do not degrade to float64 through a persistent
// backend.
func testIntValuesSurviveAsInt(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	cp := sampleCheckpoint(checkpoint.NewID(1))
	cp.ChannelValues = map[string]any{"n": 42, "big": int64(1 << 62)}
	if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if got, ok := tup.Checkpoint.ChannelValues["n"].(int); !ok || got != 42 {
		t.Fatalf("channel n = %v (%T), want int(42)", tup.Checkpoint.ChannelValues["n"], tup.Checkpoint.ChannelValues["n"])
	}
	if got, ok := tup.Checkpoint.ChannelValues["big"].(int64); !ok || got != int64(1<<62) {
		t.Fatalf("channel big = %v (%T), want int64(1<<62)", tup.Checkpoint.ChannelValues["big"], tup.Checkpoint.ChannelValues["big"])
	}
}

// testParentLinks verifies D3: a Put whose config carries a CheckpointID
// records that position as the new checkpoint's ParentConfig.
func testParentLinks(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	first := sampleCheckpoint(checkpoint.NewID(1))
	cfg1, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, first, checkpoint.Metadata{Source: "input", Step: -1}, nil)
	if err != nil {
		t.Fatalf("Put first: %v", err)
	}
	second := sampleCheckpoint(checkpoint.NewID(2))
	if _, err := s.Put(ctx, cfg1, second, checkpoint.Metadata{Source: "loop", Step: 0}, nil); err != nil {
		t.Fatalf("Put second: %v", err)
	}

	tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: second.ID})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if tup.ParentConfig == nil || tup.ParentConfig.CheckpointID != first.ID || tup.ParentConfig.ThreadID != "t1" {
		t.Fatalf("ParentConfig = %+v, want checkpoint_id %q", tup.ParentConfig, first.ID)
	}
}

// testListOrderBeforeLimit verifies List ordering (newest checkpoint ID
// first) and the Before (strictly older) / Limit filters.
func testListOrderBeforeLimit(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	ids := make([]string, 3)
	for i := range ids {
		ids[i] = checkpoint.NewID(i + 1)
		cp := sampleCheckpoint(ids[i])
		if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{Source: "loop", Step: i}, nil); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	// Another thread must not leak into t1's history.
	if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t2"}, sampleCheckpoint(checkpoint.NewID(9)), checkpoint.Metadata{}, nil); err != nil {
		t.Fatalf("Put t2: %v", err)
	}

	all, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("List returned %d tuples, want 3", len(all))
	}
	for i, tup := range all {
		if want := ids[len(ids)-1-i]; tup.Checkpoint.ID != want {
			t.Fatalf("List[%d].ID = %q, want %q (newest first)", i, tup.Checkpoint.ID, want)
		}
	}

	before, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{
		Before: &checkpoint.Config{ThreadID: "t1", CheckpointID: ids[2]},
	})
	if err != nil {
		t.Fatalf("List Before: %v", err)
	}
	if len(before) != 2 || before[0].Checkpoint.ID != ids[1] || before[1].Checkpoint.ID != ids[0] {
		t.Fatalf("List Before(%q) = %v, want [%q %q]", ids[2], tupleIDs(before), ids[1], ids[0])
	}

	limited, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{Limit: 1})
	if err != nil {
		t.Fatalf("List Limit: %v", err)
	}
	if len(limited) != 1 || limited[0].Checkpoint.ID != ids[2] {
		t.Fatalf("List Limit(1) = %v, want [%q]", tupleIDs(limited), ids[2])
	}
}

func tupleIDs(tups []checkpoint.Tuple) []string {
	ids := make([]string, len(tups))
	for i, tup := range tups {
		ids[i] = tup.Checkpoint.ID
	}
	return ids
}

// testListFilter ports the four queries of Python's test_search
// (test_sync.py:214-260) onto the Go Metadata keys source/step (Go Metadata
// is a closed struct, so Python's free-form metadata keys are not portable).
//
// Divergence: Python's list(None, filter=...) searches across ALL threads,
// and list with only a thread_id searches across all namespaces
// (test_sync.py:253-259). Go's List always lists exactly one thread, and
// Config.CheckpointNS is an exact namespace match — "" selects the root
// namespace only. The Go assertions below are therefore scoped per thread
// and per namespace.
func testListFilter(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	put := func(cfg checkpoint.Config, md checkpoint.Metadata) string {
		t.Helper()
		cp := sampleCheckpoint(checkpoint.NewID(int(md.Step + 10)))
		if _, err := s.Put(ctx, cfg, cp, md, nil); err != nil {
			t.Fatalf("Put %+v: %v", cfg, err)
		}
		return cp.ID
	}
	id1 := put(checkpoint.Config{ThreadID: "thread-1"}, checkpoint.Metadata{Source: "input", Step: 2})
	id2 := put(checkpoint.Config{ThreadID: "thread-2"}, checkpoint.Metadata{Source: "loop", Step: 1})
	id3 := put(checkpoint.Config{ThreadID: "thread-2", CheckpointNS: "inner"}, checkpoint.Metadata{Source: "loop", Step: 1})

	list := func(cfg checkpoint.Config, filter map[string]any) []checkpoint.Tuple {
		t.Helper()
		tups, err := s.List(ctx, cfg, checkpoint.ListOptions{Filter: filter})
		if err != nil {
			t.Fatalf("List(%+v, filter=%v): %v", cfg, filter, err)
		}
		return tups
	}

	// query_1 {"source": "input"}: exactly the thread-1 checkpoint.
	if tups := list(checkpoint.Config{ThreadID: "thread-1"}, map[string]any{"source": "input"}); len(tups) != 1 || tups[0].Checkpoint.ID != id1 {
		t.Fatalf("List(source=input) = %v, want [%q]", tupleIDs(tups), id1)
	}
	// query_2 {"step": 1}: exactly the thread-2 root-namespace checkpoint.
	if tups := list(checkpoint.Config{ThreadID: "thread-2"}, map[string]any{"step": 1}); len(tups) != 1 || tups[0].Checkpoint.ID != id2 {
		t.Fatalf("List(step=1) = %v, want [%q]", tupleIDs(tups), id2)
	}
	// query_3 {} (empty filter): everything in the listed thread/namespace.
	if tups := list(checkpoint.Config{ThreadID: "thread-1"}, map[string]any{}); len(tups) != 1 {
		t.Fatalf("List(thread-1, empty filter) = %v, want 1 tuple", tupleIDs(tups))
	}
	if tups := list(checkpoint.Config{ThreadID: "thread-2"}, map[string]any{}); len(tups) != 1 {
		t.Fatalf("List(thread-2 root, empty filter) = %v, want 1 tuple", tupleIDs(tups))
	}
	if tups := list(checkpoint.Config{ThreadID: "thread-2", CheckpointNS: "inner"}, map[string]any{}); len(tups) != 1 || tups[0].Checkpoint.ID != id3 {
		t.Fatalf("List(thread-2 inner, empty filter) = %v, want [%q]", tupleIDs(tups), id3)
	}
	// query_4 {"source": "update", "step": 1}: no match.
	if tups := list(checkpoint.Config{ThreadID: "thread-2"}, map[string]any{"source": "update", "step": 1}); len(tups) != 0 {
		t.Fatalf("List(source=update,step=1) = %v, want empty", tupleIDs(tups))
	}
	// Listing thread-2 without a namespace returns only the root-namespace
	// checkpoint (Go exact-ns semantics; Python would return both).
	if tups := list(checkpoint.Config{ThreadID: "thread-2"}, nil); len(tups) != 1 || tups[0].Checkpoint.ID != id2 {
		t.Fatalf("List(thread-2, no ns) = %v, want [%q] (root ns only)", tupleIDs(tups), id2)
	}
}

// testPutWritesRoundTrip: PutWrites writes are visible on
// GetTuple.PendingWrites, stamped with the given taskID, in insertion order.
// (Plain channels only: reserved-channel writes occupy fixed negative slots
// that sort before positional ones under ORDER BY task_id, idx, so mixing
// them would make insertion order backend-specific.)
func testPutWritesRoundTrip(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{Source: "loop", Step: 0}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	writes := []checkpoint.Write{
		{Channel: "a", Value: "va"},
		{Channel: "b", Value: 2},
		{Channel: "c", Value: []string{"x"}},
	}
	if err := s.PutWrites(ctx, cfg, writes, "task-9", ""); err != nil {
		t.Fatalf("PutWrites: %v", err)
	}

	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if len(tup.PendingWrites) != len(writes) {
		t.Fatalf("PendingWrites = %+v, want %d writes", tup.PendingWrites, len(writes))
	}
	for i, w := range tup.PendingWrites {
		if w.TaskID != "task-9" {
			t.Fatalf("PendingWrites[%d].TaskID = %q, want task-9", i, w.TaskID)
		}
		if w.Channel != writes[i].Channel || !reflect.DeepEqual(w.Value, writes[i].Value) {
			t.Fatalf("PendingWrites[%d] = %+v, want channel %q value %v (insertion order)", i, w, writes[i].Channel, writes[i].Value)
		}
	}
}

// testPutWritesBatchRule pins Python's put_writes dedup rule: re-writing the
// same task's regular channels is ignored (first-write-wins); a batch whose
// channels are ALL reserved REPLACES the stored value. (Python's
// SqliteSaver/PostgresSaver decide per batch via INSERT OR IGNORE /
// INSERT OR REPLACE; InMemorySaver dedups per write slot — both agree on
// this scenario.)
func testPutWritesBatchRule(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Mixed batch (state key + __tasks__, both positional since __tasks__ is
	// not a reserved slot): duplicate PutWrites for the same task are
	// ignored — the first write wins.
	mixed := []checkpoint.Write{
		{Channel: "state_key", Value: "first"},
		{Channel: checkpoint.ReservedTasks, Value: types.Send{Node: "n1"}},
	}
	if err := s.PutWrites(ctx, cfg, mixed, "task-1", ""); err != nil {
		t.Fatalf("PutWrites mixed: %v", err)
	}
	dupe := []checkpoint.Write{
		{Channel: "state_key", Value: "second"},
		{Channel: checkpoint.ReservedTasks, Value: types.Send{Node: "n2"}},
	}
	if err := s.PutWrites(ctx, cfg, dupe, "task-1", ""); err != nil {
		t.Fatalf("PutWrites duplicate mixed: %v", err)
	}

	// all-reserved batch: re-writing the same task REPLACES the stored value.
	if err := s.PutWrites(ctx, cfg, []checkpoint.Write{
		{Channel: checkpoint.ReservedInterrupt, Value: types.Interrupt{Value: "v1", ID: "i1"}},
	}, "task-2", ""); err != nil {
		t.Fatalf("PutWrites reserved: %v", err)
	}
	if err := s.PutWrites(ctx, cfg, []checkpoint.Write{
		{Channel: checkpoint.ReservedInterrupt, Value: types.Interrupt{Value: "v2", ID: "i1"}},
	}, "task-2", ""); err != nil {
		t.Fatalf("PutWrites reserved replace: %v", err)
	}

	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	byChannel := map[string]any{}
	for _, w := range tup.PendingWrites {
		if w.TaskID == "" {
			t.Fatalf("write not stamped with task ID: %+v", w)
		}
		byChannel[w.Channel] = w.Value
	}
	if got := byChannel["state_key"]; got != "first" {
		t.Fatalf("state_key = %v, want %q (first write wins)", got, "first")
	}
	if got, ok := byChannel[checkpoint.ReservedTasks].(types.Send); !ok || got.Node != "n1" {
		t.Fatalf("__tasks__ = %v, want Send{n1} (ignored duplicate)", byChannel[checkpoint.ReservedTasks])
	}
	got, ok := byChannel[checkpoint.ReservedInterrupt].(types.Interrupt)
	if !ok || got.Value != "v2" {
		t.Fatalf("__interrupt__ = %v, want Interrupt{v2} (replaced)", byChannel[checkpoint.ReservedInterrupt])
	}
	if len(tup.PendingWrites) != 3 {
		t.Fatalf("PendingWrites = %+v, want exactly 3 rows (duplicates ignored/replaced)", tup.PendingWrites)
	}
}

// testPutWritesTasksAllSurvive pins the Python-parity __tasks__ rule:
// `__tasks__` is NOT in WRITES_IDX_MAP, so several `__tasks__` writes in ONE
// batch (a multi-destination Command.Goto's routing) each take their own
// positional idx and ALL survive the round trip, in insertion order. (A
// reserved-slot mapping would collapse the batch to a single row.)
func testPutWritesTasksAllSurvive(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	writes := []checkpoint.Write{
		{Channel: checkpoint.ReservedTasks, Value: types.Send{Node: "n1"}},
		{Channel: checkpoint.ReservedTasks, Value: types.Send{Node: "n2"}},
		{Channel: checkpoint.ReservedTasks, Value: types.Send{Node: "n3"}},
	}
	if err := s.PutWrites(ctx, cfg, writes, "task-1", ""); err != nil {
		t.Fatalf("PutWrites: %v", err)
	}

	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if len(tup.PendingWrites) != len(writes) {
		t.Fatalf("PendingWrites = %+v, want all %d __tasks__ writes to survive", tup.PendingWrites, len(writes))
	}
	for i, w := range tup.PendingWrites {
		if w.TaskID != "task-1" || w.Channel != checkpoint.ReservedTasks {
			t.Fatalf("PendingWrites[%d] = %+v, want task-1 __tasks__ write", i, w)
		}
		want := writes[i].Value.(types.Send)
		if got, ok := w.Value.(types.Send); !ok || got.Node != want.Node {
			t.Fatalf("PendingWrites[%d].Value = %v, want %v (insertion order)", i, w.Value, want)
		}
	}
}

// testPutWritesTaskPath: PutWrites stamps every write with the given
// taskPath, and it round-trips through GetTuple.
func testPutWritesTaskPath(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	writes := []checkpoint.Write{
		{Channel: "a", Value: 1},
		{Channel: "b", Value: 2},
	}
	if err := s.PutWrites(ctx, cfg, writes, "task-1", "path/a"); err != nil {
		t.Fatalf("PutWrites: %v", err)
	}
	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if len(tup.PendingWrites) != len(writes) {
		t.Fatalf("PendingWrites = %+v, want %d writes", tup.PendingWrites, len(writes))
	}
	for i, w := range tup.PendingWrites {
		if w.TaskPath != "path/a" {
			t.Fatalf("PendingWrites[%d].TaskPath = %q, want %q", i, w.TaskPath, "path/a")
		}
	}
}

// testPutWritesMissingCheckpoint: PutWrites against an unknown checkpoint is
// an error.
func testPutWritesMissingCheckpoint(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)
	err := s.PutWrites(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: "nope"}, []checkpoint.Write{{Channel: "c", Value: 1}}, "task-1", "")
	if err == nil {
		t.Fatalf("PutWrites against unknown checkpoint: got nil error")
	}
}

// testDeleteThread verifies DeleteThread removes a thread's checkpoints and
// writes without touching other threads.
func testDeleteThread(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	for _, thread := range []string{"t1", "t2"} {
		cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: thread}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
		if err != nil {
			t.Fatalf("Put %s: %v", thread, err)
		}
		if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c", Value: "v"}}, "task-1", ""); err != nil {
			t.Fatalf("PutWrites %s: %v", thread, err)
		}
	}

	if err := s.DeleteThread(ctx, "t1"); err != nil {
		t.Fatalf("DeleteThread: %v", err)
	}
	tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup != nil {
		t.Fatalf("deleted thread: GetTuple = %v, %v; want nil, nil", tup, err)
	}
	list, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil || len(list) != 0 {
		t.Fatalf("deleted thread: List = %v, %v; want empty", list, err)
	}
	other, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t2"})
	if err != nil || other == nil {
		t.Fatalf("other thread must survive: tup=%v err=%v", other, err)
	}
	if len(other.PendingWrites) != 1 {
		t.Fatalf("other thread's writes must survive: %+v", other.PendingWrites)
	}
}

// testConcurrentPut hammers one Saver from multiple goroutines (distinct
// threads) and verifies all writes land.
func testConcurrentPut(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	const goroutines = 8
	const putsPerGoroutine = 5
	errs := make(chan error, goroutines*putsPerGoroutine)
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			threadID := fmt.Sprintf("thread-%d", g)
			cfg := checkpoint.Config{ThreadID: threadID}
			for i := 0; i < putsPerGoroutine; i++ {
				cp := sampleCheckpoint(checkpoint.NewID(g*putsPerGoroutine + i + 1))
				next, err := s.Put(ctx, cfg, cp, checkpoint.Metadata{Source: "loop", Step: i}, nil)
				if err != nil {
					errs <- fmt.Errorf("Put %s: %w", threadID, err)
					return
				}
				if err := s.PutWrites(ctx, next, []checkpoint.Write{{Channel: "c", Value: i}}, "task-1", ""); err != nil {
					errs <- fmt.Errorf("PutWrites %s: %w", threadID, err)
					return
				}
				cfg = next
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent put: %v", err)
	}
	if t.Failed() {
		return
	}
	for g := 0; g < goroutines; g++ {
		list, err := s.List(ctx, checkpoint.Config{ThreadID: fmt.Sprintf("thread-%d", g)}, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("List thread-%d: %v", g, err)
		}
		if len(list) != putsPerGoroutine {
			t.Fatalf("thread-%d has %d checkpoints, want %d", g, len(list), putsPerGoroutine)
		}
	}
}

// testSerdeRoundTrip round-trips channel values covering the whole serde
// registry (messages.Message, []messages.Message, types.Send,
// types.Interrupt, time.Time, []byte, int64, int, []string), a planned task
// with a typed Arg, and pending-write values, asserting exact values after a
// Put/GetTuple/PutWrites cycle.
func testSerdeRoundTrip(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()
	s := newSaver(t)

	ts := time.Date(2026, 8, 8, 12, 0, 0, 123456789, time.UTC)
	cp := sampleCheckpoint(checkpoint.NewID(1))
	cp.TS = ts
	cp.ChannelValues = map[string]any{
		"msg":       messages.AI("hi"),
		"msgs":      []messages.Message{messages.Human("a"), messages.AI("b")},
		"send":      types.Send{Node: "n1", Arg: map[string]any{"x": "y"}},
		"interrupt": types.Interrupt{Value: "why?", ID: "int-1"},
		"when":      ts,
		"blob":      []byte{1, 2, 3},
		"big":       int64(1 << 62),
		"n":         42,
		"tags":      []string{"a", "b"},
	}
	cp.Next = []checkpoint.PlannedTask{
		{ID: "task-1", Node: "node1", Arg: map[string]any{"when": ts, "big": int64(9)}},
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{Source: "loop", Step: 0}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	writes := []checkpoint.Write{
		{Channel: "blob", Value: []byte{9, 8}},
		{Channel: "big", Value: int64(1 << 60)},
		{Channel: "tags", Value: []string{"x"}},
		{Channel: checkpoint.ReservedInterrupt, Value: types.Interrupt{Value: "v", ID: "i"}},
	}
	if err := s.PutWrites(ctx, cfg, writes, "task-1", ""); err != nil {
		t.Fatalf("PutWrites: %v", err)
	}

	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if !reflect.DeepEqual(tup.Checkpoint, cp) {
		t.Fatalf("checkpoint mismatch:\n got %+v\nwant %+v", tup.Checkpoint, cp)
	}
	if len(tup.PendingWrites) != len(writes) {
		t.Fatalf("PendingWrites = %+v, want %d writes", tup.PendingWrites, len(writes))
	}
	byChannel := map[string]any{}
	for _, w := range tup.PendingWrites {
		byChannel[w.Channel] = w.Value
	}
	for _, w := range writes {
		if got, ok := byChannel[w.Channel]; !ok || !reflect.DeepEqual(got, w.Value) {
			t.Fatalf("write channel %q = %v (%T), want %v (%T)", w.Channel, got, got, w.Value, w.Value)
		}
	}
}
