package sqlite_test

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// newSaver opens a Saver on path and registers its cleanup.
func newSaver(t *testing.T, path string) *sqlite.Saver {
	t.Helper()
	s, err := sqlite.New(path, serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("New(%q): %v", path, err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// dbPath returns a fresh database file path inside the test's temp dir.
func dbPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "checkpoints.db")
}

// sampleCheckpoint returns a checkpoint exercising every projection field:
// channel values spanning plain JSON and registry types, version maps, a
// planned task with a typed arg, and fixed timestamps for exact comparison.
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
		},
		ChannelVersions: map[string]int64{"msgs": 3, "note": 1},
		VersionsSeen:    map[string]map[string]int64{"node1": {"msgs": 2}},
		Next: []checkpoint.PlannedTask{
			{ID: "task-1", Node: "node1", Arg: map[string]any{"limit": 3}},
		},
	}
}

// TestPutGetRoundTrip verifies that a checkpoint with messages, interrupts,
// typed slices and ints in its channel values survives a Put/GetTuple cycle
// byte-for-value, addressable both as "latest" and by ID.
func TestPutGetRoundTrip(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

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

// TestIntChannelValuesSurviveAsInt pins the serde contract that plain `int`
// and `int64` channel values do not degrade to float64 through SQLite.
func TestIntChannelValuesSurviveAsInt(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

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

// TestParentLinks verifies D3: a Put whose config carries a CheckpointID
// records that position as the new checkpoint's ParentConfig.
func TestParentLinks(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

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

// TestListNewestFirstBeforeLimit verifies List ordering (newest checkpoint ID
// first) and the Before/Limit filters.
func TestListNewestFirstBeforeLimit(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

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

// TestPutWritesBatchRule pins the Python BATCH-level insert rule: a batch
// mixing state keys with reserved channels uses INSERT OR IGNORE
// (first-write-wins); a batch whose channels are ALL reserved uses
// INSERT OR REPLACE under the reserved negative idx.
func TestPutWritesBatchRule(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Mixed batch (state key + reserved __tasks__): duplicate PutWrites for
	// the same task are ignored — the first write wins.
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

	// All-reserved batch: re-writing the same task REPLACES the stored value.
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
		t.Fatalf("state_key = %v, want %q (INSERT OR IGNORE: first write wins)", got, "first")
	}
	if got, ok := byChannel[checkpoint.ReservedTasks].(types.Send); !ok || got.Node != "n1" {
		t.Fatalf("__tasks__ = %v, want Send{n1} (ignored duplicate)", byChannel[checkpoint.ReservedTasks])
	}
	got, ok := byChannel[checkpoint.ReservedInterrupt].(types.Interrupt)
	if !ok || got.Value != "v2" {
		t.Fatalf("__interrupt__ = %v, want Interrupt{v2} (INSERT OR REPLACE)", byChannel[checkpoint.ReservedInterrupt])
	}
	if len(tup.PendingWrites) != 3 {
		t.Fatalf("PendingWrites = %+v, want exactly 3 rows (duplicates ignored/replaced)", tup.PendingWrites)
	}
}

// TestPutWritesMissingCheckpoint verifies PutWrites against an unknown
// checkpoint is an error, matching MemorySaver.
func TestPutWritesMissingCheckpoint(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))
	err := s.PutWrites(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: "nope"}, []checkpoint.Write{{Channel: "c", Value: 1}}, "task-1", "")
	if err == nil {
		t.Fatalf("PutWrites against unknown checkpoint: got nil error")
	}
}

// TestTaskPathColumnMigration verifies D3: a database whose writes table was
// created before the M5 schema evolution (no task_path column) is upgraded in
// place on open (ALTER TABLE): pre-existing rows read back with an empty
// TaskPath, and new PutWrites round-trip their task path.
func TestTaskPathColumnMigration(t *testing.T) {
	ctx := context.Background()
	path := dbPath(t)
	cpID := checkpoint.NewID(1)

	// Build an old-schema database: the pre-M5 writes table (no task_path
	// column) with one row.
	ser := serde.NewJSONSerializer()
	typ, data, err := ser.DumpsTyped("old-value")
	if err != nil {
		t.Fatalf("DumpsTyped: %v", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	if _, err := db.Exec(`CREATE TABLE writes (
	    thread_id TEXT NOT NULL,
	    checkpoint_ns TEXT NOT NULL DEFAULT '',
	    checkpoint_id TEXT NOT NULL,
	    task_id TEXT NOT NULL,
	    idx INTEGER NOT NULL,
	    channel TEXT NOT NULL,
	    type TEXT,
	    value BLOB,
	    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
	)`); err != nil {
		t.Fatalf("create old writes table: %v", err)
	}
	if _, err := db.Exec(`INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"t1", "", cpID, "old-task", 0, "c", typ, data); err != nil {
		t.Fatalf("insert old row: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close old db: %v", err)
	}

	// Reopen through the Saver: the missing column is added on setup.
	s := newSaver(t, path)
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(cpID), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	// The pre-existing row reads back with an empty TaskPath.
	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if len(tup.PendingWrites) != 1 || tup.PendingWrites[0].TaskID != "old-task" || tup.PendingWrites[0].TaskPath != "" {
		t.Fatalf("old row = %+v, want task old-task with empty TaskPath", tup.PendingWrites)
	}

	// New writes round-trip their task path.
	if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c2", Value: "new"}}, "task-2", "p"); err != nil {
		t.Fatalf("PutWrites: %v", err)
	}
	tup, err = s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if len(tup.PendingWrites) != 2 {
		t.Fatalf("PendingWrites = %+v, want 2 writes", tup.PendingWrites)
	}
	byTask := map[string]checkpoint.Write{}
	for _, w := range tup.PendingWrites {
		byTask[w.TaskID] = w
	}
	if byTask["task-2"].TaskPath != "p" {
		t.Fatalf("task-2 TaskPath = %q, want %q", byTask["task-2"].TaskPath, "p")
	}
}

// TestDeleteThread verifies DeleteThread removes a thread's checkpoints and
// writes without touching other threads.
func TestDeleteThread(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

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

// TestConcurrentAccess hammers a single WAL-mode database file from multiple
// goroutines (distinct threads) and verifies all writes land.
func TestConcurrentAccess(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

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
		t.Errorf("concurrent access: %v", err)
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

// TestInMemoryDatabase verifies `:memory:` databases work end to end.
func TestInMemoryDatabase(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, ":memory:")

	cp := sampleCheckpoint(checkpoint.NewID(1))
	if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{Source: "input", Step: -1}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if !reflect.DeepEqual(tup.Checkpoint, cp) {
		t.Fatalf("checkpoint mismatch:\n got %+v\nwant %+v", tup.Checkpoint, cp)
	}
}

// TestReducerChannelRoundTrip is the review-mandated critical test: an
// AppendSliceReducer channel holding a typed []string is checkpointed to a
// SQLite file, the saver is closed and REOPENED, and the run continues —
// folding more writes into the restored slice. If the serde corrupted
// []string into []any, AppendSliceReducer's type check would fail the fold.
func TestReducerChannelRoundTrip(t *testing.T) {
	ctx := context.Background()
	path := dbPath(t)

	calls := 0
	build := func(saver checkpoint.Saver) *graph.CompiledGraph {
		t.Helper()
		g := graph.NewStateGraph()
		g.AddReducer("log", channels.AppendSliceReducer)
		g.AddNode("record", func(_ context.Context, _ map[string]any) (any, error) {
			calls++
			return map[string]any{"log": []string{fmt.Sprintf("call-%d", calls)}}, nil
		})
		g.AddEdge(types.START, "record")
		g.AddEdge("record", types.END)
		cg, err := g.Compile(graph.WithCheckpointer(saver))
		if err != nil {
			t.Fatalf("Compile: %v", err)
		}
		return cg
	}

	// First turn: checkpoint the []string channel value to disk.
	s1, err := sqlite.New(path, serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res1, err := build(s1).InvokeWithOptions(ctx, map[string]any{}, graph.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke: %v", err)
	}
	if got, ok := res1.Values["log"].([]string); !ok || !reflect.DeepEqual(got, []string{"call-1"}) {
		t.Fatalf("first turn log = %v (%T), want []string{call-1}", res1.Values["log"], res1.Values["log"])
	}
	if err := s1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Second turn, after a close/reopen: state is restored through the SQLite
	// saver and both the input write and the node's write must fold into the
	// restored []string (non-empty input starts a NEW turn per D2).
	s2 := newSaver(t, path)
	res2, err := build(s2).InvokeWithOptions(ctx, map[string]any{"log": []string{"from-input"}}, graph.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("second Invoke (fold into restored []string): %v", err)
	}
	want := []string{"call-1", "from-input", "call-2"}
	got, ok := res2.Values["log"].([]string)
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("second turn log = %v (%T), want %v", res2.Values["log"], res2.Values["log"], want)
	}
}
