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
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/savertest"
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

// TestSaverContract runs the shared Saver contract suite against a fresh
// in-memory SQLite database per subtest.
func TestSaverContract(t *testing.T) {
	savertest.Run(t, func(t *testing.T) checkpoint.Saver {
		s, err := sqlite.New(":memory:", serde.NewJSONSerializer())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		t.Cleanup(func() { s.Close() })
		return s
	})
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
