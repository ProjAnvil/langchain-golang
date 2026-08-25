package sqlite_test

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite"
)

// rawExec runs one statement against the database file at path through its
// own connection (the Saver's pool is unexported), for tests that must plant
// rows or schema shapes the Saver API cannot produce.
func rawExec(t *testing.T, path, query string, args ...any) {
	t.Helper()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open(%q): %v", path, err)
	}
	defer db.Close()
	if _, err := db.Exec(query, args...); err != nil {
		t.Fatalf("exec %q: %v", query, err)
	}
}

// TestNewErrors covers New's failure modes: a nil Serializer, a path whose
// parent directory does not exist (setup fails), and a database whose writes
// table is actually a view (the task_path migration cannot ALTER it).
func TestNewErrors(t *testing.T) {
	ser := serde.NewJSONSerializer()

	if _, err := sqlite.New(":memory:", nil); err == nil {
		t.Fatal("New with nil Serializer: expected error, got nil")
	}

	missing := filepath.Join(t.TempDir(), "no-such-dir", "checkpoints.db")
	if _, err := sqlite.New(missing, ser); err == nil {
		t.Fatalf("New(%q) with missing parent dir: expected error, got nil", missing)
	}

	// A pre-existing VIEW named writes makes CREATE TABLE IF NOT EXISTS a
	// no-op, then the task_path migration's ALTER TABLE fails because views
	// cannot be altered.
	path := dbPath(t)
	rawExec(t, path, `CREATE VIEW writes AS SELECT 'x' AS task_id`)
	if _, err := sqlite.New(path, ser); err == nil {
		t.Fatal("New with writes as a view: expected task_path migration error, got nil")
	} else if !strings.Contains(err.Error(), "task_path") {
		t.Fatalf("New with writes as a view: error = %v, want a task_path migration failure", err)
	}
}

// TestPutWritesResolvesLatestCheckpoint exercises resolveCheckpointID's
// empty-CheckpointID branch: the writes attach to the thread's LATEST
// checkpoint, and an empty thread is an error (matching MemorySaver).
func TestPutWritesResolvesLatestCheckpoint(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

	thread := checkpoint.Config{ThreadID: "t1"}
	first, err := s.Put(ctx, thread, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put first: %v", err)
	}
	second, err := s.Put(ctx, first, sampleCheckpoint(checkpoint.NewID(2)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put second: %v", err)
	}

	// No CheckpointID: resolves to the latest (second) checkpoint.
	if err := s.PutWrites(ctx, checkpoint.Config{ThreadID: "t1"}, []checkpoint.Write{{Channel: "c", Value: "v"}}, "task-1", ""); err != nil {
		t.Fatalf("PutWrites without CheckpointID: %v", err)
	}
	tup, err := s.GetTuple(ctx, second)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple(second): tup=%v err=%v", tup, err)
	}
	if len(tup.PendingWrites) != 1 || tup.PendingWrites[0].Channel != "c" {
		t.Fatalf("second PendingWrites = %+v, want the write on the latest checkpoint", tup.PendingWrites)
	}
	tup, err = s.GetTuple(ctx, first)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple(first): tup=%v err=%v", tup, err)
	}
	if len(tup.PendingWrites) != 0 {
		t.Fatalf("first PendingWrites = %+v, want none", tup.PendingWrites)
	}

	// No checkpoints at all: an error, not a silent insert.
	if err := s.PutWrites(ctx, checkpoint.Config{ThreadID: "empty"}, []checkpoint.Write{{Channel: "c", Value: "v"}}, "task-1", ""); err == nil {
		t.Fatal("PutWrites on a thread with no checkpoints: expected error, got nil")
	}
}

// TestPutMergesNewVersionsIntoNilChannelVersions covers the branch where the
// checkpoint has no ChannelVersions map but newVersions must be recorded: a
// fresh map is allocated and merged.
func TestPutMergesNewVersionsIntoNilChannelVersions(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

	cp := checkpoint.Checkpoint{V: 1, ID: checkpoint.NewID(1)}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, map[string]int64{"c": 4})
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if got := tup.Checkpoint.ChannelVersions["c"]; got != 4 {
		t.Fatalf("ChannelVersions = %v, want c:4 merged from newVersions", tup.Checkpoint.ChannelVersions)
	}
}

// TestPutEncodeErrors verifies that values the Serializer cannot round-trip
// (here a chan, which has no lossless JSON form) fail Put with a wrapped
// error — from a channel value and from a planned task's arg alike — instead
// of being persisted lossily.
func TestPutEncodeErrors(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

	cp := sampleCheckpoint(checkpoint.NewID(1))
	cp.ChannelValues["bad"] = make(chan int)
	if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, nil); err == nil {
		t.Fatal("Put with unserializable channel value: expected error, got nil")
	} else if !strings.Contains(err.Error(), `channel "bad"`) {
		t.Fatalf("Put error = %v, want it to name channel %q", err, "bad")
	}

	cp = sampleCheckpoint(checkpoint.NewID(2))
	cp.Next[0].Arg["bad"] = make(chan int)
	if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, nil); err == nil {
		t.Fatal("Put with unserializable task arg: expected error, got nil")
	} else if !strings.Contains(err.Error(), `arg "bad"`) {
		t.Fatalf("Put error = %v, want it to name arg %q", err, "bad")
	}
}

// TestPutWritesEncodeError verifies an unserializable write value fails
// PutWrites with a wrapped error naming the channel.
func TestPutWritesEncodeError(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	err = s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c", Value: make(chan int)}}, "task-1", "")
	if err == nil {
		t.Fatal("PutWrites with unserializable value: expected error, got nil")
	}
	if !strings.Contains(err.Error(), `channel "c"`) {
		t.Fatalf("PutWrites error = %v, want it to name channel %q", err, "c")
	}
}

// TestClosedSaverErrors verifies every Saver method reports the closed
// database instead of panicking or silently succeeding.
func TestClosedSaverErrors(t *testing.T) {
	ctx := context.Background()
	s, err := sqlite.New(":memory:", serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	cfg := checkpoint.Config{ThreadID: "t1", CheckpointID: "c1"}
	if _, err := s.GetTuple(ctx, cfg); err == nil {
		t.Error("GetTuple on closed Saver: expected error, got nil")
	}
	if _, err := s.List(ctx, cfg, checkpoint.ListOptions{}); err == nil {
		t.Error("List on closed Saver: expected error, got nil")
	}
	if _, err := s.Put(ctx, cfg, sampleCheckpoint("c2"), checkpoint.Metadata{}, nil); err == nil {
		t.Error("Put on closed Saver: expected error, got nil")
	}
	if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c", Value: 1}}, "task-1", ""); err == nil {
		t.Error("PutWrites (explicit ID) on closed Saver: expected error, got nil")
	}
	if err := s.PutWrites(ctx, checkpoint.Config{ThreadID: "t1"}, []checkpoint.Write{{Channel: "c", Value: 1}}, "task-1", ""); err == nil {
		t.Error("PutWrites (latest-ID resolution) on closed Saver: expected error, got nil")
	}
	if err := s.DeleteThread(ctx, "t1"); err == nil {
		t.Error("DeleteThread on closed Saver: expected error, got nil")
	}
	if err := s.DeleteForRuns(ctx, []string{"r1"}); err == nil {
		t.Error("DeleteForRuns on closed Saver: expected error, got nil")
	}
	if err := s.CopyThread(ctx, "t1", "t2"); err == nil {
		t.Error("CopyThread on closed Saver: expected error, got nil")
	}
	if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err == nil {
		t.Error("Prune on closed Saver: expected error, got nil")
	}
}

// TestManagementDroppedTables forces the management methods' statements to
// fail independently by dropping tables (or rejecting deletes via trigger)
// out from under an open Saver — the same technique as
// TestDeleteThreadDroppedTables.
func TestManagementDroppedTables(t *testing.T) {
	ctx := context.Background()

	// DeleteForRuns's first statement (the writes DELETE) selects from
	// checkpoints, so dropping either table fails it; a delete-rejecting
	// trigger on checkpoints fails the second statement after the first
	// succeeded (the trigger needs a matching row to fire on).
	t.Run("delete_for_runs writes dropped", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, `DROP TABLE writes`)
		if err := s.DeleteForRuns(ctx, []string{"r1"}); err == nil {
			t.Fatal("DeleteForRuns with dropped writes table: expected error, got nil")
		}
	})
	t.Run("delete_for_runs checkpoints delete rejected", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{RunID: "r1"}, nil); err != nil {
			t.Fatalf("Put: %v", err)
		}
		rawExec(t, path, `CREATE TRIGGER reject_cp_delete BEFORE DELETE ON checkpoints BEGIN SELECT RAISE(ABORT, 'rejected'); END`)
		if err := s.DeleteForRuns(ctx, []string{"r1"}); err == nil {
			t.Fatal("DeleteForRuns with rejecting checkpoints trigger: expected error, got nil")
		}
	})

	t.Run("copy_thread checkpoints dropped", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, `DROP TABLE checkpoints`)
		if err := s.CopyThread(ctx, "t1", "t2"); err == nil {
			t.Fatal("CopyThread with dropped checkpoints table: expected error, got nil")
		}
	})
	t.Run("copy_thread writes dropped", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, `DROP TABLE writes`)
		if err := s.CopyThread(ctx, "t1", "t2"); err == nil {
			t.Fatal("CopyThread with dropped writes table: expected error, got nil")
		}
	})

	t.Run("prune keep_latest writes dropped", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, `DROP TABLE writes`)
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err == nil {
			t.Fatal("Prune(keep_latest) with dropped writes table: expected error, got nil")
		}
	})
	t.Run("prune keep_latest checkpoints delete rejected", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		cfg := checkpoint.Config{ThreadID: "t1"}
		for i := 1; i <= 2; i++ {
			next, err := s.Put(ctx, cfg, sampleCheckpoint(checkpoint.NewID(i)), checkpoint.Metadata{}, nil)
			if err != nil {
				t.Fatalf("Put %d: %v", i, err)
			}
			cfg = next
		}
		rawExec(t, path, `CREATE TRIGGER reject_cp_delete BEFORE DELETE ON checkpoints BEGIN SELECT RAISE(ABORT, 'rejected'); END`)
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err == nil {
			t.Fatal("Prune(keep_latest) with rejecting checkpoints trigger: expected error, got nil")
		}
	})
	t.Run("prune delete checkpoints dropped", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, `DROP TABLE checkpoints`)
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneDeleteAll); err == nil {
			t.Fatal("Prune(delete) with dropped checkpoints table: expected error, got nil")
		}
	})
}

// TestDeleteThreadDroppedTables forces the two DELETE statements to fail
// independently by dropping each table out from under an open Saver.
func TestDeleteThreadDroppedTables(t *testing.T) {
	ctx := context.Background()

	t.Run("checkpoints dropped", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, `DROP TABLE checkpoints`)
		if err := s.DeleteThread(ctx, "t1"); err == nil {
			t.Fatal("DeleteThread with dropped checkpoints table: expected error, got nil")
		}
	})

	t.Run("writes dropped", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, `DROP TABLE writes`)
		if err := s.DeleteThread(ctx, "t1"); err == nil {
			t.Fatal("DeleteThread with dropped writes table: expected error, got nil")
		}
	})
}

// TestCorruptCheckpointRows plants checkpoints rows the Saver API could never
// write (foreign blob types, malformed JSON, unknown serde tags) and verifies
// reads fail with descriptive errors rather than returning corrupted state.
func TestCorruptCheckpointRows(t *testing.T) {
	ctx := context.Background()
	insert := `INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES ('t1', '', 'c1', NULL, ?, ?, ?)`

	tests := []struct {
		name    string
		typ     string
		blob    string
		md      any // metadata blob; nil = SQL NULL
		wantErr string
	}{
		{
			name:    "unknown blob type",
			typ:     "pickle",
			blob:    `{"v":1,"id":"c1"}`,
			wantErr: `unknown checkpoint blob type "pickle"`,
		},
		{
			name:    "malformed checkpoint JSON",
			typ:     "json",
			blob:    `not json`,
			wantErr: "decode checkpoint",
		},
		{
			name:    "unknown channel value serde tag",
			typ:     "json",
			blob:    `{"v":1,"id":"c1","channel_values":{"c":{"type":"bogus","data":{}}}}`,
			wantErr: `channel "c"`,
		},
		{
			name:    "unknown task arg serde tag",
			typ:     "json",
			blob:    `{"v":1,"id":"c1","next":[{"id":"t","node":"n","arg":{"a":{"type":"bogus","data":{}}}}]}`,
			wantErr: `arg "a"`,
		},
		{
			name:    "malformed metadata JSON",
			typ:     "json",
			blob:    `{"v":1,"id":"c1"}`,
			md:      `not json`,
			wantErr: "decode metadata",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := dbPath(t)
			s := newSaver(t, path)
			rawExec(t, path, insert, tt.typ, tt.blob, tt.md)

			if _, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"}); err == nil {
				t.Fatalf("GetTuple: expected error containing %q, got nil", tt.wantErr)
			} else if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("GetTuple error = %v, want it to contain %q", err, tt.wantErr)
			}
			// List must surface the same corruption instead of skipping it.
			if _, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{}); err == nil {
				t.Fatal("List: expected error on corrupt row, got nil")
			}
		})
	}
}

// TestNullMetadataAndTypeColumns covers two legacy-row shapes: a NULL
// metadata blob decodes as zero Metadata (databases written before metadata
// existed), while a NULL blob type is a scan error.
func TestNullMetadataAndTypeColumns(t *testing.T) {
	ctx := context.Background()
	insert := `INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES ('t1', '', 'c1', NULL, ?, ?, ?)`

	t.Run("null metadata decodes as zero", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, insert, "json", `{"v":1,"id":"c1"}`, nil)

		tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"})
		if err != nil || tup == nil {
			t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
		}
		if tup.Metadata.Source != "" || tup.Metadata.Step != 0 || tup.Metadata.Parents != nil {
			t.Fatalf("Metadata = %+v, want zero value for a NULL metadata blob", tup.Metadata)
		}
	})

	t.Run("null blob type is a scan error", func(t *testing.T) {
		path := dbPath(t)
		s := newSaver(t, path)
		rawExec(t, path, insert, nil, `{"v":1,"id":"c1"}`, nil)

		if _, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{}); err == nil {
			t.Fatal("List: expected scan error for NULL type column, got nil")
		}
	})
}

// TestCorruptWriteRows plants writes rows the Saver API could never write and
// verifies reads fail descriptively. A dropped writes table exercises the
// loadWrites query error path.
func TestCorruptWriteRows(t *testing.T) {
	ctx := context.Background()
	insertWrite := `INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, task_path) VALUES ('t1', '', ?, 'task-1', 0, 'c', ?, ?, '')`

	setup := func(t *testing.T) (*sqlite.Saver, string, checkpoint.Config) {
		t.Helper()
		path := dbPath(t)
		s := newSaver(t, path)
		cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		return s, path, cfg
	}

	t.Run("unknown serde tag", func(t *testing.T) {
		s, path, cfg := setup(t)
		rawExec(t, path, insertWrite, cfg.CheckpointID, "bogus", `{}`)
		if _, err := s.GetTuple(ctx, cfg); err == nil {
			t.Fatal("GetTuple: expected serde decode error, got nil")
		} else if !strings.Contains(err.Error(), `channel "c"`) {
			t.Fatalf("GetTuple error = %v, want it to name channel %q", err, "c")
		}
	})

	t.Run("null type column", func(t *testing.T) {
		s, path, cfg := setup(t)
		rawExec(t, path, insertWrite, cfg.CheckpointID, nil, `{}`)
		if _, err := s.GetTuple(ctx, cfg); err == nil {
			t.Fatal("GetTuple: expected scan error for NULL type column, got nil")
		}
	})

	t.Run("writes table dropped", func(t *testing.T) {
		s, path, cfg := setup(t)
		rawExec(t, path, `DROP TABLE writes`)
		if _, err := s.GetTuple(ctx, cfg); err == nil {
			t.Fatal("GetTuple: expected load-writes query error, got nil")
		}
	})

	t.Run("put writes with dropped writes table", func(t *testing.T) {
		s, path, cfg := setup(t)
		rawExec(t, path, `DROP TABLE writes`)
		// resolveCheckpointID reads the (intact) checkpoints table, then the
		// INSERT prepare fails on the missing writes table.
		if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c", Value: 1}}, "task-1", ""); err == nil {
			t.Fatal("PutWrites with dropped writes table: expected error, got nil")
		}
	})

	t.Run("insert rejected by trigger", func(t *testing.T) {
		s, path, cfg := setup(t)
		// A BEFORE INSERT trigger with RAISE(ABORT) fails the Exec even under
		// INSERT OR IGNORE conflict resolution.
		rawExec(t, path, `CREATE TRIGGER reject_writes BEFORE INSERT ON writes BEGIN SELECT RAISE(ABORT, 'rejected'); END`)
		err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c", Value: 1}}, "task-1", "")
		if err == nil {
			t.Fatal("PutWrites with rejecting trigger: expected error, got nil")
		}
		if !strings.Contains(err.Error(), `channel "c"`) {
			t.Fatalf("PutWrites error = %v, want it to name channel %q", err, "c")
		}
	})
}

// TestListFilterWithLimit covers the in-process LIMIT break: with a Filter,
// the SQL query has no LIMIT, so filtering and capping happen row by row.
func TestListFilterWithLimit(t *testing.T) {
	ctx := context.Background()
	s := newSaver(t, dbPath(t))

	cfg := checkpoint.Config{ThreadID: "t1"}
	for i := 1; i <= 3; i++ {
		next, err := s.Put(ctx, cfg, sampleCheckpoint(checkpoint.NewID(i)), checkpoint.Metadata{Source: "loop", Step: i - 1}, nil)
		if err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
		cfg = next
	}

	list, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{
		Filter: map[string]any{"source": "loop"},
		Limit:  2,
	})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("List with filter+limit returned %d tuples, want 2 (filter before limit)", len(list))
	}
	// Newest first: steps 2 and 1.
	for i, wantStep := range []int{2, 1} {
		if list[i].Metadata.Step != wantStep {
			t.Fatalf("list[%d].Metadata.Step = %d, want %d", i, list[i].Metadata.Step, wantStep)
		}
	}
}
