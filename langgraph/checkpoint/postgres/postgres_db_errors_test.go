package postgres_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
)

// failSerde rejects every encode/decode — used to drive the serde error
// branches that the real JSON serializer cannot reach with ordinary values.
type failSerde struct{}

func (failSerde) DumpsTyped(any) (string, []byte, error) { return "", nil, errors.New("failSerde") }
func (failSerde) LoadsTyped(string, []byte) (any, error) { return nil, errors.New("failSerde") }

// newIsolatedPool creates a throwaway database on the shared embedded
// instance and returns a pool connected to it, so a test can destroy schema
// objects (drop tables, create conflicting relations) without affecting the
// other tests sharing the default database. Skips in -short mode via newPool.
func newIsolatedPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	ctx := context.Background()
	admin := newPool(t)
	name := fmt.Sprintf("pgiso_%d", time.Now().UnixNano())
	if _, err := admin.Exec(ctx, `CREATE DATABASE `+name); err != nil {
		t.Fatalf("create database %s: %v", name, err)
	}
	dsn := fmt.Sprintf("postgres://postgres:postgres@localhost:%d/%s?sslmode=disable", testPort, name)
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New(%s): %v", name, err)
	}
	t.Cleanup(func() {
		pool.Close()
		if _, err := admin.Exec(context.Background(), `DROP DATABASE `+name); err != nil {
			t.Logf("drop database %s: %v", name, err)
		}
	})
	return pool
}

// TestDeadPoolErrors: every public method must surface the connection error,
// not panic or hang, when the server is unreachable. pgxpool.New is lazy, so
// a pool on a dead port builds fine and fails on first use.
func TestDeadPoolErrors(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping embedded-postgres test in -short mode")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, "postgres://postgres:postgres@localhost:1/postgres?sslmode=disable")
	if err != nil {
		t.Fatalf("pgxpool.New (lazy) = %v, want nil error", err)
	}
	defer pool.Close()
	s := postgres.New(pool, serde.NewJSONSerializer())

	if err := s.Setup(ctx); err == nil {
		t.Error("Setup on a dead pool = nil, want a connect error")
	}
	if _, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: "cp1"}); err == nil {
		t.Error("GetTuple on a dead pool = nil, want a connect error")
	}
	if _, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{}); err == nil {
		t.Error("List on a dead pool = nil, want a connect error")
	}
	if err := s.DeleteThread(ctx, "t1"); err == nil {
		t.Error("DeleteThread on a dead pool = nil, want a connect error")
	}
	err = s.PutWrites(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: "cp1"},
		[]checkpoint.Write{{Channel: "a", Value: 1}}, "task-1", "")
	if err == nil {
		t.Error("PutWrites on a dead pool = nil, want a connect error")
	}
}

// TestSetupMigrationVersionReadError: when the migrations table exists but
// lacks the v column, Setup must fail reading the version — the CREATE TABLE
// IF NOT EXISTS is a no-op, so the SELECT is the first failing statement.
func TestSetupMigrationVersionReadError(t *testing.T) {
	pool := newIsolatedPool(t)
	ctx := context.Background()
	if _, err := pool.Exec(ctx, `CREATE TABLE checkpoint_migrations (x INTEGER PRIMARY KEY)`); err != nil {
		t.Fatalf("create broken migrations table: %v", err)
	}
	s := postgres.New(pool, serde.NewJSONSerializer())
	err := s.Setup(ctx)
	if err == nil || !strings.Contains(err.Error(), "read migration version") {
		t.Fatalf("Setup error = %v, want a read-migration-version error", err)
	}
}

// TestSetupMigrationExecError: a leftover VIEW named checkpoint_blobs makes
// v2's CREATE TABLE IF NOT EXISTS a no-op, then v4's ALTER TABLE fails — the
// migration error must name the failing version.
func TestSetupMigrationExecError(t *testing.T) {
	pool := newIsolatedPool(t)
	ctx := context.Background()
	if _, err := pool.Exec(ctx, postgres.Migrations[0]); err != nil {
		t.Fatalf("create migrations table: %v", err)
	}
	if _, err := pool.Exec(ctx, `CREATE VIEW checkpoint_blobs AS SELECT 1 AS v`); err != nil {
		t.Fatalf("create conflicting view: %v", err)
	}
	s := postgres.New(pool, serde.NewJSONSerializer())
	err := s.Setup(ctx)
	if err == nil || !strings.Contains(err.Error(), "migration 4") {
		t.Fatalf("Setup error = %v, want a migration-4 error", err)
	}
}

// TestSetupMigrationRecordError: an extra NOT NULL column on the migrations
// table lets every migration statement succeed (all are IF NOT EXISTS / no-op
// here) but breaks the version INSERT — Setup must report the record failure.
func TestSetupMigrationRecordError(t *testing.T) {
	pool := newIsolatedPool(t)
	ctx := context.Background()
	if _, err := pool.Exec(ctx,
		`CREATE TABLE checkpoint_migrations (v INTEGER PRIMARY KEY, extra TEXT NOT NULL)`); err != nil {
		t.Fatalf("create rigged migrations table: %v", err)
	}
	s := postgres.New(pool, serde.NewJSONSerializer())
	err := s.Setup(ctx)
	if err == nil || !strings.Contains(err.Error(), "record migration 0") {
		t.Fatalf("Setup error = %v, want a record-migration-0 error", err)
	}
}

// TestPutWritesResolvesLatestCheckpoint: an empty cfg.CheckpointID resolves
// to the thread's latest checkpoint (MemorySaver parity), so the writes land
// on the most recent row.
func TestPutWritesResolvesLatestCheckpoint(t *testing.T) {
	s := newEmptySaver(t)
	ctx := context.Background()

	cp := checkpoint.Checkpoint{
		V:  1,
		ID: checkpoint.NewID(1),
		TS: time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
	}
	if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t-resolve"}, cp, checkpoint.Metadata{}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}
	threadOnly := checkpoint.Config{ThreadID: "t-resolve"} // no CheckpointID
	if err := s.PutWrites(ctx, threadOnly,
		[]checkpoint.Write{{Channel: "a", Value: 42}}, "task-1", ""); err != nil {
		t.Fatalf("PutWrites without checkpoint ID: %v", err)
	}
	tup, err := s.GetTuple(ctx, threadOnly)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if len(tup.PendingWrites) != 1 || tup.PendingWrites[0].Channel != "a" || tup.PendingWrites[0].Value != 42 {
		t.Fatalf("PendingWrites = %+v, want one write {a 42} on the latest checkpoint", tup.PendingWrites)
	}
}

// TestPutWritesEncodeError: with the checkpoint resolvable, a serde failure
// on a write value must surface as an encode error naming the channel.
func TestPutWritesEncodeError(t *testing.T) {
	s := newEmptySaver(t)
	pool := newPool(t)
	ctx := context.Background()

	cp := checkpoint.Checkpoint{
		V:  1,
		ID: checkpoint.NewID(1),
		TS: time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t-enc"}, cp, checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	failing := postgres.New(pool, failSerde{})
	err = failing.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "ch", Value: 1}}, "task-1", "")
	if err == nil || !strings.Contains(err.Error(), `encode write 0 to channel "ch"`) {
		t.Fatalf("PutWrites error = %v, want an encode error for channel ch", err)
	}
}

// TestListUnmarshalableFilter: a filter value that cannot marshal to JSON is
// an error, never silently dropped.
func TestListUnmarshalableFilter(t *testing.T) {
	s := newEmptySaver(t)
	_, err := s.List(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{
		Filter: map[string]any{"bad": func() {}},
	})
	if err == nil || !strings.Contains(err.Error(), "encode list filter") {
		t.Fatalf("List error = %v, want an encode-list-filter error", err)
	}
}

// TestEmptyBlobTypeSkipped: blob rows with type "empty" are tombstones
// (Python's _load_blobs) — they are skipped, not decoded, so the channel is
// simply absent from the restored checkpoint.
func TestEmptyBlobTypeSkipped(t *testing.T) {
	s := newEmptySaver(t)
	pool := newPool(t)
	ctx := context.Background()

	cp := checkpoint.Checkpoint{
		V:               1,
		ID:              checkpoint.NewID(1),
		TS:              time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
		ChannelValues:   map[string]any{"ch": []string{"a"}},
		ChannelVersions: map[string]int64{"ch": 5},
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t-emptyblob"}, cp, checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := pool.Exec(ctx,
		`UPDATE checkpoint_blobs SET type = 'empty' WHERE thread_id = 't-emptyblob'`); err != nil {
		t.Fatalf("mark blob empty: %v", err)
	}
	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if v, ok := tup.Checkpoint.ChannelValues["ch"]; ok {
		t.Fatalf("channel ch = %v, want absent (empty blob rows are skipped)", v)
	}
}

// TestCorruptedStorageErrors pins fail-loud behavior when rows in the
// database cannot be decoded: unknown serde type tags on blobs/writes and
// JSONB values that are not objects in the checkpoints/metadata columns.
func TestCorruptedStorageErrors(t *testing.T) {
	ctx := context.Background()

	putCheckpoint := func(t *testing.T, threadID string) checkpoint.Config {
		t.Helper()
		s := newEmptySaver(t)
		cp := checkpoint.Checkpoint{
			V:  1,
			ID: checkpoint.NewID(1),
			TS: time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
		}
		cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: threadID}, cp, checkpoint.Metadata{}, nil)
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		return cfg
	}

	t.Run("unknown blob type", func(t *testing.T) {
		s := newEmptySaver(t)
		pool := newPool(t)
		cp := checkpoint.Checkpoint{
			V:               1,
			ID:              checkpoint.NewID(1),
			TS:              time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
			ChannelValues:   map[string]any{"ch": []string{"a"}},
			ChannelVersions: map[string]int64{"ch": 1},
		}
		cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t-badtype"}, cp, checkpoint.Metadata{}, nil)
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		if _, err := pool.Exec(ctx,
			`UPDATE checkpoint_blobs SET type = 'bogus' WHERE thread_id = 't-badtype'`); err != nil {
			t.Fatalf("corrupt blob type: %v", err)
		}
		_, err = s.GetTuple(ctx, cfg)
		if err == nil || !strings.Contains(err.Error(), `decode channel "ch"`) {
			t.Fatalf("GetTuple error = %v, want a decode-channel error", err)
		}
	})

	t.Run("unknown write type", func(t *testing.T) {
		s := newEmptySaver(t)
		pool := newPool(t)
		cfg := putCheckpoint(t, "t-badwrite")
		if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "a", Value: 1}}, "task-1", ""); err != nil {
			t.Fatalf("PutWrites: %v", err)
		}
		if _, err := pool.Exec(ctx,
			`UPDATE checkpoint_writes SET type = 'bogus' WHERE thread_id = 't-badwrite'`); err != nil {
			t.Fatalf("corrupt write type: %v", err)
		}
		_, err := s.GetTuple(ctx, cfg)
		if err == nil || !strings.Contains(err.Error(), `decode write to channel "a"`) {
			t.Fatalf("GetTuple error = %v, want a decode-write error", err)
		}
	})

	t.Run("non-object metadata", func(t *testing.T) {
		s := newEmptySaver(t)
		pool := newPool(t)
		cfg := putCheckpoint(t, "t-badmd")
		if _, err := pool.Exec(ctx,
			`UPDATE checkpoints SET metadata = '"oops"'::jsonb WHERE thread_id = 't-badmd'`); err != nil {
			t.Fatalf("corrupt metadata: %v", err)
		}
		_, err := s.GetTuple(ctx, cfg)
		if err == nil || !strings.Contains(err.Error(), "decode metadata") {
			t.Fatalf("GetTuple error = %v, want a decode-metadata error", err)
		}
	})

	t.Run("non-object checkpoint document", func(t *testing.T) {
		s := newEmptySaver(t)
		pool := newPool(t)
		cfg := putCheckpoint(t, "t-badcp")
		if _, err := pool.Exec(ctx,
			`UPDATE checkpoints SET checkpoint = '"oops"'::jsonb WHERE thread_id = 't-badcp'`); err != nil {
			t.Fatalf("corrupt checkpoint document: %v", err)
		}
		_, err := s.GetTuple(ctx, cfg)
		if err == nil || !strings.Contains(err.Error(), "decode checkpoint") {
			t.Fatalf("GetTuple error = %v, want a decode-checkpoint error", err)
		}
		if _, err := s.List(ctx, checkpoint.Config{ThreadID: "t-badcp"}, checkpoint.ListOptions{}); err == nil {
			t.Fatal("List over a corrupted row = nil error, want the decode error to propagate")
		}
	})
}

// TestDroppedBlobsTableError: GetTuple must surface a load-blobs error when
// the blobs table is gone (the checkpoints row itself reads fine).
func TestDroppedBlobsTableError(t *testing.T) {
	pool := newIsolatedPool(t)
	ctx := context.Background()
	s := postgres.New(pool, serde.NewJSONSerializer())
	if err := s.Setup(ctx); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	cp := checkpoint.Checkpoint{
		V:               1,
		ID:              checkpoint.NewID(1),
		TS:              time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
		ChannelValues:   map[string]any{"ch": []string{"a"}},
		ChannelVersions: map[string]int64{"ch": 1},
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := pool.Exec(ctx, `DROP TABLE checkpoint_blobs`); err != nil {
		t.Fatalf("drop checkpoint_blobs: %v", err)
	}
	_, err = s.GetTuple(ctx, cfg)
	if err == nil || !strings.Contains(err.Error(), "load blobs") {
		t.Fatalf("GetTuple error = %v, want a load-blobs error", err)
	}
}

// TestDroppedWritesTableError: with the writes table gone, GetTuple fails in
// loadWrites, and PutWrites — which resolves the checkpoint ID from the
// intact checkpoints table first — fails on the batch exec.
func TestDroppedWritesTableError(t *testing.T) {
	pool := newIsolatedPool(t)
	ctx := context.Background()
	s := postgres.New(pool, serde.NewJSONSerializer())
	if err := s.Setup(ctx); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	cp := checkpoint.Checkpoint{
		V:  1,
		ID: checkpoint.NewID(1),
		TS: time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := pool.Exec(ctx, `DROP TABLE checkpoint_writes`); err != nil {
		t.Fatalf("drop checkpoint_writes: %v", err)
	}
	_, err = s.GetTuple(ctx, cfg)
	if err == nil || !strings.Contains(err.Error(), "load writes") {
		t.Fatalf("GetTuple error = %v, want a load-writes error", err)
	}
	err = s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "a", Value: 1}}, "task-1", "")
	if err == nil || !strings.Contains(err.Error(), "put writes") {
		t.Fatalf("PutWrites error = %v, want a put-writes error", err)
	}
}

// TestDeleteThreadDroppedTableError: DeleteThread deletes the three tables in
// order inside one transaction; a missing table must abort with an error.
func TestDeleteThreadDroppedTableError(t *testing.T) {
	pool := newIsolatedPool(t)
	ctx := context.Background()
	s := postgres.New(pool, serde.NewJSONSerializer())
	if err := s.Setup(ctx); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	if _, err := pool.Exec(ctx, `DROP TABLE checkpoint_writes`); err != nil {
		t.Fatalf("drop checkpoint_writes: %v", err)
	}
	err := s.DeleteThread(ctx, "t1")
	if err == nil || !strings.Contains(err.Error(), `delete thread "t1"`) {
		t.Fatalf("DeleteThread error = %v, want a delete-thread error", err)
	}
}
