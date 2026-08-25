package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
	"time"

	_ "modernc.org/sqlite" // database/sql driver name `sqlite` (pure Go, no cgo)

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// setupSQL mirrors Python's `SqliteSaver.setup` exactly: WAL journal mode and
// the two-table schema (`libs/checkpoint-sqlite/.../sqlite/__init__.py`).
const setupSQL = `
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    task_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
`

const (
	insertCheckpointSQL = `INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)`
	insertOrIgnoreSQL   = `INSERT OR IGNORE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, task_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
	insertOrReplaceSQL  = `INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, task_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
	selectWritesSQL     = `SELECT task_id, task_path, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx`
)

// checkpointBlobType is the `type` column value for checkpoint blobs: the
// storedCheckpoint projection encoded as plain JSON.
const checkpointBlobType = "json"

// reservedWriteIdx maps reserved channels to the negative write idx used with
// INSERT OR REPLACE, approximating Python's WRITES_IDX_MAP
// `{ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}`
// (`libs/checkpoint/langgraph/checkpoint/base/__init__.py:795`): the Go
// runtime has no SCHEDULED counterpart, so the -2 slot stays empty, and
// checkpoint.ReservedError is currently unused by the executor — its mapping
// is kept for forward compatibility.
//
// TASKS is NOT in Python's map — Python's TASKS writes go through the
// positional idx — and neither is Go's `__tasks__`: it takes the positional
// idx like any regular channel, so multiple `__tasks__` writes in one batch
// (a multi-destination Command.Goto) each occupy their own slot and ALL
// survive, matching Python and MemorySaver. (An earlier revision assigned
// `__tasks__` the reserved idx -2, collapsing same-batch `__tasks__` writes
// to one row; see doc.go for the pre-1.0 behavior-change note.)
// `__resume__` at idx -4 is lossless by construction: like Python
// (one RESUME write per interrupt() call whose value is the WHOLE
// accumulated scratchpad list, types.py:905-925), the Go executor persists
// the consumed resume prefix as a SINGLE write carrying the full ordered
// list (see graph.persistInterruptAndResume), so last-write-wins has
// nothing to collapse.
var reservedWriteIdx = map[string]int{
	checkpoint.ReservedError:     -1,
	checkpoint.ReservedInterrupt: -3,
	checkpoint.ReservedResume:    -4,
}

// Saver is a checkpoint.Saver backed by a SQLite database. It is safe for
// concurrent use: the underlying pool is limited to a single connection (the
// Go equivalent of Python's single connection guarded by a lock), which also
// keeps `:memory:` databases consistent across calls.
type Saver struct {
	db    *sql.DB
	serde checkpoint.Serializer
}

var _ checkpoint.Saver = (*Saver)(nil)

// New opens (creating if needed) the SQLite database at path — or an
// in-memory database for `:memory:` — applies the schema, and returns a Saver
// persisting through serde. An empty path opens a private temporary on-disk
// database that is deleted when the Saver is closed (modernc.org/sqlite
// empty-DSN behavior). The caller must Close the Saver.
func New(path string, serde checkpoint.Serializer) (*Saver, error) {
	if serde == nil {
		return nil, errors.New("sqlite: New requires a non-nil checkpoint.Serializer")
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("sqlite: open %q: %w", path, err)
	}
	// One connection serializes all access (SQLite allows a single writer)
	// and gives `:memory:` a single, stable database.
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(setupSQL); err != nil {
		db.Close()
		return nil, fmt.Errorf("sqlite: setup %q: %w", path, err)
	}
	if err := ensureTaskPathColumn(db); err != nil {
		db.Close()
		return nil, err
	}
	return &Saver{db: db, serde: serde}, nil
}

// ensureTaskPathColumn adds the task_path column to writes tables created
// before the M5 schema evolution (Python added task_path in migration v9;
// Go sqlite savers created before M5 lack it).
func ensureTaskPathColumn(db *sql.DB) error {
	rows, err := db.Query(`PRAGMA table_info(writes)`)
	if err != nil {
		return fmt.Errorf("sqlite: inspect writes schema: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name, typ string
		var notNull int
		var dflt any
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &dflt, &pk); err != nil {
			return fmt.Errorf("sqlite: inspect writes schema: %w", err)
		}
		if name == "task_path" {
			return rows.Err()
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("sqlite: inspect writes schema: %w", err)
	}
	if _, err := db.Exec(`ALTER TABLE writes ADD COLUMN task_path TEXT NOT NULL DEFAULT ''`); err != nil {
		return fmt.Errorf("sqlite: add writes.task_path column: %w", err)
	}
	return nil
}

// Close closes the underlying database.
func (s *Saver) Close() error {
	return s.db.Close()
}

// GetTuple implements checkpoint.Saver.
func (s *Saver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	var row *sql.Row
	if cfg.CheckpointID != "" {
		row = s.db.QueryRowContext(ctx,
			`SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?`,
			cfg.ThreadID, cfg.CheckpointNS, cfg.CheckpointID)
	} else {
		row = s.db.QueryRowContext(ctx,
			`SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1`,
			cfg.ThreadID, cfg.CheckpointNS)
	}
	tup, err := s.buildTuple(ctx, cfg.CheckpointNS, row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return tup, nil
}

// buildTuple scans a single checkpoints row and assembles its full Tuple.
// Safe on the single pooled connection: *sql.Row.Scan fully consumes the row.
func (s *Saver) buildTuple(ctx context.Context, ns string, row *sql.Row) (*checkpoint.Tuple, error) {
	raw, err := scanCheckpointRow(row)
	if err != nil {
		return nil, err
	}
	return s.finishTuple(ctx, ns, raw)
}

// List implements checkpoint.Saver: newest (highest checkpoint ID) first,
// restricted to IDs strictly before opts.Before and to metadata containing
// opts.Filter, capped by opts.Limit. With a non-empty Filter the SQL query
// omits LIMIT — filtering happens in process and must precede the limit,
// mirroring Python's WHERE-before-LIMIT ordering.
func (s *Saver) List(ctx context.Context, cfg checkpoint.Config, opts checkpoint.ListOptions) ([]checkpoint.Tuple, error) {
	query := `SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?`
	args := []any{cfg.ThreadID, cfg.CheckpointNS}
	if opts.Before != nil && opts.Before.CheckpointID != "" {
		query += ` AND checkpoint_id < ?`
		args = append(args, opts.Before.CheckpointID)
	}
	query += ` ORDER BY checkpoint_id DESC`
	if opts.Limit > 0 && len(opts.Filter) == 0 {
		query += ` LIMIT ?`
		args = append(args, opts.Limit)
	}
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list thread %q: %w", cfg.ThreadID, err)
	}
	// Scan every row BEFORE decoding any of them: with a single pooled
	// connection, running the per-checkpoint writes query while this result
	// set is still open would deadlock the pool.
	var raw []rawCheckpointRow
	for rows.Next() {
		r, err := scanCheckpointRow(rows)
		if err != nil {
			rows.Close()
			return nil, fmt.Errorf("sqlite: list thread %q: %w", cfg.ThreadID, err)
		}
		raw = append(raw, r)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return nil, fmt.Errorf("sqlite: list thread %q: %w", cfg.ThreadID, err)
	}
	rows.Close()

	out := make([]checkpoint.Tuple, 0, len(raw))
	for _, r := range raw {
		tup, err := s.finishTuple(ctx, cfg.CheckpointNS, r)
		if err != nil {
			return nil, err
		}
		if !checkpoint.MetadataMatchesFilter(tup.Metadata, opts.Filter) {
			continue
		}
		out = append(out, *tup)
		if opts.Limit > 0 && len(opts.Filter) > 0 && len(out) >= opts.Limit {
			break
		}
	}
	return out, nil
}

// Put implements checkpoint.Saver. cfg.CheckpointID, when non-empty, is
// recorded as the new checkpoint's parent link (D3); newVersions is merged
// into the stored ChannelVersions.
func (s *Saver) Put(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64) (checkpoint.Config, error) {
	stored := cp
	if len(newVersions) > 0 {
		stored.ChannelVersions = maps.Clone(cp.ChannelVersions)
		if stored.ChannelVersions == nil {
			stored.ChannelVersions = map[string]int64{}
		}
		maps.Copy(stored.ChannelVersions, newVersions)
	}
	blob, err := s.encodeCheckpoint(stored)
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("sqlite: encode checkpoint %q: %w", cp.ID, err)
	}
	mdBlob, err := encodeMetadata(md)
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("sqlite: encode metadata for %q: %w", cp.ID, err)
	}
	var parent *string
	if cfg.CheckpointID != "" {
		parent = &cfg.CheckpointID
	}
	if _, err := s.db.ExecContext(ctx, insertCheckpointSQL,
		cfg.ThreadID, cfg.CheckpointNS, cp.ID, parent, checkpointBlobType, blob, mdBlob); err != nil {
		return checkpoint.Config{}, fmt.Errorf("sqlite: put checkpoint %q: %w", cp.ID, err)
	}
	return checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS, CheckpointID: cp.ID}, nil
}

// PutWrites implements checkpoint.Saver. The insert mode is a BATCH-level
// decision mirroring Python's `SqliteSaver.put_writes`: when EVERY write in
// the batch targets a reserved channel, rows use INSERT OR REPLACE with the
// reserved negative idx (re-invocation overwrites); otherwise rows use
// INSERT OR IGNORE with the positional idx (first write wins). Reserved
// channels keep their negative idx even in mixed batches, like Python's
// `WRITES_IDX_MAP.get(channel, idx)`.
func (s *Saver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	cpID, err := s.resolveCheckpointID(ctx, cfg)
	if err != nil {
		return err
	}
	query := insertOrIgnoreSQL
	if allReserved(writes) {
		query = insertOrReplaceSQL
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: put writes for checkpoint %q: %w", cpID, err)
	}
	defer tx.Rollback()
	stmt, err := tx.PrepareContext(ctx, query)
	if err != nil {
		return fmt.Errorf("sqlite: put writes for checkpoint %q: %w", cpID, err)
	}
	defer stmt.Close()
	for i, w := range writes {
		idx := i
		if reserved, ok := reservedWriteIdx[w.Channel]; ok {
			idx = reserved
		}
		typ, data, err := s.serde.DumpsTyped(w.Value)
		if err != nil {
			return fmt.Errorf("sqlite: encode write %d to channel %q: %w", i, w.Channel, err)
		}
		if _, err := stmt.ExecContext(ctx, cfg.ThreadID, cfg.CheckpointNS, cpID, taskID, idx, w.Channel, typ, data, taskPath); err != nil {
			return fmt.Errorf("sqlite: put write %d to channel %q: %w", i, w.Channel, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: put writes for checkpoint %q: %w", cpID, err)
	}
	return nil
}

// DeleteThread implements checkpoint.Saver.
func (s *Saver) DeleteThread(ctx context.Context, threadID string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: delete thread %q: %w", threadID, err)
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, `DELETE FROM checkpoints WHERE thread_id = ?`, threadID); err != nil {
		return fmt.Errorf("sqlite: delete thread %q: %w", threadID, err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM writes WHERE thread_id = ?`, threadID); err != nil {
		return fmt.Errorf("sqlite: delete thread %q: %w", threadID, err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: delete thread %q: %w", threadID, err)
	}
	return nil
}

// DeleteForRuns implements checkpoint.Saver: every checkpoint whose metadata
// run_id is listed, plus its writes, across all threads and namespaces
// (json_extract reads the metadata blob's run_id; NULL — no run_id — never
// matches because runIDs entries are non-empty strings and a SQL NULL never
// equals an IN-list element). The DeltaChannel warning on the Saver
// interface (base/__init__.py:340-346) applies: deleting a run can sever a
// live thread's delta-channel ancestor history — documented only, mirroring
// Python.
func (s *Saver) DeleteForRuns(ctx context.Context, runIDs []string) error {
	if len(runIDs) == 0 {
		return nil
	}
	placeholders := make([]string, len(runIDs))
	args := make([]any, len(runIDs))
	for i, id := range runIDs {
		placeholders[i] = "?"
		args[i] = id
	}
	in := strings.Join(placeholders, ", ")
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: delete for runs: %w", err)
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM writes WHERE (thread_id, checkpoint_ns, checkpoint_id) IN (SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints WHERE json_extract(metadata, '$.run_id') IN (`+in+`))`,
		args...); err != nil {
		return fmt.Errorf("sqlite: delete writes for runs: %w", err)
	}
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM checkpoints WHERE json_extract(metadata, '$.run_id') IN (`+in+`)`,
		args...); err != nil {
		return fmt.Errorf("sqlite: delete checkpoints for runs: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: delete for runs: %w", err)
	}
	return nil
}

// CopyThread implements checkpoint.Saver: INSERT ... SELECT with the thread
// replaced, preserving checkpoint IDs, namespaces, parent links, and writes
// (Python's copy_thread must carry the complete parent chain,
// base/__init__.py:361-371). A nonexistent source inserts nothing.
func (s *Saver) CopyThread(ctx context.Context, srcThreadID, dstThreadID string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: copy thread %q -> %q: %w", srcThreadID, dstThreadID, err)
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx,
		`INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) SELECT ?, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ?`,
		dstThreadID, srcThreadID); err != nil {
		return fmt.Errorf("sqlite: copy checkpoints %q -> %q: %w", srcThreadID, dstThreadID, err)
	}
	if _, err := tx.ExecContext(ctx,
		`INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, task_path) SELECT ?, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, task_path FROM writes WHERE thread_id = ?`,
		dstThreadID, srcThreadID); err != nil {
		return fmt.Errorf("sqlite: copy writes %q -> %q: %w", srcThreadID, dstThreadID, err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: copy thread %q -> %q: %w", srcThreadID, dstThreadID, err)
	}
	return nil
}

// Prune implements checkpoint.Saver. keep_latest keeps, per namespace, the
// row with the maximum checkpoint_id (IDs are lexicographically ordered,
// checkpoint.NewID); delete removes the whole thread. The writes DELETE runs
// BEFORE the checkpoints DELETE so its NOT IN subquery still sees the full
// pre-prune table. The DeltaChannel warning on the Saver interface
// (base/__init__.py:387-413) applies: naive keep_latest can sever
// delta-channel ancestor history — documented only, mirroring Python.
func (s *Saver) Prune(ctx context.Context, threadIDs []string, strategy checkpoint.PruneStrategy) error {
	switch strategy {
	case checkpoint.PruneKeepLatest, checkpoint.PruneDeleteAll:
	default:
		return fmt.Errorf("sqlite: unknown prune strategy %q", strategy)
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: prune: %w", err)
	}
	defer tx.Rollback()
	const keepLatestPerNS = `(SELECT checkpoint_ns, MAX(checkpoint_id) FROM checkpoints WHERE thread_id = ? GROUP BY checkpoint_ns)`
	for _, threadID := range threadIDs {
		if strategy == checkpoint.PruneDeleteAll {
			for _, table := range []string{"checkpoints", "writes"} {
				if _, err := tx.ExecContext(ctx, `DELETE FROM `+table+` WHERE thread_id = ?`, threadID); err != nil {
					return fmt.Errorf("sqlite: prune(delete) thread %q: %w", threadID, err)
				}
			}
			continue
		}
		if _, err := tx.ExecContext(ctx,
			`DELETE FROM writes WHERE thread_id = ? AND (checkpoint_ns, checkpoint_id) NOT IN `+keepLatestPerNS,
			threadID, threadID); err != nil {
			return fmt.Errorf("sqlite: prune writes for thread %q: %w", threadID, err)
		}
		if _, err := tx.ExecContext(ctx,
			`DELETE FROM checkpoints WHERE thread_id = ? AND (checkpoint_ns, checkpoint_id) NOT IN `+keepLatestPerNS,
			threadID, threadID); err != nil {
			return fmt.Errorf("sqlite: prune checkpoints for thread %q: %w", threadID, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: prune: %w", err)
	}
	return nil
}

// resolveCheckpointID resolves cfg to a stored checkpoint ID — cfg's own ID,
// or the latest for the thread/namespace when empty — and errors when no such
// checkpoint exists, matching MemorySaver's PutWrites behavior.
func (s *Saver) resolveCheckpointID(ctx context.Context, cfg checkpoint.Config) (string, error) {
	if cfg.CheckpointID != "" {
		var id string
		err := s.db.QueryRowContext(ctx,
			`SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?`,
			cfg.ThreadID, cfg.CheckpointNS, cfg.CheckpointID).Scan(&id)
		if errors.Is(err, sql.ErrNoRows) {
			return "", fmt.Errorf("sqlite: PutWrites: no checkpoint %q for thread %q (ns %q)",
				cfg.CheckpointID, cfg.ThreadID, cfg.CheckpointNS)
		}
		if err != nil {
			return "", fmt.Errorf("sqlite: PutWrites: %w", err)
		}
		return id, nil
	}
	var id string
	err := s.db.QueryRowContext(ctx,
		`SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1`,
		cfg.ThreadID, cfg.CheckpointNS).Scan(&id)
	if errors.Is(err, sql.ErrNoRows) {
		return "", fmt.Errorf("sqlite: PutWrites: no checkpoint %q for thread %q (ns %q)",
			cfg.CheckpointID, cfg.ThreadID, cfg.CheckpointNS)
	}
	if err != nil {
		return "", fmt.Errorf("sqlite: PutWrites: %w", err)
	}
	return id, nil
}

// allReserved reports whether every write in the batch targets a reserved
// channel (the empty batch vacuously qualifies, as in Python's `all(...)`,
// and inserts nothing either way).
func allReserved(writes []checkpoint.Write) bool {
	for _, w := range writes {
		if _, ok := reservedWriteIdx[w.Channel]; !ok {
			return false
		}
	}
	return true
}

// rowScanner abstracts *sql.Row / *sql.Rows so one scan path serves both
// GetTuple and List.
type rowScanner interface {
	Scan(dest ...any) error
}

// rawCheckpointRow is one scanned checkpoints row, not yet decoded.
type rawCheckpointRow struct {
	threadID string
	cpID     string
	parent   sql.NullString
	typ      string
	blob     []byte
	mdBlob   []byte
}

// scanCheckpointRow scans one row of (thread_id, checkpoint_id,
// parent_checkpoint_id, type, checkpoint, metadata).
func scanCheckpointRow(row rowScanner) (rawCheckpointRow, error) {
	var r rawCheckpointRow
	err := row.Scan(&r.threadID, &r.cpID, &r.parent, &r.typ, &r.blob, &r.mdBlob)
	return r, err
}

// finishTuple decodes a scanned checkpoints row and assembles the full Tuple,
// including pending writes ordered by task_id, idx as in Python.
func (s *Saver) finishTuple(ctx context.Context, ns string, r rawCheckpointRow) (*checkpoint.Tuple, error) {
	cp, err := s.decodeCheckpoint(r.typ, r.blob)
	if err != nil {
		return nil, fmt.Errorf("sqlite: decode checkpoint %q: %w", r.cpID, err)
	}
	md, err := decodeMetadata(r.mdBlob)
	if err != nil {
		return nil, fmt.Errorf("sqlite: decode metadata for %q: %w", r.cpID, err)
	}
	writes, err := s.loadWrites(ctx, r.threadID, ns, r.cpID)
	if err != nil {
		return nil, err
	}
	tup := &checkpoint.Tuple{
		Config:        checkpoint.Config{ThreadID: r.threadID, CheckpointNS: ns, CheckpointID: r.cpID},
		Checkpoint:    cp,
		Metadata:      md,
		PendingWrites: writes,
	}
	if r.parent.Valid {
		tup.ParentConfig = &checkpoint.Config{ThreadID: r.threadID, CheckpointNS: ns, CheckpointID: r.parent.String}
	}
	return tup, nil
}

// loadWrites returns the pending writes recorded against a checkpoint,
// ordered by (task_id, idx) exactly like Python's `SqliteSaver`.
func (s *Saver) loadWrites(ctx context.Context, threadID, ns, cpID string) ([]checkpoint.Write, error) {
	rows, err := s.db.QueryContext(ctx, selectWritesSQL, threadID, ns, cpID)
	if err != nil {
		return nil, fmt.Errorf("sqlite: load writes for checkpoint %q: %w", cpID, err)
	}
	defer rows.Close()

	var out []checkpoint.Write
	for rows.Next() {
		var w checkpoint.Write
		var typ string
		var data []byte
		if err := rows.Scan(&w.TaskID, &w.TaskPath, &w.Channel, &typ, &data); err != nil {
			return nil, fmt.Errorf("sqlite: load writes for checkpoint %q: %w", cpID, err)
		}
		v, err := s.serde.LoadsTyped(typ, data)
		if err != nil {
			return nil, fmt.Errorf("sqlite: decode write to channel %q: %w", w.Channel, err)
		}
		w.Value = v
		out = append(out, w)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite: load writes for checkpoint %q: %w", cpID, err)
	}
	return out, nil
}

// encodeValue runs one channel value / write value through the serde.
func (s *Saver) encodeValue(v any) (storedValue, error) {
	typ, data, err := s.serde.DumpsTyped(v)
	if err != nil {
		return storedValue{}, err
	}
	return storedValue{Type: typ, Data: data}, nil
}

// decodeValue restores a value encoded by encodeValue.
func (s *Saver) decodeValue(sv storedValue) (any, error) {
	return s.serde.LoadsTyped(sv.Type, sv.Data)
}

// storedCheckpoint is the JSON-safe projection of checkpoint.Checkpoint
// persisted in the checkpoints table's `checkpoint` blob column, encoded as
// plain JSON (blob `type` = "json"):
//   - V, ID: plain fields; TS as RFC3339Nano via encoding/json's time.Time
//     handling.
//   - ChannelValues: each value individually through the Serializer (type tag
//     plus data) so typed values — `[]string`, `[]messages.Message`, `int`,
//     `types.Interrupt`, ... — round-trip exactly instead of degrading to
//     JSON-decoded `[]any` / `float64`.
//   - ChannelVersions / VersionsSeen: plain JSON integer maps (Go-typed
//     fields, so encoding/json restores them exactly).
//   - Next: planned tasks; Arg values through the Serializer like channel
//     values.
type storedCheckpoint struct {
	V               int                         `json:"v"`
	ID              string                      `json:"id"`
	TS              time.Time                   `json:"ts"`
	ChannelValues   map[string]storedValue      `json:"channel_values,omitempty"`
	ChannelVersions map[string]int64            `json:"channel_versions,omitempty"`
	VersionsSeen    map[string]map[string]int64 `json:"versions_seen,omitempty"`
	Next            []storedTask                `json:"next,omitempty"`
}

// storedValue is one serde-typed value embedded in the checkpoint blob.
type storedValue struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// storedTask is the projection of checkpoint.PlannedTask.
type storedTask struct {
	ID   string                 `json:"id"`
	Node string                 `json:"node"`
	Arg  map[string]storedValue `json:"arg,omitempty"`
}

// encodeCheckpoint projects cp into its JSON-safe stored form and marshals it.
func (s *Saver) encodeCheckpoint(cp checkpoint.Checkpoint) ([]byte, error) {
	proj := storedCheckpoint{
		V:               cp.V,
		ID:              cp.ID,
		TS:              cp.TS,
		ChannelVersions: cp.ChannelVersions,
		VersionsSeen:    cp.VersionsSeen,
	}
	if cp.ChannelValues != nil {
		proj.ChannelValues = make(map[string]storedValue, len(cp.ChannelValues))
		for channel, v := range cp.ChannelValues {
			sv, err := s.encodeValue(v)
			if err != nil {
				return nil, fmt.Errorf("channel %q: %w", channel, err)
			}
			proj.ChannelValues[channel] = sv
		}
	}
	if cp.Next != nil {
		proj.Next = make([]storedTask, len(cp.Next))
		for i, task := range cp.Next {
			st := storedTask{ID: task.ID, Node: task.Node}
			if task.Arg != nil {
				st.Arg = make(map[string]storedValue, len(task.Arg))
				for k, v := range task.Arg {
					sv, err := s.encodeValue(v)
					if err != nil {
						return nil, fmt.Errorf("next task %q arg %q: %w", task.ID, k, err)
					}
					st.Arg[k] = sv
				}
			}
			proj.Next[i] = st
		}
	}
	return json.Marshal(proj)
}

// decodeCheckpoint restores a Checkpoint from its stored blob. typ must be
// checkpointBlobType — the projection is always plain JSON.
func (s *Saver) decodeCheckpoint(typ string, blob []byte) (checkpoint.Checkpoint, error) {
	if typ != checkpointBlobType {
		return checkpoint.Checkpoint{}, fmt.Errorf("unknown checkpoint blob type %q", typ)
	}
	var proj storedCheckpoint
	if err := json.Unmarshal(blob, &proj); err != nil {
		return checkpoint.Checkpoint{}, err
	}
	cp := checkpoint.Checkpoint{
		V:               proj.V,
		ID:              proj.ID,
		TS:              proj.TS,
		ChannelVersions: proj.ChannelVersions,
		VersionsSeen:    proj.VersionsSeen,
	}
	if proj.ChannelValues != nil {
		cp.ChannelValues = make(map[string]any, len(proj.ChannelValues))
		for channel, sv := range proj.ChannelValues {
			v, err := s.decodeValue(sv)
			if err != nil {
				return checkpoint.Checkpoint{}, fmt.Errorf("channel %q: %w", channel, err)
			}
			cp.ChannelValues[channel] = v
		}
	}
	if proj.Next != nil {
		cp.Next = make([]checkpoint.PlannedTask, len(proj.Next))
		for i, st := range proj.Next {
			task := checkpoint.PlannedTask{ID: st.ID, Node: st.Node}
			if st.Arg != nil {
				task.Arg = make(map[string]any, len(st.Arg))
				for k, sv := range st.Arg {
					v, err := s.decodeValue(sv)
					if err != nil {
						return checkpoint.Checkpoint{}, fmt.Errorf("next task %q arg %q: %w", st.ID, k, err)
					}
					task.Arg[k] = v
				}
			}
			cp.Next[i] = task
		}
	}
	return cp, nil
}

// storedMetadata is the plain-JSON projection of checkpoint.Metadata persisted
// in the `metadata` blob column, mirroring Python's `json.dumps(metadata)`.
type storedMetadata struct {
	Source  string            `json:"source"`
	Step    int               `json:"step"`
	Parents map[string]string `json:"parents,omitempty"`
	RunID   string            `json:"run_id,omitempty"`
}

func encodeMetadata(md checkpoint.Metadata) ([]byte, error) {
	return json.Marshal(storedMetadata{Source: md.Source, Step: md.Step, Parents: md.Parents, RunID: md.RunID})
}

func decodeMetadata(blob []byte) (checkpoint.Metadata, error) {
	if blob == nil {
		return checkpoint.Metadata{}, nil
	}
	var stored storedMetadata
	if err := json.Unmarshal(blob, &stored); err != nil {
		return checkpoint.Metadata{}, err
	}
	return checkpoint.Metadata{Source: stored.Source, Step: stored.Step, Parents: stored.Parents, RunID: stored.RunID}, nil
}
