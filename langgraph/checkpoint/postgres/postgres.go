package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// Write-path SQL, mirroring Python base.py:131-159 with $n placeholders.
const (
	upsertBlobSQL = `
    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING`
	upsertCheckpointSQL = `
    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
    DO UPDATE SET
        checkpoint = EXCLUDED.checkpoint,
        metadata = EXCLUDED.metadata`
	upsertWritesSQL = `
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel,
        type = EXCLUDED.type,
        blob = EXCLUDED.blob`
	insertWritesSQL = `
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING`
)

// reservedWriteIdx maps reserved channels to their negative write idx,
// matching Python's WRITES_IDX_MAP {ERROR:-1, SCHEDULED:-2, INTERRUPT:-3,
// RESUME:-4} (langgraph/checkpoint/base/__init__.py:795) — Go has no
// SCHEDULED counterpart, so -2 is free. Divergence: Python's map has NO
// TASKS entry — its TASKS writes use the positional idx — but the Go
// executor persists Command.Goto routing as __tasks__ writes, so Go maps
// __tasks__ to -2, matching the sqlite saver's established approximation.
// Collision risk, shared with sqlite: multiple __tasks__ writes in one batch
// all map to idx -2, so an all-reserved batch's upsert keeps only the last
// such write and a mixed batch's ON CONFLICT DO NOTHING keeps only the
// first. RESUME at -4 is lossless: the executor persists the consumed resume
// prefix as ONE write carrying the whole ordered list (Python parity,
// types.py:905-925), so last-write-wins has nothing to collapse.
var reservedWriteIdx = map[string]int{
	checkpoint.ReservedError:     -1,
	checkpoint.ReservedTasks:     -2,
	checkpoint.ReservedInterrupt: -3,
	checkpoint.ReservedResume:    -4,
}

// Saver is a checkpoint.Saver backed by PostgreSQL. It is safe for
// concurrent use (pgxpool manages the connections). The caller must call
// Setup once before first use, and Close when done.
type Saver struct {
	pool  *pgxpool.Pool
	serde checkpoint.Serializer
}

var _ checkpoint.Saver = (*Saver)(nil)

// New returns a Saver on pool, persisting through serde. Both arguments are
// required; New panics on a nil pool or nil serde (programmer error —
// Python's PostgresSaver likewise cannot function without a connection).
func New(pool *pgxpool.Pool, serde checkpoint.Serializer) *Saver {
	if pool == nil {
		panic("postgres: New requires a non-nil *pgxpool.Pool")
	}
	if serde == nil {
		panic("postgres: New requires a non-nil checkpoint.Serializer")
	}
	return &Saver{pool: pool, serde: serde}
}

// NewFromConnString opens a pgxpool from dsn and returns a Saver on it.
func NewFromConnString(ctx context.Context, dsn string, serde checkpoint.Serializer) (*Saver, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: connect: %w", err)
	}
	return New(pool, serde), nil
}

// Close closes the underlying pool.
func (s *Saver) Close() { s.pool.Close() }

// Setup applies pending schema migrations. It MUST be called explicitly once
// before first use (Python parity: PostgresSaver.setup). It does NOT wrap
// migrations in a transaction — v6–v8 are CREATE INDEX CONCURRENTLY, which
// Postgres forbids inside a transaction block.
func (s *Saver) Setup(ctx context.Context) error {
	if _, err := s.pool.Exec(ctx, Migrations[0]); err != nil {
		return fmt.Errorf("postgres saver: setup migrations table: %w", err)
	}
	var version int
	err := s.pool.QueryRow(ctx,
		`SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1`).Scan(&version)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		version = -1
	case err != nil:
		return fmt.Errorf("postgres saver: read migration version: %w", err)
	}
	for v := version + 1; v < len(Migrations); v++ {
		if _, err := s.pool.Exec(ctx, Migrations[v]); err != nil {
			return fmt.Errorf("postgres saver: migration %d: %w", v, err)
		}
		if _, err := s.pool.Exec(ctx,
			`INSERT INTO checkpoint_migrations (v) VALUES ($1)`, v); err != nil {
			return fmt.Errorf("postgres saver: record migration %d: %w", v, err)
		}
	}
	return nil
}

// isInline reports whether a channel value stays inline in the checkpoints
// JSONB document. Only JSON-native primitives inline — nil, string, bool,
// float64; int/int64 are serde-enveloped (not JSON-native) and everything
// composite goes to checkpoint_blobs (Python parity: __init__.py:316-319
// inlines primitives, sends dict/list to blobs).
func isInline(v any) bool {
	switch v.(type) {
	case nil, string, bool, float64:
		return true
	default:
		return false
	}
}

// splitChannelValues partitions values into inline primitives (kept in the
// checkpoints JSONB document) and blob values (stored per-version in
// checkpoint_blobs).
func splitChannelValues(values map[string]any) (inline, blobs map[string]any) {
	inline = map[string]any{}
	blobs = map[string]any{}
	for k, v := range values {
		if isInline(v) {
			inline[k] = v
		} else {
			blobs[k] = v
		}
	}
	return inline, blobs
}

// Put implements checkpoint.Saver. Every composite channel gets a blob row
// attempt at its current version — newVersions's version when the channel was
// just bumped, else the version already in cp.ChannelVersions — with
// ON CONFLICT DO NOTHING (immutable versioned rows), so re-Putting unchanged
// values is a no-op while a first-time write of a pre-versioned value still
// persists (Python's _dump_blobs iterates new_versions only and can lose
// such values; the savertest contract forbids losing channel data). A
// composite channel value with NO version anywhere is inlined into the
// checkpoint JSON as a serde typed envelope instead (see doc.go). The
// checkpoints row upserts.
func (s *Saver) Put(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64) (checkpoint.Config, error) {
	stored := cp
	if len(newVersions) > 0 {
		stored.ChannelVersions = maps.Clone(cp.ChannelVersions)
		if stored.ChannelVersions == nil {
			stored.ChannelVersions = map[string]int64{}
		}
		maps.Copy(stored.ChannelVersions, newVersions)
	}
	inline, blobValues := splitChannelValues(stored.ChannelValues)
	for channel, v := range blobValues {
		if _, ok := newVersions[channel]; ok {
			continue // versioned: blob row written below
		}
		if _, ok := stored.ChannelVersions[channel]; ok {
			continue // versioned: blob row written below at the stored version
		}
		// Unversioned composite: no blob row can be addressed without a
		// version, and adding one would corrupt ChannelVersions. Python
		// silently drops such values; Go inlines them as serde envelopes.
		typ, data, err := s.serde.DumpsTyped(v)
		if err != nil {
			return checkpoint.Config{}, fmt.Errorf("postgres saver: encode unversioned channel %q: %w", channel, err)
		}
		inline[channel] = storedValue{Type: typ, Data: data}
		delete(blobValues, channel)
	}
	cpJSON, err := s.encodeCheckpoint(stored, inline)
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("postgres saver: encode checkpoint %q: %w", cp.ID, err)
	}
	mdJSON, err := json.Marshal(storedMetadata{Source: md.Source, Step: md.Step, Parents: md.Parents})
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("postgres saver: encode metadata for %q: %w", cp.ID, err)
	}
	var parent *string
	if cfg.CheckpointID != "" {
		parent = &cfg.CheckpointID
	}
	batch := &pgx.Batch{}
	// Every channel left in blobValues has a version — from newVersions when
	// just bumped, else the stored one. Sorted for deterministic batch order.
	for _, channel := range slices.Sorted(maps.Keys(blobValues)) {
		ver, ok := newVersions[channel]
		if !ok {
			ver = stored.ChannelVersions[channel]
		}
		typ, data, err := s.serde.DumpsTyped(blobValues[channel])
		if err != nil {
			return checkpoint.Config{}, fmt.Errorf("postgres saver: encode channel %q: %w", channel, err)
		}
		batch.Queue(upsertBlobSQL, cfg.ThreadID, cfg.CheckpointNS, channel, ver, typ, data)
	}
	batch.Queue(upsertCheckpointSQL, cfg.ThreadID, cfg.CheckpointNS, cp.ID, parent, string(cpJSON), string(mdJSON))
	br := s.pool.SendBatch(ctx, batch)
	defer br.Close()
	for range batch.Len() {
		if _, err := br.Exec(); err != nil {
			return checkpoint.Config{}, fmt.Errorf("postgres saver: put checkpoint %q: %w", cp.ID, err)
		}
	}
	if err := br.Close(); err != nil {
		return checkpoint.Config{}, fmt.Errorf("postgres saver: put checkpoint %q: %w", cp.ID, err)
	}
	return checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS, CheckpointID: cp.ID}, nil
}

// PutWrites implements checkpoint.Saver with the Python BATCH-level insert
// rule (base.py:363-367): an all-reserved batch UPSERTs under the reserved
// negative idx (re-invocation overwrites); any other batch INSERTs with
// ON CONFLICT DO NOTHING at the positional idx (first write wins).
func (s *Saver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	if len(writes) == 0 {
		return nil // no-op, matching Python's executemany with an empty batch
	}
	cpID, err := s.resolveCheckpointID(ctx, cfg)
	if err != nil {
		return err
	}
	query := insertWritesSQL
	if allReserved(writes) {
		query = upsertWritesSQL
	}
	batch := &pgx.Batch{}
	for i, w := range writes {
		idx := i
		if reserved, ok := reservedWriteIdx[w.Channel]; ok {
			idx = reserved
		}
		typ, data, err := s.serde.DumpsTyped(w.Value)
		if err != nil {
			return fmt.Errorf("postgres saver: encode write %d to channel %q: %w", i, w.Channel, err)
		}
		batch.Queue(query, cfg.ThreadID, cfg.CheckpointNS, cpID, taskID, taskPath, idx, w.Channel, typ, data)
	}
	br := s.pool.SendBatch(ctx, batch)
	defer br.Close()
	for range batch.Len() {
		if _, err := br.Exec(); err != nil {
			return fmt.Errorf("postgres saver: put writes for checkpoint %q: %w", cpID, err)
		}
	}
	if err := br.Close(); err != nil {
		return fmt.Errorf("postgres saver: put writes for checkpoint %q: %w", cpID, err)
	}
	return nil
}

// GetTuple implements checkpoint.Saver.
func (s *Saver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	var row pgx.Row
	if cfg.CheckpointID != "" {
		row = s.pool.QueryRow(ctx,
			`SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3`,
			cfg.ThreadID, cfg.CheckpointNS, cfg.CheckpointID)
	} else {
		row = s.pool.QueryRow(ctx,
			`SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2 ORDER BY checkpoint_id DESC LIMIT 1`,
			cfg.ThreadID, cfg.CheckpointNS)
	}
	var cpID string
	var parent *string
	var cpJSON, mdJSON []byte
	if err := row.Scan(&cpID, &parent, &cpJSON, &mdJSON); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("postgres saver: get checkpoint: %w", err)
	}
	return s.assemble(ctx, cfg.ThreadID, cfg.CheckpointNS, cpID, parent, cpJSON, mdJSON)
}

// List implements checkpoint.Saver: newest checkpoint ID first, with
// Before, metadata @> Filter (server-side JSONB containment) and Limit.
func (s *Saver) List(ctx context.Context, cfg checkpoint.Config, opts checkpoint.ListOptions) ([]checkpoint.Tuple, error) {
	query, args, err := listQuery(cfg, opts)
	if err != nil {
		return nil, err
	}
	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: list thread %q: %w", cfg.ThreadID, err)
	}
	type rawRow struct {
		cpID   string
		parent *string
		cpJSON []byte
		mdJSON []byte
	}
	// Scan every row BEFORE issuing the per-checkpoint blobs/writes queries.
	var raw []rawRow
	for rows.Next() {
		var r rawRow
		if err := rows.Scan(&r.cpID, &r.parent, &r.cpJSON, &r.mdJSON); err != nil {
			rows.Close()
			return nil, fmt.Errorf("postgres saver: list thread %q: %w", cfg.ThreadID, err)
		}
		raw = append(raw, r)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return nil, fmt.Errorf("postgres saver: list thread %q: %w", cfg.ThreadID, err)
	}
	rows.Close()

	out := make([]checkpoint.Tuple, 0, len(raw))
	for _, r := range raw {
		tup, err := s.assemble(ctx, cfg.ThreadID, cfg.CheckpointNS, r.cpID, r.parent, r.cpJSON, r.mdJSON)
		if err != nil {
			return nil, err
		}
		out = append(out, *tup)
	}
	return out, nil
}

// listQuery builds the checkpoints SELECT with Before / metadata @> Filter /
// Limit predicates. Filter marshals to a JSONB containment argument
// (Python's `metadata @> %s`, base.py:655); a non-JSON-marshalable filter
// value (e.g. a func) is an error, never silently dropped.
func listQuery(cfg checkpoint.Config, opts checkpoint.ListOptions) (string, []any, error) {
	query := `SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2`
	args := []any{cfg.ThreadID, cfg.CheckpointNS}
	if opts.Before != nil && opts.Before.CheckpointID != "" {
		args = append(args, opts.Before.CheckpointID)
		query += fmt.Sprintf(` AND checkpoint_id < $%d`, len(args))
	}
	if len(opts.Filter) > 0 {
		filterJSON, err := json.Marshal(opts.Filter)
		if err != nil {
			return "", nil, fmt.Errorf("postgres saver: encode list filter: %w", err)
		}
		args = append(args, string(filterJSON))
		query += fmt.Sprintf(` AND metadata @> $%d::jsonb`, len(args))
	}
	query += ` ORDER BY checkpoint_id DESC`
	if opts.Limit > 0 {
		args = append(args, opts.Limit)
		query += fmt.Sprintf(` LIMIT $%d`, len(args))
	}
	return query, args, nil
}

// assemble decodes one checkpoints row and merges in its blob channel
// values and pending writes — the read side splits into 3 queries
// (checkpoints / blobs / writes) instead of Python's single nested
// array_agg SELECT (base.py:93-118).
func (s *Saver) assemble(ctx context.Context, threadID, ns, cpID string, parent *string, cpJSON, mdJSON []byte) (*checkpoint.Tuple, error) {
	cp, err := s.decodeCheckpoint(cpJSON)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: decode checkpoint %q: %w", cpID, err)
	}
	blobValues, err := s.loadBlobs(ctx, threadID, ns, cp.ChannelVersions)
	if err != nil {
		return nil, err
	}
	if cp.ChannelValues == nil && len(blobValues) > 0 {
		cp.ChannelValues = map[string]any{}
	}
	// Blob values win over the inline reading (base.py:581-583 merge order).
	maps.Copy(cp.ChannelValues, blobValues)
	var md checkpoint.Metadata
	var storedMd storedMetadata
	if len(mdJSON) > 0 {
		if err := json.Unmarshal(mdJSON, &storedMd); err != nil {
			return nil, fmt.Errorf("postgres saver: decode metadata for %q: %w", cpID, err)
		}
		md = checkpoint.Metadata{Source: storedMd.Source, Step: storedMd.Step, Parents: storedMd.Parents}
	}
	writes, err := s.loadWrites(ctx, threadID, ns, cpID)
	if err != nil {
		return nil, err
	}
	tup := &checkpoint.Tuple{
		Config:        checkpoint.Config{ThreadID: threadID, CheckpointNS: ns, CheckpointID: cpID},
		Checkpoint:    cp,
		Metadata:      md,
		PendingWrites: writes,
	}
	if parent != nil {
		tup.ParentConfig = &checkpoint.Config{ThreadID: threadID, CheckpointNS: ns, CheckpointID: *parent}
	}
	return tup, nil
}

// loadBlobs fetches the blob rows for exactly the (channel, version) pairs
// in versions. Rows with type "empty" are skipped (Python's _load_blobs,
// base.py:375-384).
func (s *Saver) loadBlobs(ctx context.Context, threadID, ns string, versions map[string]int64) (map[string]any, error) {
	if len(versions) == 0 {
		return nil, nil
	}
	channels := make([]string, 0, len(versions))
	vers := make([]int64, 0, len(versions))
	for _, ch := range slices.Sorted(maps.Keys(versions)) {
		channels = append(channels, ch)
		vers = append(vers, versions[ch])
	}
	rows, err := s.pool.Query(ctx,
		`SELECT channel, type, blob FROM checkpoint_blobs WHERE thread_id = $1 AND checkpoint_ns = $2 AND (channel, version) IN (SELECT * FROM unnest($3::text[], $4::bigint[]))`,
		threadID, ns, channels, vers)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: load blobs: %w", err)
	}
	defer rows.Close()
	out := map[string]any{}
	for rows.Next() {
		var channel, typ string
		var blob []byte
		if err := rows.Scan(&channel, &typ, &blob); err != nil {
			return nil, fmt.Errorf("postgres saver: load blobs: %w", err)
		}
		if typ == "empty" {
			continue
		}
		v, err := s.serde.LoadsTyped(typ, blob)
		if err != nil {
			return nil, fmt.Errorf("postgres saver: decode channel %q: %w", channel, err)
		}
		out[channel] = v
	}
	return out, rows.Err()
}

// loadWrites returns pending writes ordered by (task_id, idx), Python parity.
func (s *Saver) loadWrites(ctx context.Context, threadID, ns, cpID string) ([]checkpoint.Write, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT task_id, task_path, channel, type, blob FROM checkpoint_writes WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3 ORDER BY task_id, idx`,
		threadID, ns, cpID)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: load writes for checkpoint %q: %w", cpID, err)
	}
	defer rows.Close()
	var out []checkpoint.Write
	for rows.Next() {
		var w checkpoint.Write
		var typ string
		var data []byte
		if err := rows.Scan(&w.TaskID, &w.TaskPath, &w.Channel, &typ, &data); err != nil {
			return nil, fmt.Errorf("postgres saver: load writes for checkpoint %q: %w", cpID, err)
		}
		v, err := s.serde.LoadsTyped(typ, data)
		if err != nil {
			return nil, fmt.Errorf("postgres saver: decode write to channel %q: %w", w.Channel, err)
		}
		w.Value = v
		out = append(out, w)
	}
	return out, rows.Err()
}

// DeleteThread implements checkpoint.Saver.
func (s *Saver) DeleteThread(ctx context.Context, threadID string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("postgres saver: delete thread %q: %w", threadID, err)
	}
	defer tx.Rollback(ctx)
	for _, table := range []string{"checkpoints", "checkpoint_blobs", "checkpoint_writes"} {
		if _, err := tx.Exec(ctx, `DELETE FROM `+table+` WHERE thread_id = $1`, threadID); err != nil {
			return fmt.Errorf("postgres saver: delete thread %q: %w", threadID, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("postgres saver: delete thread %q: %w", threadID, err)
	}
	return nil
}

// resolveCheckpointID resolves cfg to a stored checkpoint ID — cfg's own ID,
// or the latest for the thread/namespace when empty — and errors when no
// such checkpoint exists, matching MemorySaver's PutWrites behavior.
func (s *Saver) resolveCheckpointID(ctx context.Context, cfg checkpoint.Config) (string, error) {
	var row pgx.Row
	if cfg.CheckpointID != "" {
		row = s.pool.QueryRow(ctx,
			`SELECT checkpoint_id FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3`,
			cfg.ThreadID, cfg.CheckpointNS, cfg.CheckpointID)
	} else {
		row = s.pool.QueryRow(ctx,
			`SELECT checkpoint_id FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2 ORDER BY checkpoint_id DESC LIMIT 1`,
			cfg.ThreadID, cfg.CheckpointNS)
	}
	var id string
	if err := row.Scan(&id); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return "", fmt.Errorf("postgres saver: PutWrites: no checkpoint %q for thread %q (ns %q)",
				cfg.CheckpointID, cfg.ThreadID, cfg.CheckpointNS)
		}
		return "", fmt.Errorf("postgres saver: PutWrites: %w", err)
	}
	return id, nil
}

// allReserved reports whether every write in the batch targets a reserved
// channel (the empty batch vacuously qualifies, as in Python's `all(...)`).
func allReserved(writes []checkpoint.Write) bool {
	for _, w := range writes {
		if _, ok := reservedWriteIdx[w.Channel]; !ok {
			return false
		}
	}
	return true
}

// storedCheckpoint is the JSONB projection of checkpoint.Checkpoint persisted
// in the checkpoints table's checkpoint column. Unlike the sqlite
// projection, ChannelValues holds only INLINE entries: JSON primitives as
// plain JSON, plus unversioned composite values as serde typed envelopes
// (storedValue — Python drops such values; Go preserves them, see Put).
// Versioned composite values live in checkpoint_blobs. Next task args remain
// serde-typed envelopes (storedValue), exactly as in the sqlite saver.
type storedCheckpoint struct {
	V               int                         `json:"v"`
	ID              string                      `json:"id"`
	TS              time.Time                   `json:"ts"`
	ChannelValues   map[string]any              `json:"channel_values,omitempty"`
	ChannelVersions map[string]int64            `json:"channel_versions,omitempty"`
	VersionsSeen    map[string]map[string]int64 `json:"versions_seen,omitempty"`
	Next            []storedTask                `json:"next,omitempty"`
}

// storedValue is one serde-typed value embedded in the checkpoint document.
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

// storedMetadata is the plain-JSON projection of checkpoint.Metadata.
type storedMetadata struct {
	Source  string            `json:"source"`
	Step    int               `json:"step"`
	Parents map[string]string `json:"parents,omitempty"`
}

// encodeCheckpoint marshals the projection with inline channel values.
func (s *Saver) encodeCheckpoint(cp checkpoint.Checkpoint, inline map[string]any) ([]byte, error) {
	proj := storedCheckpoint{
		V:               cp.V,
		ID:              cp.ID,
		TS:              cp.TS,
		ChannelVersions: cp.ChannelVersions,
		VersionsSeen:    cp.VersionsSeen,
	}
	if len(inline) > 0 {
		proj.ChannelValues = inline
	}
	if cp.Next != nil {
		proj.Next = make([]storedTask, len(cp.Next))
		for i, task := range cp.Next {
			st := storedTask{ID: task.ID, Node: task.Node}
			if task.Arg != nil {
				st.Arg = make(map[string]storedValue, len(task.Arg))
				for k, v := range task.Arg {
					typ, data, err := s.serde.DumpsTyped(v)
					if err != nil {
						return nil, fmt.Errorf("next task %q arg %q: %w", task.ID, k, err)
					}
					st.Arg[k] = storedValue{Type: typ, Data: data}
				}
			}
			proj.Next[i] = st
		}
	}
	return json.Marshal(proj)
}

// decodeCheckpoint restores a Checkpoint from its JSONB document; ChannelValues
// contains only the inline entries (blob values are merged by assemble).
// Inline entries are plain JSON primitives, except unversioned composite
// values which Put stores as serde typed envelopes — those are decoded back
// here. Plain primitives never decode to a JSON object, so any object value
// in this map is an envelope Put wrote.
func (s *Saver) decodeCheckpoint(blob []byte) (checkpoint.Checkpoint, error) {
	var proj storedCheckpoint
	if err := json.Unmarshal(blob, &proj); err != nil {
		return checkpoint.Checkpoint{}, err
	}
	for k, v := range proj.ChannelValues {
		m, ok := v.(map[string]any)
		if !ok {
			continue // plain inline primitive
		}
		raw, err := json.Marshal(m)
		if err != nil {
			return checkpoint.Checkpoint{}, fmt.Errorf("re-encode inline envelope %q: %w", k, err)
		}
		var sv storedValue
		if err := json.Unmarshal(raw, &sv); err != nil || sv.Type == "" {
			return checkpoint.Checkpoint{}, fmt.Errorf("decode inline envelope %q: %w", k, err)
		}
		val, err := s.serde.LoadsTyped(sv.Type, sv.Data)
		if err != nil {
			return checkpoint.Checkpoint{}, fmt.Errorf("decode inline channel %q: %w", k, err)
		}
		proj.ChannelValues[k] = val
	}
	cp := checkpoint.Checkpoint{
		V:               proj.V,
		ID:              proj.ID,
		TS:              proj.TS,
		ChannelValues:   proj.ChannelValues,
		ChannelVersions: proj.ChannelVersions,
		VersionsSeen:    proj.VersionsSeen,
	}
	if proj.Next != nil {
		cp.Next = make([]checkpoint.PlannedTask, len(proj.Next))
		for i, st := range proj.Next {
			task := checkpoint.PlannedTask{ID: st.ID, Node: st.Node}
			if st.Arg != nil {
				task.Arg = make(map[string]any, len(st.Arg))
				for k, sv := range st.Arg {
					v, err := s.serde.LoadsTyped(sv.Type, sv.Data)
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
