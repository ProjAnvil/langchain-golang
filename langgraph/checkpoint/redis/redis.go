package redis

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"sort"
	"time"

	goredis "github.com/redis/go-redis/v9"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// reservedWriteIdx maps reserved channels to the negative write idx used in
// overwrite mode, approximating Python's WRITES_IDX_MAP
// `{ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}`
// (`libs/checkpoint/langgraph/checkpoint/base/__init__.py:795`), exactly as
// the sqlite/postgres savers do: the Go runtime has no SCHEDULED counterpart,
// so the -2 slot stays empty. `__tasks__` is NOT in the map (Python parity):
// its writes take the positional idx, so several `__tasks__` writes in one
// batch each occupy their own slot and ALL survive.
var reservedWriteIdx = map[string]int{
	checkpoint.ReservedError:     -1,
	checkpoint.ReservedInterrupt: -3,
	checkpoint.ReservedResume:    -4,
}

// Saver is a checkpoint.Saver backed by Redis, the Go port of Python's
// langgraph-checkpoint-redis RedisSaver. It is safe for concurrent use (the
// go-redis client manages the connection pool).
//
// Unlike Python's saver it needs no Setup call: the storage model uses only
// plain Redis commands (hashes, strings, sorted sets), so there are no
// RediSearch indexes to create. See doc.go for the key layout and the
// documented divergences from Python.
type Saver struct {
	client        goredis.UniversalClient
	serde         checkpoint.Serializer
	ttl           time.Duration
	refreshOnRead bool
	ownsClient    bool
}

var _ checkpoint.Saver = (*Saver)(nil)

// Option configures a Saver.
type Option func(*Saver)

// WithTTL applies an expiration to every checkpoint, write, and ordering key
// the Saver writes (EXPIRE on Put/PutWrites), the Go counterpart of Python's
// `ttl={"default_ttl": minutes}` config. TTL application is best-effort,
// mirroring Python's expire_with_retry semantics: failures never fail the
// write itself. A non-positive ttl (the default) disables TTL management.
func WithTTL(ttl time.Duration) Option {
	return func(s *Saver) { s.ttl = ttl }
}

// WithRefreshOnRead re-applies the configured TTL to a checkpoint's keys
// whenever it is read (GetTuple/List), mirroring Python's
// `ttl={"refresh_on_read": True}` config. It is a no-op without WithTTL.
func WithRefreshOnRead(refresh bool) Option {
	return func(s *Saver) { s.refreshOnRead = refresh }
}

// New returns a Saver on client, persisting through serde. Both arguments
// are required; New panics on a nil client or nil serde (programmer error —
// Python's RedisSaver likewise cannot function without a connection). The
// caller keeps ownership of client; Close does not close it.
func New(client goredis.UniversalClient, serde checkpoint.Serializer, opts ...Option) *Saver {
	if client == nil {
		panic("redis: New requires a non-nil redis.UniversalClient")
	}
	if serde == nil {
		panic("redis: New requires a non-nil checkpoint.Serializer")
	}
	s := &Saver{client: client, serde: serde}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// NewFromConnString parses connString as a Redis URL (`redis://`,
// `rediss://`), verifies connectivity, and returns a Saver on the new
// client. The Saver owns the client; Close closes it.
func NewFromConnString(ctx context.Context, connString string, serde checkpoint.Serializer, opts ...Option) (*Saver, error) {
	cfg, err := goredis.ParseURL(connString)
	if err != nil {
		return nil, fmt.Errorf("redis saver: parse connection string: %w", err)
	}
	client := goredis.NewClient(cfg)
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("redis saver: connect: %w", err)
	}
	s := New(client, serde, opts...)
	s.ownsClient = true
	return s, nil
}

// Close closes the underlying client when the Saver owns it
// (NewFromConnString); for caller-supplied clients it is a no-op.
func (s *Saver) Close() error {
	if s.ownsClient {
		return s.client.Close()
	}
	return nil
}

// GetTuple implements checkpoint.Saver: the checkpoint identified by
// cfg.CheckpointID, or the latest (highest ID) for the thread/namespace when
// it is empty; (nil, nil) when no matching checkpoint exists.
func (s *Saver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	id := cfg.CheckpointID
	if id == "" {
		ids, err := s.client.ZRevRange(ctx, zsetKey(cfg.ThreadID, cfg.CheckpointNS), 0, 0).Result()
		if err != nil {
			return nil, fmt.Errorf("redis: latest checkpoint for thread %q (ns %q): %w", cfg.ThreadID, cfg.CheckpointNS, err)
		}
		if len(ids) == 0 {
			return nil, nil
		}
		id = ids[0]
	}
	return s.loadTuple(ctx, cfg.ThreadID, cfg.CheckpointNS, id)
}

// List implements checkpoint.Saver: newest (highest checkpoint ID) first,
// restricted to IDs strictly before opts.Before and to metadata containing
// opts.Filter, capped by opts.Limit. Filtering happens in process and
// precedes the limit, mirroring Python's WHERE-before-LIMIT ordering.
func (s *Saver) List(ctx context.Context, cfg checkpoint.Config, opts checkpoint.ListOptions) ([]checkpoint.Tuple, error) {
	// All members share score 0, so ZREVRANGE returns them in reverse
	// lexicographic member order — newest checkpoint ID first, since
	// checkpoint.NewID makes lexicographic order match chronological order.
	ids, err := s.client.ZRevRange(ctx, zsetKey(cfg.ThreadID, cfg.CheckpointNS), 0, -1).Result()
	if err != nil {
		return nil, fmt.Errorf("redis: list thread %q (ns %q): %w", cfg.ThreadID, cfg.CheckpointNS, err)
	}
	out := make([]checkpoint.Tuple, 0, len(ids))
	for _, id := range ids {
		if opts.Before != nil && opts.Before.CheckpointID != "" && id >= opts.Before.CheckpointID {
			continue
		}
		tup, err := s.loadTuple(ctx, cfg.ThreadID, cfg.CheckpointNS, id)
		if err != nil {
			return nil, err
		}
		if tup == nil {
			continue // index entry without a hash (e.g. TTL-expired): skip
		}
		if !checkpoint.MetadataMatchesFilter(tup.Metadata, opts.Filter) {
			continue
		}
		out = append(out, *tup)
		if opts.Limit > 0 && len(out) >= opts.Limit {
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
		return checkpoint.Config{}, fmt.Errorf("redis: encode checkpoint %q: %w", cp.ID, err)
	}
	mdBlob, err := encodeMetadata(md)
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("redis: encode metadata for %q: %w", cp.ID, err)
	}
	key := checkpointKey(cfg.ThreadID, cfg.CheckpointNS, cp.ID)
	zkey := zsetKey(cfg.ThreadID, cfg.CheckpointNS)
	pipe := s.client.Pipeline()
	// The parent field is always written (empty when there is no parent) so a
	// re-Put of the same checkpoint ID fully replaces the previous record,
	// matching sqlite's INSERT OR REPLACE row semantics.
	pipe.HSet(ctx, key,
		fieldParent, cfg.CheckpointID,
		fieldType, checkpointBlobType,
		fieldCheckpoint, blob,
		fieldMetadata, mdBlob)
	pipe.ZAdd(ctx, zkey, goredis.Z{Score: 0, Member: cp.ID})
	if _, err := pipe.Exec(ctx); err != nil {
		return checkpoint.Config{}, fmt.Errorf("redis: put checkpoint %q: %w", cp.ID, err)
	}
	s.applyTTL(ctx, key, zkey)
	return checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS, CheckpointID: cp.ID}, nil
}

// PutWrites implements checkpoint.Saver. The write mode is a BATCH-level
// decision mirroring Python's put_writes (and the Go sqlite/postgres
// savers): when EVERY write in the batch targets a reserved channel, slots
// are overwritten (SET); otherwise the first write to a slot wins
// (SET NX). Reserved channels keep their negative idx even in mixed
// batches, like Python's `WRITES_IDX_MAP.get(channel, idx)`.
func (s *Saver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	cpID, err := s.resolveCheckpointID(ctx, cfg)
	if err != nil {
		return err
	}
	replace := allReserved(writes)

	// Encode every write before touching Redis so a serialization failure
	// cannot leave a half-written batch.
	blobs := make([][]byte, len(writes))
	keys := make([]string, len(writes))
	for i, w := range writes {
		idx := i
		if reserved, ok := reservedWriteIdx[w.Channel]; ok {
			idx = reserved
		}
		typ, data, err := s.serde.DumpsTyped(w.Value)
		if err != nil {
			return fmt.Errorf("redis: encode write %d to channel %q: %w", i, w.Channel, err)
		}
		blob, err := json.Marshal(storedWrite{
			TaskID: taskID, TaskPath: taskPath, Channel: w.Channel,
			Idx: idx, Type: typ, Data: data,
		})
		if err != nil {
			return fmt.Errorf("redis: encode write %d to channel %q: %w", i, w.Channel, err)
		}
		blobs[i] = blob
		keys[i] = writeKey(cfg.ThreadID, cfg.CheckpointNS, cpID, taskID, idx)
	}

	pipe := s.client.Pipeline()
	for i := range writes {
		if replace {
			pipe.Set(ctx, keys[i], blobs[i], 0)
		} else {
			pipe.SetNX(ctx, keys[i], blobs[i], 0)
		}
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("redis: put writes for checkpoint %q: %w", cpID, err)
	}
	s.applyTTL(ctx, keys...)
	return nil
}

// DeleteThread implements checkpoint.Saver: every checkpoint hash, pending
// write, and ordering zset of the thread, across all namespaces.
func (s *Saver) DeleteThread(ctx context.Context, threadID string) error {
	patterns := []string{
		checkpointScanPattern(threadID),
		writeScanPattern(threadID, "", ""),
		zsetScanPattern(threadID),
	}
	for _, pattern := range patterns {
		keys, err := s.scanKeys(ctx, pattern)
		if err != nil {
			return fmt.Errorf("redis: delete thread %q: %w", threadID, err)
		}
		if len(keys) == 0 {
			continue
		}
		if err := s.client.Del(ctx, keys...).Err(); err != nil {
			return fmt.Errorf("redis: delete thread %q: %w", threadID, err)
		}
	}
	return nil
}

// resolveCheckpointID resolves cfg to a stored checkpoint ID — cfg's own ID,
// or the latest for the thread/namespace when empty — and errors when no such
// checkpoint exists, matching MemorySaver's PutWrites behavior.
func (s *Saver) resolveCheckpointID(ctx context.Context, cfg checkpoint.Config) (string, error) {
	if cfg.CheckpointID != "" {
		n, err := s.client.Exists(ctx, checkpointKey(cfg.ThreadID, cfg.CheckpointNS, cfg.CheckpointID)).Result()
		if err != nil {
			return "", fmt.Errorf("redis: PutWrites: %w", err)
		}
		if n == 0 {
			return "", fmt.Errorf("redis: PutWrites: no checkpoint %q for thread %q (ns %q)",
				cfg.CheckpointID, cfg.ThreadID, cfg.CheckpointNS)
		}
		return cfg.CheckpointID, nil
	}
	ids, err := s.client.ZRevRange(ctx, zsetKey(cfg.ThreadID, cfg.CheckpointNS), 0, 0).Result()
	if err != nil {
		return "", fmt.Errorf("redis: PutWrites: %w", err)
	}
	if len(ids) == 0 {
		return "", fmt.Errorf("redis: PutWrites: no checkpoint %q for thread %q (ns %q)",
			cfg.CheckpointID, cfg.ThreadID, cfg.CheckpointNS)
	}
	return ids[0], nil
}

// loadTuple reads one checkpoint hash and its pending writes and assembles
// the full Tuple. It returns (nil, nil) when the hash does not exist.
func (s *Saver) loadTuple(ctx context.Context, threadID, ns, cpID string) (*checkpoint.Tuple, error) {
	key := checkpointKey(threadID, ns, cpID)
	fields, err := s.client.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("redis: load checkpoint %q: %w", cpID, err)
	}
	if len(fields) == 0 {
		return nil, nil
	}
	cp, err := s.decodeCheckpoint(fields[fieldType], []byte(fields[fieldCheckpoint]))
	if err != nil {
		return nil, fmt.Errorf("redis: decode checkpoint %q: %w", cpID, err)
	}
	md, err := decodeMetadata([]byte(fields[fieldMetadata]))
	if err != nil {
		return nil, fmt.Errorf("redis: decode metadata for %q: %w", cpID, err)
	}
	writes, writeKeys, err := s.loadWrites(ctx, threadID, ns, cpID)
	if err != nil {
		return nil, err
	}
	tup := &checkpoint.Tuple{
		Config:        checkpoint.Config{ThreadID: threadID, CheckpointNS: ns, CheckpointID: cpID},
		Checkpoint:    cp,
		Metadata:      md,
		PendingWrites: writes,
	}
	if parent := fields[fieldParent]; parent != "" {
		tup.ParentConfig = &checkpoint.Config{ThreadID: threadID, CheckpointNS: ns, CheckpointID: parent}
	}
	if s.refreshOnRead {
		s.applyTTL(ctx, append(writeKeys, key, zsetKey(threadID, ns))...)
	}
	return tup, nil
}

// loadWrites returns the pending writes recorded against a checkpoint,
// ordered by (task_id, idx) exactly like the sqlite/postgres savers, plus
// the keys they were read from (for TTL refresh).
func (s *Saver) loadWrites(ctx context.Context, threadID, ns, cpID string) ([]checkpoint.Write, []string, error) {
	keys, err := s.scanKeys(ctx, writeScanPattern(threadID, ns, cpID))
	if err != nil {
		return nil, nil, fmt.Errorf("redis: load writes for checkpoint %q: %w", cpID, err)
	}
	if len(keys) == 0 {
		return nil, nil, nil
	}
	pipe := s.client.Pipeline()
	cmds := make([]*goredis.StringCmd, len(keys))
	for i, key := range keys {
		cmds[i] = pipe.Get(ctx, key)
	}
	// Each command carries its own error, so the pipeline's aggregate result
	// (the first failing command's error) adds nothing; check per command.
	_, _ = pipe.Exec(ctx)
	stored := make([]storedWrite, 0, len(keys))
	liveKeys := make([]string, 0, len(keys))
	for i, cmd := range cmds {
		raw, err := cmd.Result()
		if errors.Is(err, goredis.Nil) {
			continue // vanished between SCAN and GET (e.g. TTL): skip
		}
		if err != nil {
			return nil, nil, fmt.Errorf("redis: load writes for checkpoint %q: %w", cpID, err)
		}
		var sw storedWrite
		if err := json.Unmarshal([]byte(raw), &sw); err != nil {
			return nil, nil, fmt.Errorf("redis: decode write %q for checkpoint %q: %w", keys[i], cpID, err)
		}
		stored = append(stored, sw)
		liveKeys = append(liveKeys, keys[i])
	}
	sort.Slice(stored, func(i, j int) bool {
		if stored[i].TaskID != stored[j].TaskID {
			return stored[i].TaskID < stored[j].TaskID
		}
		return stored[i].Idx < stored[j].Idx
	})
	out := make([]checkpoint.Write, len(stored))
	for i, sw := range stored {
		v, err := s.serde.LoadsTyped(sw.Type, sw.Data)
		if err != nil {
			return nil, nil, fmt.Errorf("redis: decode write to channel %q: %w", sw.Channel, err)
		}
		out[i] = checkpoint.Write{
			TaskID:   sw.TaskID,
			TaskPath: sw.TaskPath,
			Channel:  sw.Channel,
			Value:    v,
		}
	}
	return out, liveKeys, nil
}

// scanKeys collects every key matching pattern with SCAN.
func (s *Saver) scanKeys(ctx context.Context, pattern string) ([]string, error) {
	var keys []string
	var cursor uint64
	for {
		batch, next, err := s.client.Scan(ctx, cursor, pattern, 100).Result()
		if err != nil {
			return nil, err
		}
		keys = append(keys, batch...)
		cursor = next
		if cursor == 0 {
			return keys, nil
		}
	}
}

// applyTTL best-effort EXPIREs keys, mirroring Python's expire_with_retry
// semantics: TTL failures never fail the operation that wrote the keys.
func (s *Saver) applyTTL(ctx context.Context, keys ...string) {
	if s.ttl <= 0 || len(keys) == 0 {
		return
	}
	pipe := s.client.Pipeline()
	for _, key := range keys {
		pipe.Expire(ctx, key, s.ttl)
	}
	_, _ = pipe.Exec(ctx) //nolint:errcheck // best-effort by design
}

// allReserved reports whether every write in the batch targets a reserved
// channel (the empty batch vacuously qualifies, as in Python's `all(...)`,
// and writes nothing either way).
func allReserved(writes []checkpoint.Write) bool {
	for _, w := range writes {
		if _, ok := reservedWriteIdx[w.Channel]; ok {
			continue
		}
		return false
	}
	return true
}
