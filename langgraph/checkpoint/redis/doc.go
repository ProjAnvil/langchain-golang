// Package redis implements checkpoint.Saver on Redis, the Go port of
// Python's langgraph-checkpoint-redis (RedisSaver): durable versioned
// checkpoints and pending writes in a Redis instance.
//
// This package is its own Go module (like the sibling sqlite/postgres
// savers). Consumers add it with:
//
//	go get github.com/projanvil/langchain-golang/langgraph/checkpoint/redis
//
// Run this module's tests from its own directory (`make test-redis` from the
// repo root does the same): the root module's `go test ./...` does not cross
// the nested-module boundary. The tests run fully offline against miniredis.
//
// # Storage model (no RediSearch / RedisJSON)
//
// Python's saver stores checkpoints as RedisJSON documents and queries them
// through RediSearch (FT.*) indexes. This port deliberately uses plain Redis
// commands only, so it works against any Redis-compatible server (and
// miniredis, which supports neither module):
//
//	checkpoint:{thread}:{ns}:{checkpoint_id}                        HASH
//	    fields: parent_checkpoint_id, type, checkpoint, metadata.
//	    `checkpoint`/`metadata` are the same plain-JSON projections the
//	    sqlite saver uses (each channel value individually serde-typed).
//	checkpoint_zset:{thread}:{ns}                                   ZSET
//	    members are checkpoint IDs, all at score 0. Equal scores order
//	    members lexicographically, and checkpoint.NewID makes lexicographic
//	    order match chronological order, so ZREVRANGE yields newest-first
//	    and the latest checkpoint is ZREVRANGE key 0 0.
//	checkpoint_write:{thread}:{ns}:{checkpoint_id}:{task_id}:{idx}  STRING
//	    one pending write per key: a JSON envelope with the serde-typed
//	    value plus task_id/task_path/channel/idx bookkeeping. First-write-
//	    wins slots use SET NX; all-reserved batches overwrite (SET),
//	    mirroring Python's put_writes batch rule. Key components are
//	    storage-safe escaped (`\` and `:` are backslash-escaped, the
//	    structural counterpart of Python's to_storage_safe_str/id).
//
// Reads construct keys from the Config and enumerate writes/threads with
// SCAN; keys are never parsed back into components.
//
// # Documented divergences from Python
//
//   - No RediSearch/RedisJSON: querying is replaced by the per-(thread,
//     namespace) zset plus SCAN enumeration. Consequently there is no Setup
//     call and no index schema at all.
//   - No empty-string sentinels: Python's `__empty__` / empty-UUID sentinels
//     exist because RediSearch cannot index empty strings; plain Redis keys
//     handle empty components fine, so only structural escaping is applied.
//   - TTL config is a time.Duration (WithTTL) instead of Python's
//     `{"default_ttl": minutes}` dict; application is best-effort EXPIRE on
//     Put/PutWrites, with optional refresh on read (WithRefreshOnRead).
//     Python's ttl=-1 PERSIST case has no Go counterpart.
//   - Go has no sync/async split and no shallow saver variant: Python's
//     AsyncRedisSaver / ShallowRedisSaver / AsyncShallowRedisSaver are not
//     ported, and neither is the Redis Store (vector search) or the message
//     exporter middleware — checkpoint saving only.
//   - checkpoint.Metadata is a closed struct (Source/Step/Parents), so
//     ListOptions.Filter keys are limited to those three, as in the other
//     Go savers.
//   - Cross-language database sharing is NOT possible: beyond the different
//     storage layout, the serde byte formats differ (Go JSON typed envelopes
//     vs Python's JsonPlusRedisSerializer).
package redis
