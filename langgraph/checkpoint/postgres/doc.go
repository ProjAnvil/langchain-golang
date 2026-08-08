// Package postgres implements checkpoint.Saver on PostgreSQL, the Go port
// of Python's langgraph-checkpoint-postgres (BasePostgresSaver). It uses
// pgx/v5 with pgxpool and mirrors Python's four-table schema
// (checkpoints / checkpoint_blobs / checkpoint_writes /
// checkpoint_migrations) with per-version channel blobs.
//
// Setup must be called explicitly once before first use; it applies pending
// migrations WITHOUT a transaction (migrations v6–v8 are CREATE INDEX
// CONCURRENTLY, which cannot run inside a transaction block).
//
// Documented divergences from Python:
//   - checkpoint_blobs.version is BIGINT (Go channel versions are int64);
//     Python stores the column as TEXT holding decimal strings
//     (base.py casts versions via cast(str, ver)).
//   - Cross-language database sharing is NOT possible: the serde byte
//     formats differ (Go JSON typed envelopes vs Python msgpack), on top of
//     the version column type divergence. A database written by one
//     language cannot be read by the other.
//   - Channel values inline only JSON primitives (nil/string/bool/float64)
//     into the checkpoints JSONB document; int/int64 (serde-enveloped, not
//     JSON-native), maps, slices and all registry types go to
//     checkpoint_blobs (Python inlines str/int/float/bool and sends
//     dict/list to blobs).
//   - checkpoint.Metadata is a closed struct (Source/Step/Parents), so
//     ListOptions.Filter keys are limited to those three.
//   - Null bytes in strings: Python silently strips \u0000 from metadata
//     strings before writing (langgraph/checkpoint/base/__init__.py's
//     get_checkpoint_metadata, lines 762 and 772); Go fails loudly instead —
//     Put returns an error rather than silently mutating user data.
//   - No Shallow saver variant and no delta channel history fast path:
//     Python's ShallowPostgresSaver and _DeltaSnapshot have no Go
//     counterparts.
package postgres
