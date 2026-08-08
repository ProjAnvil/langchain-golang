package postgres

// Migrations mirrors Python's MIGRATIONS
// (libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py:43-91)
// statement-for-statement; the slice index IS the schema version. The only
// deliberate deviation is v2's version column: BIGINT instead of TEXT,
// because Go channel versions are int64 (documented in doc.go — one of the
// two reasons cross-language database sharing is impossible).
//
// Setup executes these WITHOUT a transaction: v6–v8 are CREATE INDEX
// CONCURRENTLY, which Postgres forbids inside a transaction block.
var Migrations = []string{
	// v0
	`CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);`,
	// v1
	`CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);`,
	// v2 — deviation: version BIGINT (Python: version TEXT NOT NULL)
	`CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version BIGINT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);`,
	// v3
	`CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);`,
	// v4
	`ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP not null;`,
	// v5 — no-op migration, kept so migration table versions stay aligned
	// with Python (mirrors base.py:78-80).
	`SELECT 1;`,
	// v6
	`
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
    `,
	// v7
	`
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
    `,
	// v8
	`
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
    `,
	// v9
	`ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT '';`,
}
