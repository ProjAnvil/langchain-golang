package postgres_test

import (
	"context"
	"flag"
	"fmt"
	"maps"
	"os"
	"slices"
	"testing"
	"time"

	embeddedpostgres "github.com/fergusstrange/embedded-postgres"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/savertest"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
)

// testDSN points at the package-wide embedded Postgres instance started in
// TestMain. Port 55433 avoids clashing with a locally installed Postgres.
const testPort = 55433

var testDSN = fmt.Sprintf("postgres://postgres:postgres@localhost:%d/postgres?sslmode=disable", testPort)

func TestMain(m *testing.M) {
	flag.Parse()
	if testing.Short() {
		// No database in -short mode; every test below skips itself.
		os.Exit(m.Run())
	}
	db := embeddedpostgres.NewDatabase(embeddedpostgres.DefaultConfig().Port(testPort))
	if err := db.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "embedded postgres start: %v\n", err)
		os.Exit(1)
	}
	code := m.Run()
	_ = db.Stop()
	os.Exit(code)
}

// newPool returns a pool on the shared embedded instance; the caller
// truncates via newEmptySaver. Skips in -short mode.
func newPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping embedded-postgres test in -short mode")
	}
	pool, err := pgxpool.New(context.Background(), testDSN)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// newEmptySaver returns a Saver on EMPTY storage (the savertest factory
// contract): shared tables are truncated between subtests.
func newEmptySaver(t *testing.T) checkpoint.Saver {
	t.Helper()
	pool := newPool(t)
	s := postgres.New(pool, serde.NewJSONSerializer())
	ctx := context.Background()
	if err := s.Setup(ctx); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	if _, err := pool.Exec(ctx, `TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes`); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	return s
}

func TestPostgresSaverContract(t *testing.T) {
	savertest.Run(t, newEmptySaver)
}

// TestSetupIdempotent: applying migrations repeatedly is a no-op after the
// first success — the migrations table holds exactly v0..v9. (Python's
// test_nonnull_migrations, test_sync.py:277, is a static lint over the
// MIGRATIONS list; that intent is covered by Task 3's TestMigrations static
// assertions and is not repeated here.)
func TestSetupIdempotent(t *testing.T) {
	pool := newPool(t) // also skips in -short mode
	ctx := context.Background()

	s, err := postgres.NewFromConnString(ctx, testDSN, serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("NewFromConnString: %v", err)
	}
	defer s.Close()
	if err := s.Setup(ctx); err != nil {
		t.Fatalf("Setup (1st): %v", err)
	}
	if err := s.Setup(ctx); err != nil {
		t.Fatalf("Setup (2nd): %v", err)
	}

	assertMigrationCount := func(wantCount int) {
		t.Helper()
		var count, maxV int
		if err := pool.QueryRow(ctx,
			`SELECT count(*), max(v) FROM checkpoint_migrations`).Scan(&count, &maxV); err != nil {
			t.Fatalf("query checkpoint_migrations: %v", err)
		}
		if count != wantCount || maxV != 9 {
			t.Fatalf("checkpoint_migrations count/max(v) = (%d, %d), want (%d, 9)", count, maxV, wantCount)
		}
	}
	assertMigrationCount(10)

	// A fresh Saver instance over the same database must not re-apply anything.
	s3, err := postgres.NewFromConnString(ctx, testDSN, serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("NewFromConnString (3rd): %v", err)
	}
	defer s3.Close()
	if err := s3.Setup(ctx); err != nil {
		t.Fatalf("Setup (3rd): %v", err)
	}
	assertMigrationCount(10)
}

// TestInlineBlobSplit pins the inline/checkpoint_blobs boundary (D6): only
// JSON-native primitives (nil, string, bool, float64) stay inline in the
// checkpoints JSONB document; int, maps and slices go to checkpoint_blobs.
func TestInlineBlobSplit(t *testing.T) {
	s := newEmptySaver(t)
	pool := newPool(t)
	ctx := context.Background()

	values := map[string]any{
		"s":    "x",
		"b":    true,
		"f":    1.5,
		"nilv": nil,
		"n":    7,
		"m":    map[string]any{"a": 1},
		"l":    []any{1},
	}
	newVersions := map[string]int64{"s": 1, "b": 1, "f": 1, "nilv": 1, "n": 1, "m": 1, "l": 1}
	cp := checkpoint.Checkpoint{
		V:             1,
		ID:            checkpoint.NewID(1),
		TS:            time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
		ChannelValues: values,
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{Source: "loop", Step: 0}, newVersions)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Blob channels: exactly l, m, n (int goes to blobs — the key Go/Python
	// boundary; Python's int is JSON-native, Go's int is serde-enveloped).
	rows, err := pool.Query(ctx,
		`SELECT channel FROM checkpoint_blobs WHERE thread_id = $1 ORDER BY channel`, "t1")
	if err != nil {
		t.Fatalf("query checkpoint_blobs: %v", err)
	}
	defer rows.Close()
	var blobChannels []string
	for rows.Next() {
		var ch string
		if err := rows.Scan(&ch); err != nil {
			t.Fatalf("scan blob channel: %v", err)
		}
		blobChannels = append(blobChannels, ch)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate blob channels: %v", err)
	}
	if want := []string{"l", "m", "n"}; !slices.Equal(blobChannels, want) {
		t.Fatalf("blob channels = %v, want %v", blobChannels, want)
	}

	// Inline channel_values in the JSONB document: exactly b, f, nilv, s.
	var cvJSON map[string]any
	if err := pool.QueryRow(ctx,
		`SELECT checkpoint->'channel_values' FROM checkpoints WHERE thread_id = $1`, "t1").Scan(&cvJSON); err != nil {
		t.Fatalf("query inline channel_values: %v", err)
	}
	inlineKeys := slices.Sorted(maps.Keys(cvJSON))
	if want := []string{"b", "f", "nilv", "s"}; !slices.Equal(inlineKeys, want) {
		t.Fatalf("inline channel_values keys = %v, want %v", inlineKeys, want)
	}
	if cvJSON["s"] != "x" || cvJSON["b"] != true || cvJSON["f"] != 1.5 || cvJSON["nilv"] != nil {
		t.Fatalf("inline channel_values = %v, want s=x b=true f=1.5 nilv=nil", cvJSON)
	}

	// Round trip: n restores as int, m as map[string]any.
	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if got, ok := tup.Checkpoint.ChannelValues["n"].(int); !ok || got != 7 {
		t.Fatalf("channel n = %v (%T), want int(7)", tup.Checkpoint.ChannelValues["n"], tup.Checkpoint.ChannelValues["n"])
	}
	if got, ok := tup.Checkpoint.ChannelValues["m"].(map[string]any); !ok || got["a"] != 1 {
		t.Fatalf("channel m = %v (%T), want map[string]any{a:1}", tup.Checkpoint.ChannelValues["m"], tup.Checkpoint.ChannelValues["m"])
	}
}

// TestMetadataContainmentFilter exercises the server-side metadata @> JSONB
// containment filter against a real Postgres, including a nested-object
// (parents) containment.
func TestMetadataContainmentFilter(t *testing.T) {
	s := newEmptySaver(t)
	ctx := context.Background()

	put := func(id int, md checkpoint.Metadata) {
		t.Helper()
		cp := checkpoint.Checkpoint{
			V:  1,
			ID: checkpoint.NewID(id),
			TS: time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
		}
		if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, md, nil); err != nil {
			t.Fatalf("Put %+v: %v", md, err)
		}
	}
	put(1, checkpoint.Metadata{Source: "input", Step: -1})
	put(2, checkpoint.Metadata{Source: "loop", Step: 0, Parents: map[string]string{"": "p1"}})

	list := func(filter map[string]any) []checkpoint.Tuple {
		t.Helper()
		tups, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{Filter: filter})
		if err != nil {
			t.Fatalf("List filter=%v: %v", filter, err)
		}
		return tups
	}
	if tups := list(map[string]any{"source": "loop"}); len(tups) != 1 {
		t.Fatalf("List(source=loop) = %d tuples, want 1", len(tups))
	}
	if tups := list(map[string]any{"parents": map[string]string{"": "p1"}}); len(tups) != 1 {
		t.Fatalf("List(parents@>{\"\":\"p1\"}) = %d tuples, want 1 (nested containment)", len(tups))
	}
	if tups := list(map[string]any{"step": 2}); len(tups) != 0 {
		t.Fatalf("List(step=2) = %d tuples, want 0", len(tups))
	}
}

// TestPerVersionBlobDedup: checkpoint_blobs rows are immutable and keyed by
// (channel, version) — re-Putting a checkpoint whose channel version is
// unchanged does not duplicate the blob row (ON CONFLICT DO NOTHING); bumping
// the version adds exactly one new row.
func TestPerVersionBlobDedup(t *testing.T) {
	s := newEmptySaver(t)
	pool := newPool(t)
	ctx := context.Background()

	big1 := make([]string, 100)
	for i := range big1 {
		big1[i] = fmt.Sprintf("value-%d", i)
	}
	putBig := func(id int, version int64, value []string) checkpoint.Config {
		t.Helper()
		cp := checkpoint.Checkpoint{
			V:               1,
			ID:              checkpoint.NewID(id),
			TS:              time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
			ChannelValues:   map[string]any{"big": value},
			ChannelVersions: map[string]int64{"big": version},
		}
		cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, map[string]int64{"big": version})
		if err != nil {
			t.Fatalf("Put cp%d: %v", id, err)
		}
		return cfg
	}
	blobCount := func() int {
		t.Helper()
		var n int
		if err := pool.QueryRow(ctx,
			`SELECT count(*) FROM checkpoint_blobs WHERE thread_id = 't1' AND channel = 'big'`).Scan(&n); err != nil {
			t.Fatalf("count big blobs: %v", err)
		}
		return n
	}
	assertBigValue := func(cfg checkpoint.Config, want []string) {
		t.Helper()
		tup, err := s.GetTuple(ctx, cfg)
		if err != nil || tup == nil {
			t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
		}
		got, ok := tup.Checkpoint.ChannelValues["big"].([]string)
		if !ok || !slices.Equal(got, want) {
			t.Fatalf("channel big round trip mismatch (len got %d, want %d)", len(got), len(want))
		}
	}

	// cp1: big at version 1.
	cfg1 := putBig(1, 1, big1)
	if n := blobCount(); n != 1 {
		t.Fatalf("after cp1: big blob rows = %d, want 1", n)
	}
	// cp2: new checkpoint ID, big STILL at version 1 and again in newVersions —
	// deduped by ON CONFLICT DO NOTHING.
	cfg2 := putBig(2, 1, big1)
	if n := blobCount(); n != 1 {
		t.Fatalf("after cp2: big blob rows = %d, want 1 (per-version dedup)", n)
	}
	// cp3: big advances to version 2 with a new value.
	big2 := append(slices.Clone(big1), "value-100")
	cfg3 := putBig(3, 2, big2)
	if n := blobCount(); n != 2 {
		t.Fatalf("after cp3: big blob rows = %d, want 2 (new version adds one row)", n)
	}

	assertBigValue(cfg1, big1)
	assertBigValue(cfg2, big1)
	assertBigValue(cfg3, big2)
}

// TestNullCharsRejectedDivergence pins a deliberate divergence from Python
// (test_sync.py:262 test_null_chars): Python silently strips \x00 from
// metadata strings before writing (checkpoint/base/__init__.py:762,772); Go
// fails loud — Postgres rejects \u0000 in JSONB strings, and the saver
// surfaces that error instead of silently rewriting user data. The
// divergence is listed in doc.go.
func TestNullCharsRejectedDivergence(t *testing.T) {
	s := newEmptySaver(t)
	ctx := context.Background()

	base := checkpoint.Checkpoint{
		V:  1,
		ID: checkpoint.NewID(1),
		TS: time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
	}

	t.Run("channel value", func(t *testing.T) {
		cp := base
		cp.ChannelValues = map[string]any{"bad": "has\x00null"}
		_, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, map[string]int64{"bad": 1})
		if err == nil {
			t.Fatal("Put with \\x00 in a channel value: got nil error, want fail-loud rejection")
		}
	})

	t.Run("metadata source", func(t *testing.T) {
		cp := base
		cp.ID = checkpoint.NewID(2)
		_, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{Source: "bad\x00source"}, nil)
		if err == nil {
			t.Fatal("Put with \\x00 in Metadata.Source: got nil error, want fail-loud rejection")
		}
	})

	t.Run("metadata parents", func(t *testing.T) {
		cp := base
		cp.ID = checkpoint.NewID(3)
		_, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp,
			checkpoint.Metadata{Parents: map[string]string{"": "bad\x00parent"}}, nil)
		if err == nil {
			t.Fatal("Put with \\x00 in Metadata.Parents: got nil error, want fail-loud rejection")
		}
	})
}

// TestPutWritesTaskPathStored: the v9 task_path column actually takes effect —
// a direct SQL read shows PutWrites(..., "path/a") persisted as 'path/a' and
// an empty taskPath as ”.
func TestPutWritesTaskPathStored(t *testing.T) {
	s := newEmptySaver(t)
	pool := newPool(t)
	ctx := context.Background()

	cp := checkpoint.Checkpoint{
		V:  1,
		ID: checkpoint.NewID(1),
		TS: time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC),
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, cp, checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "a", Value: 1}}, "task-1", "path/a"); err != nil {
		t.Fatalf("PutWrites task-1: %v", err)
	}
	if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "b", Value: 2}}, "task-2", ""); err != nil {
		t.Fatalf("PutWrites task-2: %v", err)
	}

	taskPath := func(taskID string) string {
		t.Helper()
		var path string
		if err := pool.QueryRow(ctx,
			`SELECT task_path FROM checkpoint_writes WHERE thread_id = 't1' AND task_id = $1`, taskID).Scan(&path); err != nil {
			t.Fatalf("query task_path for %q: %v", taskID, err)
		}
		return path
	}
	if got := taskPath("task-1"); got != "path/a" {
		t.Fatalf("task-1 task_path = %q, want %q", got, "path/a")
	}
	if got := taskPath("task-2"); got != "" {
		t.Fatalf("task-2 task_path = %q, want empty string", got)
	}
}
