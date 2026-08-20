package redis_test

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/redis"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/savertest"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// newSaver starts a fresh miniredis server and returns a Saver on it,
// registering cleanup for both.
func newSaver(t *testing.T, opts ...redis.Option) (*redis.Saver, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	s := redis.New(goredis.NewClient(&goredis.Options{Addr: mr.Addr()}), serde.NewJSONSerializer(), opts...)
	return s, mr
}

// sampleCheckpoint returns a checkpoint exercising every projection field:
// channel values spanning plain JSON and registry types, version maps, a
// planned task with a typed arg, and a fixed timestamp for exact comparison.
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
// miniredis server per subtest.
func TestSaverContract(t *testing.T) {
	savertest.Run(t, func(t *testing.T) checkpoint.Saver {
		s, _ := newSaver(t)
		return s
	})
}

// TestNewPanics covers New's programmer-error guards.
func TestNewPanics(t *testing.T) {
	ser := serde.NewJSONSerializer()
	mr := miniredis.RunT(t)
	client := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})

	assertPanics := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic", name)
			}
		}()
		fn()
	}
	assertPanics("nil client", func() { redis.New(nil, ser) })
	assertPanics("nil serde", func() { redis.New(client, nil) })
}

// TestNewFromConnString covers the URL constructor: a working server
// round-trips a checkpoint, Close (owning the client) breaks further calls,
// and both a malformed URL and an unreachable server fail at construction.
func TestNewFromConnString(t *testing.T) {
	ctx := context.Background()
	ser := serde.NewJSONSerializer()

	mr := miniredis.RunT(t)
	s, err := redis.NewFromConnString(ctx, "redis://"+mr.Addr(), ser)
	if err != nil {
		t.Fatalf("NewFromConnString: %v", err)
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{Source: "loop", Step: 0}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	tup, err := s.GetTuple(ctx, cfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}

	// Close owns the client: subsequent operations fail.
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := s.GetTuple(ctx, cfg); err == nil {
		t.Fatal("GetTuple after Close: expected error, got nil")
	}

	if _, err := redis.NewFromConnString(ctx, "://not a url", ser); err == nil {
		t.Fatal("NewFromConnString with malformed URL: expected error, got nil")
	}

	// A closed server's address refuses the Ping.
	dead := miniredis.RunT(t)
	addr := dead.Addr()
	dead.Close()
	if _, err := redis.NewFromConnString(ctx, "redis://"+addr, ser); err == nil {
		t.Fatal("NewFromConnString to a dead server: expected error, got nil")
	}
}

// TestCloseCallerSuppliedClient verifies Close is a no-op for savers built
// on a caller-supplied client: the client stays usable afterwards.
func TestCloseCallerSuppliedClient(t *testing.T) {
	ctx := context.Background()
	s, mr := newSaver(t)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put after Close of caller-supplied client: %v", err)
	}
	if tup, err := s.GetTuple(ctx, cfg); err != nil || tup == nil {
		t.Fatalf("GetTuple after Close of caller-supplied client: tup=%v err=%v", tup, err)
	}
	if len(mr.Keys()) == 0 {
		t.Fatal("expected keys in the server after Put")
	}
}

// TestTTL verifies WithTTL: Put expires the checkpoint hash and ordering
// zset, PutWrites expires the write keys, entries actually disappear after
// the TTL elapses, and reads with WithRefreshOnRead extend the TTL.
func TestTTL(t *testing.T) {
	ctx := context.Background()

	t.Run("keys expire", func(t *testing.T) {
		s, mr := newSaver(t, redis.WithTTL(time.Minute))
		cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c", Value: "v"}}, "task-1", ""); err != nil {
			t.Fatalf("PutWrites: %v", err)
		}
		// Every key the saver wrote carries the TTL.
		for _, key := range mr.Keys() {
			if got := mr.TTL(key); got != time.Minute {
				t.Fatalf("TTL(%q) = %v, want 1m", key, got)
			}
		}
		// After the TTL elapses the checkpoint is gone.
		mr.FastForward(2 * time.Minute)
		tup, err := s.GetTuple(ctx, cfg)
		if err != nil || tup != nil {
			t.Fatalf("GetTuple after expiry: tup=%v err=%v, want nil, nil", tup, err)
		}
	})

	t.Run("refresh on read", func(t *testing.T) {
		s, mr := newSaver(t, redis.WithTTL(time.Minute), redis.WithRefreshOnRead(true))
		cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c", Value: "v"}}, "task-1", ""); err != nil {
			t.Fatalf("PutWrites: %v", err)
		}
		mr.FastForward(50 * time.Second)
		if _, err := s.GetTuple(ctx, cfg); err != nil {
			t.Fatalf("GetTuple: %v", err)
		}
		// The read refreshed every key of the checkpoint back to the full TTL.
		for _, key := range mr.Keys() {
			if got := mr.TTL(key); got != time.Minute {
				t.Fatalf("after refresh TTL(%q) = %v, want 1m", key, got)
			}
		}
		// List refreshes too.
		mr.FastForward(50 * time.Second)
		if _, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{}); err != nil {
			t.Fatalf("List: %v", err)
		}
		for _, key := range mr.Keys() {
			if got := mr.TTL(key); got != time.Minute {
				t.Fatalf("after List refresh TTL(%q) = %v, want 1m", key, got)
			}
		}
	})

	t.Run("no ttl configured", func(t *testing.T) {
		s, mr := newSaver(t)
		if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil); err != nil {
			t.Fatalf("Put: %v", err)
		}
		for _, key := range mr.Keys() {
			if got := mr.TTL(key); got != 0 {
				t.Fatalf("TTL(%q) = %v, want none", key, got)
			}
		}
	})
}

// TestPutWritesResolvesLatestCheckpoint exercises resolveCheckpointID's
// empty-CheckpointID branch: the writes attach to the thread's LATEST
// checkpoint, and an empty thread is an error (matching MemorySaver).
func TestPutWritesResolvesLatestCheckpoint(t *testing.T) {
	ctx := context.Background()
	s, _ := newSaver(t)

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
	s, _ := newSaver(t)

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

// TestRePutReplacesRecord verifies a re-Put of the same checkpoint ID fully
// replaces the previous record — including clearing a parent link — matching
// sqlite's INSERT OR REPLACE row semantics.
func TestRePutReplacesRecord(t *testing.T) {
	ctx := context.Background()
	s, _ := newSaver(t)

	parent, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put parent: %v", err)
	}
	child := sampleCheckpoint(checkpoint.NewID(2))
	if _, err := s.Put(ctx, parent, child, checkpoint.Metadata{Source: "loop", Step: 0}, nil); err != nil {
		t.Fatalf("Put child: %v", err)
	}
	// Re-put the same child ID from a config without a parent.
	if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, child, checkpoint.Metadata{Source: "loop", Step: 0}, nil); err != nil {
		t.Fatalf("re-Put child: %v", err)
	}
	tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: child.ID})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple: tup=%v err=%v", tup, err)
	}
	if tup.ParentConfig != nil {
		t.Fatalf("ParentConfig = %+v, want nil after parentless re-Put", tup.ParentConfig)
	}
}

// TestEscapedComponents verifies storage-safe key escaping: thread IDs,
// namespaces, checkpoint IDs and task IDs containing the key separator ":"
// and the Redis glob metacharacters "*", "?", "[", "]" and "\" round-trip
// correctly and stay isolated from lookalike threads (DeleteThread included).
func TestEscapedComponents(t *testing.T) {
	ctx := context.Background()
	s, _ := newSaver(t)

	// The "plain" thread is the concatenation of the weird thread and ns:
	// without escaping their keys would collide.
	weird := checkpoint.Config{ThreadID: `a:b*[c]?\d`, CheckpointNS: "e:f"}
	plain := checkpoint.Config{ThreadID: `a:b*[c]?\d:e`, CheckpointNS: "f"}

	cfgW, err := s.Put(ctx, weird, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{Source: "input", Step: -1}, nil)
	if err != nil {
		t.Fatalf("Put weird: %v", err)
	}
	if _, err := s.Put(ctx, plain, sampleCheckpoint(checkpoint.NewID(2)), checkpoint.Metadata{Source: "loop", Step: 0}, nil); err != nil {
		t.Fatalf("Put plain: %v", err)
	}
	if err := s.PutWrites(ctx, cfgW, []checkpoint.Write{{Channel: "c", Value: "w"}}, "task:[1]*", ""); err != nil {
		t.Fatalf("PutWrites weird task: %v", err)
	}

	tup, err := s.GetTuple(ctx, cfgW)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple weird: tup=%v err=%v", tup, err)
	}
	if tup.Config != cfgW {
		t.Fatalf("Config = %+v, want %+v", tup.Config, cfgW)
	}
	if len(tup.PendingWrites) != 1 || tup.PendingWrites[0].Value != "w" {
		t.Fatalf("PendingWrites = %+v, want the weird-thread write", tup.PendingWrites)
	}
	list, err := s.List(ctx, plain, checkpoint.ListOptions{})
	if err != nil || len(list) != 1 {
		t.Fatalf("List plain: %v, %v; want exactly its own checkpoint", list, err)
	}

	// DeleteThread removes only the exact thread, glob metacharacters and
	// all: the lookalike thread must survive.
	if err := s.DeleteThread(ctx, weird.ThreadID); err != nil {
		t.Fatalf("DeleteThread: %v", err)
	}
	if tup, err := s.GetTuple(ctx, cfgW); err != nil || tup != nil {
		t.Fatalf("deleted weird thread: GetTuple = %v, %v; want nil, nil", tup, err)
	}
	if tup, err := s.GetTuple(ctx, plain); err != nil || tup == nil {
		t.Fatalf("plain thread must survive DeleteThread(%q): tup=%v err=%v", weird.ThreadID, tup, err)
	}
}

// TestListSkipsOrphanIndexEntries covers the TTL/partial-failure edge where
// the ordering zset references a checkpoint hash that no longer exists:
// List skips the orphan instead of erroring.
func TestListSkipsOrphanIndexEntries(t *testing.T) {
	ctx := context.Background()
	s, mr := newSaver(t)

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	// Plant an orphan index entry (highest ID, no hash behind it).
	orphan := checkpoint.NewID(2)
	if err := goredis.NewClient(&goredis.Options{Addr: mr.Addr()}).ZAdd(ctx,
		"checkpoint_zset:t1:", goredis.Z{Score: 0, Member: orphan}).Err(); err != nil {
		t.Fatalf("plant orphan: %v", err)
	}

	list, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 1 || list[0].Checkpoint.ID != cfg.CheckpointID {
		t.Fatalf("List = %+v, want only the real checkpoint (orphan skipped)", list)
	}
	// "Latest" resolves to the orphan, whose hash is missing: (nil, nil).
	if tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"}); err != nil || tup != nil {
		t.Fatalf("GetTuple latest with orphan head: tup=%v err=%v, want nil, nil", tup, err)
	}
}

// TestScanPagination forces scanKeys past one SCAN batch: 150 checkpoints
// (and their writes) exceed the 100-key SCAN count, exercising the cursor
// loop in both List and DeleteThread.
func TestScanPagination(t *testing.T) {
	ctx := context.Background()
	s, _ := newSaver(t)

	cfg := checkpoint.Config{ThreadID: "t1"}
	for i := 1; i <= 150; i++ {
		next, err := s.Put(ctx, cfg, sampleCheckpoint(checkpoint.NewID(i)), checkpoint.Metadata{Source: "loop", Step: i}, nil)
		if err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
		cfg = next
	}
	list, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 150 {
		t.Fatalf("List returned %d tuples, want 150", len(list))
	}
	if err := s.DeleteThread(ctx, "t1"); err != nil {
		t.Fatalf("DeleteThread: %v", err)
	}
	list, err = s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil || len(list) != 0 {
		t.Fatalf("List after DeleteThread = %v, %v; want empty", list, err)
	}
}
