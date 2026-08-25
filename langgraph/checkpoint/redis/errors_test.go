package redis_test

import (
	"context"
	"strings"
	"testing"

	goredis "github.com/redis/go-redis/v9"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/redis"
)

// rawClient opens a second client on the test server, for tests that must
// plant keys the Saver API would never write.
func rawClient(mrAddr string) goredis.UniversalClient {
	return goredis.NewClient(&goredis.Options{Addr: mrAddr})
}

// TestServerFailures injects a server-side error (miniredis SetError) and
// verifies every Saver method reports it instead of panicking or silently
// succeeding.
func TestServerFailures(t *testing.T) {
	ctx := context.Background()
	s, mr := newSaver(t)

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}

	mr.SetError("boom")
	defer mr.SetError("")

	if _, err := s.GetTuple(ctx, cfg); err == nil {
		t.Error("GetTuple (by ID): expected error, got nil")
	}
	if _, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"}); err == nil {
		t.Error("GetTuple (latest): expected error, got nil")
	}
	if _, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{}); err == nil {
		t.Error("List: expected error, got nil")
	}
	if _, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(2)), checkpoint.Metadata{}, nil); err == nil {
		t.Error("Put: expected error, got nil")
	}
	if err := s.PutWrites(ctx, cfg, []checkpoint.Write{{Channel: "c", Value: 1}}, "task-1", ""); err == nil {
		t.Error("PutWrites (explicit ID): expected error, got nil")
	}
	if err := s.PutWrites(ctx, checkpoint.Config{ThreadID: "t1"}, []checkpoint.Write{{Channel: "c", Value: 1}}, "task-1", ""); err == nil {
		t.Error("PutWrites (latest-ID resolution): expected error, got nil")
	}
	if err := s.DeleteThread(ctx, "t1"); err == nil {
		t.Error("DeleteThread: expected error, got nil")
	}
	if err := s.DeleteForRuns(ctx, []string{"r1"}); err == nil {
		t.Error("DeleteForRuns: expected error, got nil")
	}
	if err := s.CopyThread(ctx, "t1", "t2"); err == nil {
		t.Error("CopyThread: expected error, got nil")
	}
	if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err == nil {
		t.Error("Prune (keep_latest): expected error, got nil")
	}
	if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneDeleteAll); err == nil {
		t.Error("Prune (delete): expected error, got nil")
	}

	// The error is transient: clearing it restores service.
	mr.SetError("")
	if _, err := s.GetTuple(ctx, cfg); err != nil {
		t.Errorf("GetTuple after clearing server error: %v", err)
	}
}

// TestServerClosed verifies operations against a server that went away fail
// with an error (connection loss), never a panic.
func TestServerClosed(t *testing.T) {
	ctx := context.Background()
	s, mr := newSaver(t)
	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	mr.Close()

	if _, err := s.GetTuple(ctx, cfg); err == nil {
		t.Error("GetTuple: expected connection error, got nil")
	}
	if _, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{}); err == nil {
		t.Error("List: expected connection error, got nil")
	}
	if err := s.DeleteThread(ctx, "t1"); err == nil {
		t.Error("DeleteThread: expected connection error, got nil")
	}
}

// TestManagementMalformedStorage plants keys the Saver API would never write
// (wrong-type keys, malformed metadata, keys with too few/many components)
// and verifies the management methods fail descriptively instead of
// panicking or silently corrupting state — the same technique as
// TestCorruptCheckpointHashes.
func TestManagementMalformedStorage(t *testing.T) {
	ctx := context.Background()

	// DeleteForRuns: a checkpoint-prefixed STRING key fails the metadata
	// HGet with WRONGTYPE.
	t.Run("delete_for_runs wrong-type checkpoint key", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.Set(ctx, "checkpoint:t1::c1", "x", 0).Err(); err != nil {
			t.Fatalf("plant string: %v", err)
		}
		if err := s.DeleteForRuns(ctx, []string{"r1"}); err == nil {
			t.Fatal("DeleteForRuns with string checkpoint key: expected error, got nil")
		}
	})

	// DeleteForRuns: a hash whose metadata field is not JSON fails decode.
	t.Run("delete_for_runs malformed metadata", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.HSet(ctx, "checkpoint:t1::c1", "metadata", "not json").Err(); err != nil {
			t.Fatalf("plant hash: %v", err)
		}
		if err := s.DeleteForRuns(ctx, []string{"r1"}); err == nil {
			t.Fatal("DeleteForRuns with malformed metadata: expected error, got nil")
		}
	})

	// DeleteForRuns: a matching hash whose key has too few components is a
	// malformed-key error.
	t.Run("delete_for_runs malformed key", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.HSet(ctx, "checkpoint:t1", "metadata", `{"source":"x","step":0,"run_id":"r1"}`).Err(); err != nil {
			t.Fatalf("plant hash: %v", err)
		}
		if err := s.DeleteForRuns(ctx, []string{"r1"}); err == nil {
			t.Fatal("DeleteForRuns with malformed key: expected error, got nil")
		}
	})

	// DeleteForRuns: the delete pipeline fails when the ordering zset key is
	// a STRING (ZRem on a wrong-type key).
	t.Run("delete_for_runs wrong-type zset", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.HSet(ctx, "checkpoint:t1::c1", "metadata", `{"source":"x","step":0,"run_id":"r1"}`).Err(); err != nil {
			t.Fatalf("plant hash: %v", err)
		}
		if err := raw.Set(ctx, "checkpoint_zset:t1:", "x", 0).Err(); err != nil {
			t.Fatalf("plant string zset: %v", err)
		}
		if err := s.DeleteForRuns(ctx, []string{"r1"}); err == nil {
			t.Fatal("DeleteForRuns with wrong-type zset: expected error, got nil")
		}
	})

	// CopyThread: a scanned key with too few components is a malformed-key
	// error (glob `*` matches the empty component, so "checkpoint:t1:" scans).
	t.Run("copy_thread malformed key", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.HSet(ctx, "checkpoint:t1:", "metadata", `{}`).Err(); err != nil {
			t.Fatalf("plant hash: %v", err)
		}
		if err := s.CopyThread(ctx, "t1", "t2"); err == nil {
			t.Fatal("CopyThread with malformed key: expected error, got nil")
		}
	})

	// CopyThread: a STRING where the checkpoint hash should be fails HGetAll.
	t.Run("copy_thread wrong-type checkpoint key", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.Set(ctx, "checkpoint:t1::c1", "x", 0).Err(); err != nil {
			t.Fatalf("plant string: %v", err)
		}
		if err := s.CopyThread(ctx, "t1", "t2"); err == nil {
			t.Fatal("CopyThread with string checkpoint key: expected error, got nil")
		}
	})

	// CopyThread: the write pipeline fails when the destination zset key is a
	// STRING (ZAdd on a wrong-type key).
	t.Run("copy_thread wrong-type zset", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.HSet(ctx, "checkpoint:t1::c1", "metadata", `{}`).Err(); err != nil {
			t.Fatalf("plant hash: %v", err)
		}
		if err := raw.Set(ctx, "checkpoint_zset:t2:", "x", 0).Err(); err != nil {
			t.Fatalf("plant string zset: %v", err)
		}
		if err := s.CopyThread(ctx, "t1", "t2"); err == nil {
			t.Fatal("CopyThread with wrong-type zset: expected error, got nil")
		}
	})

	// CopyThread: a write key holding a HASH fails the per-write GET.
	t.Run("copy_thread wrong-type write key", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.HSet(ctx, "checkpoint:t1::c1", "metadata", `{}`).Err(); err != nil {
			t.Fatalf("plant hash: %v", err)
		}
		if err := raw.HSet(ctx, "checkpoint_write:t1::c1:task-1:0", "x", "y").Err(); err != nil {
			t.Fatalf("plant write hash: %v", err)
		}
		if err := s.CopyThread(ctx, "t1", "t2"); err == nil {
			t.Fatal("CopyThread with wrong-type write key: expected error, got nil")
		}
	})

	// Prune: a scanned zset key with too many components is a malformed-key
	// error (a legit namespace containing ':' would be key-escaped).
	t.Run("prune malformed zset key", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.ZAdd(ctx, "checkpoint_zset:t1:a:b", goredis.Z{Score: 0, Member: "c1"}).Err(); err != nil {
			t.Fatalf("plant zset: %v", err)
		}
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err == nil {
			t.Fatal("Prune with malformed zset key: expected error, got nil")
		}
	})

	// Prune: a STRING where the ordering zset should be fails ZRange.
	t.Run("prune wrong-type zset", func(t *testing.T) {
		s, mr := newSaver(t)
		raw := rawClient(mr.Addr())
		if err := raw.Set(ctx, "checkpoint_zset:t1:", "x", 0).Err(); err != nil {
			t.Fatalf("plant string zset: %v", err)
		}
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err == nil {
			t.Fatal("Prune with wrong-type zset: expected error, got nil")
		}
	})
}

// TestCorruptCheckpointHashes plants checkpoint hashes the Saver API could
// never write (foreign blob types, malformed JSON, unknown serde tags) and
// verifies reads fail with descriptive errors rather than returning
// corrupted state.
func TestCorruptCheckpointHashes(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name    string
		fields  map[string]any
		wantErr string
	}{
		{
			name: "unknown blob type",
			fields: map[string]any{
				"type":       "pickle",
				"checkpoint": `{"v":1,"id":"c1"}`,
			},
			wantErr: `unknown checkpoint blob type "pickle"`,
		},
		{
			name: "malformed checkpoint JSON",
			fields: map[string]any{
				"type":       "json",
				"checkpoint": `not json`,
			},
			wantErr: "decode checkpoint",
		},
		{
			name: "unknown channel value serde tag",
			fields: map[string]any{
				"type":       "json",
				"checkpoint": `{"v":1,"id":"c1","channel_values":{"c":{"type":"bogus","data":{}}}}`,
			},
			wantErr: `channel "c"`,
		},
		{
			name: "unknown task arg serde tag",
			fields: map[string]any{
				"type":       "json",
				"checkpoint": `{"v":1,"id":"c1","next":[{"id":"t","node":"n","arg":{"a":{"type":"bogus","data":{}}}}]}`,
			},
			wantErr: `arg "a"`,
		},
		{
			name: "malformed metadata JSON",
			fields: map[string]any{
				"type":       "json",
				"checkpoint": `{"v":1,"id":"c1"}`,
				"metadata":   `not json`,
			},
			wantErr: "decode metadata",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, mr := newSaver(t)
			raw := rawClient(mr.Addr())
			if err := raw.HSet(ctx, "checkpoint:t1::c1", tt.fields).Err(); err != nil {
				t.Fatalf("plant hash: %v", err)
			}
			if err := raw.ZAdd(ctx, "checkpoint_zset:t1:", goredis.Z{Score: 0, Member: "c1"}).Err(); err != nil {
				t.Fatalf("plant zset entry: %v", err)
			}

			if _, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"}); err == nil {
				t.Fatalf("GetTuple: expected error containing %q, got nil", tt.wantErr)
			} else if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("GetTuple error = %v, want it to contain %q", err, tt.wantErr)
			}
			// List must surface the same corruption instead of skipping it.
			if _, err := s.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{}); err == nil {
				t.Fatal("List: expected error on corrupt hash, got nil")
			}
		})
	}
}

// TestCorruptWriteValues plants checkpoint_write keys the Saver API could
// never write and verifies reads fail descriptively (or skip, for a write
// that vanished between SCAN and GET).
func TestCorruptWriteValues(t *testing.T) {
	ctx := context.Background()

	setup := func(t *testing.T) (*redis.Saver, goredis.UniversalClient, checkpoint.Config) {
		t.Helper()
		s, mr := newSaver(t)
		cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		return s, rawClient(mr.Addr()), cfg
	}
	writeKey := func(cfg checkpoint.Config) string {
		return "checkpoint_write:t1::" + cfg.CheckpointID + ":task-1:0"
	}

	t.Run("malformed write JSON", func(t *testing.T) {
		s, raw, cfg := setup(t)
		if err := raw.Set(ctx, writeKey(cfg), `not json`, 0).Err(); err != nil {
			t.Fatalf("plant write: %v", err)
		}
		if _, err := s.GetTuple(ctx, cfg); err == nil {
			t.Fatal("GetTuple: expected decode error, got nil")
		} else if !strings.Contains(err.Error(), "decode write") {
			t.Fatalf("GetTuple error = %v, want a decode write failure", err)
		}
	})

	t.Run("unknown serde tag", func(t *testing.T) {
		s, raw, cfg := setup(t)
		blob := `{"task_id":"task-1","task_path":"","channel":"c","idx":0,"type":"bogus","data":{}}`
		if err := raw.Set(ctx, writeKey(cfg), blob, 0).Err(); err != nil {
			t.Fatalf("plant write: %v", err)
		}
		if _, err := s.GetTuple(ctx, cfg); err == nil {
			t.Fatal("GetTuple: expected serde decode error, got nil")
		} else if !strings.Contains(err.Error(), `channel "c"`) {
			t.Fatalf("GetTuple error = %v, want it to name channel %q", err, "c")
		}
	})

	t.Run("wrong key type", func(t *testing.T) {
		s, raw, cfg := setup(t)
		// A hash where loadWrites GETs a string: WRONGTYPE per command.
		if err := raw.HSet(ctx, writeKey(cfg), "f", "v").Err(); err != nil {
			t.Fatalf("plant write: %v", err)
		}
		if _, err := s.GetTuple(ctx, cfg); err == nil {
			t.Fatal("GetTuple: expected WRONGTYPE error, got nil")
		}
	})
}

// TestPutEncodeErrors verifies that values the Serializer cannot round-trip
// (here a chan, which has no lossless JSON form) fail Put with a wrapped
// error — from a channel value and from a planned task's arg alike — instead
// of being persisted lossily.
func TestPutEncodeErrors(t *testing.T) {
	ctx := context.Background()
	s, _ := newSaver(t)

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
// PutWrites with a wrapped error naming the channel, before anything is
// written.
func TestPutWritesEncodeError(t *testing.T) {
	ctx := context.Background()
	s, mr := newSaver(t)

	cfg, err := s.Put(ctx, checkpoint.Config{ThreadID: "t1"}, sampleCheckpoint(checkpoint.NewID(1)), checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	keysBefore := len(mr.Keys())
	err = s.PutWrites(ctx, cfg, []checkpoint.Write{
		{Channel: "good", Value: 1},
		{Channel: "bad", Value: make(chan int)},
	}, "task-1", "")
	if err == nil {
		t.Fatal("PutWrites with unserializable value: expected error, got nil")
	}
	if !strings.Contains(err.Error(), `channel "bad"`) {
		t.Fatalf("PutWrites error = %v, want it to name channel %q", err, "bad")
	}
	if len(mr.Keys()) != keysBefore {
		t.Fatalf("failed batch left keys behind: before %d, after %d", keysBefore, len(mr.Keys()))
	}
}
