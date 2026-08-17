package graph

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

func TestCheckpointSinkSyncPutCheckpoint(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	sink := newCheckpointSink(saver, DurabilitySync, context.Background(), nil)

	cp := checkpoint.Checkpoint{
		V:             1,
		ID:            "cp-1",
		TS:            time.Now(),
		ChannelValues: map[string]any{"x": 42},
	}
	cfg := checkpoint.Config{ThreadID: "t1"}

	resultCfg, err := sink.putCheckpoint(context.Background(), cfg, cp, checkpoint.Metadata{Source: "input", Step: -1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resultCfg.CheckpointID != "cp-1" {
		t.Fatalf("expected checkpoint ID cp-1, got %s", resultCfg.CheckpointID)
	}

	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatal("checkpoint not found in saver")
	}

	if err := sink.flush(); err != nil {
		t.Fatal(err)
	}
}

func TestCheckpointSinkSyncPutWrites(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	saver.Put(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.Checkpoint{
		V: 1, ID: "cp-1", TS: time.Now(),
	}, checkpoint.Metadata{Source: "input", Step: -1}, nil)

	sink := newCheckpointSink(saver, DurabilitySync, context.Background(), nil)

	writes := []checkpoint.Write{{Channel: "msg", Value: "hello"}}
	err := sink.putWrites(context.Background(), checkpoint.Config{ThreadID: "t1", CheckpointID: "cp-1"}, writes, "task-1")
	if err != nil {
		t.Fatal(err)
	}

	tup, _ := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if len(tup.PendingWrites) == 0 {
		t.Fatal("expected pending writes in saver")
	}

	sink.flush()
}

func TestCheckpointSinkAsyncOrdering(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	// Seed a checkpoint
	saver.Put(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.Checkpoint{
		V: 1, ID: "cp-1", TS: time.Now(),
	}, checkpoint.Metadata{Source: "input", Step: -1}, nil)

	sink := newCheckpointSink(saver, DurabilityAsync, context.Background(), nil)

	// Submit 3 PutWrites + 1 PutCheckpoint
	for i := 0; i < 3; i++ {
		err := sink.putWrites(context.Background(), checkpoint.Config{ThreadID: "t1", CheckpointID: "cp-1"},
			[]checkpoint.Write{{Channel: "ch", Value: i}}, "task-write")
		if err != nil {
			t.Fatal(err)
		}
	}

	_, err := sink.putCheckpoint(context.Background(), checkpoint.Config{ThreadID: "t1", CheckpointID: "cp-1"},
		checkpoint.Checkpoint{V: 1, ID: "cp-2", TS: time.Now(), ChannelValues: map[string]any{"ch": 3}},
		checkpoint.Metadata{Source: "loop", Step: 0}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := sink.flush(); err != nil {
		t.Fatal(err)
	}

	// Verify checkpoint cp-2 exists
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatal("expected checkpoint after flush")
	}
}

func TestCheckpointSinkAsyncPanicRecovery(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	saver.Put(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.Checkpoint{
		V: 1, ID: "cp-1", TS: time.Now(),
	}, checkpoint.Metadata{Source: "input", Step: -1}, nil)

	// Create a saver wrapper that panics on first Put
	callCount := 0
	panicky := &panickySaver{Saver: saver, panicOnPutCount: 1, callCount: &callCount}

	sink := newCheckpointSink(panicky, DurabilityAsync, context.Background(), nil)

	// Submit a Put that will panic
	sink.putCheckpoint(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.Checkpoint{
		V: 1, ID: "cp-99", TS: time.Now(),
	}, checkpoint.Metadata{Source: "loop", Step: 0}, nil)

	// Submit a PutWrites that should still be processed (per-request recover)
	err := sink.putWrites(context.Background(), checkpoint.Config{ThreadID: "t1", CheckpointID: "cp-1"},
		[]checkpoint.Write{{Channel: "ch", Value: "after-panic"}}, "task-survive")
	if err != nil {
		t.Fatal(err)
	}

	flushErr := sink.flush()
	if flushErr == nil {
		t.Fatal("expected flush to return the panic error")
	}
}

type panickySaver struct {
	checkpoint.Saver
	panicOnPutCount int
	callCount       *int
}

func (p *panickySaver) Put(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, nv map[string]int64) (checkpoint.Config, error) {
	*p.callCount++
	if *p.callCount == p.panicOnPutCount {
		panic("simulated saver panic")
	}
	return p.Saver.Put(ctx, cfg, cp, md, nv)
}

func TestCheckpointSinkAsyncNoLeak(t *testing.T) {
	before := runtime.NumGoroutine()
	saver := checkpoint.NewMemorySaver()
	saver.Put(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.Checkpoint{
		V: 1, ID: "cp-1", TS: time.Now(),
	}, checkpoint.Metadata{Source: "input", Step: -1}, nil)

	for i := 0; i < 5; i++ {
		sink := newCheckpointSink(saver, DurabilityAsync, context.Background(), nil)
		sink.putWrites(context.Background(), checkpoint.Config{ThreadID: "t1", CheckpointID: "cp-1"},
			[]checkpoint.Write{{Channel: "ch", Value: i}}, "task")
		sink.flush()
	}

	time.Sleep(50 * time.Millisecond)
	after := runtime.NumGoroutine()
	if after > before {
		t.Errorf("goroutine leak: before=%d after=%d", before, after)
	}
}

func TestCloneAnyShallow(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		if got := cloneAnyShallow(nil); got != nil {
			t.Fatalf("cloneAnyShallow(nil) = %v, want nil", got)
		}
	})
	t.Run("scalar shared as-is", func(t *testing.T) {
		if got := cloneAnyShallow(42); got != 42 {
			t.Fatalf("cloneAnyShallow(42) = %v, want 42", got)
		}
	})
	t.Run("slice gets a new backing array", func(t *testing.T) {
		orig := []int{1, 2, 3}
		cloned, ok := cloneAnyShallow(orig).([]int)
		if !ok || len(cloned) != 3 {
			t.Fatalf("cloneAnyShallow(slice) = %v, want an []int of length 3", cloned)
		}
		orig[0] = 99
		if cloned[0] != 1 {
			t.Fatalf("cloned slice shares its backing array with the original: %v", cloned)
		}
	})
	t.Run("map gets a new map", func(t *testing.T) {
		orig := map[string]int{"a": 1}
		cloned, ok := cloneAnyShallow(orig).(map[string]int)
		if !ok || cloned["a"] != 1 {
			t.Fatalf("cloneAnyShallow(map) = %v, want a map[string]int copy", cloned)
		}
		orig["b"] = 2
		if _, exists := cloned["b"]; exists {
			t.Fatalf("cloned map shares storage with the original: %v", cloned)
		}
	})
}

func TestSinkRequestExecuteUnknownKind(t *testing.T) {
	req := sinkRequest{kind: sinkRequestKind(99)}
	if err := req.execute(checkpoint.NewMemorySaver(), context.Background()); err == nil {
		t.Fatal("execute() error = nil, want an unknown-kind error")
	}
}

func TestCheckpointSinkUnknownDurabilityMode(t *testing.T) {
	// An unrecognized durability mode is a no-op: no saver calls, no error.
	sink := &checkpointSink{mode: Durability("weird")}
	cfg, err := sink.putCheckpoint(context.Background(), checkpoint.Config{ThreadID: "t"},
		checkpoint.Checkpoint{V: 1, ID: "cp", TS: time.Now()}, checkpoint.Metadata{}, nil)
	if err != nil {
		t.Fatalf("putCheckpoint() error = %v", err)
	}
	if cfg.CheckpointID != "cp" {
		t.Fatalf("putCheckpoint() cfg = %+v, want the new checkpoint ID", cfg)
	}
	if err := sink.putWrites(context.Background(), checkpoint.Config{ThreadID: "t"},
		[]checkpoint.Write{{Channel: "ch", Value: 1}}, "task"); err != nil {
		t.Fatalf("putWrites() error = %v", err)
	}
}

func TestCheckpointSinkAsyncRequestsAfterFlush(t *testing.T) {
	// After flush the worker has exited: further requests must return
	// immediately via the workerDone guard instead of blocking on writeCh.
	saver := checkpoint.NewMemorySaver()
	sink := newCheckpointSink(saver, DurabilityAsync, context.Background(), nil)
	if err := sink.flush(); err != nil {
		t.Fatalf("flush() error = %v", err)
	}
	if _, err := sink.putCheckpoint(context.Background(), checkpoint.Config{ThreadID: "t"},
		checkpoint.Checkpoint{V: 1, ID: "cp", TS: time.Now()}, checkpoint.Metadata{}, nil); err != nil {
		t.Fatalf("putCheckpoint() after flush error = %v", err)
	}
	if err := sink.putWrites(context.Background(), checkpoint.Config{ThreadID: "t"},
		[]checkpoint.Write{{Channel: "ch", Value: 1}}, "task"); err != nil {
		t.Fatalf("putWrites() after flush error = %v", err)
	}
}

func TestCheckpointSinkAsyncWorkerErrorSurfacesAtFlush(t *testing.T) {
	sink := newCheckpointSink(&putErrSaver{Saver: checkpoint.NewMemorySaver()}, DurabilityAsync, context.Background(), nil)
	if _, err := sink.putCheckpoint(context.Background(), checkpoint.Config{ThreadID: "t"},
		checkpoint.Checkpoint{V: 1, ID: "cp", TS: time.Now()}, checkpoint.Metadata{}, nil); err != nil {
		t.Fatalf("putCheckpoint() error = %v, want nil (async defers errors to flush)", err)
	}
	if err := sink.flush(); !errors.Is(err, errSaverBoom) {
		t.Fatalf("flush() error = %v, want %v", err, errSaverBoom)
	}
}

// newExitSinkWithState builds an exit-mode sink whose flush context carries a
// runState with one delta channel (snapshotFrequency 1) that received an
// Overwrite, so flushExit exercises the overwrite/snapshot bookkeeping.
func newExitSinkWithState(t *testing.T, saver checkpoint.Saver) *checkpointSink {
	t.Helper()
	rs := newRunState(map[string]channels.Channel{
		"d": channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 1),
	})
	if _, err := rs.applyWrites([]taskWrites{{node: "n", update: map[string]any{
		"d": channels.NewOverwrite([]int{2}),
		"x": 1,
	}}}); err != nil {
		t.Fatalf("applyWrites() error = %v", err)
	}
	sink := &checkpointSink{saver: saver, mode: DurabilityExit}
	sink.setFlushContext(context.Background(), Options{ThreadID: "t"}, rs,
		checkpoint.Config{ThreadID: "t", CheckpointID: "final"}, checkpoint.Metadata{Source: "loop", Step: 0})
	sink.accumulateExitWrites([]checkpoint.Write{{Channel: "x", Value: 1}}, "task-1")
	return sink
}

func TestFlushExitNilSaverOrRunState(t *testing.T) {
	if err := (&checkpointSink{mode: DurabilityExit}).flushExit(); err != nil {
		t.Fatalf("flushExit() with nil saver error = %v, want nil", err)
	}
	if err := (&checkpointSink{saver: checkpoint.NewMemorySaver(), mode: DurabilityExit}).flushExit(); err != nil {
		t.Fatalf("flushExit() with nil run state error = %v, want nil", err)
	}
}

func TestFlushExitSuccess(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	sink := newExitSinkWithState(t, saver)
	if err := sink.flushExit(); err != nil {
		t.Fatalf("flushExit() error = %v", err)
	}
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("final checkpoint not persisted: tup = %v, err = %v", tup, err)
	}
	if tup.Config.CheckpointID != "final" {
		t.Fatalf("final checkpoint ID = %q, want %q", tup.Config.CheckpointID, "final")
	}
}

func TestFlushExitSaverErrors(t *testing.T) {
	t.Run("stub checkpoint put error", func(t *testing.T) {
		sink := newExitSinkWithState(t, &putErrSaver{Saver: checkpoint.NewMemorySaver()})
		if err := sink.flushExit(); !errors.Is(err, errSaverBoom) {
			t.Fatalf("flushExit() error = %v, want %v", err, errSaverBoom)
		}
	})
	t.Run("delta writes put error", func(t *testing.T) {
		sink := newExitSinkWithState(t, &putWritesErrSaver{Saver: checkpoint.NewMemorySaver()})
		if err := sink.flushExit(); !errors.Is(err, errSaverBoom) {
			t.Fatalf("flushExit() error = %v, want %v", err, errSaverBoom)
		}
	})
	t.Run("final checkpoint put error", func(t *testing.T) {
		sink := newExitSinkWithState(t, &failNthPutSaver{Saver: checkpoint.NewMemorySaver(), n: 2})
		if err := sink.flushExit(); !errors.Is(err, errSaverBoom) {
			t.Fatalf("flushExit() error = %v, want %v", err, errSaverBoom)
		}
	})
}
