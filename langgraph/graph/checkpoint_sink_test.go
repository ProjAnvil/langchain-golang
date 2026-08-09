package graph

import (
	"context"
	"runtime"
	"testing"
	"time"

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
