package graph

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// listErrSaver fails every List call.
type listErrSaver struct{ checkpoint.Saver }

func (s *listErrSaver) List(context.Context, checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error) {
	return nil, errSaverBoom
}

func TestGetStateErrors(t *testing.T) {
	t.Run("saver GetTuple error", func(t *testing.T) {
		cg := compileLinear(t, noopNode, WithCheckpointer(&getTupleErrSaver{Saver: checkpoint.NewMemorySaver()}))
		if _, err := cg.GetState(context.Background(), checkpoint.Config{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
			t.Fatalf("GetState() error = %v, want %v", err, errSaverBoom)
		}
	})
	t.Run("pinned checkpoint not found", func(t *testing.T) {
		cg := compileLinear(t, noopNode, WithCheckpointer(checkpoint.NewMemorySaver()))
		_, err := cg.GetState(context.Background(), checkpoint.Config{ThreadID: "t", CheckpointID: "nope"})
		if err == nil || !strings.Contains(err.Error(), `"nope"`) {
			t.Fatalf("GetState() error = %v, want it to name the missing checkpoint", err)
		}
	})
}

func TestGetStateHistoryListError(t *testing.T) {
	cg := compileLinear(t, noopNode, WithCheckpointer(&listErrSaver{Saver: checkpoint.NewMemorySaver()}))
	if _, err := cg.GetStateHistory(context.Background(), checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("GetStateHistory() error = %v, want %v", err, errSaverBoom)
	}
}

func TestUpdateStateErrors(t *testing.T) {
	t.Run("saver GetTuple error", func(t *testing.T) {
		cg := compileLinear(t, noopNode, WithCheckpointer(&getTupleErrSaver{Saver: checkpoint.NewMemorySaver()}))
		if _, err := cg.UpdateState(context.Background(), checkpoint.Config{ThreadID: "t"}, map[string]any{"x": 1}, "a"); !errors.Is(err, errSaverBoom) {
			t.Fatalf("UpdateState() error = %v, want %v", err, errSaverBoom)
		}
	})
	t.Run("no checkpoint for thread", func(t *testing.T) {
		cg := compileLinear(t, noopNode, WithCheckpointer(checkpoint.NewMemorySaver()))
		_, err := cg.UpdateState(context.Background(), checkpoint.Config{ThreadID: "t"}, map[string]any{"x": 1}, "a")
		if err == nil || !strings.Contains(err.Error(), "no checkpoint found") {
			t.Fatalf("UpdateState() error = %v, want a no-checkpoint error", err)
		}
	})
	t.Run("pinned checkpoint not found", func(t *testing.T) {
		cg := compileLinear(t, noopNode, WithCheckpointer(checkpoint.NewMemorySaver()))
		_, err := cg.UpdateState(context.Background(), checkpoint.Config{ThreadID: "t", CheckpointID: "nope"}, map[string]any{"x": 1}, "a")
		if err == nil || !strings.Contains(err.Error(), `"nope"`) {
			t.Fatalf("UpdateState() error = %v, want it to name the missing checkpoint", err)
		}
	})
	t.Run("applyWrites error", func(t *testing.T) {
		cg := newErrUpdateChannelGraph(t, checkpoint.NewMemorySaver())
		if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 1}, Options{ThreadID: "t"}); err != nil {
			t.Fatalf("InvokeWithOptions() error = %v", err)
		}
		if _, err := cg.UpdateState(context.Background(), checkpoint.Config{ThreadID: "t"}, map[string]any{"bad": 1}, "a"); !errors.Is(err, errSaverBoom) {
			t.Fatalf("UpdateState() error = %v, want %v", err, errSaverBoom)
		}
	})
	t.Run("staticNext error", func(t *testing.T) {
		// Node "a" has no outgoing edge: UpdateState cannot re-resolve Next.
		saver := checkpoint.NewMemorySaver()
		g := NewStateGraph()
		g.AddNode("a", noopNode)
		g.AddEdge(types.START, "a")
		cg, err := g.Compile(WithCheckpointer(saver))
		if err != nil {
			t.Fatalf("Compile() error = %v", err)
		}
		// The run itself fails at staticNext, but only after the input
		// checkpoint is saved, so the thread has a checkpoint to update.
		if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 1}, Options{ThreadID: "t"}); err == nil {
			t.Fatal("InvokeWithOptions() error = nil, want a no-outgoing-edge error")
		}
		if _, err := cg.UpdateState(context.Background(), checkpoint.Config{ThreadID: "t"}, map[string]any{"x": 2}, "a"); err == nil ||
			!strings.Contains(err.Error(), "no outgoing edge") {
			t.Fatalf("UpdateState() error = %v, want a no-outgoing-edge error", err)
		}
	})
}

func TestBulkUpdateStateInnerErrorPropagates(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddNode("b", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 1}, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("InvokeWithOptions() error = %v", err)
	}
	_, err = cg.BulkUpdateState(context.Background(), checkpoint.Config{ThreadID: "t"}, [][]BulkUpdate{
		{{Values: map[string]any{"x": 2}, AsNode: "a"}},
		{{Values: map[string]any{"x": 3}, AsNode: "ghost"}},
	})
	if err == nil || !strings.Contains(err.Error(), `"ghost"`) {
		t.Fatalf("BulkUpdateState() error = %v, want it to name the unknown node", err)
	}
}

// deltaSnapshotBlob produces the snapshot blob a DeltaChannel persists when
// its cadence fires, used to hand-craft ancestor checkpoints.
func deltaSnapshotBlob(t *testing.T, seed []int) any {
	t.Helper()
	ch := channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 1).FromCheckpoint(nil)
	if _, err := ch.Update([]any{seed}); err != nil {
		t.Fatalf("Update() error = %v", err)
	}
	d, ok := channels.AsDelta(ch)
	if !ok {
		t.Fatal("AsDelta() ok = false, want a delta channel")
	}
	blob, ok := d.SnapshotBlob()
	if !ok {
		t.Fatal("SnapshotBlob() ok = false, want a blob for an available channel")
	}
	return blob
}

// TestReconstructDeltaChannelsAncestorWalk hand-crafts a three-checkpoint
// chain where the latest checkpoint stores neither delta channel: GetState
// must rebuild d1 from a plain-value seed at the grandparent plus replayed
// writes, and d2 from a snapshot blob at the parent plus replayed writes.
func TestReconstructDeltaChannelsAncestorWalk(t *testing.T) {
	ctx := context.Background()
	saver := checkpoint.NewMemorySaver()

	put := func(parent string, cp checkpoint.Checkpoint) checkpoint.Config {
		t.Helper()
		cfg, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: parent}, cp, checkpoint.Metadata{Source: "loop"}, nil)
		if err != nil {
			t.Fatalf("Put() error = %v", err)
		}
		return cfg
	}
	newCp := func(id string, values map[string]any) checkpoint.Checkpoint {
		return checkpoint.Checkpoint{
			V:               1,
			ID:              id,
			TS:              time.Now(),
			ChannelValues:   values,
			ChannelVersions: map[string]int64{},
			VersionsSeen:    map[string]map[string]int64{},
		}
	}

	// cp0 (grandparent): plain (non-blob) d1 value — the migration seed.
	put("", newCp("cp0", map[string]any{"d1": []int{1}}))
	// cp1 (parent): d2 snapshot blob; d1 still sentinel-only. A later d1
	// write is recorded against cp1.
	put("cp0", newCp("cp1", map[string]any{"d2": deltaSnapshotBlob(t, []int{9})}))
	cfg1 := checkpoint.Config{ThreadID: "t", CheckpointID: "cp1"}
	if err := saver.PutWrites(ctx, cfg1, []checkpoint.Write{{Channel: "d1", Value: []int{0}}}, "task-0", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}
	// cp2 (latest): neither delta channel in values; more writes.
	put("cp1", newCp("cp2", map[string]any{"x": "latest"}))
	cfg2 := checkpoint.Config{ThreadID: "t", CheckpointID: "cp2"}
	if err := saver.PutWrites(ctx, cfg2, []checkpoint.Write{
		{Channel: "d1", Value: []int{2}},
		{Channel: "d2", Value: []int{5}},
	}, "task-1", ""); err != nil {
		t.Fatalf("PutWrites() error = %v", err)
	}

	// The graph only needs the delta channel prototypes registered; the
	// hand-crafted checkpoints drive the reconstruction.
	g := NewStateGraph()
	g.AddChannel("d1", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 100))
	g.AddChannel("d2", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 100))
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	snap, err := cg.GetState(ctx, cfg2)
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got := snap.Values["d1"]; !reflect.DeepEqual(got, []int{1, 0, 2}) {
		t.Fatalf("d1 = %v, want [1 0 2] (plain-value seed + replayed ancestor writes)", got)
	}
	if got := snap.Values["d2"]; !reflect.DeepEqual(got, []int{9, 5}) {
		t.Fatalf("d2 = %v, want [9 5] (snapshot blob seed + replayed writes)", got)
	}
}

// failAncestorTupleSaver fails GetTuple for one specific checkpoint ID.
type failAncestorTupleSaver struct {
	checkpoint.Saver
	badID string
}

func (s *failAncestorTupleSaver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	if cfg.CheckpointID == s.badID {
		return nil, errSaverBoom
	}
	return s.Saver.GetTuple(ctx, cfg)
}

// TestReconstructDeltaChannelsAncestorLoadFailure verifies that a failed
// ancestor lookup stops the parent-chain walk silently: GetState still
// succeeds with the delta channel left unreconstructed.
func TestReconstructDeltaChannelsAncestorLoadFailure(t *testing.T) {
	ctx := context.Background()
	base := checkpoint.NewMemorySaver()
	saver := &failAncestorTupleSaver{Saver: base, badID: "cp0"}

	if _, err := base.Put(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.Checkpoint{
		V: 1, ID: "cp0", TS: time.Now(),
		ChannelValues:   map[string]any{"d1": []int{1}},
		ChannelVersions: map[string]int64{},
		VersionsSeen:    map[string]map[string]int64{},
	}, checkpoint.Metadata{Source: "loop"}, nil); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if _, err := base.Put(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: "cp0"}, checkpoint.Checkpoint{
		V: 1, ID: "cp1", TS: time.Now(),
		ChannelValues:   map[string]any{"x": 1},
		ChannelVersions: map[string]int64{},
		VersionsSeen:    map[string]map[string]int64{},
	}, checkpoint.Metadata{Source: "loop"}, nil); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	g := NewStateGraph()
	g.AddChannel("d1", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 100))
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: "cp1"})
	if err != nil {
		t.Fatalf("GetState() error = %v, want nil (ancestor load failure is non-fatal)", err)
	}
	if _, present := snap.Values["d1"]; present {
		t.Fatalf("d1 = %v, want it unreconstructed (the ancestor walk broke at cp0)", snap.Values["d1"])
	}
}

// TestReconstructDeltaChannelsCancelledWalk verifies that a cancelled context
// aborts the parent-chain walk: reconstruction stops at the current tuple.
func TestReconstructDeltaChannelsCancelledWalk(t *testing.T) {
	ctx := context.Background()
	saver := checkpoint.NewMemorySaver()
	if _, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.Checkpoint{
		V: 1, ID: "cp0", TS: time.Now(),
		ChannelValues:   map[string]any{"d1": []int{1}},
		ChannelVersions: map[string]int64{},
		VersionsSeen:    map[string]map[string]int64{},
	}, checkpoint.Metadata{Source: "loop"}, nil); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	childCfg, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: "cp0"}, checkpoint.Checkpoint{
		V: 1, ID: "cp1", TS: time.Now(),
		ChannelValues:   map[string]any{},
		ChannelVersions: map[string]int64{},
		VersionsSeen:    map[string]map[string]int64{},
	}, checkpoint.Metadata{Source: "loop"}, nil)
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	tup, err := saver.GetTuple(ctx, childCfg)
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %v, %v", tup, err)
	}

	g := NewStateGraph()
	g.AddChannel("d1", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 100))
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	values := map[string]any{}
	cg.reconstructDeltaChannels(cancelled, tup, values)
	if _, present := values["d1"]; present {
		t.Fatalf("d1 = %v, want it unreconstructed under a cancelled context", values["d1"])
	}
}
