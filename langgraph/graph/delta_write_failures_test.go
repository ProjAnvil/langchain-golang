package graph

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// errDeltaWriteBoom is the injected saver failure (#9235 test suite), the Go shape
// of Python's ConnectionError("... was not saved").
var errDeltaWriteBoom = errors.New("boom: not saved")

// logPlainGraph builds a -> b -> c where every node writes its letter to both
// `log` (a DeltaChannel whose cadence never fires here) and `plain` (a plain
// BinaryOperator reducer) — the Go shape of Python's
// test_delta_channel_write_failures.py graph. Node writes are ints (the node
// letter's rune) so the delta batch reducer and AppendSliceReducer fold them
// identically.
func logPlainGraph(t *testing.T, logFreq int, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, logFreq))
	g.AddReducer("plain", channels.AppendSliceReducer)
	for _, name := range []string{"a", "b", "c"} {
		letter := int(name[0])
		g.AddNode(name, func(_ runtime.Runtime, _ map[string]any) (any, error) {
			return map[string]any{"log": []int{letter}, "plain": []int{letter}}, nil
		})
	}
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", "c")
	g.AddEdge("c", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// failWritesOnceSaver fails PutWrites once for the batch carrying (channel,
// value), then delegates — the Go shape of Python's _FailsTheWriteOfBOnce.
type failWritesOnceSaver struct {
	*checkpoint.MemorySaver
	failed  bool
	channel string
	value   any
}

func (s *failWritesOnceSaver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	if !s.failed {
		for _, w := range writes {
			if w.Channel == s.channel && reflect.DeepEqual(w.Value, s.value) {
				s.failed = true
				return errDeltaWriteBoom
			}
		}
	}
	return s.MemorySaver.PutWrites(ctx, cfg, writes, taskID, taskPath)
}

// failSaveOnceSaver fails Put once for the checkpoint whose ChannelValues
// carry (channel, marker), then delegates — the Go shape of Python's
// _FailsTheSaveAfterAOnce (which keys on checkpoint["channel_values"]).
type failSaveOnceSaver struct {
	*checkpoint.MemorySaver
	failed  bool
	channel string
	marker  any
}

func (s *failSaveOnceSaver) Put(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64) (checkpoint.Config, error) {
	if !s.failed && reflect.DeepEqual(cp.ChannelValues[s.channel], s.marker) {
		s.failed = true
		return checkpoint.Config{}, errDeltaWriteBoom
	}
	return s.MemorySaver.Put(ctx, cfg, cp, md, newVersions)
}

// assertHistoryConsistent asserts the #9235 invariant over the whole thread
// history: every saved checkpoint must read the delta channel exactly as full
// as the plain one — no checkpoint is saved past a failed delta write.
func assertHistoryConsistent(t *testing.T, cg *CompiledGraph, threadID string) {
	t.Helper()
	history, err := cg.GetStateHistory(t.Context(), checkpoint.Config{ThreadID: threadID}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	if len(history) == 0 {
		t.Fatal("expected checkpoints, got none")
	}
	for _, snap := range history {
		log, _ := snap.Values["log"].([]int)
		plain, _ := snap.Values["plain"].([]int)
		if !reflect.DeepEqual(log, plain) {
			t.Fatalf("checkpoint %s (source=%s step=%d) reads log=%v but plain=%v — a checkpoint was saved past a failed delta write",
				snap.Config.CheckpointID, snap.Metadata.Source, snap.Metadata.Step, log, plain)
		}
	}
}

// logOf returns the thread's current log value.
func logOf(t *testing.T, cg *CompiledGraph, threadID string) []int {
	t.Helper()
	snap, err := cg.GetState(t.Context(), checkpoint.Config{ThreadID: threadID})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	log, _ := snap.Values["log"].([]int)
	return log
}

// finalLog is the expected completed-thread log value ([a b c]).
func finalLog() []int { return []int{'a', 'b', 'c'} }

// TestFailedDeltaWriteIsRecoveredNotLost covers the #9235 write-failure
// scenario on every durability mode: the first run fails with the injected
// error, every saved checkpoint stays consistent (log == plain — nothing was
// saved past the failed write), and a retry completes the thread. Under
// sync/async the failed run's checkpoint is repaired with a full snapshot and
// the retry CONTINUES from it (Python re-runs the failed task — see
// DIVERGENCES.md); only exit durability saves nothing before the failed
// write, so its retry starts over with the original input (Python passes
// INPUT only for the exit case).
func TestFailedDeltaWriteIsRecoveredNotLost(t *testing.T) {
	cases := []struct {
		durability Durability
		retryInput map[string]any
	}{
		{DurabilitySync, nil},
		{DurabilityAsync, nil},
		{DurabilityExit, map[string]any{"log": []int{}, "plain": []int{}}},
	}
	for _, tc := range cases {
		t.Run(string(tc.durability), func(t *testing.T) {
			threadID := "t-write-" + string(tc.durability)
			saver := &failWritesOnceSaver{MemorySaver: checkpoint.NewMemorySaver(), channel: "log", value: []int{'b'}}
			cg := logPlainGraph(t, 100, WithCheckpointer(saver), WithDurability(tc.durability))
			ctx := t.Context()
			cfg := Options{ThreadID: threadID}
			input := map[string]any{"log": []int{}, "plain": []int{}}

			if _, err := cg.InvokeWithOptions(ctx, input, cfg); !errors.Is(err, errDeltaWriteBoom) {
				t.Fatalf("[%s] Invoke() error = %v, want %v", tc.durability, err, errDeltaWriteBoom)
			}
			assertHistoryConsistent(t, cg, threadID)

			if _, err := cg.InvokeWithOptions(ctx, tc.retryInput, cfg); err != nil {
				t.Fatalf("[%s] retry Invoke() error = %v", tc.durability, err)
			}
			if got := logOf(t, cg, threadID); !reflect.DeepEqual(got, finalLog()) {
				t.Fatalf("[%s] after retry log = %v, want %v", tc.durability, got, finalLog())
			}
		})
	}
}

// TestFailedCheckpointSaveIsNotBuiltOn covers the #9235 save-failure scenario:
// a failed checkpoint save must surface (or stop) the run instead of stacking
// later checkpoints whose parent is missing, and a retry finishes the thread.
func TestFailedCheckpointSaveIsNotBuiltOn(t *testing.T) {
	for _, d := range []Durability{DurabilitySync, DurabilityAsync} {
		t.Run(string(d), func(t *testing.T) {
			threadID := "t-save-" + string(d)
			saver := &failSaveOnceSaver{MemorySaver: checkpoint.NewMemorySaver(), channel: "plain", marker: []int{'a'}}
			cg := logPlainGraph(t, 100, WithCheckpointer(saver), WithDurability(d))
			ctx := t.Context()
			cfg := Options{ThreadID: threadID}
			input := map[string]any{"log": []int{}, "plain": []int{}}

			if _, err := cg.InvokeWithOptions(ctx, input, cfg); err == nil {
				t.Fatalf("[%s] Invoke() succeeded, want the injected save failure", d)
			}
			assertHistoryConsistent(t, cg, threadID)

			if _, err := cg.InvokeWithOptions(ctx, nil, cfg); err != nil {
				t.Fatalf("[%s] retry Invoke() error = %v", d, err)
			}
			if got := logOf(t, cg, threadID); !reflect.DeepEqual(got, finalLog()) {
				t.Fatalf("[%s] after retry log = %v, want %v", d, got, finalLog())
			}
		})
	}
}

// TestDeltaAsyncRunTerminatesAfterFailure: after an async-mode saver failure
// the run must terminate (the sink worker stops consuming), never hang — the
// Go analog of Python's single-background-thread hang regression (the Go sink
// owns a dedicated worker goroutine, so the test pins termination only).
func TestDeltaAsyncRunTerminatesAfterFailure(t *testing.T) {
	saver := &failWritesOnceSaver{MemorySaver: checkpoint.NewMemorySaver(), channel: "log", value: []int{'b'}}
	cg := logPlainGraph(t, 100, WithCheckpointer(saver), WithDurability(DurabilityAsync))
	done := make(chan error, 1)
	go func() {
		_, err := cg.InvokeWithOptions(context.Background(), map[string]any{"log": []int{}, "plain": []int{}}, Options{ThreadID: "t-hang"})
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, errDeltaWriteBoom) {
			t.Fatalf("Invoke() error = %v, want %v", err, errDeltaWriteBoom)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("async invoke hung after a failed delta write")
	}
}

// pauseExitRepairGraph builds START -> w1 -> w2 -> (a || b): `a` interrupts
// in-node, `b` writes the delta `log` (snapshotFrequency=2 — the pause save
// suppresses its snapshot, b's write anchors on the checkpoint), and `w1`
// also writes `other`, a delta channel whose cadence never fires and which
// receives no anchored pause write — so `other` stays in the pause
// checkpoint's repair payload while `log` is excluded from it.
func pauseExitRepairGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 2))
	g.AddChannel("other", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddNode("w1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"w1"}, "other": []string{"w1"}}, nil
	})
	g.AddNode("w2", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"w2"}}, nil
	})
	g.AddNode("a", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(rt, "stop")
		return nil, nil
	})
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"b"}}, nil
	})
	g.AddEdge(types.START, "w1")
	g.AddEdge("w1", "w2")
	g.AddEdge("w2", "a")
	g.AddEdge("w2", "b")
	g.AddEdge("a", types.END)
	g.AddEdge("b", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestExitPauseWriteFailureRepairsPauseCheckpoint pins the exit-mode branch
// of putPauseWrites to the same repair treatment its sibling write paths
// (sync putWrites, the async worker, putPauseCheckpointExit's accumulated
// writes) already perform (#9235): a PutWrites failure against the pause
// checkpoint must poison the run AND re-put the checkpoint with its forced
// delta snapshots — minus the channels suppressed by the pause save, which
// stay blob-free so their anchored writes survive the walk.
func TestExitPauseWriteFailureRepairsPauseCheckpoint(t *testing.T) {
	ctx := t.Context()
	saver := &failWritesOnceSaver{MemorySaver: checkpoint.NewMemorySaver(), channel: "log", value: []string{"b"}}
	cg := pauseExitRepairGraph(t, WithCheckpointer(saver), WithDurability(DurabilityExit))
	cfg := Options{ThreadID: "t"}

	// The pause persists under exit mode; b's anchored delta write fails.
	_, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []string{"in"}}, cfg)
	if !errors.Is(err, errDeltaWriteBoom) {
		t.Fatalf("pause Invoke() error = %v, want %v", err, errDeltaWriteBoom)
	}

	pauseTup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || pauseTup == nil {
		t.Fatalf("GetTuple(latest) = (%v, %v), want the pause checkpoint", pauseTup, err)
	}
	// The suppressed channel stays blob-free through the repair (its blob
	// would predate b's anchored write and strand it).
	if _, isBlob := channels.UnwrapDeltaSnapshot(pauseTup.Checkpoint.ChannelValues["log"]); isBlob {
		t.Fatalf("pause checkpoint %s carries a log blob after repair — suppressed channels must stay sentinel", pauseTup.Checkpoint.ID)
	}
	// The repair re-put happened: `other` (not suppressed, below cadence —
	// sentinel as-put) is materialized by the repair payload, exactly as a
	// failed write against a sync-mode checkpoint would leave it.
	if _, isBlob := channels.UnwrapDeltaSnapshot(pauseTup.Checkpoint.ChannelValues["other"]); !isBlob {
		t.Fatalf("pause checkpoint %s left as-put — the exit-mode failed pause write must re-put it with the repair payload (materialize `other`) like the sibling write paths", pauseTup.Checkpoint.ID)
	}

	// Reads stay consistent: b's write failed, so `log` reads its full
	// pre-pause value from the accumulated writes anchored on the pause.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got, _ := snap.Values["log"].([]string); !reflect.DeepEqual(got, []string{"in", "w1", "w2"}) {
		t.Fatalf("log at the pause checkpoint = %v, want [in w1 w2]", got)
	}
	if got, _ := snap.Values["other"].([]string); !reflect.DeepEqual(got, []string{"w1"}) {
		t.Fatalf("other at the pause checkpoint = %v, want [w1]", got)
	}
}
