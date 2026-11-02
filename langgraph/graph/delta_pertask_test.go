package graph

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestDeltaChannelPerTaskWritesReconstruction verifies that with
// snapshotFrequency > 1 (cadence never firing in two supersteps), GetState
// reconstructs the delta channel purely from per-task writes persisted in the
// normal commit flow — no checkpoint carries a snapshot blob, so the value can
// only come from the ancestor-write walk over each task's PutWrites.
//
// This is the load-bearing assertion for the per-task writes added in Task 4:
// before that change the normal flow persisted no per-task writes, and with the
// DeltaChannel.Checkpoint sentinel flip (no forced first snapshot) the value
// would be empty.
func TestDeltaChannelPerTaskWritesReconstruction(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	// freq=10: cadence never fires across the two supersteps (n1, n2).
	cg := deltaGraphFreq(t, 10, WithCheckpointer(saver))
	ctx := t.Context()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	// No checkpoint should carry a snapshot blob for "items" (cadence 10 never
	// fired), proving reconstruction must come from per-task writes.
	history, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(history) == 0 {
		t.Fatal("expected checkpoints, got none")
	}
	for i, h := range history {
		if blob, ok := h.Checkpoint.ChannelValues["items"]; ok {
			t.Fatalf("checkpoint %d (source=%s step=%d) unexpectedly carries an items blob %v — cadence should not have fired", i, h.Metadata.Source, h.Metadata.Step, blob)
		}
	}

	// GetState must reconstruct items=[1,2,3] via the per-task-write ancestor walk.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	got, ok := snap.Values["items"].([]int)
	if !ok {
		t.Fatalf("GetState().Values['items'] = %v (%T), want []int{1,2,3}", snap.Values["items"], snap.Values["items"])
	}
	if want := []int{1, 2, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("GetState().Values['items'] = %v, want %v", got, want)
	}
	// The non-delta channel ("msg", a LastValue) is stored in ChannelValues and
	// needs no reconstruction; assert it round-trips too.
	if got, want := snap.Values["msg"], "hello"; got != want {
		t.Fatalf("GetState().Values['msg'] = %v, want %q", got, want)
	}
}

// TestDeltaChannelInputWritesReconstruction verifies that delta-channel INPUT
// writes are persisted (NullTaskID) so GetState reconstructs them across turns
// even when every checkpoint stores the channel as a sentinel (cadence never
// fires). A single-node graph appends a constant on each turn; the accumulated
// value (input writes + node writes) is rebuilt by the ancestor walk.
func TestDeltaChannelInputWritesReconstruction(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	// BatchReducer passed directly to NewDeltaChannel (C4); freq=10 so cadence
	// never fires across the turns below.
	g.AddChannel("items", channels.NewDeltaChannel(
		func(existing any, updates []any) (any, error) {
			base, _ := existing.([]int)
			out := make([]int, len(base))
			copy(out, base)
			for _, u := range updates {
				out = append(out, u.([]int)...)
			}
			return out, nil
		},
		func() any { return []int{} },
		10,
	))
	g.AddNode("n1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"items": []int{10}}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := t.Context()

	// Turn 1 (fresh): input writes items=[1]; n1 appends [10] -> [1,10].
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{1}}, Options{ThreadID: "t2"}); err != nil {
		t.Fatalf("invoke turn 1: %v", err)
	}
	// Turn 2 (new turn): input writes items=[2]; n1 appends [10] -> [2,10].
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"items": []int{2}}, Options{ThreadID: "t2"}); err != nil {
		t.Fatalf("invoke turn 2: %v", err)
	}

	// GetState reconstructs the full accumulated history across both turns:
	// turn-1 input [1], turn-1 node [10], turn-2 input [2], turn-2 node [10].
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t2"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	got, ok := snap.Values["items"].([]int)
	if !ok {
		t.Fatalf("GetState().Values['items'] = %v (%T), want []int{1,10,2,10}", snap.Values["items"], snap.Values["items"])
	}
	if want := []int{1, 10, 2, 10}; !reflect.DeepEqual(got, want) {
		t.Fatalf("GetState().Values['items'] = %v, want %v", got, want)
	}
}

// interruptAfterDeltaGraph builds START -> a -> b -> END paused by
// interrupt_after("a"): both nodes write their letter to `log` (a DeltaChannel
// whose cadence never fires here) and to `plain` (an AppendSliceReducer), so
// every checkpoint must read log == plain — any divergence means a delta
// write the run failed to persist beside its checkpoint.
func interruptAfterDeltaGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	return interruptAfterDeltaGraphFreq(t, 100, opts...)
}

// interruptAfterDeltaGraphFreq is interruptAfterDeltaGraph with the delta
// channel's snapshot cadence parameterized: freq=100 (the default helper)
// keeps the channel below cadence for the whole run, while freq=1 fires a
// snapshot blob at every checkpoint that folds a write.
func interruptAfterDeltaGraphFreq(t *testing.T, freq int, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, freq))
	g.AddReducer("plain", channels.AppendSliceReducer)
	for _, name := range []string{"a", "b"} {
		letter := int(name[0])
		g.AddNode(name, func(_ runtime.Runtime, _ map[string]any) (any, error) {
			return map[string]any{"log": []int{letter}, "plain": []int{letter}}, nil
		})
	}
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	opts = append(opts, WithInterruptAfter("a"))
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestInterruptAfterPersistsSuperstepWrites pins the interrupt_after pause to
// Python's write durability: Python persists each task's writes as the task
// completes (put_writes, _loop.py:515-543) — before the superstep's loop
// checkpoint save and the interrupt_after pause that follows it (after_tick,
// _loop.py:766-775) — and under exit durability _suppress_interrupt
// force-persists the accumulated writes at the pause (_loop.py:1513-1517).
// Go's interrupt_after block returned early, skipping the commit path's
// per-task putWrites: the pause checkpoint (whose values fold the superstep)
// stored below-cadence delta channels as sentinels with no anchored writes,
// so reads and resumes at the pause reconstructed them short.
func TestInterruptAfterPersistsSuperstepWrites(t *testing.T) {
	for _, d := range []Durability{DurabilitySync, DurabilityAsync, DurabilityExit} {
		t.Run(string(d), func(t *testing.T) {
			ctx := t.Context()
			saver := checkpoint.NewMemorySaver()
			cg := interruptAfterDeltaGraph(t, WithCheckpointer(saver), WithDurability(d))
			cfg := Options{ThreadID: "t"}

			res, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []int{}, "plain": []int{}}, cfg)
			if err != nil {
				t.Fatalf("[%s] pause Invoke() error = %v", d, err)
			}
			if len(res.Interrupts) != 1 {
				t.Fatalf("[%s] interrupts = %+v, want the interrupt_after pause", d, res.Interrupts)
			}

			// At the pause, `a` has committed: the below-cadence delta channel
			// must read exactly as full as the plain one — at the latest
			// checkpoint and across the whole history that exists so far.
			snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
			if err != nil {
				t.Fatalf("GetState() error = %v", err)
			}
			if got, _ := snap.Values["log"].([]int); !reflect.DeepEqual(got, []int{'a'}) {
				t.Fatalf("[%s] log at the interrupt_after pause = %v, want [a] — the superstep's writes must anchor on the pause checkpoint", d, got)
			}
			if got, _ := snap.Values["plain"].([]int); !reflect.DeepEqual(got, []int{'a'}) {
				t.Fatalf("[%s] plain at the interrupt_after pause = %v, want [a]", d, got)
			}
			assertHistoryConsistent(t, cg, "t")

			// Resume (nil input): `b` runs; both channels read the full run.
			if _, err := cg.InvokeWithOptions(ctx, nil, cfg); err != nil {
				t.Fatalf("[%s] resume Invoke() error = %v", d, err)
			}
			if got := logOf(t, cg, "t"); !reflect.DeepEqual(got, []int{'a', 'b'}) {
				t.Fatalf("[%s] log after resume = %v, want [a b]", d, got)
			}
			snap, err = cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
			if err != nil {
				t.Fatalf("GetState() after resume error = %v", err)
			}
			if got, _ := snap.Values["plain"].([]int); !reflect.DeepEqual(got, []int{'a', 'b'}) {
				t.Fatalf("[%s] plain after resume = %v, want [a b]", d, got)
			}
			// Sync/async anchor each write on the checkpoint that folds it,
			// so log == plain holds across the whole history. Exit mode
			// anchors the resumed run's accumulated writes on its START
			// checkpoint (Python _put_exit_delta_writes' anchor-on-start,
			// _loop.py). The LIVE read above — the latest checkpoint is the
			// pause checkpoint — folds those anchored writes in, matching
			// how a Python read of the latest checkpoint applies its pending
			// writes; no such equivalence is claimed for history reads of
			// the pause checkpoint, so the strict per-checkpoint invariant
			// is asserted only before the resume there (above).
			if d != DurabilityExit {
				assertHistoryConsistent(t, cg, "t")
			}
		})
	}
}

// TestInterruptAfterCadenceFiresAtPause is the cadence counterpart of
// TestInterruptAfterPersistsSuperstepWrites: the delta channel's snapshot
// cadence is 1, so the pause checkpoint stores a real `log` blob ALONGSIDE
// the per-task writes newly anchored on it for the same superstep. Reads and
// history must stay consistent: the present() shadowing rule (snapshot.go's
// walkKeys skips channels the checkpoint already carries) means the blob wins
// and the anchored writes for the blob-stored channel are shadowed, not
// double-applied on top of it.
func TestInterruptAfterCadenceFiresAtPause(t *testing.T) {
	for _, d := range []Durability{DurabilitySync, DurabilityAsync, DurabilityExit} {
		t.Run(string(d), func(t *testing.T) {
			ctx := t.Context()
			saver := checkpoint.NewMemorySaver()
			cg := interruptAfterDeltaGraphFreq(t, 1, WithCheckpointer(saver), WithDurability(d))
			thread := "cadence-pause"
			cfg := Options{ThreadID: thread}

			res, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []int{}, "plain": []int{}}, cfg)
			if err != nil {
				t.Fatalf("[%s] pause Invoke() error = %v", d, err)
			}
			if len(res.Interrupts) != 1 {
				t.Fatalf("[%s] interrupts = %+v, want the interrupt_after pause", d, res.Interrupts)
			}

			// Cadence 1 fired: the pause checkpoint (List is newest-first, so
			// history[0]) must carry a real `log` snapshot blob — the
			// {"type": "__delta_snapshot__", "value": …} envelope of
			// channels/delta.go — not the sentinel (absent key) the freq-100
			// variant stores, while `a`'s per-task writes are anchored on it
			// all the same.
			history, err := saver.List(ctx, checkpoint.Config{ThreadID: thread}, checkpoint.ListOptions{})
			if err != nil {
				t.Fatalf("List() error = %v", err)
			}
			if len(history) == 0 {
				t.Fatal("expected checkpoints, got none")
			}
			raw, ok := history[0].Checkpoint.ChannelValues["log"].(map[string]any)
			if !ok || raw["type"] != "__delta_snapshot__" {
				t.Fatalf("[%s] pause checkpoint carries log entry %v (envelope=%t), want a __delta_snapshot__ blob — cadence 1 must fire at the pause", d, history[0].Checkpoint.ChannelValues["log"], ok)
			}
			blob, ok := raw["value"].([]int)
			if !ok || !reflect.DeepEqual(blob, []int{'a'}) {
				t.Fatalf("[%s] pause checkpoint log blob value = %v ([]int=%t), want [a]", d, raw["value"], ok)
			}

			// The blob shadows the anchored writes: reads at the pause see
			// [a] exactly once on both channels, and so does every history
			// read.
			snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: thread})
			if err != nil {
				t.Fatalf("GetState() error = %v", err)
			}
			if got, _ := snap.Values["log"].([]int); !reflect.DeepEqual(got, []int{'a'}) {
				t.Fatalf("[%s] log at the interrupt_after pause = %v, want [a] — the blob must shadow, not stack on, the anchored writes", d, got)
			}
			if got, _ := snap.Values["plain"].([]int); !reflect.DeepEqual(got, []int{'a'}) {
				t.Fatalf("[%s] plain at the interrupt_after pause = %v, want [a]", d, got)
			}
			assertHistoryConsistent(t, cg, thread)

			// Resume (nil input): `b` runs; both channels read the full run —
			// the shadowed anchored writes neither double-apply nor vanish.
			if _, err := cg.InvokeWithOptions(ctx, nil, cfg); err != nil {
				t.Fatalf("[%s] resume Invoke() error = %v", d, err)
			}
			if got := logOf(t, cg, thread); !reflect.DeepEqual(got, []int{'a', 'b'}) {
				t.Fatalf("[%s] log after resume = %v, want [a b]", d, got)
			}
			snap, err = cg.GetState(ctx, checkpoint.Config{ThreadID: thread})
			if err != nil {
				t.Fatalf("GetState() after resume error = %v", err)
			}
			if got, _ := snap.Values["plain"].([]int); !reflect.DeepEqual(got, []int{'a', 'b'}) {
				t.Fatalf("[%s] plain after resume = %v, want [a b]", d, got)
			}
			// Same exit-mode caveat as TestInterruptAfterPersistsSuperstepWrites:
			// exit anchors the resumed run's accumulated writes on its START
			// (pause) checkpoint, so the strict per-checkpoint history
			// invariant is asserted only for sync/async after the resume.
			if d != DurabilityExit {
				assertHistoryConsistent(t, cg, thread)
			}
		})
	}
}
