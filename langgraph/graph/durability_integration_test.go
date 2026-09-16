package graph

import (
	"context"
	"reflect"
	_runtime "runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestAsyncDurabilityEndToEnd verifies that a graph compiled with
// DurabilityAsync persists checkpoints correctly after invoke returns.
func TestAsyncDurabilityEndToEnd(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"x": 42}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityAsync))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatal(err)
	}

	// Checkpoint should be persisted after invoke returns (flush in defer)
	tup, err := saver.GetTuple(t.Context(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatal("expected checkpoint after async invoke")
	}
}

// TestExitDurabilityEndToEnd verifies that a graph compiled with
// DurabilityExit persists a single final checkpoint after invoke returns.
func TestExitDurabilityEndToEnd(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"x": 99}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityExit))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatal(err)
	}

	// Final checkpoint should exist after invoke returns
	tup, err := saver.GetTuple(t.Context(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatal("expected final checkpoint after exit-mode invoke")
	}
}

// TestAsyncDurabilityNoGoroutineLeak verifies no goroutine leaks across
// multiple async-mode invokes.
func TestAsyncDurabilityNoGoroutineLeak(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityAsync))
	if err != nil {
		t.Fatal(err)
	}

	before := _runtime.NumGoroutine()
	for range 10 {
		_, err = cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"})
		if err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(100 * time.Millisecond) // allow cleanup
	after := _runtime.NumGoroutine()
	if after > before {
		t.Errorf("goroutine leak: before=%d after=%d", before, after)
	}
}

// countingSaver counts Put/PutWrites calls, delegating to the wrapped saver.
type countingSaver struct {
	checkpoint.Saver
	puts       atomic.Int64
	putWrites  atomic.Int64
	perNSPuts  map[string]*atomic.Int64
	perNSMutex sync.Mutex
}

func newCountingSaver(s checkpoint.Saver) *countingSaver {
	return &countingSaver{Saver: s, perNSPuts: map[string]*atomic.Int64{}}
}

func (s *countingSaver) Put(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, nv map[string]int64) (checkpoint.Config, error) {
	s.puts.Add(1)
	s.perNSMutex.Lock()
	if s.perNSPuts[cfg.CheckpointNS] == nil {
		s.perNSPuts[cfg.CheckpointNS] = &atomic.Int64{}
	}
	s.perNSMutex.Unlock()
	s.perNSPuts[cfg.CheckpointNS].Add(1)
	return s.Saver.Put(ctx, cfg, cp, md, nv)
}

func (s *countingSaver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	s.putWrites.Add(1)
	return s.Saver.PutWrites(ctx, cfg, writes, taskID, taskPath)
}

// putsInNS reports how many Put calls landed in namespace ns.
func (s *countingSaver) putsInNS(ns string) int64 {
	s.perNSMutex.Lock()
	defer s.perNSMutex.Unlock()
	if c := s.perNSPuts[ns]; c != nil {
		return c.Load()
	}
	return 0
}

// durabilityGraph builds a two-node graph (a -> b) whose nodes each write one
// key, so a full sync run persists an input checkpoint plus one loop
// checkpoint per superstep (3 Puts total).
func durabilityGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	})
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"y": 2}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestRunDurabilityOverridesCompiled verifies that Options.Durability (set via
// WithRunDurability) overrides the compiled WithDurability value for a single
// run, mirroring Python's invoke(..., durability=...) runtime option: a
// sync-compiled graph invoked with DurabilityExit persists only the exit-mode
// flush (stub + final checkpoint), skipping the per-superstep puts.
func TestRunDurabilityOverridesCompiled(t *testing.T) {
	ctx := t.Context()

	// Baseline: compiled default (sync), no run override. Two supersteps =>
	// input + 2 loop checkpoints = 3 Puts.
	syncSaver := newCountingSaver(checkpoint.NewMemorySaver())
	syncCg := durabilityGraph(t, WithCheckpointer(syncSaver))
	if _, err := syncCg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("sync Invoke() error = %v", err)
	}
	if got := syncSaver.puts.Load(); got != 3 {
		t.Fatalf("sync run Put calls = %d, want 3 (input + 2 loop checkpoints)", got)
	}

	// Per-run override to exit: the same sync-compiled graph defers everything
	// to the exit flush — stub anchor + final checkpoint = 2 Puts.
	exitSaver := newCountingSaver(checkpoint.NewMemorySaver())
	exitCg := durabilityGraph(t, WithCheckpointer(exitSaver))
	if _, err := exitCg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t2"}.WithRunDurability(DurabilityExit)); err != nil {
		t.Fatalf("exit-override Invoke() error = %v", err)
	}
	if got := exitSaver.puts.Load(); got != 2 {
		t.Fatalf("exit-override run Put calls = %d, want 2 (stub + final exit flush only)", got)
	}
	tup, err := exitSaver.GetTuple(ctx, checkpoint.Config{ThreadID: "t2"})
	if err != nil || tup == nil {
		t.Fatalf("exit-override run left no latest checkpoint (tup=%v err=%v)", tup, err)
	}
	if v, _ := tup.Checkpoint.ChannelValues["y"].(int); v != 2 {
		t.Fatalf("exit-override final checkpoint y = %v, want 2", tup.Checkpoint.ChannelValues["y"])
	}

	// Per-run override to async over an exit-compiled graph: per-superstep
	// puts happen again (buffered, flushed before return).
	asyncSaver := newCountingSaver(checkpoint.NewMemorySaver())
	asyncCg := durabilityGraph(t, WithCheckpointer(asyncSaver), WithDurability(DurabilityExit))
	if _, err := asyncCg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t3"}.WithRunDurability(DurabilityAsync)); err != nil {
		t.Fatalf("async-override Invoke() error = %v", err)
	}
	if got := asyncSaver.puts.Load(); got != 3 {
		t.Fatalf("async-override run Put calls = %d, want 3 (input + 2 loop checkpoints)", got)
	}
}

// TestRunDurabilityDefaultUnchanged verifies the default stays the compiled
// value: a graph compiled with DurabilityAsync and invoked with a zero
// Options.Durability keeps async behavior (input + loop puts via the
// background worker), NOT Python's runtime "async" default flipping a
// sync-compiled graph.
func TestRunDurabilityDefaultUnchanged(t *testing.T) {
	ctx := t.Context()
	saver := newCountingSaver(checkpoint.NewMemorySaver())
	cg := durabilityGraph(t, WithCheckpointer(saver), WithDurability(DurabilityAsync))
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := saver.puts.Load(); got != 3 {
		t.Fatalf("async-compiled run without override Put calls = %d, want 3 (compiled default honored)", got)
	}
	// And a graph compiled without WithDurability keeps its sync default.
	syncSaver := newCountingSaver(checkpoint.NewMemorySaver())
	syncCg := durabilityGraph(t, WithCheckpointer(syncSaver))
	if _, err := syncCg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t2"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := syncSaver.puts.Load(); got != 3 {
		t.Fatalf("default-compiled run Put calls = %d, want 3 (sync default unchanged)", got)
	}
}

// TestRunDurabilityInvalidValue verifies an unknown per-run durability value
// errors the run instead of silently no-op'ing every checkpoint write.
func TestRunDurabilityInvalidValue(t *testing.T) {
	cg := durabilityGraph(t, WithCheckpointer(checkpoint.NewMemorySaver()))
	_, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"}.WithRunDurability(Durability("bogus")))
	if err == nil || !strings.Contains(err.Error(), "durability") {
		t.Fatalf("Invoke() error = %v, want an error naming the invalid durability", err)
	}
}

// TestRunDurabilityPropagatesToSubgraphs verifies the per-run override
// propagates into subgraph runs, mirroring Python setting
// CONFIG_KEY_DURABILITY for subgraphs (main.py:2862-2864): a sync-compiled
// parent+child invoked with DurabilityExit defers the child's writes too
// (single exit flush in the child namespace).
func TestRunDurabilityPropagatesToSubgraphs(t *testing.T) {
	ctx := t.Context()
	saver := newCountingSaver(checkpoint.NewMemorySaver())

	// A two-node child (two supersteps), so sync vs exit differ in the
	// child namespace too: sync persists input + 2 loop checkpoints, exit
	// only the stub + final flush.
	child := NewStateGraph()
	child.AddNode("c1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"child": 1}, nil
	})
	child.AddNode("c2", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"child": 2}, nil
	})
	child.AddEdge(types.START, "c1")
	child.AddEdge("c1", "c2")
	child.AddEdge("c2", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}
	top := NewStateGraph()
	top.AddSubgraph("sub", childCG)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	// Sync baseline: root ns gets input + 1 loop checkpoint (the sub->END
	// superstep commits once); the child ns gets input + 2 loop checkpoints.
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("sync Invoke() error = %v", err)
	}
	syncRootPuts := saver.putsInNS("")
	if syncRootPuts != 2 {
		t.Fatalf("sync run root-ns Put calls = %d, want 2", syncRootPuts)
	}
	syncChildPuts := saver.puts.Load() - syncRootPuts
	if syncChildPuts != 3 {
		t.Fatalf("sync run child-ns Put calls = %d, want 3 (input + 2 loop)", syncChildPuts)
	}

	// Exit override propagates: the child namespace gets only its exit flush
	// (stub anchor + final checkpoint).
	exitSaver := newCountingSaver(checkpoint.NewMemorySaver())
	exitTop := NewStateGraph()
	exitTop.AddSubgraph("sub", childCG)
	exitTop.AddEdge(types.START, "sub")
	exitTop.AddEdge("sub", types.END)
	exitCg, err := exitTop.Compile(WithCheckpointer(exitSaver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := exitCg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t2"}.WithRunDurability(DurabilityExit)); err != nil {
		t.Fatalf("exit-override Invoke() error = %v", err)
	}
	exitChildPuts := exitSaver.puts.Load() - exitSaver.putsInNS("")
	if exitChildPuts != 2 {
		t.Fatalf("exit-override run child-ns Put calls = %d, want 2 (stub + final exit flush only)", exitChildPuts)
	}
}

// interruptResumeGraph builds a -> ask -> b where "ask" interrupts in-node,
// so a pause happens in the second superstep with one committed superstep
// ("a") already behind it. opts are appended after the checkpointer.
func interruptResumeGraph(t *testing.T, saver checkpoint.Saver, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	})
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "who?")
		return map[string]any{"answer": v}, nil
	})
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"done": true}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "ask")
	g.AddEdge("ask", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(append([]CompileOption{WithCheckpointer(saver)}, opts...)...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestPauseDurabilityMatrixRootInterrupt is the T15 PR-1 regression gate: a
// pause is a state that must be immediately resumable, so under EVERY
// durability mode the pause checkpoint and its pending writes must be durably
// persisted before Invoke returns (Python forces the write in
// _suppress_interrupt, _loop.py:1319-1329). Before the fix, async mode raced
// the direct PutWrites against the queued checkpoint Put and exit mode never
// persisted the checkpoint at all — both surfaced "PutWrites: no checkpoint".
func TestPauseDurabilityMatrixRootInterrupt(t *testing.T) {
	for _, mode := range []Durability{DurabilitySync, DurabilityAsync, DurabilityExit} {
		t.Run(string(mode), func(t *testing.T) {
			ctx := t.Context()
			saver := checkpoint.NewMemorySaver()
			cg := interruptResumeGraph(t, saver, WithDurability(mode))

			res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t-" + string(mode)})
			if err != nil {
				t.Fatalf("pause Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 1 || res.Interrupts[0].Value != "who?" {
				t.Fatalf("expected one interrupt (who?), got %+v", res.Interrupts)
			}

			// Same-instance resume: the pause state is durable, a scalar resume
			// answers the single pending interrupt, and the run completes.
			res, err = cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t-" + string(mode), Resume: "42"})
			if err != nil {
				t.Fatalf("resume Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 0 {
				t.Fatalf("resume re-paused with %+v", res.Interrupts)
			}
			if v, _ := res.Values["answer"].(string); v != "42" {
				t.Fatalf("resumed answer = %v, want 42", res.Values["answer"])
			}
			if v, _ := res.Values["x"].(int); v != 1 {
				t.Fatalf("resumed x = %v, want 1", res.Values["x"])
			}
			if v, _ := res.Values["done"].(bool); !v {
				t.Fatalf("resumed done = %v, want true", res.Values["done"])
			}
		})
	}
}

// TestPauseDurabilityCrossInstanceResume resumes a paused thread through a
// DIFFERENT CompiledGraph instance sharing the saver, proving the persisted
// pause state is complete without any in-memory carryover.
func TestPauseDurabilityCrossInstanceResume(t *testing.T) {
	for _, mode := range []Durability{DurabilityAsync, DurabilityExit} {
		t.Run(string(mode), func(t *testing.T) {
			ctx := t.Context()
			saver := checkpoint.NewMemorySaver()
			first := interruptResumeGraph(t, saver, WithDurability(mode))

			res, err := first.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "x-" + string(mode)})
			if err != nil {
				t.Fatalf("pause Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 1 {
				t.Fatalf("expected one interrupt, got %+v", res.Interrupts)
			}

			second := interruptResumeGraph(t, saver, WithDurability(mode))
			res, err = second.InvokeWithOptions(ctx, nil, Options{ThreadID: "x-" + string(mode), Resume: "42"})
			if err != nil {
				t.Fatalf("cross-instance resume Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 0 {
				t.Fatalf("cross-instance resume re-paused with %+v", res.Interrupts)
			}
			if v, _ := res.Values["answer"].(string); v != "42" {
				t.Fatalf("cross-instance resumed answer = %v, want 42", res.Values["answer"])
			}
		})
	}
}

// TestPauseDurabilityBoundaryInterrupts runs the interrupt_before pause
// through the durability matrix (nil resume, Python's invoke(None, config)):
// the boundary pause checkpoint and its ReservedInterrupt pending write must
// be durable in every mode.
func TestPauseDurabilityBoundaryInterrupts(t *testing.T) {
	for _, mode := range []Durability{DurabilitySync, DurabilityAsync, DurabilityExit} {
		t.Run(string(mode), func(t *testing.T) {
			ctx := t.Context()
			saver := checkpoint.NewMemorySaver()
			g := NewStateGraph()
			g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
				return map[string]any{"x": 1}, nil
			})
			g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) {
				return map[string]any{"done": true}, nil
			})
			g.AddEdge(types.START, "a")
			g.AddEdge("a", "b")
			g.AddEdge("b", types.END)
			cg, err := g.Compile(WithCheckpointer(saver), WithDurability(mode), WithInterruptBefore("b"))
			if err != nil {
				t.Fatalf("Compile() error = %v", err)
			}

			res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "b-" + string(mode)})
			if err != nil {
				t.Fatalf("pause Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 1 {
				t.Fatalf("expected one boundary interrupt, got %+v", res.Interrupts)
			}

			res, err = cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "b-" + string(mode)})
			if err != nil {
				t.Fatalf("resume Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 0 {
				t.Fatalf("resume re-paused with %+v", res.Interrupts)
			}
			if v, _ := res.Values["done"].(bool); !v {
				t.Fatalf("resumed done = %v, want true", res.Values["done"])
			}
		})
	}
}

// TestPauseDurabilityExitDeltaChannelSurvivesPause pins the exit-mode pause
// path's accumulated-write anchoring: with a never-snapshotted delta channel
// (snapshotFrequency=100), the input delta write deferred by exit mode must
// still be materialized when the pause forces a synchronous write, anchored on
// the pause checkpoint itself (no stub), so GetState's ancestor walk
// reconstructs the value from the paused thread. (Resume-time in-memory delta
// reconstruction across a pause is a pre-existing limitation shared with sync
// mode and is out of scope here.)
func TestPauseDurabilityExitDeltaChannelSurvivesPause(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddChannel("msgs", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(rt, "q")
		return nil, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityExit))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{"msgs": []string{"t1", "turn"}}, Options{ThreadID: "dp"})
	if err != nil {
		t.Fatalf("pause Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("expected one interrupt, got %+v", res.Interrupts)
	}

	// The pause checkpoint must be the namespace's LATEST checkpoint (an
	// anchor minted after it would shadow it and break resume).
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "dp"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = (%v, %v), want the pause checkpoint", tup, err)
	}
	if len(tup.Checkpoint.Next) != 1 || tup.Checkpoint.Next[0].Node != "ask" {
		t.Fatalf("latest checkpoint Next = %+v, want the paused ask task", tup.Checkpoint.Next)
	}

	// GetState reconstructs the delta channel from the pause state.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "dp"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	got, _ := snap.Values["msgs"].([]string)
	if !reflect.DeepEqual(got, []string{"t1", "turn"}) {
		t.Fatalf("paused msgs = %v, want [t1 turn] (input delta write lost at pause)", snap.Values["msgs"])
	}
}

// stringBatchReducer is a BatchReducer that concatenates existing (a []string)
// with each update in the batch, mirroring intBatchReducer for string tokens.
// NewDeltaChannel takes a BatchReducer directly, NOT BatchFromReducer(Reducer).
func stringBatchReducer(existing any, updates []any) (any, error) {
	base, _ := existing.([]string)
	out := make([]string, len(base))
	copy(out, base)
	for _, u := range updates {
		add, _ := u.([]string)
		out = append(out, add...)
	}
	return out, nil
}

// exitDeltaGraph builds a single-node graph whose "msgs" channel is a
// DeltaChannel with snapshotFrequency=100, so it NEVER snapshots mid-loop and
// forces state reconstruction via the checkpoint ancestor-write walk. The
// "appender" node appends a constant token on every run, so each turn
// contributes both an input delta write and a node delta write.
func exitDeltaGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("msgs", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddNode("appender", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"msgs": []string{"node"}}, nil
	})
	g.AddEdge(types.START, "appender")
	g.AddEdge("appender", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestExitDurabilityDeltaChannelMultiTurn verifies the flushExit anchor fix: in
// exit mode, a second turn's accumulated delta writes must anchor on the
// PERSISTED parent checkpoint (initialCfg) — not the not-yet-persisted final
// checkpoint (flushCfg) — so they survive and GetState can reconstruct BOTH
// turns via the ancestor-write walk.
//
// snapshotFrequency=100 ensures "msgs" is never force-snapshotted, so every
// value lives only as per-checkpoint delta writes — the exact path the anchor
// bug corrupted. Before the fix, turn 2's flushExit PutWrites'd against a
// non-existent checkpoint (the in-memory-only flushCfg), which errored and
// aborted before the final checkpoint was saved; GetState then returned only
// turn 1's state. With the error-surfacing fix (named return in run), that
// PutWrites failure is now returned from Invoke itself.
func TestExitDurabilityDeltaChannelMultiTurn(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := exitDeltaGraph(t, WithCheckpointer(saver), WithDurability(DurabilityExit))
	ctx := t.Context()

	// Turn 1: input delta write ["t1","turn"]; node appends ["node"].
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"msgs": []string{"t1", "turn"}}, Options{ThreadID: "multi"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}

	// Turn 2: input delta write ["t2","turn"]; node appends ["node"] again.
	// A new turn loads the turn-1 checkpoint as its persisted parent, so this
	// flush is the one that exercises the persisted-parent anchor.
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"msgs": []string{"t2", "turn"}}, Options{ThreadID: "multi"}); err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}

	// GetState reconstructs "msgs" by walking the checkpoint parent chain and
	// replaying ancestor delta writes. Both turns must be present.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "multi"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	msgs, ok := snap.Values["msgs"]
	if !ok {
		t.Fatal("GetState().Values missing 'msgs'")
	}
	got, ok := msgs.([]string)
	if !ok {
		t.Fatalf("msgs is %T, want []string", msgs)
	}
	// Replay order: turn-1 input, turn-1 node, turn-2 input, turn-2 node.
	// Missing turn-2 tokens means the anchor bug lost the second turn's writes.
	want := []string{"t1", "turn", "node", "t2", "turn", "node"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("msgs = %v, want %v (turn-2 writes lost = anchor bug)", got, want)
	}
}
