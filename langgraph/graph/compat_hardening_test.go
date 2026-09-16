package graph

import (
	"context"
	"maps"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// PR-5 compatibility and composition hardening (design §6.8-§6.13): the
// scenarios PR-3/4 did not cover — cross-"process" resume, legacy-namespace
// data, cache interaction, streaming shapes, and GetState's pending
// interrupts.

// TestCrossProcessResumeRootInterrupt (§6.8, root half): resuming must depend
// ONLY on the persisted thread state, not on any in-memory run state of the
// CompiledGraph that paused. Two separately Compiled instances over one shared
// MemorySaver simulate two processes: the second Invoke carries nothing but
// Options{ThreadID, Resume} and a root-level interrupt answers through.
func TestCrossProcessResumeRootInterrupt(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	var askRuns atomic.Int32
	build := func() *CompiledGraph {
		t.Helper()
		g := NewStateGraph()
		g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
			askRuns.Add(1)
			v := Interrupt(rt, "root-question")
			return map[string]any{"answer": v}, nil
		})
		g.AddEdge(types.START, "ask")
		g.AddEdge("ask", types.END)
		cg, err := g.Compile(WithCheckpointer(saver))
		if err != nil {
			t.Fatalf("Compile() error = %v", err)
		}
		return cg
	}

	cg1 := build()
	res, err := cg1.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first-process Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].Value != "root-question" {
		t.Fatalf("Interrupts = %+v, want the single root interrupt", res.Interrupts)
	}

	cg2 := build() // a fresh Compile over the same saver: no shared run state
	res2, err := cg2.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "42"})
	if err != nil {
		t.Fatalf("second-process resume error = %v, want completion", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	if got := res2.Values["answer"]; got != "42" {
		t.Fatalf("Values[answer] = %v, want 42 (resume value reached the re-run node)", got)
	}
	if askRuns.Load() != 2 {
		t.Fatalf("ask runs = %d, want 2 (paused once, re-run once)", askRuns.Load())
	}
}

// TestCrossProcessResumeSubgraphInterrupt (§6.8, subgraph half): the §6.8
// contract with the interrupt raised inside a subgraph — the second Compile
// must reconstruct the parent pause checkpoint (ReservedInterrupt copy +
// Metadata.Parents pin), re-dispatch the subgraph task under the same planned
// ID, and forward the resume payload into the pinned child.
func TestCrossProcessResumeSubgraphInterrupt(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	var askRuns, afterRuns atomic.Int32
	var resumedWith any
	buildChild := func() *CompiledGraph {
		g := NewStateGraph()
		g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
			askRuns.Add(1)
			v := Interrupt(rt, "child-question")
			resumedWith = v
			return map[string]any{"answer": v}, nil
		})
		g.AddEdge(types.START, "ask")
		g.AddEdge("ask", types.END)
		cg, err := g.Compile()
		if err != nil {
			t.Fatalf("child Compile() error = %v", err)
		}
		return cg
	}
	build := func() *CompiledGraph {
		g := NewStateGraph()
		g.AddSubgraph("sub", buildChild())
		g.AddNode("after", func(_ runtime.Runtime, _ map[string]any) (any, error) {
			afterRuns.Add(1)
			return nil, nil
		})
		g.AddEdge(types.START, "sub")
		g.AddEdge("sub", "after")
		g.AddEdge("after", types.END)
		cg, err := g.Compile(WithCheckpointer(saver))
		if err != nil {
			t.Fatalf("Compile() error = %v", err)
		}
		return cg
	}

	cg1 := build()
	res, err := cg1.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first-process Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].Value != "child-question" {
		t.Fatalf("Interrupts = %+v, want the propagated child interrupt", res.Interrupts)
	}

	cg2 := build()
	res2, err := cg2.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "42"})
	if err != nil {
		t.Fatalf("second-process resume error = %v, want completion", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	if resumedWith != "42" {
		t.Fatalf("child Interrupt() returned %v, want the forwarded 42", resumedWith)
	}
	if got := res2.Values["answer"]; got != "42" {
		t.Fatalf("Values[answer] = %v, want 42 (child values merged into the parent)", got)
	}
	if askRuns.Load() != 2 {
		t.Fatalf("ask runs = %d, want 2 (paused once, re-run once)", askRuns.Load())
	}
	if afterRuns.Load() != 1 {
		t.Fatalf("after runs = %d, want 1 (ran once, after the resumed superstep)", afterRuns.Load())
	}
}

// TestLegacyNamespacedSubgraphInterruptResume (§6.9): read-side compatibility
// for a thread persisted before per-task subgraph namespacing AND before
// subgraph interrupts paused the parent — the parent pause checkpoint records
// the child position under the node-only Parents key ("sub"), the child's
// pause checkpoint lives in the legacy node-only namespace, and the pending
// interrupts carry no NS (pre-NS-stamp data). A scalar resume must pin the
// child to the legacy pause checkpoint (loaded from the legacy ns), run it to
// completion, and write every NEW checkpoint under the per-task namespace.
func TestLegacyNamespacedSubgraphInterruptResume(t *testing.T) {
	ctx := t.Context()

	var askRuns atomic.Int32
	child := compileChild(t, "ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		askRuns.Add(1)
		v := Interrupt(rt, "legacy-question")
		return map[string]any{"answer": v}, nil
	})
	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	top.AddSubgraph("sub", child)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	// Hand-craft the pre-T14e thread: a child pause checkpoint under the
	// node-only ns "sub" (its pending write keyed by the planned ask task's
	// ID, exactly as the pre-namespacing executor stamped it), and a parent
	// pause checkpoint in the root ns whose Parents pin uses the node-only
	// key and whose pending writes carry the interrupt copy (NS-less, as
	// pre-NS data decodes). The zero-prefixed IDs sort below every NewID
	// value so the resumed run's checkpoints rank newest.
	const (
		legacyChildPauseID  = "0000000000000-000000-0000000000000002"
		legacyParentPauseID = "0000000000000-000000-0000000000000003"
		askTaskID           = "0000000000000001"
		subTaskID           = "0000000000000002"
	)
	legacyInterrupt := types.Interrupt{Value: "legacy-question", ID: "ask-1"}
	legacyChildPause := checkpoint.Checkpoint{
		V:               1,
		ID:              legacyChildPauseID,
		ChannelValues:   map[string]any{},
		ChannelVersions: map[string]int64{},
		Next:            []checkpoint.PlannedTask{{ID: askTaskID, Node: "ask"}},
	}
	if _, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: "sub"}, legacyChildPause, checkpoint.Metadata{Source: "loop", Step: 0}, nil); err != nil {
		t.Fatal(err)
	}
	if err := saver.PutWrites(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: "sub", CheckpointID: legacyChildPauseID},
		interruptWrites([]types.Interrupt{legacyInterrupt}), askTaskID, ""); err != nil {
		t.Fatal(err)
	}
	legacyParentPause := checkpoint.Checkpoint{
		V:               1,
		ID:              legacyParentPauseID,
		ChannelValues:   map[string]any{},
		ChannelVersions: map[string]int64{},
		Next:            []checkpoint.PlannedTask{{ID: subTaskID, Node: "sub"}},
	}
	if _, err := saver.Put(ctx, checkpoint.Config{ThreadID: "t1"}, legacyParentPause,
		checkpoint.Metadata{Source: "loop", Step: 0, Parents: map[string]string{"sub": legacyChildPauseID}}, nil); err != nil {
		t.Fatal(err)
	}
	if err := saver.PutWrites(ctx, checkpoint.Config{ThreadID: "t1", CheckpointID: legacyParentPauseID},
		interruptWrites([]types.Interrupt{legacyInterrupt}), subTaskID, ""); err != nil {
		t.Fatal(err)
	}

	res, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "legacy-answer"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v, want completion from the legacy pin", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res.Interrupts)
	}
	if got := res.Values["answer"]; got != "legacy-answer" {
		t.Fatalf("Values[answer] = %v, want the resumed value", res.Values)
	}
	if askRuns.Load() != 1 {
		t.Fatalf("ask runs = %d, want 1 (executed once, from the pinned legacy pause)", askRuns.Load())
	}

	// New writes land under the per-task namespace keyed by the re-dispatched
	// task's ID ("sub:<subTaskID>"), while the legacy namespace is untouched
	// apart from its original checkpoint (no migration).
	newNS := "sub:" + subTaskID
	newTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: newNS}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(ns=%q) error = %v", newNS, err)
	}
	if len(newTups) == 0 {
		t.Fatalf("per-task ns %q holds no checkpoints, want the resumed child run's writes", newNS)
	}
	legacyTups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: "sub"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(legacy ns) error = %v", err)
	}
	if len(legacyTups) != 1 {
		t.Fatalf("legacy ns holds %d checkpoints after resume, want 1 (untouched)", len(legacyTups))
	}
}

// countingCache wraps a Cache backend counting Get/Set calls per namespace,
// for observing the cache interaction of resumed subgraph tasks.
type countingCache struct {
	mu   sync.Mutex
	gets map[string]int
	sets map[string]int
}

func newCountingCache() *countingCache {
	return &countingCache{gets: map[string]int{}, sets: map[string]int{}}
}

func (c *countingCache) Get(_ context.Context, ns, _ string) ([]checkpoint.Write, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.gets[ns]++
	return nil, false, nil
}

func (c *countingCache) Set(_ context.Context, ns, _ string, _ []checkpoint.Write, _ time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sets[ns]++
	return nil
}

func (c *countingCache) Clear(context.Context, string) error { return nil }

func (c *countingCache) snapshot() (gets, sets map[string]int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return maps.Clone(c.gets), maps.Clone(c.sets)
}

// TestSubgraphInterruptCacheResumeSkipsCache (§6.11): a subgraph node carrying
// a CachePolicy interrupts on its first run — the pause stores NO cache entry
// (interrupted tasks store nothing) — and the resume must NOT hit the cache:
// the resumed task is classified as an interrupt task, which skips the lookup,
// so the child re-executes and consumes the resume value. A cache hit here
// would silently drop the answer.
func TestSubgraphInterruptCacheResumeSkipsCache(t *testing.T) {
	ctx := t.Context()

	var askRuns atomic.Int32
	var resumedWith any
	child := compileChild(t, "ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		askRuns.Add(1)
		v := Interrupt(rt, "cached-q")
		resumedWith = v
		return map[string]any{"answer": v}, nil
	})

	cache := newCountingCache()
	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	// AddSubgraph has no policies variant; register the same wrapper node
	// AddSubgraph installs (invokeSubgraph) with the CachePolicy attached —
	// identical runtime behavior plus the policy.
	top.AddNodeWithPolicies("sub", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return invokeSubgraph(rt, "sub", child, state)
	}, NodePolicies{Cache: &CachePolicy{}})
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(saver), WithCache(cache))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want the child interrupt", res.Interrupts)
	}
	if askRuns.Load() != 1 {
		t.Fatalf("ask runs after pause = %d, want 1", askRuns.Load())
	}
	// The interrupted miss stored nothing.
	_, sets := cache.snapshot()
	if sets["writes/sub"] != 0 {
		t.Fatalf("cache Set calls for writes/sub = %d after the pause, want 0 (interrupted tasks store nothing)", sets["writes/sub"])
	}

	res2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "42"})
	if err != nil {
		t.Fatalf("resume error = %v, want completion", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	if resumedWith != "42" {
		t.Fatalf("child Interrupt() returned %v, want 42 (the child re-executed, not cache-hit)", resumedWith)
	}
	if got := res2.Values["answer"]; got != "42" {
		t.Fatalf("Values[answer] = %v, want 42", got)
	}
	if askRuns.Load() != 2 {
		t.Fatalf("ask runs after resume = %d, want 2 (resume re-executes the interrupted subgraph)", askRuns.Load())
	}
	// The resumed task never consulted the cache (interrupt tasks skip the
	// lookup), and its completion stored nothing either (resumed tasks store
	// nothing — a resume-derived entry would poison later fresh runs).
	gets, sets := cache.snapshot()
	if gets["writes/sub"] != 1 {
		t.Fatalf("cache Get calls for writes/sub = %d total, want 1 (fresh run only; the resume skipped the lookup)", gets["writes/sub"])
	}
	if sets["writes/sub"] != 0 {
		t.Fatalf("cache Set calls for writes/sub = %d after resume, want 0 (resumed tasks store nothing)", sets["writes/sub"])
	}
}

// TestStreamPauseChunkCarriesSubgraphInterrupt (§6.12, Stream half): the
// pause chunks of a Stream run carry the subgraph-raised interrupt — the
// updates interrupt chunk ({"__interrupt__": [...]}) and the values chunk
// (state with the "__interrupt__" key merged) both name the child's interrupt
// with its full nested NS. With Subgraphs: true the child's own chunks carry
// the child namespace.
func TestStreamPauseChunkCarriesSubgraphInterrupt(t *testing.T) {
	ctx := t.Context()

	child := compileChild(t, "ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "stream-q")
		return map[string]any{"answer": v}, nil
	})
	top := NewStateGraph()
	top.AddSubgraph("sub", child)
	top.AddEdge(types.START, "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(ctx, map[string]any{}, StreamOptions{
		Options:   Options{ThreadID: "t1"},
		Modes:     []StreamMode{StreamUpdates, StreamValues},
		Subgraphs: true,
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}

	var sawRootUpdatesInterrupt, sawRootValuesInterrupt, sawChildPauseChunk bool
	for _, ch := range chunks {
		if strings.HasPrefix(ch.Namespace, "sub") {
			// The child's own pause emission (its run paused first, under its
			// namespaced emitter) — including its interrupt chunks.
			if payload, ok := ch.Payload.(map[string]any); ok {
				if _, ok := payload[checkpoint.ReservedInterrupt]; ok {
					sawChildPauseChunk = true
				}
			}
			continue
		}
		payload, ok := ch.Payload.(map[string]any)
		if !ok {
			continue
		}
		raw, ok := payload[checkpoint.ReservedInterrupt]
		if !ok {
			continue
		}
		interrupts, ok := raw.([]types.Interrupt)
		if !ok || len(interrupts) != 1 {
			t.Fatalf("chunk (ns %q, mode %s) __interrupt__ payload = %#v, want one interrupt", ch.Namespace, ch.Mode, raw)
		}
		intr := interrupts[0]
		if intr.Value != "stream-q" {
			t.Fatalf("interrupt Value = %v, want the child's", intr.Value)
		}
		// The parent's pause chunk is a ROOT-namespace emission (the parent
		// pauses; the child already returned), and the interrupt it carries
		// is the propagated child interrupt with its full nested NS
		// (<sub>:<taskID>/ask:<taskID>).
		childNS, askSeg, ok := strings.Cut(intr.NS, "/")
		if !ok || !strings.HasPrefix(childNS, "sub:") || !perTaskNSPattern.MatchString(childNS) ||
			!strings.HasPrefix(askSeg, "ask:") || !perTaskNSPattern.MatchString(askSeg) {
			t.Fatalf("interrupt NS = %q, want the nested child task namespace", intr.NS)
		}
		switch ch.Mode {
		case StreamUpdates:
			sawRootUpdatesInterrupt = true
		case StreamValues:
			sawRootValuesInterrupt = true
			if ch.Namespace != "" {
				t.Fatalf("values pause chunk namespace = %q, want the root emission", ch.Namespace)
			}
		}
	}
	if !sawRootUpdatesInterrupt || !sawRootValuesInterrupt {
		t.Fatalf("root pause chunks: updates-interrupt=%v values-interrupt=%v, want both carrying the child interrupt",
			sawRootUpdatesInterrupt, sawRootValuesInterrupt)
	}
	if !sawChildPauseChunk {
		t.Fatalf("no pause chunk carried the child's namespace, want the child's own emission under Subgraphs:true")
	}

	// Resume through the stream completes with the merged answer.
	chunks2, err := collectStream(t, cg.Stream(ctx, nil, StreamOptions{
		Options: Options{ThreadID: "t1", Resume: "42"},
		Modes:   []StreamMode{StreamValues},
	}))
	if err != nil {
		t.Fatalf("resume Stream() error = %v", err)
	}
	var last map[string]any
	for _, ch := range chunks2 {
		if ch.Mode == StreamValues {
			if v, ok := ch.Payload.(map[string]any); ok {
				last = v
			}
		}
	}
	if got := last["answer"]; got != "42" {
		t.Fatalf("final streamed Values[answer] = %v, want 42", got)
	}
}

// eventRecorder collects RawEvents for the InvokeStream pairing test.
type eventRecorder struct {
	mu     sync.Mutex
	events []RawEvent
}

func (r *eventRecorder) EmitRawEvent(e RawEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

// TestInvokeStreamNodeEventsPairedAcrossSubgraphPanic (§6.12, InvokeStream
// half): when the AddSubgraph wrapper panics with the subgraph-interrupt
// signal (a child pausing), the emitting task wrapper still balances
// RawNodeStart/RawNodeEnd — the End fires after runTask converts the panic
// into the interrupts outcome, so every node's start/end pair stays matched.
func TestInvokeStreamNodeEventsPairedAcrossSubgraphPanic(t *testing.T) {
	ctx := t.Context()

	child := compileChild(t, "ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "pair-q")
		return map[string]any{"answer": v}, nil
	})
	top := NewStateGraph()
	top.AddNode("pre", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	top.AddSubgraph("sub", child)
	top.AddEdge(types.START, "pre")
	top.AddEdge("pre", "sub")
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	rec := &eventRecorder{}
	res, err := cg.InvokeStream(ctx, map[string]any{}, Options{ThreadID: "t1"}, rec)
	if err != nil {
		t.Fatalf("InvokeStream() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want the propagated child interrupt", res.Interrupts)
	}

	open := map[string]int{}
	for _, e := range rec.events {
		switch e.Kind {
		case RawNodeStart:
			open[e.Node]++
		case RawNodeEnd:
			open[e.Node]--
		}
		if open[e.Node] < 0 {
			t.Fatalf("RawNodeEnd for %q without a matching start: %+v", e.Node, rec.events)
		}
	}
	// The paused run dispatched pre and sub once each; the sub node's panic
	// (the child pausing) must not unbalance its pair.
	for _, node := range []string{"pre", "sub"} {
		if open[node] != 0 {
			t.Fatalf("node %q has %d unmatched RawNodeStart events (%+v)", node, open[node], rec.events)
		}
	}
	starts := map[string]int{}
	for _, e := range rec.events {
		if e.Kind == RawNodeStart {
			starts[e.Node]++
		}
	}
	if starts["sub"] != 1 || starts["pre"] != 1 {
		t.Fatalf("node starts = %v, want one dispatch each for pre and sub", starts)
	}

	// Resume through InvokeStream: the re-dispatched sub node re-pairs, the
	// child completes, and the run finishes.
	rec2 := &eventRecorder{}
	res2, err := cg.InvokeStream(ctx, nil, Options{ThreadID: "t1", Resume: "42"}, rec2)
	if err != nil {
		t.Fatalf("resume InvokeStream() error = %v, want completion", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	open = map[string]int{}
	for _, e := range rec2.events {
		switch e.Kind {
		case RawNodeStart:
			open[e.Node]++
		case RawNodeEnd:
			open[e.Node]--
		}
		if open[e.Node] < 0 {
			t.Fatalf("resume: RawNodeEnd for %q without a matching start: %+v", e.Node, rec2.events)
		}
	}
	for node, n := range open {
		if n != 0 {
			t.Fatalf("resume: node %q has %d unmatched RawNodeStart events", node, n)
		}
	}
	if got := res2.Values["answer"]; got != "42" {
		t.Fatalf("resumed Values[answer] = %v, want 42", got)
	}
}

// TestGetStateShowsSubgraphInterruptCopies (§6.13): GetState on a thread
// paused by a subgraph interrupt reports the pending interrupts — including
// the parent-level copy of the child's interrupt, verbatim (Value/ID and the
// full nested NS), alongside the planned subgraph task. A second pause (Send
// fan-out) aggregates both copies.
func TestGetStateShowsSubgraphInterruptCopies(t *testing.T) {
	ctx := t.Context()

	cg, _ := interruptingSubgraphFixture(t)
	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res.Interrupts) != 2 {
		t.Fatalf("Interrupts = %+v, want both child interrupts", res.Interrupts)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if len(snap.Interrupts) != 2 {
		t.Fatalf("snapshot Interrupts = %+v, want the two subgraph interrupt copies", snap.Interrupts)
	}
	byNS := map[string]types.Interrupt{}
	for _, intr := range snap.Interrupts {
		byNS[intr.NS] = intr
	}
	for _, want := range res.Interrupts {
		got, ok := byNS[want.NS]
		if !ok || got.ID != want.ID || got.Value != want.Value {
			t.Fatalf("snapshot interrupts = %+v, want a verbatim copy of %+v", snap.Interrupts, want)
		}
	}
	// The planned next tasks still name the two fanned-in subgraph tasks.
	var subTasks int
	for _, st := range snap.Tasks {
		if st.Name == "sub" {
			subTasks++
		}
	}
	if subTasks != 2 {
		t.Fatalf("snapshot Tasks = %+v, want two planned sub tasks", snap.Tasks)
	}
}
