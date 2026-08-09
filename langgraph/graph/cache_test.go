package graph

import (
	"context"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// compileCacheGraph builds a single-node graph with the given cache policy
// and cache backend: START -> "node" -> END.
func compileCacheGraph(t *testing.T, fn NodeFunc, policy *CachePolicy, cache checkpoint.Cache) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNodeWithPolicies("node", fn, NodePolicies{Cache: policy})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile(WithCache(cache))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestDefaultCacheKeyDeterministic(t *testing.T) {
	k1, err := DefaultCacheKey(map[string]any{"a": 1, "b": "x"})
	if err != nil {
		t.Fatalf("DefaultCacheKey() error = %v", err)
	}
	// json.Marshal sorts map keys, so key order must not matter.
	k2, err := DefaultCacheKey(map[string]any{"b": "x", "a": 1})
	if err != nil {
		t.Fatalf("DefaultCacheKey() error = %v", err)
	}
	if k1 != k2 {
		t.Fatalf("DefaultCacheKey() not key-order deterministic: %q vs %q", k1, k2)
	}
	if len(k1) != 64 { // sha256 hex
		t.Fatalf("DefaultCacheKey() = %q, want a 64-char sha256 hex digest", k1)
	}
	k3, err := DefaultCacheKey(map[string]any{"a": 2})
	if err != nil {
		t.Fatalf("DefaultCacheKey() error = %v", err)
	}
	if k3 == k1 {
		t.Fatalf("DefaultCacheKey() collision between different inputs: %q", k3)
	}
	if _, err := DefaultCacheKey(map[string]any{"f": func() {}}); err == nil {
		t.Fatalf("DefaultCacheKey() of a non-JSON value error = nil, want an error")
	}
}

func TestCacheHitSkipsExecutionButAppliesWrites(t *testing.T) {
	var runs atomic.Int32
	cg := compileCacheGraph(t, func(_ runtime.Runtime, state map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"echo": state["x"]}, nil
	}, &CachePolicy{}, checkpoint.NewInMemoryCache())
	ctx := context.Background()

	first, err := cg.Invoke(ctx, map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	second, err := cg.Invoke(ctx, map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if got := runs.Load(); got != 1 {
		t.Fatalf("node ran %d times, want 1 (second run must be a cache hit)", got)
	}
	if !reflect.DeepEqual(first.Values, second.Values) {
		t.Fatalf("cache-hit values = %+v, want identical to first run %+v", second.Values, first.Values)
	}
	if second.Values["echo"] != 1 {
		t.Fatalf("echo = %v, want 1", second.Values["echo"])
	}
}

func TestCacheMissOnDifferentInput(t *testing.T) {
	var runs atomic.Int32
	cg := compileCacheGraph(t, func(_ runtime.Runtime, state map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"echo": state["x"]}, nil
	}, &CachePolicy{}, checkpoint.NewInMemoryCache())
	ctx := context.Background()

	if _, err := cg.Invoke(ctx, map[string]any{"x": 1}); err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	res, err := cg.Invoke(ctx, map[string]any{"x": 2})
	if err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if got := runs.Load(); got != 2 {
		t.Fatalf("node ran %d times, want 2 (different input must miss)", got)
	}
	if res.Values["echo"] != 2 {
		t.Fatalf("echo = %v, want 2", res.Values["echo"])
	}
}

func TestCacheTTLExpiryReexecutes(t *testing.T) {
	var runs atomic.Int32
	cg := compileCacheGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"done": true}, nil
	}, &CachePolicy{TTL: 20 * time.Millisecond}, checkpoint.NewInMemoryCache())
	ctx := context.Background()

	if _, err := cg.Invoke(ctx, map[string]any{"x": 1}); err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	time.Sleep(40 * time.Millisecond)
	if _, err := cg.Invoke(ctx, map[string]any{"x": 1}); err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if got := runs.Load(); got != 2 {
		t.Fatalf("node ran %d times, want 2 (expired entry must re-execute)", got)
	}
}

func TestClearCacheForcesReexecution(t *testing.T) {
	var runs atomic.Int32
	cg := compileCacheGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"done": true}, nil
	}, &CachePolicy{}, checkpoint.NewInMemoryCache())
	ctx := context.Background()

	if _, err := cg.Invoke(ctx, map[string]any{"x": 1}); err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if err := cg.ClearCache(ctx, "writes/node"); err != nil {
		t.Fatalf("ClearCache() error = %v", err)
	}
	if _, err := cg.Invoke(ctx, map[string]any{"x": 1}); err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if got := runs.Load(); got != 2 {
		t.Fatalf("node ran %d times, want 2 (ClearCache must force re-execution)", got)
	}
}

func TestCacheSendArgsKeyOnArg(t *testing.T) {
	var runs atomic.Int32
	g := NewStateGraph()
	g.AddReducer("results", channels.AppendSliceReducer)
	g.AddNode("fan", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return &types.Command{Goto: []any{
			&types.Send{Node: "work", Arg: map[string]any{"n": 1}},
			&types.Send{Node: "work", Arg: map[string]any{"n": 2}},
		}}, nil
	})
	g.AddNodeWithPolicies("work", func(_ runtime.Runtime, state map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"results": []int{state["n"].(int) * 10}}, nil
	}, NodePolicies{Cache: &CachePolicy{}})
	g.AddEdge(types.START, "fan")
	g.AddEdge("work", types.END)
	cg, err := g.Compile(WithCache(checkpoint.NewInMemoryCache()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	first, err := cg.Invoke(ctx, map[string]any{"go": true})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if got := runs.Load(); got != 2 {
		t.Fatalf("work ran %d times, want 2 (one per Send)", got)
	}
	second, err := cg.Invoke(ctx, map[string]any{"go": true})
	if err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if got := runs.Load(); got != 2 {
		t.Fatalf("work ran %d times on the second run, want 2 (both Sends must hit)", got)
	}
	// If the two Send args collided on one cache key, the second run would
	// replay the same writes twice and the results would differ from run 1.
	if !reflect.DeepEqual(first.Values, second.Values) {
		t.Fatalf("cache-hit values = %+v, want identical to first run %+v", second.Values, first.Values)
	}
	if got, want := second.Values["results"], []int{10, 20}; !reflect.DeepEqual(got, want) {
		t.Fatalf("results = %v, want %v", got, want)
	}
}

// TestCacheCompletedSiblingReplayBypassesCache: a completed sibling of an
// interrupted task is replayed from its persisted pending writes on resume —
// never re-dispatched — so it cannot consult the cache at all. Clearing the
// cache before resume proves the replay does not depend on (or miss into) it.
func TestCacheCompletedSiblingReplayBypassesCache(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cache := checkpoint.NewInMemoryCache()
	var firstRuns, secondRuns atomic.Int32
	g := NewStateGraph()
	g.AddNodeWithPolicies("first", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		firstRuns.Add(1)
		return map[string]any{"a": 1}, nil
	}, NodePolicies{Cache: &CachePolicy{}})
	g.AddNode("second", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		secondRuns.Add(1)
		Interrupt(ctx, "hold")
		return nil, nil
	})
	g.AddEdge(types.START, "first")
	g.AddEdge("first", "second")
	g.AddEdge("second", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithCache(cache))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	res, err := cg.InvokeWithOptions(ctx, map[string]any{"q": "v"}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("expected one interrupt, got %+v", res.Interrupts)
	}
	if firstRuns.Load() != 1 || secondRuns.Load() != 1 {
		t.Fatalf("runs at pause: first=%d second=%d, want 1 each", firstRuns.Load(), secondRuns.Load())
	}

	// Prove the replay path is cache-independent: wipe the cache, then resume.
	if err := cg.ClearCache(ctx, "writes/first"); err != nil {
		t.Fatalf("ClearCache() error = %v", err)
	}
	res, err = cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "go"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", res.Interrupts)
	}
	if got := firstRuns.Load(); got != 1 {
		t.Fatalf("first ran %d times, want 1 (replayed from pending writes, never re-dispatched)", got)
	}
	if got := secondRuns.Load(); got != 2 {
		t.Fatalf("second ran %d times, want 2 (re-executed with the resume value)", got)
	}
	if res.Values["a"] != 1 {
		t.Fatalf("a = %v, want 1 (replayed write must land)", res.Values["a"])
	}
}

// TestCacheInterruptResumeBypassesCache: a task resuming with a pending
// interrupt must skip the cache lookup even when a DIFFERENT run populated an
// entry for the same key — otherwise the node would be skipped and the resume
// value silently dropped.
func TestCacheInterruptResumeBypassesCache(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cache := checkpoint.NewInMemoryCache()
	var runs atomic.Int32
	g := NewStateGraph()
	g.AddNodeWithPolicies("gate", func(ctx runtime.Runtime, state map[string]any) (any, error) {
		runs.Add(1)
		if state["mode"] == "auto" {
			return map[string]any{"answer": "auto"}, nil
		}
		v := Interrupt(ctx, "question")
		return map[string]any{"answer": v}, nil
	}, NodePolicies{Cache: &CachePolicy{
		// Key on "x" only, so the "auto" and "human" runs share one cache key.
		KeyFunc: func(input map[string]any) (string, error) {
			return DefaultCacheKey(map[string]any{"x": input["x"]})
		},
	}})
	g.AddEdge(types.START, "gate")
	g.AddEdge("gate", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithCache(cache))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	// Thread "human" pauses on the in-node interrupt.
	res, err := cg.InvokeWithOptions(ctx, map[string]any{"x": 1, "mode": "human"}, Options{ThreadID: "human"})
	if err != nil {
		t.Fatalf("human Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("expected one interrupt, got %+v", res.Interrupts)
	}
	// Thread "auto" shares the cache key and completes, populating the cache.
	res, err = cg.InvokeWithOptions(ctx, map[string]any{"x": 1, "mode": "auto"}, Options{ThreadID: "auto"})
	if err != nil {
		t.Fatalf("auto Invoke() error = %v", err)
	}
	if res.Values["answer"] != "auto" {
		t.Fatalf("auto answer = %v, want auto", res.Values["answer"])
	}
	if got := runs.Load(); got != 2 {
		t.Fatalf("gate ran %d times, want 2", got)
	}

	// Resume the interrupted run: the cache holds an "auto" entry for the
	// same key, but the pending interrupt must bypass it — gate re-executes
	// and the resume value is delivered.
	res, err = cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "human", Resume: "human-answer"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", res.Interrupts)
	}
	if got := runs.Load(); got != 3 {
		t.Fatalf("gate ran %d times, want 3 (resume must re-execute, not hit the cache)", got)
	}
	if res.Values["answer"] != "human-answer" {
		t.Fatalf("answer = %v, want human-answer (the resume value must be delivered)", res.Values["answer"])
	}
}

func TestCacheKeyErrorFailsTask(t *testing.T) {
	var runs atomic.Int32
	cg := compileCacheGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"done": true}, nil
	}, &CachePolicy{}, checkpoint.NewInMemoryCache())

	// A func in state is not JSON-representable, so DefaultCacheKey fails.
	_, err := cg.Invoke(context.Background(), map[string]any{"bad": func() {}})
	if err == nil {
		t.Fatalf("Invoke() error = nil, want a wrapped key-computation error")
	}
	if !strings.Contains(err.Error(), `"node"`) || !strings.Contains(err.Error(), "cache") {
		t.Fatalf("Invoke() error = %v, want it to name the node and the cache key failure", err)
	}
	if got := runs.Load(); got != 0 {
		t.Fatalf("node ran %d times, want 0 (key failure must fail the task before execution)", got)
	}
}

func TestCacheCommandGotoCachedAndReplayed(t *testing.T) {
	var routerRuns, targetRuns atomic.Int32
	g := NewStateGraph()
	g.AddNodeWithPolicies("router", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		routerRuns.Add(1)
		return &types.Command{Update: map[string]any{"r": "yes"}, Goto: To("target")}, nil
	}, NodePolicies{Cache: &CachePolicy{}})
	g.AddNode("target", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		targetRuns.Add(1)
		return map[string]any{"done": true}, nil
	})
	g.AddEdge(types.START, "router")
	g.AddEdge("target", types.END)
	cg, err := g.Compile(WithCache(checkpoint.NewInMemoryCache()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	first, err := cg.Invoke(ctx, map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if routerRuns.Load() != 1 || targetRuns.Load() != 1 {
		t.Fatalf("runs: router=%d target=%d, want 1 each", routerRuns.Load(), targetRuns.Load())
	}

	second, err := cg.Invoke(ctx, map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if got := routerRuns.Load(); got != 1 {
		t.Fatalf("router ran %d times, want 1 (second run must be a cache hit)", got)
	}
	if got := targetRuns.Load(); got != 2 {
		t.Fatalf("target ran %d times, want 2 (the cached Command.Goto must route again)", got)
	}
	if !reflect.DeepEqual(first.Values, second.Values) {
		t.Fatalf("cache-hit values = %+v, want identical to first run %+v", second.Values, first.Values)
	}
	if second.Values["r"] != "yes" || second.Values["done"] != true {
		t.Fatalf("values = %+v, want r=yes done=true", second.Values)
	}
}

// mustStream runs Stream to completion (via stream_test.go's collectStream)
// and returns its chunks, failing on any run error.
func mustStream(t *testing.T, cg *CompiledGraph, input map[string]any, modes ...StreamMode) []StreamChunk {
	t.Helper()
	chunks, err := collectStream(t, cg.Stream(context.Background(), input, StreamOptions{Modes: modes}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	return chunks
}

func updatesChunks(chunks []StreamChunk) []any {
	var out []any
	for _, c := range chunks {
		if c.Mode == StreamUpdates {
			out = append(out, c.Payload)
		}
	}
	return out
}

func TestCacheHitEmitsUpdatesAndDebugTaskEvents(t *testing.T) {
	var runs atomic.Int32
	cg := compileCacheGraph(t, func(_ runtime.Runtime, state map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"echo": state["x"]}, nil
	}, &CachePolicy{}, checkpoint.NewInMemoryCache())
	input := map[string]any{"x": 1}

	first := updatesChunks(mustStream(t, cg, input, StreamUpdates))
	if len(first) == 0 {
		t.Fatalf("first run produced no updates chunks")
	}

	// Cache-hit run: the node does not execute, but its updates chunk is
	// still emitted from the injected writes, and debug task events still
	// fire for the prepared task.
	second := mustStream(t, cg, input, StreamUpdates, StreamDebug)
	if got := runs.Load(); got != 1 {
		t.Fatalf("node ran %d times, want 1 (second run must be a cache hit)", got)
	}
	if got := updatesChunks(second); !reflect.DeepEqual(got, first) {
		t.Fatalf("cache-hit updates chunks = %+v, want identical to first run %+v", got, first)
	}
	foundDebugTask := false
	for _, c := range second {
		if c.Mode != StreamDebug {
			continue
		}
		m, _ := c.Payload.(map[string]any)
		if m["type"] != "task" {
			continue
		}
		if payload, _ := m["payload"].(map[string]any); payload["name"] == "node" {
			foundDebugTask = true
		}
	}
	if !foundDebugTask {
		t.Fatalf("no debug task event for \"node\" on the cache-hit run; debug task events must still fire")
	}
}

func TestCacheHitEmitsNoNodeStartEnd(t *testing.T) {
	var runs atomic.Int32
	cg := compileCacheGraph(t, func(_ runtime.Runtime, state map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"echo": state["x"]}, nil
	}, &CachePolicy{}, checkpoint.NewInMemoryCache())
	ctx := context.Background()
	input := map[string]any{"x": 1}

	sink := &retryRecordingSink{}
	if _, err := cg.InvokeStream(ctx, input, Options{}, sink); err != nil {
		t.Fatalf("first InvokeStream() error = %v", err)
	}
	if sink.count(RawNodeStart, "node") != 1 || sink.count(RawNodeEnd, "node") != 1 {
		t.Fatalf("first run events: start=%d end=%d, want 1 each",
			sink.count(RawNodeStart, "node"), sink.count(RawNodeEnd, "node"))
	}

	if _, err := cg.InvokeStream(ctx, input, Options{}, sink); err != nil {
		t.Fatalf("second InvokeStream() error = %v", err)
	}
	if got := runs.Load(); got != 1 {
		t.Fatalf("node ran %d times, want 1 (second run must be a cache hit)", got)
	}
	if sink.count(RawNodeStart, "node") != 1 || sink.count(RawNodeEnd, "node") != 1 {
		t.Fatalf("cache-hit run emitted node events: start=%d end=%d, want both still 1 (a hit emits no start/end pair)",
			sink.count(RawNodeStart, "node"), sink.count(RawNodeEnd, "node"))
	}
}

func TestCachePolicyWithoutCacheBackendUnchanged(t *testing.T) {
	// A cache policy with no WithCache backend: zero behavior change, the
	// node executes on every run, and ClearCache is a no-op.
	var runs atomic.Int32
	g := NewStateGraph()
	g.AddNodeWithPolicies("node", func(_ runtime.Runtime, state map[string]any) (any, error) {
		runs.Add(1)
		return map[string]any{"echo": state["x"]}, nil
	}, NodePolicies{Cache: &CachePolicy{}})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	if _, err := cg.Invoke(ctx, map[string]any{"x": 1}); err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	res, err := cg.Invoke(ctx, map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if got := runs.Load(); got != 2 {
		t.Fatalf("node ran %d times, want 2 (no cache backend installed)", got)
	}
	if res.Values["echo"] != 1 {
		t.Fatalf("echo = %v, want 1", res.Values["echo"])
	}
	if err := cg.ClearCache(ctx, "writes/node"); err != nil {
		t.Fatalf("ClearCache() without a backend error = %v, want nil", err)
	}
}
