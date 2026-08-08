package fn

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TaskOpts bundles the optional per-task execution policies, mirroring the
// `@task(retry_policy=..., cache_policy=..., timeout=...)` decorator
// arguments.
type TaskOpts struct {
	// Retry enables automatic retry of the task's failures (see
	// graph.RetryPolicy). Nil means never retry.
	Retry *graph.RetryPolicy
	// Cache enables result caching for the task (see graph.CachePolicy). It
	// is inert unless the enclosing Entrypoint has a checkpoint.Cache
	// backend installed (EntrypointOpts.Cache) — the same "policy without
	// backend is inert" rule as graph node caching. Only SUCCESSFUL results
	// are cached (graph parity: errored/interrupted tasks store nothing).
	Cache *graph.CachePolicy
	// Timeout caps each task attempt. Go cannot kill a goroutine, so a
	// timeout can only cancel the attempt's context and stop waiting for it
	// (the abandoned goroutine still runs to completion in the background;
	// task functions should honor their context). A timed-out attempt fails
	// with context.DeadlineExceeded, which graph.DefaultRetryOn treats as
	// retryable. Zero means no timeout.
	Timeout time.Duration
}

// Task is a named, callable unit of work, created by NewTask. The name
// replaces Python's `module.qualname` introspection: it identifies the task
// in the cache namespace (`__fn_writes/<name>`) and in deterministic task
// IDs, so it must be unique within an entrypoint's call graph.
type Task[I, O any] struct {
	name string
	f    func(context.Context, I) (O, error)
	opts TaskOpts
}

// NewTask wraps f as a Task. f runs in its own goroutine per Call; it must
// be safe for concurrent use if the caller holds several Futures of the same
// task at once.
func NewTask[I, O any](name string, f func(context.Context, I) (O, error), opts TaskOpts) *Task[I, O] {
	if name == "" {
		panic("fn: task name must be non-empty")
	}
	if f == nil {
		panic("fn: task function must be non-nil")
	}
	return &Task[I, O]{name: name, f: f, opts: opts}
}

// Call starts the task and returns a Future for its result. Call may only be
// reached from within an Entrypoint function, from within another task, or
// from a StateGraph node via an Entrypoint.Invoke inside that node (the run
// dispatcher travels through the context); anywhere else Call panics —
// the Go analogue of Python's runtime error for tasks called outside an
// entrypoint/StateGraph.
//
// When the enclosing run resumed from a checkpoint whose pending writes
// contain this call's deterministic task ID, the Future is filled from the
// persisted result and f does NOT re-execute (checkpoint replay).
func (t *Task[I, O]) Call(ctx context.Context, in I) *Future[O] {
	d := dispatcherFromContext(ctx)
	if d == nil {
		panic("fn: Task.Call must be called from within an Entrypoint function, another task, or an Entrypoint invoked inside a StateGraph node")
	}
	parentPath, _ := ctx.Value(callPathKey{}).(string)
	callIdx := d.nextCallIdx(parentPath)

	// 1. Checkpoint replay: a persisted result fills the future without
	//    re-executing (pregel/_runner.py:745-756).
	if d.replay != nil {
		id := graph.FnTaskID(d.cpID, d.ns, d.step, t.name, parentPath, callIdx)
		if w, ok := d.replayWrite(id); ok {
			return replayedCall[O](ctx, d, t.name, parentPath, callIdx, w, d.replayConsumed(id))
		}
	}
	// 2. Cache lookup (independent second mechanism; only with a backend).
	key := ""
	if t.opts.Cache != nil && d.cache != nil {
		fut, k, ok := cachedCall(ctx, d, t, parentPath, callIdx, in)
		if ok {
			return fut
		}
		key = k // reuse for the post-run store: KeyFunc runs once per Call
	}
	// 3. Fresh execution in its own goroutine.
	return startTask(d, ctx, t, parentPath, callIdx, in, key)
}

// ClearCache removes every cached result of this task from cache,
// mirroring Python's `_TaskFunction.clear_cache`. It is a no-op when the
// task has no Cache policy.
func (t *Task[I, O]) ClearCache(ctx context.Context, cache checkpoint.Cache) error {
	if t.opts.Cache == nil {
		return nil
	}
	return cache.Clear(ctx, fnCacheNS(t.name))
}

// fnCacheNS is the cache namespace of a task's results (spec-pinned); the Go
// analogue of Python's `CACHE_NS_WRITES + module.qualname`.
func fnCacheNS(name string) string { return "__fn_writes/" + name }

// replayedCall fills a Future from a persisted replay write and re-records
// the outcome: a replayed result must be re-buffered into this run's result
// table so a chained pause re-appends it to the next checkpoint (F4) — the
// consumed count rides along so the queue alignment survives pause chains.
// Once the dispatcher is sealed, record drops it naturally — no race.
//
// The replayed task does not re-execute, so the Interrupt calls its original
// execution made never re-fire; consumed advances the node's shared resume
// queue past the values that execution already consumed (Go-only
// compensation for the shared queue — Python gives each @task call its own
// Pregel task and scratchpad, see checkpoint.ReservedFnConsumed).
func replayedCall[O any](ctx context.Context, d *dispatcher, name, parentPath string, callIdx int, w checkpoint.Write, consumed int) *Future[O] {
	graph.ReplayInterruptConsumption(ctx, consumed)
	var zero O
	switch w.Channel {
	case checkpoint.ReservedReturn:
		val, ok := w.Value.(O)
		if !ok {
			err := fmt.Errorf("fn: replayed result of task %q has type %T, want the declared output type", name, w.Value)
			d.record(taskResult{name: name, parentPath: parentPath, callIdx: callIdx, isErr: true, errMsg: err.Error()})
			return resolvedFuture[O](zero, err, nil)
		}
		d.record(taskResult{name: name, parentPath: parentPath, callIdx: callIdx, value: val, consumed: consumed})
		return resolvedFuture[O](val, nil, nil)
	case checkpoint.ReservedError:
		msg, ok := w.Value.(string)
		if !ok {
			msg = fmt.Sprint(w.Value)
		}
		d.record(taskResult{name: name, parentPath: parentPath, callIdx: callIdx, isErr: true, errMsg: msg, consumed: consumed})
		return resolvedFuture[O](zero, errors.New(msg), nil)
	default:
		// Unreachable: loadReplay keeps only __return__/__error__ writes.
		err := fmt.Errorf("fn: replayed write of task %q has unexpected channel %q", name, w.Channel)
		return resolvedFuture[O](zero, err, nil)
	}
}

// startTask runs the task in its own goroutine and returns a Future resolved
// by that goroutine. The goroutine writes fut's fields before close(fut.done)
// (happens-before via the channel) and is the sole closer, so done is closed
// exactly once.
func startTask[I, O any](d *dispatcher, ctx context.Context, t *Task[I, O], parentPath string, callIdx int, in I, key string) *Future[O] {
	fut := &Future[O]{done: make(chan struct{})}
	taskPath := t.name + "@" + strconv.Itoa(callIdx) // root call: "a@0" (no leading /)
	if parentPath != "" {
		taskPath = parentPath + "/" + taskPath // nested: "a@0/b@0"
	}
	taskCtx := context.WithValue(ctx, callPathKey{}, taskPath)
	// Resume values this Call's execution consumes via Interrupt, measured as
	// the node interrupt state's cursor delta over the whole attempt loop
	// (retries included). Persisted with the outcome (see persistResults) so
	// a later replay can re-skip them — replayed tasks never re-fire their
	// Interrupt calls (checkpoint.ReservedFnConsumed).
	consumedBefore := graph.InterruptConsumeCount(taskCtx)
	consumed := func() int { return graph.InterruptConsumeCount(taskCtx) - consumedBefore }
	go func() {
		defer close(fut.done)
		var retry *graph.RetryPolicy
		if t.opts.Retry != nil {
			r := t.opts.Retry.Resolved()
			retry = &r
		}
		for attempt := 1; ; attempt++ {
			val, gi, err := runAttempt(taskCtx, t, in)
			if gi != nil { // interrupt passthrough: not recorded, not retried; Get re-panics
				fut.gi = gi
				return
			}
			if err == nil {
				// Store before resolving: a cache-store failure fails the
				// task (graph node-cache parity), so it must not observe a
				// success first.
				if cerr := cacheStore(ctx, d, t, key, val); cerr != nil {
					fut.err = cerr
					d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, isErr: true, errMsg: cerr.Error(), consumed: consumed()})
					return
				}
				fut.val = val
				d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, value: val, consumed: consumed()})
				return
			}
			if taskCtx.Err() != nil { // run canceled (interrupt pause / parent cancel): give up, record nothing
				fut.err = err
				return
			}
			if retry == nil || attempt >= retry.MaxAttempts || !retry.RetryOn(err) {
				fut.err = err
				d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, isErr: true, errMsg: err.Error(), consumed: consumed()})
				return
			}
			timer := time.NewTimer(retry.BackoffDelay(attempt))
			select {
			case <-taskCtx.Done():
				timer.Stop()
				fut.err = taskCtx.Err()
				return
			case <-timer.C:
			}
		}
	}()
	return fut
}

// runAttempt executes one attempt. With a per-attempt Timeout the attempt
// runs in a child goroutine whose result channel is buffered (1), so an
// abandoned goroutine never blocks or leaks; on timeout the attempt fails
// with context.DeadlineExceeded (the goroutine itself cannot be killed — see
// TaskOpts.Timeout). Parent/run cancellation is distinguished from the
// task's own timeout and surfaced as the parent's error.
func runAttempt[I, O any](ctx context.Context, t *Task[I, O], in I) (O, *types.GraphInterrupt, error) {
	var zero O
	if t.opts.Timeout <= 0 {
		return callSafely(ctx, t, in)
	}
	attCtx, cancel := context.WithTimeout(ctx, t.opts.Timeout)
	defer cancel()
	type outcome struct {
		val O
		gi  *types.GraphInterrupt
		err error
	}
	ch := make(chan outcome, 1)
	go func() {
		v, g, e := callSafely(attCtx, t, in)
		ch <- outcome{v, g, e}
	}()
	select {
	case o := <-ch:
		return o.val, o.gi, o.err
	case <-attCtx.Done():
		if ctx.Err() != nil { // parent/run cancellation, not the task's own timeout
			return zero, nil, ctx.Err()
		}
		return zero, nil, context.DeadlineExceeded // give up waiting (the goroutine cannot be force-killed; documented)
	}
}

// callSafely invokes f, converting panics: a *types.GraphInterrupt is
// returned as gi (interrupt passthrough); any other panic becomes an
// ordinary error that participates in RetryOn decisions.
func callSafely[I, O any](ctx context.Context, t *Task[I, O], in I) (val O, gi *types.GraphInterrupt, err error) {
	defer func() {
		if r := recover(); r != nil {
			if g, ok := r.(*types.GraphInterrupt); ok {
				val = *new(O)
				gi = g
				err = nil
				return
			}
			val = *new(O)
			err = fmt.Errorf("fn: task %q panicked: %v", t.name, r)
		}
	}()
	val, err = t.f(ctx, in)
	return val, nil, err
}

// cachedCall serves a Call from the cache backend. ok=false means cache miss
// (or expired entry) and the caller proceeds to fresh execution, reusing the
// returned key for the post-run store so the KeyFunc is evaluated exactly
// once per Call. Lookup-phase failures (KeyFunc or Get error) fail the task
// with a wrapped error and are recorded as __error__ — graph node-cache
// parity ("key_func errors propagate as task errors"). A hit re-records the
// outcome into this run's result table, like replayedCall.
//
// The KeyFunc receives the call arguments packed as a single-key map,
// map[string]any{"input": in} (documented divergence: Python's key_func
// receives *args/**kwargs).
//
// Unsupported combination: a cache HIT replays the result without the
// resume-queue alignment replayedCall performs (the cached entry carries no
// consumed count), so a cached task whose function also calls Interrupt
// would misalign the node's shared resume queue — Python's baseline never
// combines cache_policy with interrupt-in-task (documented divergence).
func cachedCall[I, O any](ctx context.Context, d *dispatcher, t *Task[I, O], parentPath string, callIdx int, in I) (*Future[O], string, bool) {
	key, err := cacheKey(t, in)
	if err != nil {
		d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, isErr: true, errMsg: err.Error()})
		var zero O
		return resolvedFuture[O](zero, err, nil), "", true
	}
	writes, ok, err := d.cache.Get(ctx, fnCacheNS(t.name), key)
	if err != nil {
		werr := fmt.Errorf("fn: task %q cache get: %w", t.name, err)
		d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, isErr: true, errMsg: werr.Error()})
		var zero O
		return resolvedFuture[O](zero, werr, nil), "", true
	}
	if !ok || len(writes) == 0 || writes[0].Channel != checkpoint.ReservedReturn {
		return nil, key, false
	}
	val, ok := writes[0].Value.(O)
	if !ok {
		var zero O
		err := fmt.Errorf("fn: cached result of task %q has type %T, want the declared output type", t.name, writes[0].Value)
		d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, isErr: true, errMsg: err.Error()})
		return resolvedFuture[O](zero, err, nil), "", true
	}
	d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, value: val})
	return resolvedFuture[O](val, nil, nil), "", true
}

// cacheStore stores a successful result under the key cachedCall already
// derived for this Call. A Set failure fails the task with a wrapped error
// (graph node-cache parity); the caller records that error.
func cacheStore[I, O any](ctx context.Context, d *dispatcher, t *Task[I, O], key string, val O) error {
	if t.opts.Cache == nil || d.cache == nil {
		return nil
	}
	if err := d.cache.Set(ctx, fnCacheNS(t.name), key, []checkpoint.Write{{Channel: checkpoint.ReservedReturn, Value: val}}, t.opts.Cache.TTL); err != nil {
		return fmt.Errorf("fn: task %q cache set: %w", t.name, err)
	}
	return nil
}

// cacheKey derives the cache key for a call, wrapping KeyFunc errors (nil
// KeyFunc means graph.DefaultCacheKey).
func cacheKey[I, O any](t *Task[I, O], in I) (string, error) {
	keyFunc := t.opts.Cache.KeyFunc
	if keyFunc == nil {
		keyFunc = graph.DefaultCacheKey
	}
	key, err := keyFunc(map[string]any{"input": in})
	if err != nil {
		return "", fmt.Errorf("fn: task %q cache key: %w", t.name, err)
	}
	return key, nil
}
