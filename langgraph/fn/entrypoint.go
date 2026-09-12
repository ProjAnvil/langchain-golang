package fn

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

const (
	channelStart    = "__start__"
	channelEnd      = "__end__"
	channelPrevious = "__previous__"
	entrypointNode  = "entrypoint"
)

// EntrypointOpts bundles the optional Entrypoint backends and policies,
// mirroring `@entrypoint(checkpointer=..., store=..., cache=...,
// retry_policy=...)`.
type EntrypointOpts struct {
	// Checkpointer enables cross-invocation state (previous) and
	// interrupt/resume with task-result replay. Nil disables persistence.
	Checkpointer checkpoint.Saver
	// Store is the cross-thread BaseStore, mirroring Python's
	// `@entrypoint(store=...)`. When non-nil it is installed on the internal
	// graph via graph.WithStore and surfaced on Runtime.Store for the
	// entrypoint function and every task it dispatches (see fn.Task). Nil
	// disables the store; consumers nil-check rt.Store before use.
	Store store.Store
	// Cache is the backend for task-level Cache policies (TaskOpts.Cache)
	// and is simply installed on the internal graph via graph.WithCache.
	Cache checkpoint.Cache
	// Retry retries the workflow's failures: it is installed as the internal
	// graph's graph-level default (graph.WithDefaultRetryPolicy — Python
	// parity: func/__init__.py:607 passes retry_policy to the internal Pregel,
	// not to the node), so it retries BOTH the entrypoint function (the single
	// node, which has no own policy) AND every task without its own
	// TaskOpts.Retry (PUSH calls inherit the graph default).
	Retry *graph.RetryPolicy
	// CachePolicy caches the whole workflow's writes (the __end__/__previous__
	// result) on the internal entrypoint node, mirroring Python's
	// @entrypoint(cache_policy=...) (func/__init__.py:443, passed to the
	// internal Pregel at :606). Inert unless Cache is also installed — the
	// same "policy without backend is inert" rule as graph node caching.
	// Only SUCCESSFUL runs are cached.
	CachePolicy *graph.CachePolicy
	// Timeout caps each workflow attempt, mirroring @entrypoint(timeout=...)
	// (func/__init__.py:445): installed as the internal node's
	// graph.TimeoutPolicy. Cooperative cancellation applies — a function
	// that never observes rt.Done() overruns the deadline (see
	// graph.TimeoutPolicy).
	Timeout *graph.TimeoutPolicy
	// ContextSchema validates the run-scoped context (rt.Context, attached
	// via runtime.ContextWithValues) before the entrypoint function runs —
	// the Go port of @entrypoint(context_schema=...) validation
	// (func/__init__.py:442; Python coerces context into the schema type,
	// which has no Go analogue, so the port is a validation hook). Nil
	// disables validation. A validation error fails the invocation.
	ContextSchema func(ctx any) error
}

// Entrypoint is a function compiled to a checkpointed workflow, mirroring
// Python's `entrypoint`-decorated callable. I is the input type, O the
// return type, S the save type threaded through `previous`.
type Entrypoint[I, O, S any] struct {
	opts  EntrypointOpts
	graph *graph.CompiledGraph
}

// NewEntrypoint compiles f into an Entrypoint. prev is the save value of the
// previous invocation on the same thread (graph.Options.ThreadID); hasPrev
// is false — and prev the zero value — when there is no checkpointer, no
// ThreadID, or no prior completed invocation (Python passes None; Go uses an
// explicit bool so a zero S is never misread). For the plain form the
// returned O value is also written as the save value, so O must be
// assignable to S (use NewEntrypointFinal to decouple them); a mismatch
// surfaces as a descriptive error on the NEXT invocation's previous
// assertion, not silently.
//
// Construction returns an error if f is nil or the internal graph fails to
// compile — both caller-visible, not panics: opts.Timeout and opts.Retry are
// user-supplied and their validation errors surface at compile time, so an
// invalid policy must return an error rather than crash the program.
func NewEntrypoint[I, O, S any](opts EntrypointOpts,
	f func(rt runtime.Runtime, in I, prev S, hasPrev bool) (O, error)) (*Entrypoint[I, O, S], error) {
	if f == nil {
		return nil, errors.New("fn: entrypoint function must be non-nil")
	}
	nodeFn := func(rt runtime.Runtime, state map[string]any) (any, error) {
		if err := validateEntrypointContext(opts.ContextSchema, rt); err != nil {
			return nil, err
		}
		in, prev, hasPrev, err := entrypointInput[I, S](state)
		if err != nil {
			return nil, err
		}
		v, err := f(rt, in, prev, hasPrev)
		if err != nil {
			return nil, err
		}
		// The plain form saves the returned value itself (see NewEntrypoint);
		// a type mismatch with S surfaces on the NEXT invocation's previous
		// assertion inside entrypointInput, not here.
		return map[string]any{channelEnd: v, channelPrevious: v}, nil
	}
	return compileEntrypoint[I, O, S](opts, nodeFn)
}

// Final decouples the value returned to the caller from the value saved for
// the next invocation's `previous`, mirroring `entrypoint.final(value=,
// save=)` (`func/__init__.py:475-514`).
type Final[O, S any] struct {
	// Value is returned to the caller (written to the __end__ channel).
	Value O
	// Save is persisted for the next invocation's previous (written to the
	// __previous__ channel).
	Save S
}

// NewEntrypointFinal is NewEntrypoint for functions returning Final[O, S].
// Like NewEntrypoint it returns an error instead of panicking on a nil
// function or an invalid compile-time option.
func NewEntrypointFinal[I, O, S any](opts EntrypointOpts,
	f func(rt runtime.Runtime, in I, prev S, hasPrev bool) (Final[O, S], error)) (*Entrypoint[I, O, S], error) {
	if f == nil {
		return nil, errors.New("fn: entrypoint function must be non-nil")
	}
	nodeFn := func(rt runtime.Runtime, state map[string]any) (any, error) {
		if err := validateEntrypointContext(opts.ContextSchema, rt); err != nil {
			return nil, err
		}
		in, prev, hasPrev, err := entrypointInput[I, S](state)
		if err != nil {
			return nil, err
		}
		fin, err := f(rt, in, prev, hasPrev)
		if err != nil {
			return nil, err
		}
		return map[string]any{channelEnd: fin.Value, channelPrevious: fin.Save}, nil
	}
	return compileEntrypoint[I, O, S](opts, nodeFn)
}

// entrypointInput extracts the invocation input and the previous save value
// from the node state. A failed assertion is a serde-contract violation and
// surfaces as a descriptive error — never a silent downgrade.
func entrypointInput[I, S any](state map[string]any) (I, S, bool, error) {
	var zero I
	var prev S
	rawIn, ok := state[channelStart]
	if !ok {
		return zero, prev, false, fmt.Errorf("fn: entrypoint state is missing the %q channel", channelStart)
	}
	in, ok := rawIn.(I)
	if !ok {
		return zero, prev, false, fmt.Errorf("fn: entrypoint input has type %T, want the declared input type", rawIn)
	}
	rawPrev, ok := state[channelPrevious]
	if !ok {
		return in, prev, false, nil
	}
	p, ok := rawPrev.(S)
	if !ok {
		return zero, prev, false, fmt.Errorf("fn: previous value has type %T, want the declared save type "+
			"(plain NewEntrypoint requires the return type to be assignable to the save type; use NewEntrypointFinal to decouple them)", rawPrev)
	}
	return in, p, true, nil
}

// validateEntrypointContext runs the ContextSchema hook (nil = no
// validation), wrapping failures with the entrypoint context.
func validateEntrypointContext(schema func(any) error, rt runtime.Runtime) error {
	if schema == nil {
		return nil
	}
	if err := schema(rt.Context); err != nil {
		return fmt.Errorf("fn: entrypoint context_schema validation: %w", err)
	}
	return nil
}

// compileEntrypoint builds the single-node StateGraph every Entrypoint
// compiles to: three reserved channels (__start__ ephemeral input, __end__
// and __previous__ last-value saves) plus one "entrypoint" node (Python
// parity `func/__init__.py:576-609`). opts.Retry is installed as the
// graph-level default (not the node's policy — mirroring Python, which passes
// retry_policy to the internal Pregel), so it also reaches tasks via the
// dispatcher (see dispatcher.defaultRetry). Compile can fail on user-supplied
// policies (opts.Timeout / opts.Retry are validated there), so the error is
// returned to the constructor rather than panicking.
func compileEntrypoint[I, O, S any](opts EntrypointOpts, nodeFn graph.NodeFunc) (*Entrypoint[I, O, S], error) {
	g := graph.NewStateGraph().
		AddChannel(channelStart, channels.NewEphemeral(true)).
		AddChannel(channelEnd, channels.NewLastValue()).
		AddChannel(channelPrevious, channels.NewLastValue()).
		AddNodeWithPolicies(entrypointNode, func(rt runtime.Runtime, state map[string]any) (any, error) {
			return runWithBootstrapDispatcher(opts, nodeFn, rt, state)
		}, graph.NodePolicies{Cache: opts.CachePolicy, Timeout: opts.Timeout}).
		SetEntryPoint(entrypointNode).
		AddEdge(entrypointNode, types.END)
	var copts []graph.CompileOption
	if opts.Retry != nil {
		copts = append(copts, graph.WithDefaultRetryPolicy(opts.Retry))
	}
	if opts.Checkpointer != nil {
		copts = append(copts, graph.WithCheckpointer(opts.Checkpointer))
	}
	if opts.Store != nil {
		copts = append(copts, graph.WithStore(opts.Store))
	}
	if opts.Cache != nil {
		copts = append(copts, graph.WithCache(opts.Cache))
	}
	cg, err := g.Compile(copts...)
	if err != nil {
		return nil, fmt.Errorf("fn: entrypoint graph compile: %w", err)
	}
	return &Entrypoint[I, O, S]{opts: opts, graph: cg}, nil
}

// prepare assembles one fresh run: the input batch (with the previous save
// value injected from the thread's latest checkpoint, when any), the replay
// table loaded from that same tuple, and a run-scoped context carrying this
// run's dispatcher. The returned cancel must run before the dispatcher is
// sealed (F4 teardown order: cancel -> seal).
func (e *Entrypoint[I, O, S]) prepare(ctx context.Context, in I, opts graph.Options) (context.Context, map[string]any, context.CancelFunc, *dispatcher, error) {
	input := map[string]any{channelStart: in}
	d := newDispatcher(e.opts.Cache)
	if e.opts.Retry != nil {
		// The graph-level default also retries tasks without their own
		// policy (Python: PUSH calls inherit it). Resolved once here, before
		// the run starts, so startTask's goroutines only read it.
		r := e.opts.Retry.Resolved()
		d.defaultRetry = &r
	}
	if e.opts.Checkpointer != nil && opts.ThreadID != "" {
		tup, err := e.opts.Checkpointer.GetTuple(ctx, checkpoint.Config{ThreadID: opts.ThreadID, CheckpointID: opts.CheckpointID})
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("fn: entrypoint checkpoint load: %w", err)
		}
		if tup != nil {
			// Replay loading must finish before the run starts: Task.Call
			// reads d.replay/d.cpID/d.step lock-free, ordered after this
			// load by the run's start (happens-before).
			d.loadReplay(tup, opts)
			// Fresh rounds read previous from the pre-run checkpoint and
			// write it into the input batch; on resume the graph ignores
			// input and __previous__ is restored from the pause checkpoint
			// instead.
			if opts.Resume == nil {
				if rawPrev, ok := tup.Checkpoint.ChannelValues[channelPrevious]; ok {
					input[channelPrevious] = rawPrev
				}
			}
		}
	}
	runCtx, cancel := context.WithCancel(contextWithDispatcher(ctx, d))
	return runCtx, input, cancel, d, nil
}

// persistResults appends the run's recorded task results to the thread's
// latest checkpoint L (F4: the executor creates the pause/final checkpoint
// during the run — pause -> the pause checkpoint; completion -> the final
// loop checkpoint; failure -> the pre-run input checkpoint — so each
// result's deterministic task ID is re-stamped against L only afterwards;
// recompute-on-restore then always recomputes the same ID). A nil
// checkpointer, an empty ThreadID, or an empty result table is a no-op.
func (e *Entrypoint[I, O, S]) persistResults(ctx context.Context, opts graph.Options, d *dispatcher) error {
	if e.opts.Checkpointer == nil || opts.ThreadID == "" {
		return nil
	}
	results := d.snapshotResults()
	if len(results) == 0 {
		return nil
	}
	saver := e.opts.Checkpointer
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: opts.ThreadID})
	if err != nil {
		return err
	}
	if tup == nil {
		return nil // the run committed no checkpoint (e.g. it failed pre-loop)
	}
	return putResults(ctx, saver, tup, results)
}

// runWithBootstrapDispatcher adapts the entrypoint node to runs that did not
// start through Entrypoint.Invoke/Stream — today the internal graph composed
// as a StateGraph subgraph node (graph.AddSubgraph over the entrypoint's
// compiled graph, mirroring Python's add_node(name, entrypoint_func); the
// composition is package-internal until the compiled graph is exposed). The
// subgraph wrapper dispatches the child graph directly, so no fn-layer
// prepare ever installed a run dispatcher: Task.Call would panic. When the
// context already carries one (Entrypoint.Invoke/Stream, or an enclosing
// entrypoint invoked from a node — the divergence-6 shape), this is a no-op
// and the established F4 lifecycle is untouched.
func runWithBootstrapDispatcher(opts EntrypointOpts, nodeFn graph.NodeFunc, rt runtime.Runtime, state map[string]any) (res any, err error) {
	if dispatcherFromContext(rt) != nil {
		return nodeFn(rt, state)
	}
	d := newDispatcher(opts.Cache)
	if opts.Retry != nil {
		r := opts.Retry.Resolved()
		d.defaultRetry = &r
	}
	// The dispatcher context derives from rt (a context.Context), so the
	// node's interrupt state and ExecutionInfo survive the override and
	// Task.Call / graph.Interrupt both keep working through rt2.
	dctx, cancel := context.WithCancel(contextWithDispatcher(rt, d))
	defer func() {
		// Same pinned teardown order as Entrypoint.Invoke (cancel -> seal ->
		// persist). An interrupt panic unwinds through here: results recorded
		// before the pause persist best-effort (an error must not mask the
		// panic), then the panic continues to propagate to the graph
		// executor's recover.
		cancel()
		d.seal()
		perr := persistRunResults(rt, opts.Checkpointer, d)
		if p := recover(); p != nil {
			panic(p)
		}
		if perr != nil && err == nil {
			err = perr
		}
	}()
	loadRunReplay(rt, opts.Checkpointer, d)
	rt2 := rt.Override(runtime.WithRuntimeCtx(dctx))
	return nodeFn(rt2, state)
}

// loadRunReplay seeds d's replay table from the checkpoint position the
// surrounding graph run holds at node dispatch (rt.ExecutionInfo): a resumed
// subgraph run is pinned to its pause checkpoint, whose own pending writes
// carry the node-level interrupt writes but NOT the fn results — the
// interrupted node persisted those against the position it started from
// (persistRunResults below), which is the pause checkpoint's parent — so the
// lookup falls back one ParentConfig level. The fresh-run gates of
// dispatcher.loadReplay are intentionally bypassed (loadReplayTuple): the
// child's checkpoint namespace is per-task and a fresh task starts in an
// empty namespace, so any fn writes reachable from the start position always
// belong to this logical run's pause/resume chain.
func loadRunReplay(rt runtime.Runtime, saver checkpoint.Saver, d *dispatcher) {
	ei := rt.ExecutionInfo
	if saver == nil || ei == nil || ei.ThreadID == "" {
		return
	}
	tup := fetchTuple(rt, saver, checkpoint.Config{ThreadID: ei.ThreadID, CheckpointNS: ei.CheckpointNS, CheckpointID: ei.CheckpointID})
	if tup == nil {
		return
	}
	if !tupleHasFnWrites(tup) && tup.ParentConfig != nil {
		tup = fetchTuple(rt, saver, *tup.ParentConfig)
	}
	if tup != nil && tupleHasFnWrites(tup) {
		d.loadReplayTuple(tup)
	}
}

// persistRunResults appends d's recorded task results to the LATEST
// checkpoint of the run's namespace. At node exit — normal return or
// interrupt unwind — that is the position the run started from: the
// single-node child graph saves no checkpoint between dispatch and node
// completion, and the pause checkpoint (saved by the executor only AFTER the
// node unwinds) forks from exactly that position. Stamping the results there
// keeps the F4 invariant (IDs recompute from the tuple they are read
// against) and pairs with loadRunReplay's one-level parent walk: the next
// resume pins to the pause checkpoint and finds these writes on its parent.
//
// Durability note: this persists through the saver directly, so it requires
// the run's checkpoints to be durable at node-exit time — the default sync
// mode. Under DurabilityAsync/Exit the child's input checkpoint may not be
// flushed yet when the defer runs, the (best-effort) persist is dropped, and
// task replay degrades to re-execution. The Entrypoint.Invoke path is
// unaffected (it persists after the run returned, post-flush).
func persistRunResults(rt runtime.Runtime, saver checkpoint.Saver, d *dispatcher) error {
	ei := rt.ExecutionInfo
	if saver == nil || ei == nil || ei.ThreadID == "" {
		return nil
	}
	results := d.snapshotResults()
	if len(results) == 0 {
		return nil
	}
	tup, err := saver.GetTuple(rt, checkpoint.Config{ThreadID: ei.ThreadID, CheckpointNS: ei.CheckpointNS})
	if err != nil {
		return err
	}
	if tup == nil {
		return nil // the run committed no checkpoint (uncheckpointed composition)
	}
	return putResults(rt, saver, tup, results)
}

// fetchTuple is GetTuple with errors swallowed (a failed lookup means "no
// replay", not a failed run — the graph run owns error surfacing here).
func fetchTuple(ctx context.Context, saver checkpoint.Saver, cfg checkpoint.Config) *checkpoint.Tuple {
	tup, err := saver.GetTuple(ctx, cfg)
	if err != nil {
		return nil
	}
	return tup
}

// tupleHasFnWrites reports whether tup's pending writes carry any persisted
// task results.
func tupleHasFnWrites(tup *checkpoint.Tuple) bool {
	if tup == nil {
		return false
	}
	for _, w := range tup.PendingWrites {
		if w.Channel == checkpoint.ReservedReturn || w.Channel == checkpoint.ReservedError {
			return true
		}
	}
	return false
}

// putResults stamps each recorded result's deterministic task ID against tup
// (FnTaskID of tup's own ID/ns/step — the F4 rule) and persists it as one
// PutWrites batch: the result write (__return__/__error__) plus, when the
// execution consumed resume values via Interrupt, the ReservedFnConsumed
// count a later replay must re-skip.
func putResults(ctx context.Context, saver checkpoint.Saver, tup *checkpoint.Tuple, results []taskResult) error {
	for _, r := range results {
		id := graph.FnTaskID(tup.Checkpoint.ID, tup.Config.CheckpointNS, tup.Metadata.Step, r.name, r.parentPath, r.callIdx)
		w := checkpoint.Write{Channel: checkpoint.ReservedReturn, Value: r.value}
		if r.isErr {
			w = checkpoint.Write{Channel: checkpoint.ReservedError, Value: r.errMsg}
		}
		writes := []checkpoint.Write{w}
		if r.consumed > 0 {
			// The execution consumed resume values via Interrupt; persist the
			// count so a replay of this result re-skips them in the node's
			// shared resume queue (checkpoint.ReservedFnConsumed).
			writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedFnConsumed, Value: r.consumed})
		}
		// One PutWrites per result: PutWrites stamps a single taskID on the
		// whole batch (checkpoint/checkpoint.go + memory.go).
		if err := saver.PutWrites(ctx, tup.Config, writes, id, r.parentPath); err != nil {
			return err
		}
	}
	return nil
}

// Invoke runs the entrypoint. opts.ThreadID (+ a Checkpointer) enables
// previous-injection and resumability; opts.Resume feeds a value to a paused
// run's pending graph.Interrupt (input is ignored on resume, mirroring
// graph.Options.Resume semantics); opts.CheckpointID pins a historical
// checkpoint (time travel).
//
// When the run pauses on interrupts, Invoke returns the zero O and an
// *InterruptError carrying them (recover via errors.As). Any other run
// failure is returned as a plain error.
func (e *Entrypoint[I, O, S]) Invoke(ctx context.Context, in I, opts graph.Options) (O, error) {
	var zero O
	runCtx, input, cancel, d, err := e.prepare(ctx, in, opts)
	if err != nil {
		return zero, err
	}
	defer cancel() // panic-path backstop; the normal path cancels explicitly first
	defer func() {
		// A user panic unwinding through the graph run still gets the
		// pinned teardown (cancel -> seal -> snapshot -> PutWrites) on a
		// best-effort basis: a persist failure here must not mask the
		// original panic, which always continues to propagate.
		if p := recover(); p != nil {
			cancel()
			d.seal()
			_ = e.persistResults(ctx, opts, d)
			panic(p)
		}
	}()
	res, err := e.graph.InvokeWithOptions(runCtx, input, opts)
	// Pinned teardown order (F4): cancel the run-scoped ctx, seal the
	// dispatcher, then snapshot+persist the recorded results.
	cancel()
	d.seal()
	perr := e.persistResults(ctx, opts, d)
	if err != nil {
		return zero, err
	}
	if perr != nil {
		return zero, fmt.Errorf("fn: persisting task results: %w", perr)
	}
	if len(res.Interrupts) > 0 {
		return zero, &InterruptError{Interrupts: res.Interrupts}
	}
	v, ok := res.Values[channelEnd].(O)
	if !ok {
		return zero, fmt.Errorf("fn: entrypoint returned a value of type %T, want the declared output type", res.Values[channelEnd])
	}
	return v, nil
}

// Stream runs the entrypoint like Invoke and yields stream chunks. The mode
// is fixed to graph.StreamUpdates (Python's entrypoint default
// stream_mode="updates", `func/__init__.py:532`): each chunk's payload is
// map[string]any{"entrypoint": <return value>} for the completion chunk and
// map[string]any{"__interrupt__": []types.Interrupt{...}} on pause —
// reserved channel keys (__start__/__end__/__previous__) are filtered out,
// and individual task calls do NOT produce chunks (tasks run inside the
// node; they are not graph tasks — documented divergence from Python, whose
// PUSH tasks stream per-task updates). Early break cancels the run.
func (e *Entrypoint[I, O, S]) Stream(ctx context.Context, in I, opts graph.Options) iter.Seq2[graph.StreamChunk, error] {
	return func(yield func(graph.StreamChunk, error) bool) {
		runCtx, input, cancel, d, err := e.prepare(ctx, in, opts)
		if err != nil {
			yield(graph.StreamChunk{}, err)
			return
		}
		defer func() {
			// Same pinned teardown as Invoke (cancel -> seal -> persist),
			// best-effort: chunk/yield errors already reached the caller,
			// and a user panic unwinding through the run keeps propagating.
			cancel()
			d.seal()
			_ = e.persistResults(ctx, opts, d)
		}()
		for chunk, err := range e.graph.Stream(runCtx, input, graph.StreamOptions{
			Options: opts,
			Modes:   []graph.StreamMode{graph.StreamUpdates},
		}) {
			if err != nil {
				yield(graph.StreamChunk{}, err)
				return
			}
			// Rewrite the node update {entrypoint: {__end__: v, __previous__: s}}
			// to {entrypoint: v}; the __interrupt__ chunk carries no node key
			// and passes through unchanged.
			if p, ok := chunk.Payload.(map[string]any); ok {
				if inner, ok := p[entrypointNode].(map[string]any); ok {
					chunk.Payload = map[string]any{entrypointNode: inner[channelEnd]}
				}
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
}

// InterruptError is returned by Invoke when the run paused on one or more
// interrupts, mirroring the "__interrupt__" key of Python's invoke result.
type InterruptError struct {
	// Interrupts are the pausing interrupts, in the run's collection order.
	Interrupts []types.Interrupt
}

func (e *InterruptError) Error() string {
	return fmt.Sprintf("fn: entrypoint interrupted (%d pending)", len(e.Interrupts))
}
