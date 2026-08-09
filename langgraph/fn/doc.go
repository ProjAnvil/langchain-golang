// Package fn implements the functional API of the Go langgraph port,
// mirroring Python's `langgraph.func` (`@entrypoint` / `@task`): an
// Entrypoint wraps a plain function as a checkpointed workflow, and Task
// wraps a function as an asynchronously executed, checkpoint-replayable unit
// whose result survives interrupt/resume cycles.
//
// The package is a thin wrapper over the existing graph machinery: an
// Entrypoint compiles to a single-node graph.StateGraph with three reserved
// channels (__start__/__end__/__previous__, Python parity
// `func/__init__.py:576-609`), so interrupt/resume/stream/time-travel all
// come from the existing executor.
//
// # Documented divergences from Python
//
//  1. Timeout semantics: TaskOpts.Timeout can only cancel the attempt's
//     context and stop waiting for it — a goroutine cannot be force-killed,
//     so the abandoned attempt keeps running in the background and task
//     functions should honor their context (Python likewise does not
//     support timeout for sync task functions).
//
//  2. Cancellation of in-flight tasks: when a run pauses on an interrupt,
//     tasks that started but did not finish are canceled (run-scoped
//     context cancel + dispatcher seal); results that already completed
//     land in the checkpoint's pending writes (Python discards unfinished
//     PUSH tasks — the semantics correspond). The same applies on normal
//     return: a task still in flight because its Future was never Get'd is
//     canceled and its result is NOT persisted.
//
//  3. No checkpointer means hasPrev=false: Python passes previous=None;
//     Go uses an explicit bool so a zero save value is never misread.
//
//  4. No store: Python's `@entrypoint(checkpointer=..., store=...)`
//     cross-thread BaseStore is not ported; EntrypointOpts has no such
//     field.
//
//  5. Replayed errors lose their concrete type: only the message is
//     persisted (the __error__ write) and Get returns errors.New(msg) on
//     replay — Go cannot serialize error values (Python pickles the
//     exception object).
//
//  6. Tasks inside a StateGraph node: the Go shape is invoking an
//     Entrypoint inside the node (e.g. add.Invoke(rt, ...) within the
//     NodeFunc). Python's bare @task-in-node relies on Pregel config
//     injection and has no Go equivalent.
//
//  7. Stream is fixed to updates mode, and individual task calls produce
//     no chunks: tasks execute inside the node and are not graph tasks
//     (Python's PUSH tasks stream per-task updates).
//
//  8. Determinism constraint: across replays of the same entrypoint the
//     task call order must be deterministic (same rule as Python's
//     determinism section) — put non-deterministic logic inside tasks.
//     Interrupts must likewise surface in deterministic Get order.
//
//  9. Serde contract: I/O/S and task inputs/outputs must round-trip
//     through JSON or live in the serde closed registry; with a
//     persistent saver an unregistered type surfaces as a descriptive
//     error, never a silent downgrade.
//
//  10. Cache key input wrapping: the CachePolicy KeyFunc receives the call
//     arguments packed as map[string]any{"input": in} (Python's key_func
//     receives *args/**kwargs).
//
//  11. A failed entrypoint run poisons its thread: after an error, ANY
//     subsequent Invoke on the same ThreadID — including one with
//     brand-new input — replays and re-throws that error (replay gate 2
//     does not look at the input). Start a new round with a fresh
//     ThreadID to escape the error state (Python escapes with new input;
//     Go cannot).
//
//  12. Time-travel fork from a pause: pinning a paused checkpoint via
//     CheckpointID and feeding fresh input starts a NEW round — all tasks
//     re-execute and do NOT hit that checkpoint's pending writes (replay
//     only happens on Resume recovery or the gate-2 error retry).
//
//  13. Cache + interrupt-in-task is an unsupported combination: a cache
//     hit replays the result without resume-queue alignment (the cached
//     entry carries no consumed count), so a cached task whose function
//     also calls Interrupt would misalign the node's shared resume queue.
//     Python's baseline never combines cache_policy with
//     interrupt-in-task.
//
//  14. Concurrent resume-value consumption: when two tasks of the same
//     run both consume resume values via Interrupt, each Call's consumed
//     count is measured as the node-level cursor delta, so concurrent
//     consumers may cross-count each other. The Python baseline is the
//     sequential Call→Get pattern, where this is unreachable — keep
//     interrupt-consuming tasks sequential.
//
//  15. ReplayInterruptConsumption advances the resume cursor with no
//     upper-bound clamp; the normal path is kept correct by the
//     persistence invariant (a persisted consumed count never exceeds the
//     resume queue).
//
//  16. Default retry predicate is narrower: graph.DefaultRetryOn does NOT
//     retry errors outside its listed categories (net.Error,
//     context.DeadlineExceeded, 5xx HTTPStatus), while Python's
//     `default_retry_on` retries everything except its exclusions. Task
//     retries reuse the graph policy, so fn.Task inherits this divergence
//     (declared at graph/policy.go: DefaultRetryOn); supply RetryOn for
//     domain errors.
//
//  17. Single retry policy only: TaskOpts.Retry takes ONE *graph.RetryPolicy,
//     while Python's `@task(retry_policy=...)` accepts a SEQUENCE of policies
//     (tried in order per attempt). Modeling per-attempt policy switching has
//     no Go equivalent — a functional narrowing, not a behavior difference
//     for the single-policy case.
package fn
