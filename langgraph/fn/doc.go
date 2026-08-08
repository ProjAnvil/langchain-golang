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
// come from the existing executor. See the package documentation for the
// documented divergences (no store, timeout semantics, replayed error
// typing, StateGraph-node usage pattern).
package fn
