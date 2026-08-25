// Package types holds the shared control-flow primitives of the Go port of
// Python's `langgraph` (https://github.com/langchain-ai/langgraph): the
// START/END sentinels and the Send/Command/Interrupt values exchanged
// between graph nodes and the Pregel-style executor in langgraph/graph.
//
// This package corresponds to Python's `langgraph.types` plus
// `langgraph.constants` and `langgraph.errors` (GraphInterrupt,
// GraphRecursionError).
package types

import (
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
)

// START and END are sentinel node names, matching Python's
// `langgraph.constants.START`/`END`.
const (
	START = "__start__"
	END   = "__end__"
)

// ParentGraph is the sentinel value for Command.Graph indicating the command
// targets the closest parent graph rather than the current graph, matching
// Python's `Command.PARENT`. It is meaningful inside a subgraph (see
// graph.StateGraph.AddSubgraph): the parent graph applies the command's
// update and goto at its own level. Surfacing from a top-level graph is an
// error, as is any other non-empty Command.Graph value.
const ParentGraph = "__parent__"

// Send represents a message to dynamically invoke a node with custom input,
// matching Python's `langgraph.types.Send`. It is returned by conditional
// edge routers (or embedded in a Command's Goto) to fan out to a node with
// per-invocation input, independent from the shared graph state.
type Send struct {
	// Node is the name of the target node.
	Node string
	// Arg is the input passed to Node for this invocation. Unlike normal
	// routing (which passes the full shared graph state), a Send-driven
	// invocation receives exactly this value as its node input. It must be a
	// map[string]any to satisfy CompiledGraph's NodeFunc signature.
	Arg map[string]any
}

// Command carries state updates and/or routing decisions returned by a node,
// matching (a scoped-down subset of) Python's `langgraph.types.Command`.
// It implements messages.ToolOutput, mirroring Python's Command(ToolOutputMixin) mixin (see IsToolOutput).
type Command struct {
	// Graph selects which graph the command applies to. Empty means the
	// current graph; ParentGraph targets the closest parent graph (see
	// graph.StateGraph.AddSubgraph); any other value is an error.
	Graph string
	// Update is a partial state update, merged into channel state via each
	// key's Reducer exactly like a plain map[string]any node return value.
	Update map[string]any
	// Resume supplies the value(s) an in-progress Interrupt call should
	// resume with. Keyed by interrupt ID, or a single value to resume the
	// next (first) pending interrupt in the resumed node.
	Resume any
	// Goto specifies what to execute next: a node name, multiple node names,
	// one or more *Send values, or nil to fall back to the graph's static/
	// conditional edges.
	Goto []any
}

// Compile-time assertion mirroring Python's class Command(Generic[N],
// ToolOutputMixin) (langgraph/types.py:759).
var _ messages.ToolOutput = (*Command)(nil)

// IsToolOutput marks Command as a value tools may return directly (Python's
// ToolOutputMixin): a tool signals graph control flow by placing a *Command in
// its Result.Artifact, and tool executors recognize it via the
// messages.ToolOutput interface. Always returns true.
func (*Command) IsToolOutput() bool { return true }

// Interrupt describes a pause in graph execution surfaced to the caller,
// matching Python's `langgraph.types.Interrupt`.
type Interrupt struct {
	// Value is the value surfaced to the caller, provided by the node's
	// Interrupt(ctx, value) call.
	Value any
	// ID identifies this interrupt so a Command.Resume map can target it.
	ID string
}

// GraphInterrupt is the sentinel error a node's execution stops with when it
// calls Interrupt and no resume value is available, matching Python's
// `GraphInterrupt` (a `GraphBubbleUp` exception). CompiledGraph.Invoke
// recovers this internally; it is exported so callers constructing custom
// node functions can recognize it (e.g. in tests) via errors.As.
type GraphInterrupt struct {
	Interrupt Interrupt
}

func (e *GraphInterrupt) Error() string {
	return fmt.Sprintf("types: interrupted with value %v (id=%s)", e.Interrupt.Value, e.Interrupt.ID)
}

// GraphRecursionError reports that a run exceeded its recursion limit (the
// maximum number of supersteps) before reaching a terminal state, mirroring
// Python's langgraph.errors.GraphRecursionError.
type GraphRecursionError struct {
	// Limit is the effective recursion limit the run tripped.
	Limit int
	// Node names the node about to be dispatched when the limit tripped,
	// when a single next node is known; empty otherwise. Diagnostic only.
	Node string
}

func (e *GraphRecursionError) Error() string {
	return fmt.Sprintf("graph: recursion limit (%d) exceeded", e.Limit)
}

// CacheKey is the structured cache key a CachePolicy.KeyFunc returns,
// mirroring Python's `langgraph.types.CacheKey` (a NamedTuple of
// ns: tuple[str, ...], key: str, ttl: int | None). Unlike a bare string key,
// it lets a key function specify the cache namespace and per-entry TTL
// alongside the key.
type CacheKey struct {
	// Namespace is the hierarchical entry namespace, joined with "/" by the
	// cache consumer. Empty selects the consumer's default namespace
	// ("writes/<node>" in the graph executor, "__fn_writes/<task>" in fn).
	Namespace []string
	// Key is the flat string cache key.
	Key string
	// TTL is the entry lifetime. 0 means "no override": the consumer falls
	// back to its CachePolicy.TTL, which itself means "never expires" when 0.
	TTL time.Duration
}

// NewCacheKey returns a CacheKey with the given namespace, key, and ttl.
func NewCacheKey(ns []string, key string, ttl time.Duration) CacheKey {
	return CacheKey{Namespace: ns, Key: key, TTL: ttl}
}
