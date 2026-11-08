// Package types holds the shared control-flow primitives of the Go port of
// Python's `langgraph` (https://github.com/langchain-ai/langgraph): the
// START/END sentinels and the Send/Command/Interrupt values exchanged
// between graph nodes and the Pregel-style executor in langgraph/graph.
//
// This package corresponds to Python's `langgraph.types` plus
// `langgraph.constants` and `langgraph.errors` (GraphInterrupt,
// GraphRecursionError, GraphDrained).
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
	// NS is the full task checkpoint namespace of the interrupted task:
	// "<parentNS>/<node>:<taskID>" for a task inside a subgraph, or
	// "<node>:<taskID>" at the root level (the namespace the task's own
	// checkpoints live under, see graph.taskCheckpointNS). Boundary
	// interrupts (interrupt_before/interrupt_after) carry the node-only
	// namespace "<parentNS>/<node>". A resume map may address an interrupt
	// by NS as well as by ID. Interrupts persisted before this field existed
	// deserialize with NS == "" and are matched by ID only.
	NS string
	// ResponseSchema is the JSON Schema of the value expected when resuming
	// this interrupt, if the node provided one (Python langgraph 1.2.13
	// interrupt(response_schema=...), #8886). nil when unset. Surfaced
	// interrupts carry it so HITL consumers can render structured resume
	// forms.
	ResponseSchema map[string]any `json:"response_schema,omitempty"`
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

// UnknownInterruptError is raised by strict resume mode (langgraph 1.2.20,
// #9574, CONFIG_KEY_STRICT_RESUME="__pregel_strict_resume"): a resume map
// names an interrupt that is not pending, is empty, or a scalar resume meets
// multiple pending interrupts. Without strict mode these are silently
// ignored (or a plain error for the ambiguous case), which hides double- or
// mis-answered interrupts from clients.
type UnknownInterruptError struct {
	// Reason is "not_pending", "empty_map", or "ambiguous".
	Reason string
	// InterruptIDs carries the offending resume ids (Reason "not_pending").
	InterruptIDs []string
	// PendingIDs carries the pending interrupt ids at check time, sorted.
	PendingIDs []string
}

// Error implements the error interface.
func (e *UnknownInterruptError) Error() string {
	switch e.Reason {
	case "empty_map":
		return "The resume map names no interrupt"
	case "not_pending":
		if len(e.InterruptIDs) > 0 {
			return "Unknown or already-consumed interrupt: " + e.InterruptIDs[0]
		}
		return "Unknown or already-consumed interrupt"
	case "ambiguous":
		return "When there are multiple pending interrupts, you must specify the interrupt id when resuming. " +
			"Docs: https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts."
	default:
		return "unknown interrupt error: " + e.Reason
	}
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

// GraphDrainedError reports that a run exited early because a cooperative
// drain was requested, mirroring Python's `langgraph.errors.GraphDrained` (a
// GraphBubbleUp). The graph stopped at a superstep boundary — the requesting
// superstep fully committed and its checkpoint (with the next task set) was
// saved — so a later resume continues from the boundary; see
// graph.Options.Control and runtime.RunControl (#9281).
type GraphDrainedError struct {
	// Reason is the drain reason the requester passed to
	// RunControl.RequestDrain ("shutdown" when it passed an empty string).
	Reason string
}

func (e *GraphDrainedError) Error() string {
	return fmt.Sprintf("graph drained: %s", e.Reason)
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
