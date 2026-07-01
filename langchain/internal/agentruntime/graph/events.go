package graph

// Streaming events layer for CompiledGraph.
//
// This file adds a non-breaking, opt-in event-ified execution path
// (CompiledGraph.InvokeStream) alongside the existing Invoke/InvokeWithOptions.
// The design keeps the graph's exported surface minimal and avoids any import
// of langchain/agents (the graph lives under internal/agentruntime and must not
// reach back up to the public agents package): instead of a callback taking an
// agents.StreamEvent directly, the graph emits small private RawEvent values to
// a NodeEventSink, and CreateAgent maps those into the public
// agents.StreamEvent (see langchain/agents/stream.go).
//
// Emitter lookup is context-value based (EventSinkFromContext), so a node
// function built by CreateAgent can ask for the sink only when streaming is
// active. When no sink is installed (the plain Invoke / InvokeWithOptions
// path), EventSinkFromContext returns nil and node functions fall back to
// their non-streaming behavior with zero added overhead — the existing tests
// and call shapes are untouched.
//
// Concurrent fan-out (Send): when multiple nodes run concurrently within a
// superstep, their events interleave on the sink. Consumers can disambiguate
// via the Node field on the public StreamEvent. Start/end pairs are guaranteed
// balanced per node invocation regardless of interleaving (see
// InvokeStream's emit pair around runNode).

import "context"

// RawEventKind enumerates the graph-level event kinds emitted during a streamed
// execution. These are graph-internal; CreateAgent maps them onto the public
// agents.StreamEventType constants.
type RawEventKind string

const (
	// RawNodeStart is emitted just before a node's NodeFunc is dispatched.
	RawNodeStart RawEventKind = "node_start"
	// RawNodeEnd is emitted just after a node's NodeFunc returns successfully.
	RawNodeEnd RawEventKind = "node_end"
)

// RawEvent is a graph-internal streaming event. It carries only the
// graph-level lifecycle (node start/end); model/tool nodes emit their own
// domain events directly through the NodeEventSink they obtain from their
// context (see EmitModelDelta/EmitModelEnd/EmitToolStart/EmitToolEnd helpers
// in langchain/agents/stream.go), so this struct intentionally has no
// model/tool payload fields.
type RawEvent struct {
	Kind RawEventKind
	Node string
}

// NodeEventSink receives RawEvents during a streamed execution. Implementations
// are expected to be goroutine-safe: InvokeStream dispatches concurrent nodes
// within a superstep, and each may emit through the same sink.
type NodeEventSink interface {
	EmitRawEvent(event RawEvent)
}

// eventSinkKey is the context-value key under which InvokeStream installs the
// active NodeEventSink. Node functions (or CreateAgent's model/tool node
// builders) retrieve it via EventSinkFromContext.
type eventSinkKey struct{}

// ContextWithEventSink returns a derived ctx carrying sink, so that node
// functions invoked during InvokeStream can discover it via
// EventSinkFromContext. It is intended for use by InvokeStream; callers that
// want to drive a streamed run use InvokeStream rather than installing a sink
// manually.
func ContextWithEventSink(ctx context.Context, sink NodeEventSink) context.Context {
	if sink == nil {
		return ctx
	}
	return context.WithValue(ctx, eventSinkKey{}, sink)
}

// EventSinkFromContext returns the NodeEventSink installed in ctx by
// InvokeStream (via ContextWithEventSink), or nil when streaming is not active
// (i.e. the plain Invoke/InvokeWithOptions path). Node functions should
// nil-check the result and fall back to non-streaming behavior when nil — this
// is what keeps the non-streaming path at zero added overhead.
func EventSinkFromContext(ctx context.Context) NodeEventSink {
	sink, _ := ctx.Value(eventSinkKey{}).(NodeEventSink)
	return sink
}
