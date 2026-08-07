// Package graph is a thin alias layer over
// `github.com/projanvil/langchain-golang/langgraph/graph`; see that package
// for documentation. New code should import langgraph/graph directly.
package graph

import lcgraph "github.com/projanvil/langchain-golang/langgraph/graph"

type NodeFunc = lcgraph.NodeFunc
type ConditionalEdge = lcgraph.ConditionalEdge
type StateGraph = lcgraph.StateGraph
type CompileOption = lcgraph.CompileOption
type CompiledGraph = lcgraph.CompiledGraph
type Options = lcgraph.Options
type Result = lcgraph.Result
type RawEventKind = lcgraph.RawEventKind
type RawEvent = lcgraph.RawEvent
type NodeEventSink = lcgraph.NodeEventSink

const (
	RawNodeStart = lcgraph.RawNodeStart
	RawNodeEnd   = lcgraph.RawNodeEnd
)

var (
	To                   = lcgraph.To
	NewStateGraph        = lcgraph.NewStateGraph
	WithCheckpointer     = lcgraph.WithCheckpointer
	WithRecursionLimit   = lcgraph.WithRecursionLimit
	WithInterruptBefore  = lcgraph.WithInterruptBefore
	WithInterruptAfter   = lcgraph.WithInterruptAfter
	Interrupt            = lcgraph.Interrupt
	ContextWithEventSink = lcgraph.ContextWithEventSink
	EventSinkFromContext = lcgraph.EventSinkFromContext
)
