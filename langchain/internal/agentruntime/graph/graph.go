package graph

// Package graph implements a deliberately scoped Go port of Python's
// `langgraph.graph.StateGraph` builder plus a synchronous, in-process
// Pregel-style executor (see `langgraph.pregel`), sufficient to run the fixed
// "model node <-> tools node" shape `langchain`'s v1 agent factory needs.
//
// Scope note: this is not a general distributed graph execution engine.
// Compared to Python's langgraph:
//
//   - No typed/schema-validated state: state is a plain map[string]any, with
//     per-key Reducer functions (see the channels package) standing in for
//     Python's `Annotated[T, reducer]` state schema fields.
//   - No subgraphs: Command.Graph must be empty (targeting "the current
//     graph"); any other value is a compile/runtime error.
//   - No langgraph "stream modes" (values/updates/debug), time-travel/state
//     history, caching, or retry policies. A minimal event-ified execution
//     path (InvokeStream + the NodeEventSink in events.go) IS supported, so
//     CreateAgent's StreamEvents can observe node/model/tool lifecycle; this
//     is not a general streaming-modes surface.
//   - Checkpointing (via the checkpoint package) only keeps the single most
//     recent checkpoint per thread, enough for the "pause on Interrupt,
//     resume with Command.Resume" human-in-the-loop pattern.
//   - Concurrent execution only happens *within* a superstep (multiple nodes
//     active at once via Send or multi-destination edges); node functions
//     must treat the state map they receive as read-only and communicate
//     changes only through their return value, since it may be read
//     concurrently by sibling tasks in the same superstep.

import (
	"context"
	"fmt"
	"sync"

	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/channels"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/checkpoint"
)

// NodeFunc is a graph node, mirroring Python's node callables. It receives
// the current graph state and returns one of:
//
//   - nil: no state update.
//   - map[string]any: a partial state update, merged into state via each
//     key's Reducer.
//   - *agentruntime.Command: a state update (Command.Update) plus an optional
//     routing override (Command.Goto) that bypasses the graph's static and
//     conditional edges for this task.
//
// Any other return type is a runtime error. NodeFunc must not mutate the
// state map it receives (see the package doc comment).
type NodeFunc func(ctx context.Context, state map[string]any) (any, error)

// ConditionalEdge routes execution dynamically based on state, mirroring
// Python's `add_conditional_edges` router callables. Each returned element
// must be a string (a node name, or agentruntime.END) or a *agentruntime.Send.
type ConditionalEdge func(ctx context.Context, state map[string]any) ([]any, error)

// To is a small convenience constructing a ConditionalEdge/Command.Goto
// destination list from plain node names.
func To(names ...string) []any {
	out := make([]any, len(names))
	for i, name := range names {
		out[i] = name
	}
	return out
}

// StateGraph builds a CompiledGraph, mirroring (a scoped-down subset of)
// Python's `langgraph.graph.StateGraph` builder.
type StateGraph struct {
	nodes       map[string]NodeFunc
	reducers    map[string]channels.Reducer
	edges       map[string][]string
	conditional map[string]ConditionalEdge
	entry       string
	err         error
}

// NewStateGraph constructs an empty StateGraph builder.
func NewStateGraph() *StateGraph {
	return &StateGraph{
		nodes:       map[string]NodeFunc{},
		reducers:    map[string]channels.Reducer{},
		edges:       map[string][]string{},
		conditional: map[string]ConditionalEdge{},
	}
}

func (g *StateGraph) setErr(err error) {
	if g.err == nil {
		g.err = err
	}
}

// AddNode registers a node. Names must be unique, non-empty, and distinct
// from agentruntime.START/agentruntime.END.
func (g *StateGraph) AddNode(name string, fn NodeFunc) *StateGraph {
	if name == "" || name == agentruntime.START || name == agentruntime.END {
		g.setErr(fmt.Errorf("graph: invalid node name %q", name))
		return g
	}
	if fn == nil {
		g.setErr(fmt.Errorf("graph: node %q function must not be nil", name))
		return g
	}
	if _, exists := g.nodes[name]; exists {
		g.setErr(fmt.Errorf("graph: duplicate node %q", name))
		return g
	}
	g.nodes[name] = fn
	return g
}

// AddReducer registers the Reducer used to merge updates written to key.
// Keys without a registered reducer default to channels.LastValueReducer.
func (g *StateGraph) AddReducer(key string, reducer channels.Reducer) *StateGraph {
	g.reducers[key] = reducer
	return g
}

// AddEdge adds a static (unconditional) edge from -> to. Passing
// agentruntime.START as from sets the graph's entry point, mirroring Python's
// `add_edge(START, node)`; only one entry point is supported (call
// SetEntryPoint or AddEdge(START, ...) exactly once).
func (g *StateGraph) AddEdge(from, to string) *StateGraph {
	if from == agentruntime.START {
		return g.SetEntryPoint(to)
	}
	g.edges[from] = append(g.edges[from], to)
	return g
}

// AddConditionalEdges registers a dynamic router for the given node,
// mirroring Python's `add_conditional_edges`. Only one router may be
// registered per source node.
func (g *StateGraph) AddConditionalEdges(from string, router ConditionalEdge) *StateGraph {
	if router == nil {
		g.setErr(fmt.Errorf("graph: conditional edge router for %q must not be nil", from))
		return g
	}
	if _, exists := g.conditional[from]; exists {
		g.setErr(fmt.Errorf("graph: duplicate conditional edge for %q", from))
		return g
	}
	g.conditional[from] = router
	return g
}

// SetEntryPoint sets the node the graph starts execution from, mirroring
// Python's `add_edge(START, name)` / `set_entry_point(name)`.
func (g *StateGraph) SetEntryPoint(name string) *StateGraph {
	if g.entry != "" {
		g.setErr(fmt.Errorf("graph: entry point already set to %q", g.entry))
		return g
	}
	g.entry = name
	return g
}

// CompileOption configures Compile.
type CompileOption func(*compileOptions)

type compileOptions struct {
	checkpointer   checkpoint.Saver
	recursionLimit int
}

// WithCheckpointer installs a checkpoint.Saver, enabling Interrupt/Resume
// support (mirrors passing `checkpointer=` to Python's `.compile()`).
func WithCheckpointer(saver checkpoint.Saver) CompileOption {
	return func(o *compileOptions) { o.checkpointer = saver }
}

// WithRecursionLimit overrides the default superstep limit (100), mirroring
// Python's `recursion_limit` config option. It guards against unintentional
// infinite loops in a graph's routing.
func WithRecursionLimit(limit int) CompileOption {
	return func(o *compileOptions) { o.recursionLimit = limit }
}

const defaultRecursionLimit = 100

// Compile validates the graph and returns an executable CompiledGraph,
// mirroring Python's `StateGraph.compile()`.
func (g *StateGraph) Compile(opts ...CompileOption) (*CompiledGraph, error) {
	if g.err != nil {
		return nil, g.err
	}
	if g.entry == "" {
		return nil, fmt.Errorf("graph: entry point not set (call AddEdge(agentruntime.START, node) or SetEntryPoint)")
	}
	if g.entry != agentruntime.END {
		if _, ok := g.nodes[g.entry]; !ok {
			return nil, fmt.Errorf("graph: entry point %q is not a registered node", g.entry)
		}
	}
	for from, tos := range g.edges {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("graph: edge source %q is not a registered node", from)
		}
		for _, to := range tos {
			if to != agentruntime.END {
				if _, ok := g.nodes[to]; !ok {
					return nil, fmt.Errorf("graph: edge target %q is not a registered node", to)
				}
			}
		}
	}
	for from := range g.conditional {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("graph: conditional edge source %q is not a registered node", from)
		}
	}

	options := compileOptions{recursionLimit: defaultRecursionLimit}
	for _, opt := range opts {
		opt(&options)
	}

	return &CompiledGraph{
		nodes:          g.nodes,
		reducers:       g.reducers,
		edges:          g.edges,
		conditional:    g.conditional,
		entry:          g.entry,
		checkpointer:   options.checkpointer,
		recursionLimit: options.recursionLimit,
	}, nil
}

// CompiledGraph is an executable graph, mirroring Python's
// `CompiledStateGraph`.
type CompiledGraph struct {
	nodes          map[string]NodeFunc
	reducers       map[string]channels.Reducer
	edges          map[string][]string
	conditional    map[string]ConditionalEdge
	entry          string
	checkpointer   checkpoint.Saver
	recursionLimit int
}

// Options configures a single Invoke call.
type Options struct {
	// ThreadID identifies the conversation/run for checkpointing. Required
	// (together with a checkpointer installed via WithCheckpointer) to use
	// Resume, or for a node's Interrupt call to be resumable at all.
	ThreadID string
	// Resume supplies the value(s) to resume a previously interrupted run
	// with, mirroring Python's `Command(resume=...)`. When set, input is
	// ignored and the run continues from the checkpointed state instead.
	Resume any
}

// Result is returned by Invoke, mirroring the value/interrupt split of
// Python's `graph.invoke()` (`state, state["__interrupt__"]`).
type Result struct {
	// Values is the graph state: either the final state (Interrupts empty)
	// or the state as of the pause (Interrupts non-empty).
	Values map[string]any
	// Interrupts holds any interrupts raised in the step that paused
	// execution. Empty means the run completed normally.
	Interrupts []agentruntime.Interrupt
}

type task struct {
	node string
	arg  map[string]any // nil means "use the shared graph state"
}

// Invoke runs the graph from its entry point with input as the initial
// state, mirroring Python's `graph.invoke(input)`.
func (g *CompiledGraph) Invoke(ctx context.Context, input map[string]any) (Result, error) {
	return g.InvokeWithOptions(ctx, input, Options{})
}

// InvokeWithOptions runs the graph, optionally resuming a checkpointed,
// previously interrupted run (see Options.Resume) instead of starting fresh
// from input.
//
// This is the non-streaming path: no event sink is installed, so node
// functions observe a nil sink from EventSinkFromContext and take their
// non-streaming code path with zero added overhead.
func (g *CompiledGraph) InvokeWithOptions(ctx context.Context, input map[string]any, opts Options) (Result, error) {
	return g.run(ctx, input, opts, nil)
}

// InvokeStream runs the graph exactly like InvokeWithOptions, but additionally
// installs sink into the context passed to every node function (via
// ContextWithEventSink / EventSinkFromContext) so that node start/end (and any
// model/tool events the node emits through the sink) are observable.
//
// InvokeStream is the event-ified path; Invoke/InvokeWithOptions are
// unchanged. It emits a RawNodeStart just before dispatching each node and a
// RawNodeEnd immediately after the node returns successfully (interrupted and
// errored nodes still get a RawNodeEnd before the error/interrupt is surfaced,
// so start/end pairs are always balanced per invocation).
//
// Concurrent fan-out (Send): when multiple nodes run concurrently within a
// superstep, their events interleave on sink. Consumers can disambiguate via
// the RawEvent.Node field (mapped to agents.StreamEvent.Node by CreateAgent).
func (g *CompiledGraph) InvokeStream(ctx context.Context, input map[string]any, opts Options, sink NodeEventSink) (Result, error) {
	return g.run(ctx, input, opts, sink)
}

// run is the shared execution loop backing both InvokeWithOptions (sink==nil,
// non-streaming, zero overhead) and InvokeStream (sink!=nil, event-ified). See
// each public entry point's doc comment for the contract.
func (g *CompiledGraph) run(ctx context.Context, input map[string]any, opts Options, sink NodeEventSink) (Result, error) {
	runCtx := ctx
	if sink != nil {
		runCtx = ContextWithEventSink(ctx, sink)
	}

	var state map[string]any
	var tasks []task
	resumeValues := map[string][]any{}

	if opts.Resume != nil {
		if g.checkpointer == nil {
			return Result{}, fmt.Errorf("graph: Options.Resume requires a checkpointer (see WithCheckpointer)")
		}
		if opts.ThreadID == "" {
			return Result{}, fmt.Errorf("graph: Options.Resume requires ThreadID")
		}
		cp, ok := g.checkpointer.Get(opts.ThreadID)
		if !ok {
			return Result{}, fmt.Errorf("graph: no checkpoint found for thread %q", opts.ThreadID)
		}
		state = cloneState(cp.Values)
		tasks = []task{{node: cp.Next}}
		resumeValues[cp.Next] = resumeValuesFor(cp.PendingInterrupts, opts.Resume)
	} else {
		state = cloneState(input)
		tasks = []task{{node: g.entry}}
	}

	steps := 0
	for {
		active := make([]task, 0, len(tasks))
		for _, t := range tasks {
			if t.node != agentruntime.END {
				active = append(active, t)
			}
		}
		if len(active) == 0 {
			break
		}

		steps++
		if steps > g.recursionLimit {
			return Result{}, fmt.Errorf("graph: recursion limit (%d) exceeded", g.recursionLimit)
		}

		type outcome struct {
			update      map[string]any
			cmd         *agentruntime.Command
			interrupted *agentruntime.Interrupt
			err         error
		}
		outcomes := make([]outcome, len(active))

		var wg sync.WaitGroup
		for i, t := range active {
			wg.Add(1)
			go func(i int, t task) {
				defer wg.Done()
				if sink != nil {
					sink.EmitRawEvent(RawEvent{Kind: RawNodeStart, Node: t.node})
				}
				result, interrupted, err := g.runNode(runCtx, t, state, resumeValues[t.node])
				if sink != nil {
					// Always emit node_end so start/end pairs are balanced per
					// invocation, even on the error/interrupt paths.
					sink.EmitRawEvent(RawEvent{Kind: RawNodeEnd, Node: t.node})
				}
				if err != nil {
					outcomes[i] = outcome{err: err}
					return
				}
				if interrupted != nil {
					outcomes[i] = outcome{interrupted: interrupted}
					return
				}
				update, cmd, nerr := normalizeNodeResult(result)
				outcomes[i] = outcome{update: update, cmd: cmd, err: nerr}
			}(i, t)
		}
		wg.Wait()
		resumeValues = nil

		for _, o := range outcomes {
			if o.err != nil {
				return Result{}, o.err
			}
		}

		var interrupts []agentruntime.Interrupt
		interruptedNode := ""
		for i, o := range outcomes {
			if o.interrupted != nil {
				interrupts = append(interrupts, *o.interrupted)
				if interruptedNode == "" {
					interruptedNode = active[i].node
				}
			}
		}
		if len(interrupts) > 0 {
			if g.checkpointer != nil && opts.ThreadID != "" {
				g.checkpointer.Put(opts.ThreadID, checkpoint.Checkpoint{
					Values:            cloneState(state),
					Next:              interruptedNode,
					PendingInterrupts: interrupts,
				})
			}
			return Result{Values: state, Interrupts: interrupts}, nil
		}

		for _, o := range outcomes {
			if o.update != nil {
				if err := mergeState(state, o.update, g.reducers); err != nil {
					return Result{}, err
				}
			}
		}

		var nextTasks []task
		for i, t := range active {
			if cmd := outcomes[i].cmd; cmd != nil && len(cmd.Goto) > 0 {
				dests, err := resolveDestinations(cmd.Goto)
				if err != nil {
					return Result{}, err
				}
				nextTasks = append(nextTasks, dests...)
				continue
			}
			dests, err := g.staticNext(ctx, t.node, state)
			if err != nil {
				return Result{}, err
			}
			nextTasks = append(nextTasks, dests...)
		}
		tasks = nextTasks
	}

	if g.checkpointer != nil && opts.ThreadID != "" {
		g.checkpointer.Delete(opts.ThreadID)
	}
	return Result{Values: state}, nil
}

func (g *CompiledGraph) staticNext(ctx context.Context, nodeName string, state map[string]any) ([]task, error) {
	if router, ok := g.conditional[nodeName]; ok {
		dests, err := router(ctx, state)
		if err != nil {
			return nil, err
		}
		return resolveDestinations(dests)
	}
	if edges, ok := g.edges[nodeName]; ok && len(edges) > 0 {
		return resolveDestinations(To(edges...))
	}
	return nil, fmt.Errorf("graph: node %q has no outgoing edge (add AddEdge/AddConditionalEdges, or return a *agentruntime.Command with Goto)", nodeName)
}

func (g *CompiledGraph) runNode(ctx context.Context, t task, state map[string]any, resumeQueue []any) (result any, interrupted *agentruntime.Interrupt, err error) {
	fn, ok := g.nodes[t.node]
	if !ok {
		return nil, nil, fmt.Errorf("graph: unknown node %q", t.node)
	}

	input := state
	if t.arg != nil {
		input = t.arg
	}

	ist := &taskInterruptState{resumeQueue: resumeQueue, nodeName: t.node}
	nodeCtx := context.WithValue(ctx, interruptCtxKey{}, ist)

	defer func() {
		if r := recover(); r != nil {
			if gi, ok := r.(*agentruntime.GraphInterrupt); ok {
				interrupted = &gi.Interrupt
				result = nil
				err = nil
				return
			}
			panic(r)
		}
	}()

	result, err = fn(nodeCtx, input)
	return
}

type interruptCtxKey struct{}

type taskInterruptState struct {
	resumeQueue []any
	idx         int
	counter     int
	nodeName    string
}

// Interrupt pauses the current node's execution, matching Python's
// `langgraph.types.interrupt(value)`. On first call within a (non-resumed)
// node invocation, it panics with a *agentruntime.GraphInterrupt, which
// CompiledGraph.Invoke recovers, converting it into a paused Result;
// callers should not recover this panic themselves. When re-invoked while
// resuming (via Options.Resume), it instead returns the corresponding
// resume value, in call order, matching Python's documented behavior that a
// resumed node re-executes from the start with successive interrupt() calls
// consuming queued resume values in order.
//
// Interrupt must be called from within a NodeFunc invoked by
// CompiledGraph.Invoke/InvokeWithOptions; calling it otherwise panics with a
// plain error.
func Interrupt(ctx context.Context, value any) any {
	st, ok := ctx.Value(interruptCtxKey{}).(*taskInterruptState)
	if !ok {
		panic("agentruntime: Interrupt called outside of a graph node execution")
	}
	if st.idx < len(st.resumeQueue) {
		v := st.resumeQueue[st.idx]
		st.idx++
		st.counter++
		return v
	}
	st.counter++
	panic(&agentruntime.GraphInterrupt{Interrupt: agentruntime.Interrupt{
		Value: value,
		ID:    fmt.Sprintf("%s-%d", st.nodeName, st.counter),
	}})
}

func resumeValuesFor(pending []agentruntime.Interrupt, resume any) []any {
	if len(pending) == 0 {
		return nil
	}
	if byID, ok := resume.(map[string]any); ok {
		values := make([]any, len(pending))
		for i, p := range pending {
			values[i] = byID[p.ID]
		}
		return values
	}
	values := make([]any, len(pending))
	values[0] = resume
	return values
}

func normalizeNodeResult(result any) (map[string]any, *agentruntime.Command, error) {
	switch v := result.(type) {
	case nil:
		return nil, nil, nil
	case map[string]any:
		return v, nil, nil
	case *agentruntime.Command:
		if v.Graph != "" {
			return nil, nil, fmt.Errorf("graph: Command.Graph %q is not supported (subgraphs are out of scope for this port)", v.Graph)
		}
		return v.Update, v, nil
	default:
		return nil, nil, fmt.Errorf("graph: node returned unsupported type %T (want map[string]any or *agentruntime.Command)", result)
	}
}

func resolveDestinations(raw []any) ([]task, error) {
	tasks := make([]task, 0, len(raw))
	for _, d := range raw {
		switch v := d.(type) {
		case string:
			tasks = append(tasks, task{node: v})
		case *agentruntime.Send:
			tasks = append(tasks, task{node: v.Node, arg: v.Arg})
		case agentruntime.Send:
			tasks = append(tasks, task{node: v.Node, arg: v.Arg})
		default:
			return nil, fmt.Errorf("graph: unsupported routing destination %T (want string or *agentruntime.Send)", d)
		}
	}
	return tasks, nil
}

func mergeState(state map[string]any, update map[string]any, reducers map[string]channels.Reducer) error {
	for k, v := range update {
		reducer := reducers[k]
		if reducer == nil {
			reducer = channels.LastValueReducer
		}
		merged, err := reducer(state[k], v)
		if err != nil {
			return fmt.Errorf("graph: reducer for key %q: %w", k, err)
		}
		state[k] = merged
	}
	return nil
}

func cloneState(state map[string]any) map[string]any {
	out := make(map[string]any, len(state))
	for k, v := range state {
		out[k] = v
	}
	return out
}
