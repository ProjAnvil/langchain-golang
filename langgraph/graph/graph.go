// Package graph implements a deliberately scoped Go port of Python's
// `langgraph.graph.StateGraph` builder plus a synchronous, in-process
// Pregel-style executor (see `langgraph.pregel`), sufficient to run the fixed
// "model node <-> tools node" shape `langchain`'s v1 agent factory needs.
// It is the public home of this runtime; `langchain/internal/agentruntime`
// is now a thin alias layer over the `langgraph` packages.
//
// Scope note: this is not a general distributed graph execution engine.
// Compared to Python's langgraph:
//
//   - No typed/schema-validated state: state is a plain map[string]any at the
//     API surface; internally each key is held by a channels.Channel object
//     (LastValue by default, BinaryOperator for reducer-registered keys, or
//     an explicit prototype via StateGraph.AddChannel), standing in for
//     Python's `Annotated[T, reducer]` state schema fields.
//   - Subgraphs are supported via StateGraph.AddSubgraph: a compiled graph
//     runs as a node of a parent graph, sharing state through shared keys,
//     and a node inside it may return Command{Graph: types.ParentGraph} to
//     have the parent apply the update and routing (see ParentCommandError).
//     Any other non-empty Command.Graph value remains a runtime error.
//   - No langgraph "stream modes" (values/updates/debug), caching, or retry
//     policies. A minimal event-ified execution path (InvokeStream + the
//     NodeEventSink in events.go) IS supported, so CreateAgent's StreamEvents
//     can observe node/model/tool lifecycle; this is not a general
//     streaming-modes surface.
//   - Checkpointing (via the checkpoint package) keeps full versioned
//     history per thread: one "input" checkpoint per turn plus one "loop"
//     checkpoint per committed superstep, each carrying channel values,
//     channel versions, versions-seen bookkeeping, and the tasks planned for
//     the next superstep. Checkpoints are retained after a run completes, so
//     the state inspection APIs (GetState/GetStateHistory/UpdateState, see
//     snapshot.go) and time travel (Options.CheckpointID) work on any
//     recorded position.
//   - Concurrent execution only happens *within* a superstep (multiple nodes
//     active at once via Send or multi-destination edges); node functions
//     must treat the state map they receive as read-only and communicate
//     changes only through their return value, since it may be read
//     concurrently by sibling tasks in the same superstep.
package graph

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"
	"sync/atomic"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// NodeFunc is a graph node, mirroring Python's node callables. It receives
// the current graph state and returns one of:
//
//   - nil: no state update.
//   - map[string]any: a partial state update, merged into state via each
//     key's channel (see StateGraph.AddReducer / StateGraph.AddChannel).
//   - *types.Command: a state update (Command.Update) plus an optional
//     routing override (Command.Goto) that bypasses the graph's static and
//     conditional edges for this task. Command.Graph selects the graph the
//     command applies to: empty means the current graph, and
//     types.ParentGraph targets the closest parent graph (only meaningful
//     inside a subgraph; see StateGraph.AddSubgraph and ParentCommandError).
//     Any other value is a runtime error.
//
// Any other return type is a runtime error. NodeFunc must not mutate the
// state map it receives (see the package doc comment).
type NodeFunc func(ctx context.Context, state map[string]any) (any, error)

// ConditionalEdge routes execution dynamically based on state, mirroring
// Python's `add_conditional_edges` router callables. Each returned element
// must be a string (a node name, or types.END) or a *types.Send.
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
	nodes         map[string]NodeFunc
	channelProtos map[string]channels.Channel
	edges         map[string][]string
	conditional   map[string]ConditionalEdge
	entry         string
	err           error
}

// NewStateGraph constructs an empty StateGraph builder.
func NewStateGraph() *StateGraph {
	return &StateGraph{
		nodes:         map[string]NodeFunc{},
		channelProtos: map[string]channels.Channel{},
		edges:         map[string][]string{},
		conditional:   map[string]ConditionalEdge{},
	}
}

func (g *StateGraph) setErr(err error) {
	if g.err == nil {
		g.err = err
	}
}

// AddNode registers a node. Names must be unique, non-empty, and distinct
// from types.START/types.END.
func (g *StateGraph) AddNode(name string, fn NodeFunc) *StateGraph {
	if name == "" || name == types.START || name == types.END {
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

// AddReducer registers the Reducer used to merge updates written to key,
// mirroring Python's `Annotated[T, reducer]` state schema fields: the key's
// channel becomes a channels.BinaryOperator over reducer, folding each
// superstep's writes left-to-right in deterministic task order. Keys without
// a registered reducer or channel default to channels.NewLastValue() — last
// write wins, and more than one write to the key in a single superstep is an
// *channels.InvalidUpdateError.
func (g *StateGraph) AddReducer(key string, reducer channels.Reducer) *StateGraph {
	g.channelProtos[key] = channels.NewBinaryOperator(reducer)
	return g
}

// AddChannel registers an explicit channel prototype for key, giving direct
// access to Python's channel semantics (e.g. channels.NewEphemeral or
// channels.NewTopic for values that expire at a superstep boundary). The
// prototype itself is never mutated; each run clones it via FromCheckpoint.
// Registration is last-call-wins: whichever of AddChannel/AddReducer is
// called last for a key determines that key's channel.
func (g *StateGraph) AddChannel(key string, prototype channels.Channel) *StateGraph {
	if prototype == nil {
		g.setErr(fmt.Errorf("graph: channel prototype for key %q must not be nil", key))
		return g
	}
	g.channelProtos[key] = prototype
	return g
}

// AddEdge adds a static (unconditional) edge from -> to. Passing
// types.START as from sets the graph's entry point, mirroring Python's
// `add_edge(START, node)`; only one entry point is supported (call
// SetEntryPoint or AddEdge(START, ...) exactly once).
func (g *StateGraph) AddEdge(from, to string) *StateGraph {
	if from == types.START {
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
	checkpointer    checkpoint.Saver
	recursionLimit  int
	interruptBefore map[string]bool
	interruptAfter  map[string]bool
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

// WithInterruptBefore registers node names that the graph must pause before
// dispatching, mirroring Python's `interrupt_before=` compile argument. When
// the exec loop is about to run a superstep containing a registered node, it
// checkpoints the current state (prior nodes' updates already merged) and
// returns a paused Result whose Interrupts name the to-run node. The run is
// resumable via the same Options.Resume / ThreadID path as an in-node
// `types.Interrupt` (Options.Resume may be nil — mirroring Python's
// `invoke(None, config)` — since there is no in-node interrupt to feed a value
// back to). A checkpointer (WithCheckpointer) is required for the pause to be
// resumable; without one, the boundary still surfaces as an interrupt Result
// but is not checkpointed. In supersteps with multiple active tasks the
// checkpoint plans the full task set, so resume re-dispatches every sibling.
func WithInterruptBefore(nodes ...string) CompileOption {
	return func(o *compileOptions) {
		if o.interruptBefore == nil {
			o.interruptBefore = map[string]bool{}
		}
		for _, n := range nodes {
			o.interruptBefore[n] = true
		}
	}
}

// WithInterruptAfter registers node names that the graph must pause after
// running (and merging their state update), before dispatching their
// successors, mirroring Python's `interrupt_after=` compile argument. The
// checkpoint plans the full successor task set as the resume point. See
// WithInterruptBefore's doc comment for the resume semantics and the
// checkpointer requirement.
func WithInterruptAfter(nodes ...string) CompileOption {
	return func(o *compileOptions) {
		if o.interruptAfter == nil {
			o.interruptAfter = map[string]bool{}
		}
		for _, n := range nodes {
			o.interruptAfter[n] = true
		}
	}
}

const defaultRecursionLimit = 100

// Compile validates the graph and returns an executable CompiledGraph,
// mirroring Python's `StateGraph.compile()`.
func (g *StateGraph) Compile(opts ...CompileOption) (*CompiledGraph, error) {
	if g.err != nil {
		return nil, g.err
	}
	if g.entry == "" {
		return nil, fmt.Errorf("graph: entry point not set (call AddEdge(types.START, node) or SetEntryPoint)")
	}
	if g.entry != types.END {
		if _, ok := g.nodes[g.entry]; !ok {
			return nil, fmt.Errorf("graph: entry point %q is not a registered node", g.entry)
		}
	}
	for from, tos := range g.edges {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("graph: edge source %q is not a registered node", from)
		}
		for _, to := range tos {
			if to != types.END {
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
		nodes:           g.nodes,
		channelProtos:   g.channelProtos,
		edges:           g.edges,
		conditional:     g.conditional,
		entry:           g.entry,
		checkpointer:    options.checkpointer,
		recursionLimit:  options.recursionLimit,
		interruptBefore: options.interruptBefore,
		interruptAfter:  options.interruptAfter,
	}, nil
}

// CompiledGraph is an executable graph, mirroring Python's
// `CompiledStateGraph`.
type CompiledGraph struct {
	nodes           map[string]NodeFunc
	channelProtos   map[string]channels.Channel
	edges           map[string][]string
	conditional     map[string]ConditionalEdge
	entry           string
	checkpointer    checkpoint.Saver
	recursionLimit  int
	interruptBefore map[string]bool
	interruptAfter  map[string]bool
}

// Options configures a single Invoke call.
type Options struct {
	// ThreadID identifies the conversation/run for checkpointing. Required
	// (together with a checkpointer installed via WithCheckpointer) to use
	// Resume, or for a node's Interrupt call to be resumable at all.
	ThreadID string
	// CheckpointID pins the checkpoint the run starts from, enabling time
	// travel: instead of the thread's latest checkpoint, the run loads the
	// pinned historical one and then follows the usual rules (Resume / nil
	// input resumes from it; fresh input starts a new turn on top of its
	// state, per D2). New checkpoints fork off the pinned one — their
	// ParentConfig points at it (D3). Requires ThreadID and a checkpointer,
	// and the pinned checkpoint must exist. When the pinned checkpoint's
	// Metadata.Parents names a subgraph's namespace, a re-entered subgraph
	// resumes from that recorded child checkpoint instead of its namespace's
	// latest (see StateGraph.AddSubgraph).
	CheckpointID string
	// Resume supplies the value(s) to resume a previously interrupted run
	// with, mirroring Python's `Command(resume=...)`. When set, input is
	// ignored and the run continues from the checkpointed state instead.
	//
	// Resume may also be left nil to resume a run paused by
	// interrupt_before/interrupt_after (WithInterruptBefore /
	// WithInterruptAfter): when a checkpoint already exists for ThreadID and
	// input is nil or empty, the run continues from that checkpoint with a
	// nil resume value. This mirrors Python's `invoke(None, config)` resume
	// semantic, and is the intended resume path for boundary interrupts
	// (which have no in-node Interrupt() call to feed a value back to). An
	// explicit non-nil Resume is still required to feed a value to a node's
	// in-node Interrupt.
	//
	// Conversely, invoking with fresh (non-empty) input on a thread that
	// already has a checkpoint starts a NEW turn rather than resuming: the
	// input is applied as a write batch on top of the latest checkpointed
	// state and execution restarts from the entry point (Python parity —
	// fresh input never silently resumes).
	//
	// Resume values are matched to pending interrupts by two rules,
	// mirroring Python:
	//
	//   - A map[string]any Resume addresses interrupts by ID. An interrupt
	//     whose ID is absent from the map is NOT fed a value: its Interrupt
	//     call re-pauses the run with the same interrupt, so partially
	//     addressed resumes pause again with the unmatched interrupts.
	//   - Any non-map (scalar) Resume value is fed to a single pending
	//     interrupt. When the checkpoint has more than one pending
	//     interrupt, a scalar Resume is an error — resume with an
	//     interrupt-ID map instead.
	Resume any

	// checkpointNS namespaces the run's checkpoints within the thread. It is
	// set only internally, by the StateGraph.AddSubgraph node wrapper, to run
	// a child graph under <parentNS>/<name> (see joinCheckpointNS); callers
	// invoking a graph directly always use the root namespace.
	checkpointNS string
}

// Result is returned by Invoke, mirroring the value/interrupt split of
// Python's `graph.invoke()` (`state, state["__interrupt__"]`).
type Result struct {
	// Values is the graph state: either the final state (Interrupts empty)
	// or the state as of the pause (Interrupts non-empty).
	Values map[string]any
	// Interrupts holds any interrupts raised in the step that paused
	// execution. Empty means the run completed normally.
	Interrupts []types.Interrupt
}

type task struct {
	// id is the task's planned ID (PlannedTask.ID) when the task comes from a
	// resumed checkpoint; resume value queues key on it (D5). Empty for tasks
	// derived from input or routing within the current run.
	id   string
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
	res, err := g.run(ctx, input, opts, nil)
	if err != nil {
		return Result{}, topLevelParentCommandError(err)
	}
	return res, nil
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
	res, err := g.run(ctx, input, opts, sink)
	if err != nil {
		return Result{}, topLevelParentCommandError(err)
	}
	return res, nil
}

// topLevelParentCommandError rewrites a *ParentCommandError surfacing from a
// public entry point — meaning a node of THIS graph returned
// Command{Graph: types.ParentGraph} but the graph has no parent to apply it
// to — into a descriptive error. The *ParentCommandError stays in the wrap
// chain so callers can still recover the command via errors.As. The
// AddSubgraph wrapper bypasses this by calling run directly.
func topLevelParentCommandError(err error) error {
	var pce *ParentCommandError
	if errors.As(err, &pce) {
		return fmt.Errorf("graph: Command targeting the parent graph surfaced from the top-level graph, which has no parent: %w", err)
	}
	return err
}

// checkpointSeq supplies the clock sequence for executor-minted checkpoint
// IDs so successive checkpoints remain chronologically ordered even when
// minted within NewID's millisecond timestamp resolution (e.g. a pause
// checkpoint right after the loop checkpoint of the same step).
var checkpointSeq atomic.Int64

// run is the shared execution loop backing both InvokeWithOptions (sink==nil,
// non-streaming, zero overhead) and InvokeStream (sink!=nil, event-ified). See
// each public entry point's doc comment for the contract.
func (g *CompiledGraph) run(ctx context.Context, input map[string]any, opts Options, sink NodeEventSink) (Result, error) {
	runCtx := ctx
	if sink != nil {
		runCtx = ContextWithEventSink(ctx, sink)
	}

	checkpointing := g.checkpointer != nil && opts.ThreadID != ""
	// parentSC links a subgraph run back to the checkpointing parent that
	// dispatched it (published via the context by the parent's run): this
	// run's checkpoints record Metadata.Parents[parentSC.ns] = the parent's
	// checkpoint position at the time the child ran.
	var parentSC subgraphCheckpoint
	isSubgraph := checkpointing && opts.checkpointNS != ""
	if isSubgraph {
		parentSC, _ = ctx.Value(subgraphCheckpointKey{}).(subgraphCheckpoint)
	}

	rs := newRunState(g.channelProtos)
	var tasks []task
	resumeValues := map[string][]any{}
	// resumingNode is the node name carried over from a checkpoint whose
	// interrupt_before check must be skipped on the first superstep of a
	// resume, so that resuming an interrupt_before(N) pause actually runs N
	// instead of immediately re-pausing. It is cleared after the first
	// superstep so subsequent arrivals at N (e.g. a loop) pause normally.
	resumingNode := ""

	var tup *checkpoint.Tuple
	if checkpointing {
		var err error
		tup, err = g.checkpointer.GetTuple(ctx, checkpoint.Config{ThreadID: opts.ThreadID, CheckpointNS: opts.checkpointNS, CheckpointID: opts.CheckpointID})
		if err != nil {
			return Result{}, fmt.Errorf("graph: loading checkpoint for thread %q: %w", opts.ThreadID, err)
		}
	}
	if opts.CheckpointID != "" {
		if g.checkpointer == nil {
			return Result{}, fmt.Errorf("graph: Options.CheckpointID requires a checkpointer (see WithCheckpointer)")
		}
		if opts.ThreadID == "" {
			return Result{}, fmt.Errorf("graph: Options.CheckpointID requires ThreadID")
		}
		if tup == nil {
			return Result{}, fmt.Errorf("graph: no checkpoint %q found for thread %q", opts.CheckpointID, opts.ThreadID)
		}
	}

	// currentCfg tracks the executor's current checkpoint position (D3): every
	// save passes its CheckpointID to Put so the new checkpoint's ParentConfig
	// points at its actual predecessor. Resumed/restored runs start from the
	// loaded checkpoint's position. It is shared with child runs via the
	// published subgraphCheckpoint — read by subgraph nodes during a
	// superstep, advanced only between supersteps — so child checkpoints can
	// record the parent's position in Metadata.Parents.
	currentCfg := &checkpoint.Config{}
	if tup != nil {
		*currentCfg = tup.Config
	}

	// children accumulates the latest checkpoint of each child namespace used
	// during the run (recorded by the AddSubgraph wrapper), so this run's
	// checkpoints name each child's position in Metadata.Parents — the link a
	// later time-travel re-entry pins the child to.
	children := &childCheckpoints{}
	if checkpointing {
		// Publish this run's checkpoint identity so subgraph nodes (see
		// StateGraph.AddSubgraph) checkpoint into the same thread under
		// <parentNS>/<name>; nested runs shadow it with their own namespace.
		// parents carries the loaded checkpoint's Metadata.Parents so a
		// re-entered subgraph resumes from the position recorded at the
		// pinned/resumed parent checkpoint instead of its namespace's latest.
		var parents map[string]string
		if tup != nil {
			parents = tup.Metadata.Parents
		}
		runCtx = context.WithValue(runCtx, subgraphCheckpointKey{}, subgraphCheckpoint{
			saver:    g.checkpointer,
			threadID: opts.ThreadID,
			ns:       opts.checkpointNS,
			parents:  parents,
			current:  currentCfg,
			children: children,
		})
	}

	// save persists a checkpoint (see saveCheckpoint) and advances currentCfg
	// to it, stamping Metadata.Parents with the child namespaces used so far
	// and — for a subgraph run — the parent's current position.
	save := func(md checkpoint.Metadata, next []checkpoint.PlannedTask) error {
		md.Parents = children.snapshot()
		if isSubgraph && parentSC.current != nil && parentSC.current.CheckpointID != "" {
			if md.Parents == nil {
				md.Parents = map[string]string{}
			}
			md.Parents[parentSC.ns] = parentSC.current.CheckpointID
		}
		cfg, err := g.saveCheckpoint(ctx, opts, rs, *currentCfg, md, next)
		if err != nil {
			return err
		}
		*currentCfg = cfg
		return nil
	}

	// Run mode selection. An explicit Options.Resume always resumes (the
	// in-node Interrupt path), requiring a checkpointer + checkpoint. Nil or
	// empty input with an existing checkpoint resumes too, mirroring Python's
	// `invoke(None, config)` semantic used by interrupt_before/interrupt_after
	// (no value to feed back). Fresh (non-empty) input with an existing
	// checkpoint starts a NEW turn instead (D2): the input applies on top of
	// the latest state and execution restarts from the entry point.
	var err error
	switch {
	case opts.Resume != nil:
		if g.checkpointer == nil {
			return Result{}, fmt.Errorf("graph: Options.Resume requires a checkpointer (see WithCheckpointer)")
		}
		if opts.ThreadID == "" {
			return Result{}, fmt.Errorf("graph: Options.Resume requires ThreadID")
		}
		if tup == nil {
			return Result{}, fmt.Errorf("graph: no checkpoint found for thread %q", opts.ThreadID)
		}
		tasks, resumeValues, resumingNode, err = resumeFromTuple(rs, tup, opts.Resume)
		if err != nil {
			return Result{}, err
		}
	case tup != nil && len(input) == 0:
		tasks, resumeValues, resumingNode, err = resumeFromTuple(rs, tup, nil)
		if err != nil {
			return Result{}, err
		}
	case tup != nil:
		// New turn (D2): restore the latest state, apply the input as a
		// write batch, and start over from the entry point. The step counter
		// continues from the restored checkpoint.
		rs.restore(tup.Checkpoint)
		rs.step = tup.Metadata.Step
		if err := rs.applyWrites([]taskWrites{{node: inputNodeName, update: input}}); err != nil {
			return Result{}, err
		}
		if err := save(checkpoint.Metadata{Source: "input", Step: -1}, nil); err != nil {
			return Result{}, err
		}
		tasks = []task{{node: g.entry}}
	default:
		// Fresh start: the input is the first write batch.
		if err := rs.applyWrites([]taskWrites{{node: inputNodeName, update: input}}); err != nil {
			return Result{}, err
		}
		if checkpointing {
			if err := save(checkpoint.Metadata{Source: "input", Step: -1}, nil); err != nil {
				return Result{}, err
			}
		}
		tasks = []task{{node: g.entry}}
	}

	steps := 0
	for {
		active := make([]task, 0, len(tasks))
		for _, t := range tasks {
			if t.node != types.END {
				active = append(active, t)
			}
		}
		if len(active) == 0 {
			break
		}

		// interrupt_before: if any active task's node is registered, pause
		// before dispatching the superstep. resumingNode excludes the node
		// being re-dispatched as part of a resume from a prior
		// interrupt_before(N) pause (otherwise resuming would immediately
		// re-pause). The checkpoint plans the superstep's full active task
		// set so resume re-dispatches every sibling; the superstep's writes
		// are not committed, so the checkpoint keeps the pre-superstep
		// channel state and step.
		if pausedBefore := g.findInterruptBefore(active, resumingNode); pausedBefore != "" {
			interrupt := types.Interrupt{
				Value: fmt.Sprintf("interrupt_before: %s", pausedBefore),
				ID:    interruptBeforeID + pausedBefore,
			}
			if checkpointing {
				if err := save(checkpoint.Metadata{Source: "loop", Step: rs.step},
					plannedTasks(active)); err != nil {
					return Result{}, err
				}
				if err := persistInterrupts(ctx, g.checkpointer, *currentCfg, pausedBefore, []types.Interrupt{interrupt}); err != nil {
					return Result{}, err
				}
			}
			return Result{Values: rs.snapshot(), Interrupts: []types.Interrupt{interrupt}}, nil
		}
		resumingNode = "" // resume-only skip applies solely to the first superstep

		steps++
		if steps > g.recursionLimit {
			return Result{}, fmt.Errorf("graph: recursion limit (%d) exceeded", g.recursionLimit)
		}

		type outcome struct {
			update      map[string]any
			cmd         *types.Command
			interrupted *types.Interrupt
			err         error
		}
		outcomes := make([]outcome, len(active))

		state := rs.snapshot()
		var wg sync.WaitGroup
		for i, t := range active {
			wg.Add(1)
			go func(i int, t task) {
				defer wg.Done()
				if sink != nil {
					sink.EmitRawEvent(RawEvent{Kind: RawNodeStart, Node: t.node})
				}
				result, interrupted, err := g.runNode(runCtx, t, state, resumeValues[t.id])
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

		var interrupts []types.Interrupt
		for _, o := range outcomes {
			if o.interrupted != nil {
				interrupts = append(interrupts, *o.interrupted)
			}
		}
		if len(interrupts) > 0 {
			// In-node interrupt: the superstep is not committed. The pause
			// checkpoint keeps the pre-superstep channel state and step and
			// plans the superstep's full active task set. Completed sibling
			// tasks persist their writes (state updates + D4-normalized goto
			// Sends) so resume replays them instead of re-running the tasks;
			// interrupted tasks persist their ReservedInterrupt writes. All
			// pending writes are keyed by the task's planned ID (D5).
			if checkpointing {
				next := plannedTasks(active)
				if err := save(checkpoint.Metadata{Source: "loop", Step: rs.step}, next); err != nil {
					return Result{}, err
				}
				for i, o := range outcomes {
					taskID := next[i].ID
					if o.interrupted != nil {
						if err := persistInterrupts(ctx, g.checkpointer, *currentCfg, taskID, []types.Interrupt{*o.interrupted}); err != nil {
							return Result{}, err
						}
						continue
					}
					writes, err := completedTaskWrites(o.update, o.cmd)
					if err != nil {
						return Result{}, err
					}
					if len(writes) > 0 {
						if err := g.checkpointer.PutWrites(ctx, *currentCfg, writes, taskID); err != nil {
							return Result{}, fmt.Errorf("graph: persisting completed task writes for thread %q: %w", opts.ThreadID, err)
						}
					}
				}
			}
			return Result{Values: state, Interrupts: interrupts}, nil
		}

		// Commit the superstep: one write batch per task, in deterministic
		// active-task slice order.
		writes := make([]taskWrites, len(active))
		for i, t := range active {
			writes[i] = taskWrites{node: t.node, update: outcomes[i].update}
		}
		if err := rs.applyWrites(writes); err != nil {
			return Result{}, err
		}
		rs.step++

		merged := rs.snapshot()
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
			dests, err := g.staticNext(ctx, t.node, merged)
			if err != nil {
				return Result{}, err
			}
			nextTasks = append(nextTasks, dests...)
		}

		// interrupt_after: if any node that just ran is registered, pause
		// before dispatching its successors. The checkpoint plans the full
		// successor task set as Next so resume continues from there; the
		// already-merged state update is preserved in Values (resume does not
		// re-run the paused-from node). If there is no successor, Next is
		// empty and resume is a no-op completion.
		if pausedAfter := g.findInterruptAfter(active); pausedAfter != "" {
			planned := plannedTasks(nextTasks)
			interrupt := types.Interrupt{
				Value: fmt.Sprintf("interrupt_after: %s", pausedAfter),
				ID:    interruptAfterID + pausedAfter,
			}
			if checkpointing {
				if err := save(checkpoint.Metadata{Source: "loop", Step: rs.step}, planned); err != nil {
					return Result{}, err
				}
				if err := persistInterrupts(ctx, g.checkpointer, *currentCfg, pausedAfter, []types.Interrupt{interrupt}); err != nil {
					return Result{}, err
				}
			}
			return Result{Values: merged, Interrupts: []types.Interrupt{interrupt}}, nil
		}

		if checkpointing {
			if err := save(checkpoint.Metadata{Source: "loop", Step: rs.step}, plannedTasks(nextTasks)); err != nil {
				return Result{}, err
			}
		}
		tasks = nextTasks
	}

	// D1: checkpoints survive completion — the final loop checkpoint (with an
	// empty Next) stays in the thread's history.
	return Result{Values: rs.snapshot()}, nil
}

// plannedTasks converts resolved next-step tasks into their checkpoint
// representation, dropping END destinations.
func plannedTasks(tasks []task) []checkpoint.PlannedTask {
	var out []checkpoint.PlannedTask
	for _, t := range tasks {
		if t.node == types.END {
			continue
		}
		out = append(out, checkpoint.PlannedTask{Node: t.node, Arg: t.arg})
	}
	return out
}

// saveCheckpoint persists rs as a new checkpoint for opts.ThreadID with the
// given metadata and planned next tasks, returning the new checkpoint's
// Config. parent is the executor's current checkpoint position: its
// CheckpointID is passed to Put so the new checkpoint's ParentConfig links to
// its actual predecessor (D3). Planned task IDs bind to the new checkpoint's
// ID and the superstep the tasks will run in (md.Step + 1).
func (g *CompiledGraph) saveCheckpoint(ctx context.Context, opts Options, rs *runState, parent checkpoint.Config, md checkpoint.Metadata, next []checkpoint.PlannedTask) (checkpoint.Config, error) {
	cp := checkpoint.Checkpoint{
		V:               1,
		ID:              checkpoint.NewID(int(checkpointSeq.Add(1))),
		TS:              time.Now(),
		ChannelValues:   rs.channelValues(),
		ChannelVersions: maps.Clone(rs.versions),
		VersionsSeen:    cloneSeen(rs.seen),
	}
	for i := range next {
		next[i].ID = TaskID(cp.ID, md.Step+1, next[i].Node, next[i].Arg)
	}
	cp.Next = next
	cfg, err := g.checkpointer.Put(ctx, checkpoint.Config{ThreadID: opts.ThreadID, CheckpointNS: opts.checkpointNS, CheckpointID: parent.CheckpointID}, cp, md, nil)
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("graph: saving checkpoint for thread %q: %w", opts.ThreadID, err)
	}
	return cfg, nil
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
	return nil, fmt.Errorf("graph: node %q has no outgoing edge (add AddEdge/AddConditionalEdges, or return a *types.Command with Goto)", nodeName)
}

func (g *CompiledGraph) runNode(ctx context.Context, t task, state map[string]any, resumeQueue []any) (result any, interrupted *types.Interrupt, err error) {
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
			if gi, ok := r.(*types.GraphInterrupt); ok {
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
// node invocation, it panics with a *types.GraphInterrupt, which
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
		panic("graph: Interrupt called outside of a graph node execution")
	}
	if st.idx < len(st.resumeQueue) {
		v := st.resumeQueue[st.idx]
		st.idx++
		st.counter++
		return v
	}
	st.counter++
	panic(&types.GraphInterrupt{Interrupt: types.Interrupt{
		Value: value,
		ID:    fmt.Sprintf("%s-%d", st.nodeName, st.counter),
	}})
}

// interruptBeforeID / interruptAfterID prefix the IDs of boundary interrupts
// so the resume path can recognize a checkpoint produced by interrupt_before
// (and thus must skip that node's interrupt_before check on the first
// superstep — see resumingNode) versus interrupt_after or an in-node
// types.Interrupt (whose IDs are "<node>-<counter>", never matching
// these prefixes).
const (
	interruptBeforeID = "interrupt-before-"
	interruptAfterID  = "interrupt-after-"
)

// findInterruptBefore returns the first active task's node that is registered
// in the graph's interrupt_before set, skipping skipNode (used to avoid
// re-pausing on the node being resumed from a prior interrupt_before pause).
// Returns "" if no active task matches.
func (g *CompiledGraph) findInterruptBefore(active []task, skipNode string) string {
	if len(g.interruptBefore) == 0 {
		return ""
	}
	for _, t := range active {
		if t.node == skipNode {
			continue
		}
		if g.interruptBefore[t.node] {
			return t.node
		}
	}
	return ""
}

// findInterruptAfter returns the first active task's node that is registered
// in the graph's interrupt_after set, or "" if none match.
func (g *CompiledGraph) findInterruptAfter(active []task) string {
	if len(g.interruptAfter) == 0 {
		return ""
	}
	for _, t := range active {
		if g.interruptAfter[t.node] {
			return t.node
		}
	}
	return ""
}

func normalizeNodeResult(result any) (map[string]any, *types.Command, error) {
	switch v := result.(type) {
	case nil:
		return nil, nil, nil
	case map[string]any:
		return v, nil, nil
	case *types.Command:
		if v.Graph == types.ParentGraph {
			// D6: a parent-targeted command aborts the run; the closest
			// AddSubgraph wrapper recovers it (see ParentCommandError).
			return nil, nil, &ParentCommandError{Command: v}
		}
		if v.Graph != "" {
			return nil, nil, fmt.Errorf("graph: Command.Graph %q is not supported (only %q, targeting the parent graph)", v.Graph, types.ParentGraph)
		}
		return v.Update, v, nil
	default:
		return nil, nil, fmt.Errorf("graph: node returned unsupported type %T (want map[string]any or *types.Command)", result)
	}
}

func resolveDestinations(raw []any) ([]task, error) {
	sends, err := gotoSends(raw)
	if err != nil {
		return nil, err
	}
	tasks := make([]task, 0, len(sends))
	for _, s := range sends {
		tasks = append(tasks, task{node: s.Node, arg: s.Arg})
	}
	return tasks, nil
}
