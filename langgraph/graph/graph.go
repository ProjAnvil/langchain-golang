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
//   - Stream modes (values/updates/debug/messages/custom) ARE supported via
//     CompiledGraph.Stream (see stream.go), a Go iterator over an emission
//     layer in the run loop.
//     It coexists with the older event-ified path (InvokeStream + the
//     NodeEventSink in events.go), which CreateAgent's StreamEvents uses to
//     observe node/model/tool lifecycle; neither replaces the other.
//   - Multi-parent waiting edges (StateGraph.AddJoinEdge) are supported via
//     a barrier channel per edge (Python's NamedBarrierValue): the child
//     triggers exactly once after all parents commit; plain edges, Sends,
//     and Command.Goto into the child bypass the barrier (Python's OR
//     semantics). defer=True / NamedBarrierValueAfterFinish are NOT
//     supported (the edge-driven loop has no finish broadcast).
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
	"slices"
	"strings"
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
	policies      map[string]NodePolicies
	channelProtos map[string]channels.Channel
	edges         map[string][]string
	conditional   map[string]ConditionalEdge
	// joinEdges holds the waiting edges registered via AddJoinEdge, in
	// registration order (Compile turns each into a joinMeta + barrier
	// channel prototype).
	joinEdges []joinEdge
	entry     string
	err       error
}

// joinEdge is one AddJoinEdge waiting edge: the child triggers once ALL
// parents have committed.
type joinEdge struct {
	parents []string
	child   string
}

// joinMeta is a compiled waiting edge (see StateGraph.AddJoinEdge).
type joinMeta struct {
	key     string   // barrier channel key ("join:a+b:c")
	parents []string // parent node names, in AddJoinEdge order
	child   string   // node dispatched once the barrier fills
}

// joinKey is the barrier channel key for a waiting edge, mirroring Python's
// `join:a+b:c` naming (langgraph/graph/state.py:1547): parent names in
// AddJoinEdge order, joined with "+".
func joinKey(parents []string, child string) string {
	return "join:" + strings.Join(parents, "+") + ":" + child
}

// NewStateGraph constructs an empty StateGraph builder.
func NewStateGraph() *StateGraph {
	return &StateGraph{
		nodes:         map[string]NodeFunc{},
		policies:      map[string]NodePolicies{},
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
// from types.START/types.END. It is AddNodeWithPolicies with zero policies:
// the node is never retried and never cached.
func (g *StateGraph) AddNode(name string, fn NodeFunc) *StateGraph {
	return g.AddNodeWithPolicies(name, fn, NodePolicies{})
}

// AddNodeWithPolicies registers a node together with its per-node execution
// policies (see NodePolicies). Policies are validated at Compile time.
//
// There is deliberately no graph-level default retry (Python has
// `retry_policy=` on compile; per-node policies suffice — documented
// divergence, YAGNI).
func (g *StateGraph) AddNodeWithPolicies(name string, fn NodeFunc, policies NodePolicies) *StateGraph {
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
	g.policies[name] = policies
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

// AddJoinEdge adds a waiting edge: the child node is triggered exactly once
// after ALL parents have committed (Python's `add_edge((a, b), c)`, backed by
// a `NamedBarrierValue` channel). The parents' arrivals accumulate in a
// barrier channel named `join:a+b:c` registered at Compile time; the barrier
// is control-plane state and never appears in node inputs, snapshots, or
// stream output.
//
// WARNING (Python parity, OR semantics): a plain edge, conditional edge,
// types.Send, or Command.Goto targeting the join child BYPASSES the barrier
// and triggers the child directly. Mixing both edge kinds into one child can
// run it multiple times — that is Python's documented behavior, not a bug.
//
// Documented divergences from Python (state.py:956-966): Go requires at
// least 2 parents (Python accepts a single-element tuple as a degenerate
// waiting edge), rejects duplicate parents (Python silently set-dedups), and
// rejects types.END as the child (Python allows `add_edge((a, b), END)`).
// Node-existence is validated at Compile time, consistent with AddEdge.
func (g *StateGraph) AddJoinEdge(from []string, to string) *StateGraph {
	if len(from) < 2 {
		g.setErr(fmt.Errorf("graph: join edge into %q requires at least 2 parents, got %d", to, len(from)))
		return g
	}
	seen := make(map[string]bool, len(from))
	for _, name := range from {
		if name == "" || name == types.START || name == types.END {
			g.setErr(fmt.Errorf("graph: invalid join parent name %q", name))
			return g
		}
		if seen[name] {
			g.setErr(fmt.Errorf("graph: duplicate join parent %q", name))
			return g
		}
		seen[name] = true
	}
	if to == "" || to == types.START || to == types.END {
		g.setErr(fmt.Errorf("graph: invalid join child name %q", to))
		return g
	}
	g.joinEdges = append(g.joinEdges, joinEdge{parents: slices.Clone(from), child: to})
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
	cache           checkpoint.Cache
	recursionLimit  int
	interruptBefore map[string]bool
	interruptAfter  map[string]bool
}

// WithCheckpointer installs a checkpoint.Saver, enabling Interrupt/Resume
// support (mirrors passing `checkpointer=` to Python's `.compile()`).
func WithCheckpointer(saver checkpoint.Saver) CompileOption {
	return func(o *compileOptions) { o.checkpointer = saver }
}

// WithCache installs a checkpoint.Cache backend, enabling per-node
// CachePolicy write caching (see StateGraph.AddNodeWithPolicies and
// CachePolicy). Nodes with a cache policy but no installed backend execute
// uncached, exactly as if they had no policy.
func WithCache(c checkpoint.Cache) CompileOption {
	return func(o *compileOptions) { o.cache = c }
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
	for name, policies := range g.policies {
		if policies.Retry != nil {
			if err := policies.Retry.validate(); err != nil {
				return nil, fmt.Errorf("graph: node %q retry policy: %w", name, err)
			}
		}
	}
	for _, je := range g.joinEdges {
		for _, p := range je.parents {
			if _, ok := g.nodes[p]; !ok {
				return nil, fmt.Errorf("graph: join edge parent %q is not a registered node", p)
			}
		}
		if _, ok := g.nodes[je.child]; !ok {
			return nil, fmt.Errorf("graph: join edge child %q is not a registered node", je.child)
		}
	}

	options := compileOptions{recursionLimit: defaultRecursionLimit}
	for _, opt := range opts {
		opt(&options)
	}

	// Register one barrier channel prototype per waiting edge (Python's
	// attach_edge, state.py:1546-1561). The clone keeps the builder's own
	// channelProtos untouched so Compile stays re-entrant.
	channelProtos := maps.Clone(g.channelProtos)
	joins := make([]joinMeta, 0, len(g.joinEdges))
	joinsByParent := map[string][]string{}
	for _, je := range g.joinEdges {
		key := joinKey(je.parents, je.child)
		if _, exists := channelProtos[key]; exists {
			return nil, fmt.Errorf("graph: duplicate join channel %q (identical AddJoinEdge calls, or a user-registered AddChannel/AddReducer collision)", key)
		}
		channelProtos[key] = channels.NewBarrier(je.parents...)
		joins = append(joins, joinMeta{key: key, parents: je.parents, child: je.child})
		for _, p := range je.parents {
			joinsByParent[p] = append(joinsByParent[p], key)
		}
	}

	return &CompiledGraph{
		nodes:           g.nodes,
		policies:        g.policies,
		channelProtos:   channelProtos,
		edges:           g.edges,
		conditional:     g.conditional,
		joins:           joins,
		joinsByParent:   joinsByParent,
		entry:           g.entry,
		checkpointer:    options.checkpointer,
		cache:           options.cache,
		recursionLimit:  options.recursionLimit,
		interruptBefore: options.interruptBefore,
		interruptAfter:  options.interruptAfter,
	}, nil
}

// CompiledGraph is an executable graph, mirroring Python's
// `CompiledStateGraph`.
type CompiledGraph struct {
	nodes         map[string]NodeFunc
	policies      map[string]NodePolicies
	channelProtos map[string]channels.Channel
	edges         map[string][]string
	conditional   map[string]ConditionalEdge
	// joins/joinsByParent are the compiled waiting edges (empty for graphs
	// without AddJoinEdge): the executor appends an implicit barrier write to
	// each parent task's commit batch and dispatches join children from the
	// commit path (see run).
	joins           []joinMeta
	joinsByParent   map[string][]string
	entry           string
	checkpointer    checkpoint.Saver
	cache           checkpoint.Cache
	recursionLimit  int
	interruptBefore map[string]bool
	interruptAfter  map[string]bool
}

// ClearCache removes every cached entry in namespace ns, delegating to the
// cache backend installed via WithCache. The executor namespaces each node's
// entries as "writes/<node>". It is a no-op returning nil when no cache
// backend is installed.
func (g *CompiledGraph) ClearCache(ctx context.Context, ns string) error {
	if g.cache == nil {
		return nil
	}
	return g.cache.Clear(ctx, ns)
}

// cacheWritesNS is the cache namespace holding nodeName's cached task writes.
func cacheWritesNS(nodeName string) string {
	return "writes/" + nodeName
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
	// Resuming an in-node interrupt with a nil Resume (or nil/empty input)
	// does NOT answer it with nil: the interrupted node's resume queue is
	// left empty, so its Interrupt() call re-fires and the run pauses
	// again with the same interrupt (Python parity). Boundary interrupts
	// are unaffected — they never consult resume queues, so they resume
	// with nil exactly as before.
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
//
// Stream emission: when a *streamEmitter is installed in ctx (by Stream, or
// by a parent run's invokeSubgraph), the loop calls its hooks at the input
// batch, task dispatch/collection, superstep commit, pause, and checkpoint
// save points, and observes context cancellation at superstep boundaries. A
// nil emitter makes every hook a no-op, so Invoke/InvokeStream keep their
// exact prior behavior.
func (g *CompiledGraph) run(ctx context.Context, input map[string]any, opts Options, sink NodeEventSink) (Result, error) {
	em := emitterFromContext(ctx)
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
	// and — for a subgraph run — the parent's current position. A successful
	// save emits the stream layer's debug checkpoint chunk.
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
		em.debugCheckpoint(md, cfg, *currentCfg, g.dropJoinKeys(rs.channelValues()), next)
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
	var replayWrites []taskWrites
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
		tasks, resumeValues, resumingNode, replayWrites, err = resumeFromTuple(rs, tup, opts.Resume)
		if err != nil {
			return Result{}, err
		}
	case tup != nil && len(input) == 0:
		tasks, resumeValues, resumingNode, replayWrites, err = resumeFromTuple(rs, tup, nil)
		if err != nil {
			return Result{}, err
		}
	case tup != nil:
		// New turn (D2): restore the latest state, apply the input as a
		// write batch, and start over from the entry point. The step counter
		// continues from the restored checkpoint.
		rs.restore(tup.Checkpoint)
		rs.step = tup.Metadata.Step
		changed, err := rs.applyWrites([]taskWrites{{node: inputNodeName, update: input}})
		if err != nil {
			return Result{}, err
		}
		if changed {
			em.emitValues(rs.snapshot())
		}
		// S6: the input checkpoint continues the step counter from the
		// restored checkpoint (only a thread's FIRST input checkpoint is -1).
		if err := save(checkpoint.Metadata{Source: "input", Step: tup.Metadata.Step}, nil); err != nil {
			return Result{}, err
		}
		tasks = []task{{node: g.entry}}
	default:
		// Fresh start: the input is the first write batch.
		changed, err := rs.applyWrites([]taskWrites{{node: inputNodeName, update: input}})
		if err != nil {
			return Result{}, err
		}
		if changed {
			em.emitValues(rs.snapshot())
		}
		if checkpointing {
			if err := save(checkpoint.Metadata{Source: "input", Step: -1}, nil); err != nil {
				return Result{}, err
			}
		}
		tasks = []task{{node: g.entry}}
	}
	// Cached/replayed writes are re-streamed as updates on resume (Python
	// parity, `_loop.py:676-679`).
	for _, w := range replayWrites {
		em.emitUpdate(w.node, g.dropJoinKeys(w.update))
	}

	steps := 0
	for {
		// A streaming run (emitter installed) observes cancellation — an
		// early iterator break — at the superstep boundary, so the run
		// goroutine always terminates. Non-streaming runs are unaffected.
		if em != nil {
			if err := ctx.Err(); err != nil {
				return Result{}, err
			}
		}
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
			em.emitPause(rs.snapshot(), []types.Interrupt{interrupt})
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
			consumed    []any
			err         error
		}
		outcomes := make([]outcome, len(active))

		state := rs.snapshot()
		// Debug task events fire at dispatch, in deterministic task order —
		// including for cache-hit tasks, whose node never executes (Python
		// emits task events for prepared tasks; documented parity).
		for _, t := range active {
			em.debugTask(rs.step+1, t, state)
		}

		// Cache lookup pass (per-node CachePolicy, see WithCache), synchronous
		// and in deterministic task order, BEFORE any task dispatches — and
		// therefore before its RawNodeStart event. A hit injects the stored
		// writes as the task's outcome: the node never executes and no
		// RawNodeStart/End pair is emitted (Python parity: execution is
		// skipped, so no node events fire). Tasks resuming with a pending
		// interrupt skip the lookup — they carry a (possibly empty) resume
		// queue entry in resumeValues, see planResume — because a cache entry
		// populated by a DIFFERENT run for the same input must not skip the
		// node and silently drop the resume value. Tasks replayed from pending
		// writes never reach dispatch at all, so they bypass the cache
		// automatically.
		//
		// Because the whole lookup pass precedes the store pass, two Sends
		// with the same arg to one cached node in a single superstep both
		// execute (neither sees the other's entry yet) — Python parity.
		execute := make([]bool, len(active))
		missed := make([]bool, len(active)) // cache miss: store writes after execution
		storeKey := make([]string, len(active))
		for i, t := range active {
			policy, ok := g.policies[t.node]
			if g.cache == nil || !ok || policy.Cache == nil {
				execute[i] = true
				continue
			}
			if _, resuming := resumeValues[t.id]; resuming {
				execute[i] = true
				continue
			}
			input := state
			if t.arg != nil {
				input = t.arg
			}
			keyFunc := policy.Cache.KeyFunc
			if keyFunc == nil {
				keyFunc = DefaultCacheKey
			}
			key, err := keyFunc(input)
			if err != nil {
				outcomes[i].err = fmt.Errorf("graph: node %q cache key: %w", t.node, err)
				continue
			}
			writes, hit, err := g.cache.Get(ctx, cacheWritesNS(t.node), key)
			if err != nil {
				outcomes[i].err = fmt.Errorf("graph: node %q cache get: %w", t.node, err)
				continue
			}
			if hit {
				outcomes[i].update, outcomes[i].cmd = outcomeFromCachedWrites(writes)
				continue
			}
			execute[i] = true
			missed[i] = true
			storeKey[i] = key
		}

		var wg sync.WaitGroup
		for i, t := range active {
			if !execute[i] {
				continue
			}
			wg.Add(1)
			go func(i int, t task) {
				defer wg.Done()
				if sink != nil {
					sink.EmitRawEvent(RawEvent{Kind: RawNodeStart, Node: t.node})
				}
				update, cmd, interrupted, consumed, err := g.runTask(em.nodeContext(runCtx, t.node, rs.step+1), t, state, resumeValues[t.id])
				if sink != nil {
					// Always emit node_end so start/end pairs are balanced per
					// invocation, even on the error/interrupt paths. The pair
					// wraps the WHOLE attempt loop: retried tasks emit exactly
					// one start/end pair regardless of attempt count.
					sink.EmitRawEvent(RawEvent{Kind: RawNodeEnd, Node: t.node})
				}
				outcomes[i] = outcome{update: update, cmd: cmd, interrupted: interrupted, consumed: consumed, err: err}
			}(i, t)
		}
		wg.Wait()
		resumeValues = nil

		// Cache store pass: tasks that missed and then completed (no error,
		// no interrupt) persist their writes — via completedTaskWrites, the
		// same serializer the resume path uses — so later runs with the same
		// input replay them. Errored and interrupted tasks store nothing, and
		// neither do RESUMED tasks (missed stays false for them). That last
		// part is load-bearing correctness, not an accident: writes produced
		// from a human's resume value cached under the pre-interrupt input key
		// would poison later fresh runs with that same input.
		for i, o := range outcomes {
			if !missed[i] || o.err != nil || o.interrupted != nil {
				continue
			}
			writes, err := completedTaskWrites(o.update, o.cmd)
			if err != nil {
				return Result{}, fmt.Errorf("graph: node %q cache writes: %w", active[i].node, err)
			}
			if err := g.cache.Set(ctx, cacheWritesNS(active[i].node), storeKey[i], writes, g.policies[active[i].node].Cache.TTL); err != nil {
				return Result{}, fmt.Errorf("graph: node %q cache set: %w", active[i].node, err)
			}
		}

		// Join barrier arrivals: each completed parent task implicitly
		// writes its own name to every waiting-edge barrier it feeds
		// (Python attaches a ChannelWrite per parent at compile time,
		// state.py:1558-1561). Injecting into the task's update batch
		// (rather than a side channel) gives the write the same
		// applyWrites version bump AND the interrupt path's
		// completedTaskWrites persistence below — the "parent A arrived,
		// parent B interrupted, resume replays A's arrival" closure.
		// Interrupted/errored tasks write nothing (Python: the parent's
		// ChannelWrite never executes). Cache entries were stored BEFORE
		// this pass, so they stay free of control-plane keys; a cache-hit
		// parent still records its arrival here.
		for i, t := range active {
			if len(g.joinsByParent[t.node]) == 0 || outcomes[i].err != nil || outcomes[i].interrupted != nil {
				continue
			}
			if outcomes[i].update == nil {
				outcomes[i].update = map[string]any{}
			}
			for _, key := range g.joinsByParent[t.node] {
				outcomes[i].update[key] = t.node
			}
		}

		// Collect outcomes in deterministic task order: debug task_result per
		// task at completion, then the task's updates chunk. (Divergence from
		// Python's as-they-finish timing: Go applies writes only after all
		// tasks of the superstep complete, so updates bunch here.)
		var interrupts []types.Interrupt
		for i, o := range outcomes {
			var taskInterrupts []types.Interrupt
			if o.interrupted != nil {
				taskInterrupts = []types.Interrupt{*o.interrupted}
				interrupts = append(interrupts, *o.interrupted)
			}
			pub := g.dropJoinKeys(o.update)
			em.debugTaskResult(rs.step+1, active[i], pub, o.err, taskInterrupts)
			em.emitUpdate(active[i].node, pub)
		}
		for _, o := range outcomes {
			if o.err != nil {
				return Result{}, o.err
			}
		}
		if len(interrupts) > 0 {
			// In-node interrupt: the superstep is not committed. The pause
			// checkpoint keeps the pre-superstep channel state and step and
			// plans the superstep's full active task set. Completed sibling
			// tasks persist their writes (state updates + D4-normalized goto
			// Sends) so resume replays them instead of re-running the tasks;
			// interrupted tasks persist their ReservedInterrupt writes plus
			// one ReservedResume write carrying the ordered prefix of resume
			// values they already consumed (see persistInterruptAndResume),
			// so the next resume rebuilds their full ordered resume queue. All
			// pending writes are keyed by the task's planned ID (D5).
			if checkpointing {
				next := plannedTasks(active)
				if err := save(checkpoint.Metadata{Source: "loop", Step: rs.step}, next); err != nil {
					return Result{}, err
				}
				for i, o := range outcomes {
					taskID := next[i].ID
					if o.interrupted != nil {
						if err := persistInterruptAndResume(ctx, g.checkpointer, *currentCfg, taskID, []types.Interrupt{*o.interrupted}, o.consumed); err != nil {
							return Result{}, err
						}
						continue
					}
					writes, err := completedTaskWrites(o.update, o.cmd)
					if err != nil {
						return Result{}, err
					}
					if len(writes) > 0 {
						if err := g.checkpointer.PutWrites(ctx, *currentCfg, writes, taskID, ""); err != nil {
							return Result{}, fmt.Errorf("graph: persisting completed task writes for thread %q: %w", opts.ThreadID, err)
						}
					}
				}
			}
			em.emitPause(state, interrupts)
			return Result{Values: state, Interrupts: interrupts}, nil
		}

		// Commit the superstep: one write batch per task, in deterministic
		// active-task slice order.
		writes := make([]taskWrites, len(active))
		for i, t := range active {
			writes[i] = taskWrites{node: t.node, update: outcomes[i].update}
		}
		changed, err := rs.applyWrites(writes)
		if err != nil {
			return Result{}, err
		}
		rs.step++

		// Join children that CONSUMED their barrier this superstep reset it
		// (Python calls NamedBarrierValue.consume only for channels the
		// task was actually triggered by). The seen>=versions check is
		// exactly that: a barrier-dispatched child carries the barrier's
		// current version in versions-seen (applyWrites re-records the
		// pre-write view at commit), while a child that ran via a Send/
		// plain edge in the superstep that FILLED the barrier holds an
		// older mark — its barrier stays armed so the trigger scan below
		// still dispatches the barrier task (OR semantics;
		// TestJoinSendBypassesBarrier locks this in). Consume itself is a
		// no-op on a non-full barrier. Must run BEFORE the trigger scan,
		// or a consumed barrier would re-dispatch its child. Documented
		// divergence: Python consumes read channels at the apply_writes
		// head, before applying the superstep's writes
		// (pregel/_algo.py:285-292); Go consumes after applyWrites, which
		// differs only when a join child co-runs with its own parent in
		// one superstep (the parent's idempotent re-arrival is consumed
		// away here, kept as a new partial arrival in Python).
		// Documented divergence: Python's consume path also bumps the
		// barrier channel's version to next_version on success
		// (pregel/_algo.py:289-290); Go's Barrier.Consume only clears
		// the seen set and leaves the version untouched. No behavioral
		// difference: a re-trigger requires a fresh parent arrival,
		// which itself bumps the version.
		if len(g.joins) > 0 {
			ran := make(map[string]bool, len(active))
			for _, t := range active {
				ran[t.node] = true
			}
			for _, jm := range g.joins {
				if !ran[jm.child] {
					continue
				}
				if rs.seen[jm.child][jm.key] < rs.versions[jm.key] {
					continue // child ran via Send/edge, not via this barrier
				}
				if b, ok := rs.channels[jm.key].(interface{ Consume() bool }); ok {
					b.Consume()
				}
			}
		}

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

		// Waiting-edge triggers: a barrier filled by this superstep's
		// writes dispatches its child EXACTLY once. Versions-seen is the
		// dedup ledger: a barrier version the child already saw never
		// re-dispatches it (same-superstep multi-parent commits fold into
		// one channel update = one version bump = one dispatch). The mark
		// is recorded at dispatch time so a pause checkpoint's
		// VersionsSeen stays consistent with its planned Next. Send/
		// Command.Goto/normal-edge tasks for the same child are separate
		// entries in nextTasks on purpose (Python's OR semantics) — they
		// must NOT be deduped against this barrier task.
		for _, jm := range g.joins {
			ch, ok := rs.channels[jm.key]
			if !ok || !ch.IsAvailable() {
				continue
			}
			v := rs.versions[jm.key]
			if rs.seen[jm.child][jm.key] >= v {
				continue
			}
			if rs.seen[jm.child] == nil {
				rs.seen[jm.child] = map[string]int64{}
			}
			rs.seen[jm.child][jm.key] = v
			nextTasks = append(nextTasks, task{node: jm.child})
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
			em.emitPause(merged, []types.Interrupt{interrupt})
			return Result{Values: merged, Interrupts: []types.Interrupt{interrupt}}, nil
		}

		// values chunk for the committed superstep, gated on at least one
		// channel version having bumped (Python's updated-channels gate).
		if changed {
			em.emitValues(merged)
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

// dropJoinKeys strips join barrier channels from a task update or
// channel-value map destined for user-visible emission (updates chunks,
// debug task_result/checkpoint payloads). Internal paths — applyWrites,
// completedTaskWrites, checkpoint saves — always keep the full map. Returns
// the input unchanged when the graph has no join edges.
func (g *CompiledGraph) dropJoinKeys(m map[string]any) map[string]any {
	if len(g.joins) == 0 || len(m) == 0 {
		return m
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		if !isJoinKey(g.channelProtos, k) {
			out[k] = v
		}
	}
	return out
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
	if len(g.joinsByParent[nodeName]) > 0 {
		// Waiting-edge-only parent: its successors are dispatched by the
		// barrier trigger in the commit path, not per-parent edges.
		return nil, nil
	}
	return nil, fmt.Errorf("graph: node %q has no outgoing edge (add AddEdge/AddConditionalEdges, or return a *types.Command with Goto)", nodeName)
}

// runTask executes one task, wrapping runNode in the node's RetryPolicy
// attempt loop (installed via StateGraph.AddNodeWithPolicies). Nodes without
// a retry policy take exactly one attempt, preserving the pre-policy
// behavior.
//
// Semantics (Python parity, `pregel/_retry.py`):
//
//   - Interrupts are terminal: runNode's deferred recover converts a
//     GraphInterrupt panic into interrupted != nil BEFORE this loop sees
//     anything, so an interrupted task is never re-executed.
//   - Each attempt re-invokes the node from the start: runNode builds a
//     fresh taskInterruptState per call, so a retried resume re-feeds the
//     resume values from index 0.
//   - Node-internal emissions of a failed attempt (InvokeStream sink events,
//     messages/custom stream chunks) are NOT rolled back and therefore
//     duplicate across attempts. Outcome writes do NOT duplicate: outcomes
//     are buffered pre-commit, so a failed attempt leaves nothing to clear.
//   - Backoff sleeps select on ctx.Done(): parent cancellation aborts the
//     retry loop immediately and surfaces the PARENT's ctx error, not the
//     node's error.
//
// Events: the RawNodeStart/RawNodeEnd pair (in run's task wrapper) and the
// debug task_result emission bracket the whole attempt loop, so exactly one
// of each appears per task regardless of attempt count.
func (g *CompiledGraph) runTask(ctx context.Context, t task, state map[string]any, resumeQueue []any) (update map[string]any, cmd *types.Command, interrupted *types.Interrupt, consumed []any, err error) {
	var retry *RetryPolicy
	if policies, ok := g.policies[t.node]; ok && policies.Retry != nil {
		p := policies.Retry.withDefaults()
		retry = &p
	}
	for attempt := 1; ; attempt++ {
		result, intr, cons, rerr := g.runNode(ctx, t, state, resumeQueue)
		if intr != nil {
			return nil, nil, intr, cons, nil
		}
		if rerr == nil {
			update, cmd, nerr := normalizeNodeResult(result)
			return update, cmd, nil, nil, nerr
		}
		if retry == nil || attempt >= retry.MaxAttempts || !retry.RetryOn(rerr) {
			return nil, nil, nil, nil, rerr
		}
		timer := time.NewTimer(retry.backoff(attempt))
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, nil, nil, nil, ctx.Err()
		case <-timer.C:
		}
	}
}

// runNode runs one node invocation. On an interrupt it additionally reports
// consumed — a copy of the resume-queue prefix the invocation consumed
// before panicking (ist.resumeQueue[:ist.idx]; at the panic idx ==
// len(resumeQueue), i.e. the full ordered prefix consumed so far, including
// values carried from earlier pause/resume cycles) — so the pause path can
// persist it as a single full-list ReservedResume write. Retry/error paths discard it: an
// errored task produces no pause checkpoint, so the prefix has nowhere to
// land.
func (g *CompiledGraph) runNode(ctx context.Context, t task, state map[string]any, resumeQueue []any) (result any, interrupted *types.Interrupt, consumed []any, err error) {
	fn, ok := g.nodes[t.node]
	if !ok {
		return nil, nil, nil, fmt.Errorf("graph: unknown node %q", t.node)
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
				ist.mu.Lock()
				consumed = append([]any{}, ist.resumeQueue[:ist.idx]...)
				ist.mu.Unlock()
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

// taskInterruptState is the per-node-invocation interrupt bookkeeping. The
// mutex guards idx/counter: fn tasks call Interrupt from their own
// goroutines (task.go startTask), so the entrypoint body and in-flight task
// goroutines can touch the shared state concurrently (fn measures the
// consumption cursor via InterruptConsumeCount while a sibling task
// interrupts). resumeQueue is fixed at creation and read-only thereafter.
type taskInterruptState struct {
	mu          sync.Mutex
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
	st.mu.Lock()
	defer st.mu.Unlock()
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

// InterruptConsumeCount reports how many resume values the node invocation
// owning ctx has consumed via Interrupt so far. It returns 0 when ctx carries
// no interrupt state (outside a node execution). The fn package uses it to
// measure how many resume values a task's execution consumed, so the count
// can be persisted (checkpoint.ReservedFnConsumed) and honored on replay.
func InterruptConsumeCount(ctx context.Context) int {
	st, ok := ctx.Value(interruptCtxKey{}).(*taskInterruptState)
	if !ok {
		return 0
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.idx
}

// ReplayInterruptConsumption advances ctx's interrupt state as if n
// Interrupt calls had each consumed a queued resume value: the next Interrupt
// call skips the n values a replayed execution already consumed in an
// earlier run. The counter advances too, keeping generated interrupt IDs
// identical to a full re-execution. It is a no-op when ctx carries no
// interrupt state or n <= 0.
//
// This exists for checkpoint-replay layers (the fn package): a task whose
// persisted result is replayed does not re-execute, so its Interrupt calls
// never re-fire — without this advance, the next Interrupt call would be
// misaligned onto the resume value the replayed task already consumed.
func ReplayInterruptConsumption(ctx context.Context, n int) {
	st, ok := ctx.Value(interruptCtxKey{}).(*taskInterruptState)
	if !ok || n <= 0 {
		return
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	st.idx += n
	st.counter += n
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
