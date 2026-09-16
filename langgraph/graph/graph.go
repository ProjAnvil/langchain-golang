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
//     recorded position. The write-flush cadence (sync/async/exit) is chosen
//     at Compile time via WithDurability and can be overridden per run
//     (Options.WithRunDurability). Subgraph tasks checkpoint under their own
//     <parentNS>/<node>:<taskID> namespace (see StateGraph.AddSubgraph).
//   - Concurrent execution only happens *within* a superstep (multiple nodes
//     active at once via Send or multi-destination edges); node functions
//     must treat the state map they receive as read-only and communicate
//     changes only through their return value, since it may be read
//     concurrently by sibling tasks in the same superstep.
//   - Graph export (GetGraph/DrawMermaid, see export.go) is built from
//     registered metadata, plus best-effort probing of routers registered
//     without a path map (a single empty-state call; Python does a full
//     dry-run simulation). Subgraph nodes (AddSubgraph) draw as single nodes
//     by default and expand into ":"-prefixed inner graphs with
//     GetGraph(WithXrayDepth(...)), mirroring Python's xray. PNG rendering
//     is not supported.
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

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// NodeFunc is a graph node, mirroring Python's node callables. It receives a
// runtime.Runtime (which itself satisfies context.Context, so existing ctx
// idioms survive — pass rt wherever a context.Context is expected) and the
// current graph state, and returns one of:
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
//
// Migrating from the pre-v1 NodeFunc signature: the first parameter changed
// from `ctx context.Context` to `rt runtime.Runtime`. Since Runtime satisfies
// context.Context, every ctx usage inside a node body (ctx.Done(), ctx.Err(),
// ctx.Value(k), Interrupt(ctx, ...), and helpers that take a context.Context)
// still compiles unchanged when you rename the parameter from ctx to rt. The
// only change is the declared type of the first parameter.
type NodeFunc func(rt runtime.Runtime, state map[string]any) (any, error)

// ConditionalEdge routes execution dynamically based on state, mirroring
// Python's `add_conditional_edges` router callables. It receives a
// runtime.Runtime (which satisfies context.Context) and the current graph
// state; each returned element must be a string (a node name, or types.END)
// or a *types.Send.
type ConditionalEdge func(rt runtime.Runtime, state map[string]any) ([]any, error)

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
	// pathMaps holds the static path maps registered via
	// AddConditionalEdgesWithPathMap (keyed by source node); they make the
	// otherwise opaque routers enumerable for graph export (GetGraph) and
	// enable compile-time target validation.
	pathMaps map[string]map[string]string
	// joinEdges holds the waiting edges registered via AddJoinEdge, in
	// registration order (Compile turns each into a joinMeta + barrier
	// channel prototype).
	joinEdges []joinEdge
	entry     string
	// entryRouter, when set, is the conditional entry point registered via
	// SetConditionalEntryPoint; exactly one of entry/entryRouter is set.
	entryRouter ConditionalEdge
	// entryPathMap is the static path map registered via
	// SetConditionalEntryPointWithPathMap (see pathMaps).
	entryPathMap map[string]string
	// subgraphs records the compiled children registered via AddSubgraph,
	// keyed by node name, so GetGraph can expand them (WithXrayDepth).
	subgraphs map[string]*CompiledGraph
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
		pathMaps:      map[string]map[string]string{},
		subgraphs:     map[string]*CompiledGraph{},
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
// policies (see NodePolicies). Policies are validated at Compile time. A node
// without its own retry policy also inherits the graph-level default
// installed via WithDefaultRetryPolicy (if any).
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
	if policies.Timeout != nil {
		if err := policies.Timeout.validate(); err != nil {
			g.setErr(fmt.Errorf("graph: node %q timeout policy: %w", name, err))
			return g
		}
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
// Neither parents nor the child may be the reserved names types.START or
// types.END. defer=True / NamedBarrierValueAfterFinish are not supported
// (see the package documentation).
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
	if g.entryRouter != nil {
		g.setErr(fmt.Errorf("graph: conditional entry point already set"))
		return g
	}
	if g.entry != "" {
		g.setErr(fmt.Errorf("graph: entry point already set to %q", g.entry))
		return g
	}
	g.entry = name
	return g
}

// SetConditionalEntryPoint registers a dynamic router that picks the first
// node(s) from the post-input state, mirroring Python's
// `set_conditional_entry_point` (`add_conditional_edges(START, path)`,
// state.py:1079). The router returns node names, types.END (stop
// immediately), or *types.Send, exactly like a node router (see
// AddConditionalEdges). Python has no path_map counterpart here: Go routers
// return node names directly.
//
// Python implements this by ATTACHING a branch to START in the pregel wiring
// (attach_branch, state.py:1563); the Go executor resolves routers from a
// map at run time, so there is nothing to attach — SetConditionalEntryPoint
// is the whole mechanism.
func (g *StateGraph) SetConditionalEntryPoint(router ConditionalEdge) *StateGraph {
	if router == nil {
		g.setErr(fmt.Errorf("graph: conditional entry point router must not be nil"))
		return g
	}
	if g.entry != "" {
		g.setErr(fmt.Errorf("graph: entry point already set to %q", g.entry))
		return g
	}
	if g.entryRouter != nil {
		g.setErr(fmt.Errorf("graph: conditional entry point already set"))
		return g
	}
	g.entryRouter = router
	return g
}

// SetFinishPoint marks a node as a finish point of the graph — sugar for
// AddEdge(name, types.END), mirroring Python's `set_finish_point`
// (state.py:1103).
func (g *StateGraph) SetFinishPoint(name string) *StateGraph {
	return g.AddEdge(name, types.END)
}

// NamedNode pairs a node name with its NodeFunc for AddSequence (Go has no
// callable-name introspection, so the name is explicit, matching the named
// `(name, fn)` tuple form of Python's add_sequence).
type NamedNode struct {
	Name string
	Fn   NodeFunc
}

// AddSequence registers nodes to be executed in the given order: each node
// is added and chained to its predecessor with a static edge, mirroring
// Python's `add_sequence` (state.py:1019). An empty sequence is an error
// (Python: "Sequence requires at least one node."); duplicate names surface
// the existing duplicate-node error. Entry/finish wiring is left to the
// caller (SetEntryPoint/SetConditionalEntryPoint/SetFinishPoint), as in
// Python.
func (g *StateGraph) AddSequence(nodes ...NamedNode) *StateGraph {
	if len(nodes) == 0 {
		g.setErr(fmt.Errorf("graph: AddSequence requires at least one node"))
		return g
	}
	previous := ""
	for i, n := range nodes {
		g.AddNode(n.Name, n.Fn)
		if i > 0 {
			g.AddEdge(previous, n.Name)
		}
		previous = n.Name
	}
	return g
}

// CompileOption configures Compile.
type CompileOption func(*compileOptions)

type compileOptions struct {
	checkpointer    checkpoint.Saver
	cache           checkpoint.Cache
	store           store.Store
	recursionLimit  int
	interruptBefore map[string]bool
	interruptAfter  map[string]bool
	durability      Durability
	defaultRetry    *RetryPolicy
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

// WithStore installs a cross-thread store.Store (the langgraph BaseStore),
// mirroring Python's `StateGraph.compile(store=...)`. When non-nil, the store
// is surfaced on Runtime.Store for every node function and conditional edge
// router (see CompiledGraph.buildRuntime), enabling persistence and memory
// shared across threads. It is distinct from core/stores.BaseStore[V] (the
// generic typed KV) and from the checkpoint.Cache backend.
func WithStore(s store.Store) CompileOption {
	return func(o *compileOptions) { o.store = s }
}

// WithRecursionLimit overrides the default superstep limit (100), mirroring
// Python's `recursion_limit` config option. It guards against unintentional
// infinite loops in a graph's routing.
func WithRecursionLimit(limit int) CompileOption {
	return func(o *compileOptions) { o.recursionLimit = limit }
}

// WithDefaultRetryPolicy installs a graph-level default RetryPolicy applied
// to every node that does not carry its own retry policy
// (StateGraph.AddNodeWithPolicies), mirroring Python's graph-level default
// retry (`retry_policy=` on compile / `set_node_defaults`). A node's own
// policy always takes precedence; nodes with neither retry anywhere. Like
// Python's defaults, the policy is NOT inherited by subgraphs (each subgraph
// is compiled separately). Passing nil disables the default (and unsets a
// previous option in the same Compile call).
//
// The default is validated at Compile time; an invalid policy fails Compile
// with a "default retry policy" error.
func WithDefaultRetryPolicy(p *RetryPolicy) CompileOption {
	return func(o *compileOptions) { o.defaultRetry = p }
}

// Durability selects when the executor flushes checkpoint writes, mirroring
// Python's langgraph.types.Durability (types.py:87).
//
// Python resolves durability at RUN time (invoke/stream take a durability
// argument defaulting to "async"); this port resolves it at COMPILE time via
// WithDurability, defaulting to sync (the pre-existing Go behavior), with an
// equivalent per-run override available via Options.Durability /
// Options.WithRunDurability. A per-run value propagates into subgraph runs,
// mirroring Python's CONFIG_KEY_DURABILITY propagation (main.py:2862-2864).
type Durability string

const (
	// DurabilitySync persists changes synchronously before the next step
	// starts (the Go default).
	DurabilitySync Durability = "sync"
	// DurabilityAsync persists changes while the next step executes
	// (background goroutine, flushed before return). This is Python's
	// runtime default.
	DurabilityAsync Durability = "async"
	// DurabilityExit persists changes only when the graph exits
	// (deferred flush at end of invoke).
	DurabilityExit Durability = "exit"
)

// WithDurability sets the checkpoint write-flush mode (see Durability). The
// compiled value is the default for every run; a single run overrides it via
// Options.WithRunDurability.
func WithDurability(d Durability) CompileOption {
	return func(o *compileOptions) { o.durability = d }
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
	if g.entry == "" && g.entryRouter == nil {
		return nil, fmt.Errorf("graph: entry point not set (call AddEdge(types.START, node), SetEntryPoint, or SetConditionalEntryPoint)")
	}
	if g.entry != "" && g.entry != types.END {
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
	// Path maps are the statically enumerable form of the routers (used by
	// GetGraph), so their targets are validated like static edge targets.
	for from, pathMap := range g.pathMaps {
		for _, target := range pathMap {
			if target != types.END {
				if _, ok := g.nodes[target]; !ok {
					return nil, fmt.Errorf("graph: conditional edge %q path map target %q is not a registered node", from, target)
				}
			}
		}
	}
	for _, target := range g.entryPathMap {
		if target != types.END {
			if _, ok := g.nodes[target]; !ok {
				return nil, fmt.Errorf("graph: conditional entry point path map target %q is not a registered node", target)
			}
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

	options := compileOptions{recursionLimit: defaultRecursionLimit, durability: DurabilitySync}
	for _, opt := range opts {
		opt(&options)
	}
	var defaultRetry *RetryPolicy
	if options.defaultRetry != nil {
		if err := options.defaultRetry.validate(); err != nil {
			return nil, fmt.Errorf("graph: default retry policy: %w", err)
		}
		p := options.defaultRetry.withDefaults()
		defaultRetry = &p
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
		pathMaps:        g.pathMaps,
		joins:           joins,
		joinsByParent:   joinsByParent,
		entry:           g.entry,
		entryRouter:     g.entryRouter,
		entryPathMap:    g.entryPathMap,
		subgraphs:       g.subgraphs,
		checkpointer:    options.checkpointer,
		cache:           options.cache,
		store:           options.store,
		recursionLimit:  options.recursionLimit,
		durability:      options.durability,
		interruptBefore: options.interruptBefore,
		interruptAfter:  options.interruptAfter,
		defaultRetry:    defaultRetry,
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
	// pathMaps/entryPathMap are the static path maps registered on the
	// builder (AddConditionalEdgesWithPathMap /
	// SetConditionalEntryPointWithPathMap), kept for graph export (GetGraph).
	pathMaps     map[string]map[string]string
	entryPathMap map[string]string
	// joins/joinsByParent are the compiled waiting edges (empty for graphs
	// without AddJoinEdge): the executor appends an implicit barrier write to
	// each parent task's commit batch and dispatches join children from the
	// commit path (see run).
	joins         []joinMeta
	joinsByParent map[string][]string
	entry         string
	// entryRouter, when set, is the conditional entry point registered via
	// SetConditionalEntryPoint; exactly one of entry/entryRouter is set.
	entryRouter ConditionalEdge
	// subgraphs records the compiled children registered via
	// StateGraph.AddSubgraph, keyed by node name (copied by Compile), so
	// GetGraph can expand them (WithXrayDepth).
	subgraphs       map[string]*CompiledGraph
	checkpointer    checkpoint.Saver
	cache           checkpoint.Cache
	store           store.Store
	recursionLimit  int
	durability      Durability
	interruptBefore map[string]bool
	interruptAfter  map[string]bool
	// defaultRetry is the graph-level default retry policy installed via
	// WithDefaultRetryPolicy, defaults-resolved at Compile; nil when absent.
	// It applies only to nodes without their own retry policy (see runTask).
	defaultRetry *RetryPolicy
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
	//   - A map[string]any Resume addresses interrupts by NS first and by ID
	//     second (see types.Interrupt.NS; interrupts persisted before NS
	//     stamping match by ID only). An interrupt whose NS and ID are both
	//     absent from the map is NOT fed a value: its Interrupt() call
	//     re-pauses the run with the same interrupt, so partially addressed
	//     resumes pause again with the unmatched interrupts.
	//   - Any non-map (scalar) Resume value is fed to a single pending
	//     interrupt. When the checkpoint has more than one pending
	//     interrupt, a scalar Resume is an error — resume with an
	//     interrupt-NS/ID map instead (or narrow the matching with Graph
	//     below, which allows a scalar whenever exactly one pending
	//     interrupt lies within the named namespace).
	//
	// Interrupts raised inside a subgraph pause the parent run carrying
	// their full nested NS; a Resume for them is forwarded verbatim into the
	// child run (see StateGraph.AddSubgraph), so a scalar works for a single
	// pending subgraph interrupt at any nesting depth and a map keyed by NS
	// dispatches per interrupting task.
	Resume any

	// Graph optionally namespaces a Resume, mirroring the graph selector of
	// Python's Command(resume=..., graph=ns): when non-empty, the resume's
	// matching is restricted (and validated) to the pending interrupts whose
	// NS lies within the named checkpoint namespace — the namespace itself
	// (e.g. a subgraph task's "<node>:<taskID>") or anything nested under it
	// (e.g. an interrupt raised inside that subgraph, whose NS is
	// "<node>:<taskID>/<inner>:<taskID>"). A scalar Resume is then allowed
	// whenever exactly one pending interrupt hits, bypassing the usual
	// "multiple pending interrupts require a map" error. A non-empty Graph
	// matching no pending interrupt is a descriptive error listing the
	// available NS values.
	//
	// Requires Resume (Graph without Resume is an error), and
	// types.ParentGraph is rejected (a resume cannot climb out of the graph
	// it targets — Python errors the same way, _io.py:57-58). Graph is
	// propagated into subgraph runs together with Resume (see
	// StateGraph.AddSubgraph), so addressing a nested interrupt by any
	// ancestor namespace of its NS works at any depth.
	Graph string

	// forceResume marks Options built by the AddSubgraph wrapper for a child
	// run dispatched by a RESUMING parent: the child must enter the run loop's
	// resume branch (continuing from its own pinned pause checkpoint) instead
	// of treating the parent state as fresh input, mirroring Python's
	// CONFIG_KEY_RESUMING propagation (_loop.py:1037-1076). Internal only.
	forceResume bool

	// RecursionLimit overrides the compiled recursion limit for this single
	// invocation: > 0 takes precedence over the compile-time
	// WithRecursionLimit value; 0 (or negative) means use the compiled
	// default. Mirrors Python's runtime config {"recursion_limit": N}.
	// Propagated into subgraph runs (see StateGraph.AddSubgraph).
	RecursionLimit int

	// MaxConcurrency bounds how many nodes of one superstep run
	// concurrently. Mirrors Python Pregel's use of RunnableConfig
	// max_concurrency to cap its executor. 0 (or negative) means the
	// default bound (min(32, GOMAXPROCS+4), the same default as Python's
	// global executor).
	MaxConcurrency int

	// Durability overrides the compiled WithDurability mode for this single
	// run, mirroring Python's invoke/stream durability argument (Python
	// resolves durability at run time, defaulting to "async"; Go keeps the
	// compile-time value as the default — sync unless WithDurability says
	// otherwise — so leaving this empty changes nothing). The override
	// propagates into subgraph runs, mirroring Python's
	// CONFIG_KEY_DURABILITY propagation (main.py:2862-2864). See
	// Options.WithRunDurability.
	Durability Durability

	// checkpointNS namespaces the run's checkpoints within the thread. It is
	// set only internally, by the StateGraph.AddSubgraph node wrapper, to run
	// a child graph under <parentNS>/<name>:<taskID> (see
	// taskCheckpointNS); callers invoking a graph directly always use the
	// root namespace.
	checkpointNS string

	// pinNS names the namespace the pinned checkpoint (CheckpointID) is
	// loaded FROM when it differs from checkpointNS. Set only internally by
	// the AddSubgraph wrapper for legacy-namespace resume compatibility: a
	// pre-per-task-namespacing checkpoint records Metadata.Parents under the
	// node-only namespace, so the pinned child checkpoint must be read from
	// that namespace while the run itself writes into its per-task
	// namespace. Empty means load from checkpointNS.
	pinNS string
}

// WithRunDurability returns a copy of o with the per-run durability override
// set (see Options.Durability). It mirrors Python's invoke/stream durability
// argument, overriding the compiled WithDurability value for this one run and
// propagating into subgraph runs:
//
//	opts := graph.Options{ThreadID: "t"}.WithRunDurability(graph.DurabilityExit)
//	cg.InvokeWithOptions(ctx, input, opts)
//
// The zero Options keeps the compiled default; the Go default remains sync
// (unlike Python, whose runtime default is "async").
func (o Options) WithRunDurability(d Durability) Options {
	o.Durability = d
	return o
}

// effectiveDurability resolves the durability mode for one run: a per-run
// Options.Durability override wins over the compiled value. The empty string
// is the "use the compiled value" sentinel.
func (g *CompiledGraph) effectiveDurability(run Durability) Durability {
	if run != "" {
		return run
	}
	return g.durability
}

// validateDurability rejects unknown per-run durability values: an invalid
// mode would silently disable checkpoint persistence (every sink branch
// no-ops), so it must fail loudly instead.
func validateDurability(d Durability) error {
	switch d {
	case "", DurabilitySync, DurabilityAsync, DurabilityExit:
		return nil
	default:
		return fmt.Errorf("graph: unknown durability %q (want %q, %q, or %q)", d, DurabilitySync, DurabilityAsync, DurabilityExit)
	}
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
	// runErrorHandler marks a resumed task whose node failed with a persisted
	// ERROR write: the executor runs the node's error handler instead of the
	// node function (crash recovery — Python langgraph 1.2.0 re-executes the
	// handler, not the node; see planResume).
	runErrorHandler bool
	// resumeErrMsg carries the persisted error message for a runErrorHandler
	// task (the reconstructed NodeError's Err).
	resumeErrMsg string
}

// plannedID returns the task's deterministic planned identity: the resumed
// planned ID when the task came from a checkpoint's Next, else the TaskID
// recomputed against the planning checkpoint (the run loop's current
// checkpoint position) and the superstep the task runs in — the same formula
// saveCheckpoint stamps into Next and the commit path uses to key pending
// writes, so all three agree on one ID per task.
func (t task) plannedID(planning checkpoint.Config, step int) string {
	if t.id != "" {
		return t.id
	}
	return TaskID(planning.CheckpointID, step, t.node, t.arg)
}

// plannedTaskIDKey is the context key under which the run loop publishes the
// dispatched task's planned ID (task.plannedID), so the AddSubgraph node
// wrapper can namespace the child's checkpoints per task
// (<parentNS>/<node>:<taskID>, mirroring Python's task_checkpoint_ns,
// pregel/_algo.py:624).
type plannedTaskIDKey struct{}

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
	if _, ok := errors.AsType[*ParentCommandError](err); ok {
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
func (g *CompiledGraph) run(ctx context.Context, input map[string]any, opts Options, sink NodeEventSink) (result Result, err error) {
	em := emitterFromContext(ctx)
	runCtx := ctx
	if sink != nil {
		runCtx = ContextWithEventSink(ctx, sink)
	}

	// Per-run durability override (Python's invoke/stream durability
	// argument): an unknown value must fail loudly rather than silently
	// disabling persistence.
	if err := validateDurability(opts.Durability); err != nil {
		return Result{}, err
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
		// The pinned checkpoint is loaded from pinNS when set (legacy
		// subgraph-namespace resume compatibility, see Options.pinNS);
		// everything this run WRITES goes to opts.checkpointNS.
		loadNS := opts.checkpointNS
		if opts.pinNS != "" {
			loadNS = opts.pinNS
		}
		tup, err = g.checkpointer.GetTuple(ctx, checkpoint.Config{ThreadID: opts.ThreadID, CheckpointNS: loadNS, CheckpointID: opts.CheckpointID})
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

	// Publish the execution meta (current checkpoint position + run options)
	// so buildRuntime can populate Runtime.ExecutionInfo without threading
	// them through runTask/runNode signatures. Published for every run —
	// not just checkpointed ones — so ExecutionInfo is always available to
	// nodes (fields simply default to empty when there is no checkpointer).
	// The cfg pointer is shared with the run loop, which advances it between
	// supersteps; ExecutionInfo reads it at buildRuntime time, so it tracks
	// the latest save.
	//
	// resuming mirrors the mode-selection switch below (the two resume
	// branches): a resuming run's subgraph tasks forward the resume payload
	// into their child runs (invokeSubgraph). A forceResume whose namespace
	// holds no checkpoint degrades to a fresh-input run, which is NOT
	// resuming.
	resuming := false
	switch {
	case opts.Resume != nil || (opts.forceResume && tup != nil):
		resuming = true
	case opts.Resume == nil && !opts.forceResume && tup != nil && len(input) == 0:
		resuming = true
	}
	runCtx = context.WithValue(runCtx, executionMetaKey{}, executionMeta{
		cfg:      currentCfg,
		opts:     opts,
		resuming: resuming,
	})

	// cpSink dispatches checkpoint/per-task writes according to the
	// effective Durability mode (sync/async/exit). In sync mode it is a thin
	// wrapper around the saver. In async mode it uses a background goroutine.
	// In exit mode it accumulates writes and flushes at exit. The mode is
	// the per-run override when Options.Durability is set (Python's
	// invoke/stream durability argument), else the compiled WithDurability
	// value.
	var cpSink *checkpointSink
	if checkpointing {
		cpSink = newCheckpointSink(g.checkpointer, g.effectiveDurability(opts.Durability), runCtx, tup)
		defer func() {
			cpSink.setFlushContext(ctx, opts, rs, *currentCfg, checkpoint.Metadata{Source: "loop", Step: rs.step})
			// Surface flush errors via the named return. Only assign when the
			// invoke itself succeeded (err == nil): a flush failure must not
			// mask the actual run error. Using flushErr avoids shadowing the
			// named err inside this closure.
			if flushErr := cpSink.flush(); flushErr != nil && err == nil {
				err = flushErr
			}
		}()
	} else {
		cpSink = newCheckpointSink(g.checkpointer, DurabilitySync, runCtx, nil)
	}

	// save persists a checkpoint (see saveCheckpoint) and advances currentCfg
	// to it, stamping Metadata.Parents with the child namespaces used so far
	// and — for a subgraph run — the parent's current position. A successful
	// save emits the stream layer's debug checkpoint chunk and, when active,
	// the checkpoints chunk (a StateSnapshot). savePause is the pause-checkpoint
	// variant: it routes the write through cpSink.putPauseCheckpoint, which
	// guarantees durability before the run returns under every Durability mode
	// (see checkpointSink.putPauseCheckpoint).
	persist := func(pause bool, md checkpoint.Metadata, next []checkpoint.PlannedTask) error {
		md.Parents = children.snapshot()
		if isSubgraph && parentSC.current != nil && parentSC.current.CheckpointID != "" {
			if md.Parents == nil {
				md.Parents = map[string]string{}
			}
			md.Parents[parentSC.ns] = parentSC.current.CheckpointID
		}
		cfg, err := g.saveCheckpoint(ctx, cpSink, opts, rs, *currentCfg, md, next, pause)
		if err != nil {
			return err
		}
		em.debugCheckpoint(md, cfg, *currentCfg, g.dropJoinKeys(rs.channelValues()), next)
		em.emitCheckpointSnapshot(md, cfg, *currentCfg, rs.snapshot(), next)
		*currentCfg = cfg
		return nil
	}
	save := func(md checkpoint.Metadata, next []checkpoint.PlannedTask) error {
		return persist(false, md, next)
	}
	savePause := func(md checkpoint.Metadata, next []checkpoint.PlannedTask) error {
		return persist(true, md, next)
	}

	// Run mode selection. An explicit Options.Resume always resumes (the
	// in-node Interrupt path), requiring a checkpointer + checkpoint. Nil or
	// empty input with an existing checkpoint resumes too, mirroring Python's
	// `invoke(None, config)` semantic used by interrupt_before/interrupt_after
	// (no value to feed back). Fresh (non-empty) input with an existing
	// checkpoint starts a NEW turn instead (D2): the input applies on top of
	// the latest state and execution restarts from the entry point.
	//
	// opts.forceResume (set by the AddSubgraph wrapper when the parent run is
	// resuming) routes the child through the resume branch as well; when the
	// child's namespace holds no checkpoint yet — a subgraph task dispatched
	// for the first time by a resuming parent — it degrades to the fresh-input
	// branches below instead of erroring, so the child simply runs.
	//
	// Options.Graph names the checkpoint namespace a Resume is addressed to
	// (see its doc comment): validate it up front — without a Resume it is a
	// caller error, and types.ParentGraph cannot be resumed into (Python
	// rejects the same, _io.py:57-58).
	if opts.Graph != "" {
		if opts.Resume == nil {
			return Result{}, fmt.Errorf("graph: Options.Graph requires Options.Resume")
		}
		if opts.Graph == types.ParentGraph {
			return Result{}, fmt.Errorf("graph: Options.Graph must not be %q (a resume applies within the graph it targets; to resume the parent of a subgraph, resume the parent thread)", types.ParentGraph)
		}
	}
	var replayWrites []taskWrites
	forceDegraded := opts.forceResume && tup == nil
	switch {
	case (opts.Resume != nil || opts.forceResume) && !forceDegraded:
		if g.checkpointer == nil {
			return Result{}, fmt.Errorf("graph: Options.Resume requires a checkpointer (see WithCheckpointer)")
		}
		if opts.ThreadID == "" {
			return Result{}, fmt.Errorf("graph: Options.Resume requires ThreadID")
		}
		if tup == nil {
			return Result{}, fmt.Errorf("graph: no checkpoint found for thread %q", opts.ThreadID)
		}
		tasks, resumeValues, resumingNode, replayWrites, err = g.resumeFromTuple(rs, tup, opts.Resume, opts.Graph)
		if err != nil {
			return Result{}, err
		}
	case tup != nil && len(input) == 0:
		tasks, resumeValues, resumingNode, replayWrites, err = g.resumeFromTuple(rs, tup, nil, "")
		if err != nil {
			return Result{}, err
		}
	case tup != nil:
		// New turn (D2): restore the latest state, apply the input as a
		// write batch, and start over from the entry point. The step counter
		// continues from the restored checkpoint.
		rs.restore(tup.Checkpoint)
		rs.step = tup.Metadata.Step
		// Seed deltaCounters from the loaded checkpoint so resume continues
		// the per-channel cadence (S3). Cloned so rs.deltaCounters is
		// independent of the (shared) loaded metadata map.
		rs.deltaCounters = maps.Clone(tup.Metadata.CountersSinceDeltaSnapshot)
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
		// M2.2: persist delta-channel input writes AFTER the input checkpoint
		// save (so they anchor on it) for ancestor-walk reconstruction.
		if err := g.persistDeltaInputWrites(ctx, cpSink, *currentCfg, input); err != nil {
			return Result{}, fmt.Errorf("graph: persisting delta input writes for thread %q: %w", opts.ThreadID, err)
		}
		entryTasks, err := g.entryTasks(runCtx, rs.snapshot())
		if err != nil {
			return Result{}, fmt.Errorf("graph: conditional entry point: %w", err)
		}
		tasks = entryTasks
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
			// M2.2: persist delta-channel input writes AFTER the input checkpoint
			// save (so they anchor on it) for ancestor-walk reconstruction.
			if err := g.persistDeltaInputWrites(ctx, cpSink, *currentCfg, input); err != nil {
				return Result{}, fmt.Errorf("graph: persisting delta input writes for thread %q: %w", opts.ThreadID, err)
			}
		}
		entryTasks, err := g.entryTasks(runCtx, rs.snapshot())
		if err != nil {
			return Result{}, fmt.Errorf("graph: conditional entry point: %w", err)
		}
		tasks = entryTasks
	}
	// Cached/replayed writes are re-streamed as updates on resume (Python
	// parity, `_loop.py:676-679`).
	for _, w := range replayWrites {
		em.emitUpdate(w.node, g.dropJoinKeys(w.update))
	}

	// Effective recursion limit for this invocation: a positive
	// Options.RecursionLimit overrides the compiled WithRecursionLimit value,
	// mirroring Python's runtime config {"recursion_limit": N}.
	recursionLimit := g.recursionLimit
	if opts.RecursionLimit > 0 {
		recursionLimit = opts.RecursionLimit
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
				NS:    taskCheckpointNS(opts.checkpointNS, pausedBefore, ""),
			}
			if checkpointing {
				next := plannedTasks(active)
				// Pre-stamp the would-be dispatch identities (see
				// stampDispatchIDs): a later resume re-dispatches these
				// tasks with the checkpoint's IDs, keeping the identities
				// a non-paused run would have computed.
				stampDispatchIDs(next, active, *currentCfg, rs.step+1)
				if err := savePause(checkpoint.Metadata{Source: "loop", Step: rs.step},
					next); err != nil {
					return Result{}, err
				}
				if err := cpSink.putPauseWrites(ctx, *currentCfg, interruptWrites([]types.Interrupt{interrupt}), pausedBefore); err != nil {
					return Result{}, err
				}
			}
			em.emitPause(rs.snapshot(), []types.Interrupt{interrupt})
			return Result{Values: rs.snapshot(), Interrupts: []types.Interrupt{interrupt}}, nil
		}
		resumingNode = "" // resume-only skip applies solely to the first superstep

		steps++
		if steps > recursionLimit {
			// Best-effort diagnostics: name the pending node when exactly
			// one task is about to dispatch; leave Node empty when ambiguous.
			node := ""
			if len(active) == 1 {
				node = active[0].node
			}
			return Result{}, &types.GraphRecursionError{Limit: recursionLimit, Node: node}
		}

		type outcome struct {
			update     map[string]any
			cmd        *types.Command
			interrupts []types.Interrupt
			consumed   []any
			err        error
			// nodeErr is non-nil when this task's failure went through the
			// node's error handler (a fresh *NodeError failure, or a resumed
			// handler task): it drives the pause/abort persistence of the
			// ReservedError write below.
			nodeErr *NodeError
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
		storeNS := make([]string, len(active))
		storeKey := make([]string, len(active))
		storeTTL := make([]time.Duration, len(active))
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
			if t.runErrorHandler {
				// A resumed handler task must re-run the handler, not replay
				// some other run's cached writes for the same input.
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
			// The KeyFunc may pin a namespace and a per-entry TTL; empty
			// namespace falls back to "writes/<node>", zero TTL to the policy
			// TTL.
			ns := cacheWritesNS(t.node)
			if len(key.Namespace) > 0 {
				ns = strings.Join(key.Namespace, "/")
			}
			ttl := policy.Cache.TTL
			if key.TTL != 0 {
				ttl = key.TTL
			}
			writes, hit, err := g.cache.Get(ctx, ns, key.Key)
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
			storeNS[i] = ns
			storeKey[i] = key.Key
			storeTTL[i] = ttl
		}

		// Bounded fan-out: at most opts.MaxConcurrency (default
		// defaultSuperstepParallelism) nodes execute concurrently, matching
		// Python Pregel's executor cap. Tasks always run to completion —
		// cancellation surfaces through the task bodies, not by skipping.
		runSuperstepBounded(active, execute, superstepBound(opts.MaxConcurrency, len(active)), func(i int, t task) {
			if sink != nil {
				sink.EmitRawEvent(RawEvent{Kind: RawNodeStart, Node: t.node})
			}
			// Publish the task's planned ID on the task context (per-task
			// subgraph checkpoint namespacing; see plannedTaskIDKey).
			taskCtx := context.WithValue(em.nodeContext(runCtx, t.node, rs.step+1),
				plannedTaskIDKey{}, t.plannedID(*currentCfg, rs.step+1))
			// Per-node TracePolicy (langgraph 1.2.11 trace_policy=): wrap the
			// node context's callback manager so every event emitted within
			// this node — from the node itself or runnables it invokes —
			// carries transformed payloads before any tracer observes them.
			// An empty manager (no handlers installed, e.g. the inert carrier
			// subgraphs run under) is left untouched: events are dropped
			// there anyway, and wrapping would flip its Empty() signal.
			if tp := g.tracePolicy(t.node); tp != nil {
				if parent, ok := callbacks.ManagerFromContext(taskCtx); ok && !parent.Empty() {
					taskCtx = callbacks.ContextWithManager(taskCtx,
						callbacks.NewPayloadPolicyManager(parent, tp.ProcessInputs, tp.ProcessOutputs))
				}
			}
			var nodeErr *NodeError
			var update map[string]any
			var cmd *types.Command
			var interrupts []types.Interrupt
			var consumed []any
			var err error
			if t.runErrorHandler {
				// Resumed handler task (crash recovery): run the handler with
				// the persisted message instead of the node function —
				// Python langgraph 1.2.0 re-executes the handler, not the
				// node. Attempt is 0: the persisted write carries the message
				// only. planResume only builds these tasks for nodes that
				// still have a handler; the nil check is defensive.
				ne := &NodeError{Node: t.node, Err: errors.New(t.resumeErrMsg)}
				nodeErr = ne
				if hp := g.errorHandlerPolicy(t.node); hp != nil {
					result, herr := hp.Handler(state, ne)
					update, cmd, err = normalizeErrorHandlerResult(result, herr, ne)
				} else {
					err = fmt.Errorf("graph: task %q carries a persisted error write but the compiled graph has no error handler for the node", t.node)
				}
			} else {
				update, cmd, interrupts, consumed, err = g.runTask(taskCtx, t, state, resumeValues[t.id])
				if ne, ok := errors.AsType[*NodeError](err); ok {
					// The node's retries are exhausted and a handler is
					// installed. Persist the task's ERROR write BEFORE
					// running the handler, so a crash mid-handler resumes
					// into a handler re-run instead of a node re-run
					// (mirrors Python's committed task error write; the
					// ReservedError slot exists in every saver). Under exit
					// durability nothing persists mid-run — the pause/abort
					// barriers below persist it when the run actually stops.
					nodeErr = ne
					runHandler := true
					if checkpointing && cpSink.mode != DurabilityExit {
						writes := []checkpoint.Write{{Channel: checkpoint.ReservedError, Value: ne.Err.Error()}}
						if werr := cpSink.putWrites(ctx, *currentCfg, writes, t.plannedID(*currentCfg, rs.step+1)); werr != nil {
							runHandler = false
							err = fmt.Errorf("graph: persisting error write for thread %q: %w", opts.ThreadID, werr)
						}
					}
					if runHandler {
						result, herr := g.errorHandlerPolicy(t.node).Handler(state, ne)
						update, cmd, err = normalizeErrorHandlerResult(result, herr, ne)
					}
				}
			}
			if sink != nil {
				// Always emit node_end so start/end pairs are balanced per
				// invocation, even on the error/interrupt paths. The pair
				// wraps the WHOLE attempt loop: retried tasks emit exactly
				// one start/end pair regardless of attempt count.
				sink.EmitRawEvent(RawEvent{Kind: RawNodeEnd, Node: t.node})
			}
			outcomes[i] = outcome{update: update, cmd: cmd, interrupts: interrupts, consumed: consumed, err: err, nodeErr: nodeErr}
		})
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
			if !missed[i] || o.err != nil || len(o.interrupts) > 0 {
				continue
			}
			writes, err := completedTaskWrites(o.update, o.cmd)
			if err != nil {
				return Result{}, fmt.Errorf("graph: node %q cache writes: %w", active[i].node, err)
			}
			if err := g.cache.Set(ctx, storeNS[i], storeKey[i], writes, storeTTL[i]); err != nil {
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
			if len(g.joinsByParent[t.node]) == 0 || outcomes[i].err != nil || len(outcomes[i].interrupts) > 0 {
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
			interrupts = append(interrupts, o.interrupts...)
			pub := g.dropJoinKeys(o.update)
			em.debugTaskResult(rs.step+1, active[i], pub, o.err, o.interrupts)
			em.emitUpdate(active[i].node, pub)
		}
		for _, o := range outcomes {
			if o.err != nil {
				// Aborting with handler tasks in play: persist a pause-style
				// checkpoint naming this superstep's tasks plus each handler
				// task's ERROR write, so a later resume re-runs HANDLERS, not
				// nodes (crash recovery). The mid-superstep putWrites in the
				// task wrapper anchored on the planning checkpoint, which a
				// fresh turn's input checkpoint does not name tasks for; this
				// checkpoint does. Same shape as the in-node interrupt block
				// above. Runs only when some task took the handler path —
				// plain failures keep the bare abort, byte-for-byte.
				if checkpointing && slices.ContainsFunc(outcomes, func(o outcome) bool { return o.nodeErr != nil }) {
					next := plannedTasks(active)
					stampDispatchIDs(next, active, *currentCfg, rs.step+1)
					if err := savePause(checkpoint.Metadata{Source: "loop", Step: rs.step}, next); err != nil {
						return Result{}, err
					}
					for i, o := range outcomes {
						if o.nodeErr == nil {
							continue
						}
						if err := cpSink.putPauseWrites(ctx, *currentCfg, errorWrites(o.nodeErr), next[i].ID); err != nil {
							return Result{}, fmt.Errorf("graph: persisting error write for thread %q: %w", opts.ThreadID, err)
						}
					}
				}
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
				// Stamp the dispatch identities BEFORE saving: the pause's
				// Next must name the IDs the (possibly re-dispatched) tasks
				// of this superstep actually ran under, so the next resume
				// re-attaches those same IDs (see stampDispatchIDs).
				stampDispatchIDs(next, active, *currentCfg, rs.step+1)
				if err := savePause(checkpoint.Metadata{Source: "loop", Step: rs.step}, next); err != nil {
					return Result{}, err
				}
				for i, o := range outcomes {
					taskID := next[i].ID
					if o.nodeErr != nil {
						// The task failed and its handler produced this
						// superstep's outcome; the ERROR write rides
						// alongside the handler's completed writes so a
						// resume (or time travel) sees both. planResume
						// prefers completed writes — replay — over the ERROR
						// write; only the ERROR write alone re-runs the
						// handler.
						if err := cpSink.putPauseWrites(ctx, *currentCfg, errorWrites(o.nodeErr), taskID); err != nil {
							return Result{}, err
						}
					}
					if len(o.interrupts) > 0 {
						if err := cpSink.putPauseWrites(ctx, *currentCfg, interruptAndResumeWrites(o.interrupts, o.consumed), taskID); err != nil {
							return Result{}, err
						}
						continue
					}
					writes, err := completedTaskWrites(o.update, o.cmd)
					if err != nil {
						return Result{}, err
					}
					if len(writes) > 0 {
						if err := cpSink.putPauseWrites(ctx, *currentCfg, writes, taskID); err != nil {
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
			// Pre-stamp the successors' dispatch identities against the
			// just-committed position (see stampDispatchIDs): the next
			// dispatch — with or without an intervening resume — runs
			// under the same IDs.
			stampDispatchIDs(planned, nextTasks, *currentCfg, rs.step+1)
			interrupt := types.Interrupt{
				Value: fmt.Sprintf("interrupt_after: %s", pausedAfter),
				ID:    interruptAfterID + pausedAfter,
				NS:    taskCheckpointNS(opts.checkpointNS, pausedAfter, ""),
			}
			if checkpointing {
				if err := savePause(checkpoint.Metadata{Source: "loop", Step: rs.step}, planned); err != nil {
					return Result{}, err
				}
				if err := cpSink.putPauseWrites(ctx, *currentCfg, interruptWrites([]types.Interrupt{interrupt}), pausedAfter); err != nil {
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
			// Capture the active tasks' planned identities before the save so
			// each task's writes key under a stable, unique slot (M5: same
			// plannedTasks+ID pattern as the interrupt path at the in-node
			// interrupt block above). The loop checkpoint's Next carries the
			// SUCCESSORS (nextTasks); the active tasks ran this superstep, so
			// their IDs are stamped separately against the just-saved
			// checkpoint below.
			planned := plannedTasks(active)
			if err := save(checkpoint.Metadata{Source: "loop", Step: rs.step}, plannedTasks(nextTasks)); err != nil {
				return Result{}, err
			}
			// M2.2: persist each active task's writes so a later GetState can
			// reconstruct delta channels stored as sentinels by replaying
			// ancestor writes (mirrors Python's per-task put_writes, which
			// anchors each task's writes on the checkpoint that observed them).
			for i := range planned {
				planned[i].ID = TaskID(currentCfg.CheckpointID, rs.step+1, planned[i].Node, planned[i].Arg)
			}
			for i := range active {
				writes, err := completedTaskWrites(outcomes[i].update, outcomes[i].cmd)
				if err != nil {
					return Result{}, fmt.Errorf("graph: persisting task writes for thread %q: %w", opts.ThreadID, err)
				}
				if len(writes) > 0 {
					if err := cpSink.putWrites(ctx, *currentCfg, writes, planned[i].ID); err != nil {
						return Result{}, fmt.Errorf("graph: persisting task writes for thread %q: %w", opts.ThreadID, err)
					}
				}
			}
		}
		tasks = nextTasks
	}

	// D1: checkpoints survive completion — the final loop checkpoint (with an
	// empty Next) stays in the thread's history.
	return Result{Values: rs.snapshot()}, nil
}

// stampDispatchIDs pre-stamps each planned task with the identity its task
// dispatched under (or, for not-yet-dispatched tasks, the identity the
// upcoming dispatch will compute): task.plannedID against the run's CURRENT
// checkpoint position and the superstep the tasks run in. A pause checkpoint
// that carries these IDs keeps the original dispatch, its pending-writes
// keys, and every post-resume re-dispatch (planResume re-attaches pt.ID)
// on ONE task identity. That single identity is what lets a subgraph task
// recompute the child namespace (taskCheckpointNS) its pause checkpoint's
// Metadata.Parents pin names — across any number of pause/resume cycles —
// and it keeps an interrupt's NS stable while a pause cycle re-fires it.
// END destinations are skipped, mirroring plannedTasks, so next and the
// non-END tasks of tasks stay index-aligned.
func stampDispatchIDs(next []checkpoint.PlannedTask, tasks []task, planning checkpoint.Config, step int) {
	j := 0
	for _, t := range tasks {
		if t.node == types.END {
			continue
		}
		next[j].ID = t.plannedID(planning, step)
		j++
	}
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

// persistDeltaInputWrites persists the delta-channel entries of input as
// pending writes (NullTaskID) against cfg, so a later GetState can reconstruct
// them via the ancestor walk when the input checkpoint stored them as
// sentinels (snapshotFrequency > 1, below cadence). Non-delta input keys are
// always present in ChannelValues, so they need no write. Mirrors Python's
// put_writes(NULL_TASK_ID, delta_input) (langgraph/pregel/_loop.py:1023-1030).
// cfg must identify an already-saved checkpoint (PutWrites errors otherwise),
// so callers invoke this AFTER the input checkpoint save.
func (g *CompiledGraph) persistDeltaInputWrites(ctx context.Context, cpSink *checkpointSink, cfg checkpoint.Config, input map[string]any) error {
	var deltaInput []checkpoint.Write
	for k, v := range input {
		if proto, ok := g.channelProtos[k]; ok && channels.IsDelta(proto) {
			deltaInput = append(deltaInput, checkpoint.Write{Channel: k, Value: v})
		}
	}
	if len(deltaInput) == 0 {
		return nil
	}
	return cpSink.putWrites(ctx, cfg, deltaInput, checkpoint.NullTaskID)
}

// saveCheckpoint persists rs as a new checkpoint for opts.ThreadID with the
// given metadata and planned next tasks, returning the new checkpoint's
// Config. parent is the executor's current checkpoint position: its
// CheckpointID is passed to Put so the new checkpoint's ParentConfig links to
// its actual predecessor (D3). Planned task IDs bind to the new checkpoint's
// ID and the superstep the tasks will run in (md.Step + 1). pause selects the
// sink's pause path (putPauseCheckpoint), which makes the write durable under
// every Durability mode instead of deferring it (async/exit).
func (g *CompiledGraph) saveCheckpoint(ctx context.Context, cpSink *checkpointSink, opts Options, rs *runState, parent checkpoint.Config, md checkpoint.Metadata, next []checkpoint.PlannedTask, pause bool) (checkpoint.Config, error) {
	// Advance the per-delta-channel (updates, supersteps) counters for this
	// checkpoint, then decide which delta channels snapshot now. Mirrors
	// Python's _loop._put_checkpoint counter advancement + create_checkpoint
	// channel_values assembly (langgraph/pregel/_loop.py:1111-1155).
	newCounters := advanceDeltaCounters(rs.channels, rs.deltaCounters, rs.updatedChannels)

	// A channel snapshots when its cadence fires (DeltaChannelsToSnapshot) or
	// it received an Overwrite since the last snapshot (deltaOverwriteChs).
	channelsToSnapshot := channels.DeltaChannelsToSnapshot(rs.channels, newCounters)
	for k := range rs.deltaOverwriteChs {
		channelsToSnapshot[k] = true
	}

	// UpdateState (Source == "update") force-snapshots every available delta
	// channel. Python keys this on is_fresh_thread (the only UpdateState path
	// that needs a forced blob because there is no ancestor to replay writes
	// from); the Go executor does not yet persist per-task writes in the normal
	// flow (Task 4), so it needs the forced blob on EVERY update, not just
	// fresh threads. Input/loop checkpoints do NOT force a snapshot — they
	// rely on input writes (Task 4), never a forced blob here. Matches Python's
	// create_checkpoint_plan_for_update_state_api (langgraph/pregel/_checkpoint.py:117-146)
	// + get_delta_channels_from_all_channels.
	if md.Source == "update" {
		for name, ch := range rs.channels {
			if d, ok := channels.AsDelta(ch); ok && d.IsAvailable() {
				channelsToSnapshot[name] = true
			}
		}
	}

	// Counters reset to zero for channels that snapshot this checkpoint.
	for k := range channelsToSnapshot {
		newCounters[k] = [2]int{0, 0}
	}
	rs.deltaCounters = newCounters
	rs.deltaOverwriteChs = make(map[string]bool)

	// Persist only non-zero counters (Python omits the key entirely when all
	// entries are zero, langgraph/pregel/_loop.py:1158-1162).
	md.CountersSinceDeltaSnapshot = nonZeroCounters(newCounters)

	cp := checkpoint.Checkpoint{
		V:               1,
		ID:              checkpoint.NewID(int(checkpointSeq.Add(1))),
		TS:              time.Now(),
		ChannelValues:   rs.checkpointValues(channelsToSnapshot),
		ChannelVersions: maps.Clone(rs.versions),
		VersionsSeen:    cloneSeen(rs.seen),
	}
	// Planned-task IDs: a pre-stamped ID (the pause paths in run stamp the
	// IDs the superstep's tasks actually dispatched under — see
	// stampDispatchIDs) is kept verbatim; an empty one (commit checkpoints'
	// undispatched successors, UpdateState destinations) is stamped against
	// the NEW checkpoint, which is the checkpoint the next superstep's
	// dispatch recomputes planned IDs against (see task.plannedID).
	for i := range next {
		if next[i].ID == "" {
			next[i].ID = TaskID(cp.ID, md.Step+1, next[i].Node, next[i].Arg)
		}
	}
	cp.Next = next
	put := cpSink.putCheckpoint
	if pause {
		put = cpSink.putPauseCheckpoint
	}
	cfg, err := put(ctx, checkpoint.Config{ThreadID: opts.ThreadID, CheckpointNS: opts.checkpointNS, CheckpointID: parent.CheckpointID}, cp, md, nil)
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("graph: saving checkpoint for thread %q: %w", opts.ThreadID, err)
	}
	return cfg, nil
}

// advanceDeltaCounters advances the per-channel (updates, supersteps) counters
// for every delta channel by one superstep (and one update when the channel was
// written this superstep), returning a fresh map. It mirrors Python's
// _loop._put_checkpoint counter bump (langgraph/pregel/_loop.py:1111-1124):
// every DeltaChannel's superstep counter increments unconditionally, the update
// counter increments only when the channel is in updated. Missing prev entries
// are treated as {0, 0}. Returns nil when there are no delta channels.
func advanceDeltaCounters(chs map[string]channels.Channel, prev map[string][2]int, updated map[string]bool) map[string][2]int {
	out := make(map[string][2]int)
	for name, ch := range chs {
		_, ok := channels.AsDelta(ch)
		if !ok {
			continue
		}
		u, s := 0, 0
		if c, ok := prev[name]; ok {
			u, s = c[0], c[1]
		}
		s++
		if updated[name] {
			u++
		}
		out[name] = [2]int{u, s}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// nonZeroCounters returns a copy of counters excluding zero {0, 0} entries,
// returning nil when nothing remains. Mirrors Python's non-zero filter
// (langgraph/pregel/_loop.py:1158 and _checkpoint.py:143).
func nonZeroCounters(counters map[string][2]int) map[string][2]int {
	out := make(map[string][2]int)
	for k, v := range counters {
		if v != [2]int{0, 0} {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
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

// entryTasks resolves the run's first tasks: the static entry node, or the
// conditional entry router's destinations resolved against the post-input
// state (Python's START branch). The router receives a Runtime built for a
// synthetic START task, like node routers receive their node's.
func (g *CompiledGraph) entryTasks(ctx context.Context, state map[string]any) ([]task, error) {
	if g.entryRouter == nil {
		return []task{{node: g.entry}}, nil
	}
	rt := g.buildRuntime(ctx, task{node: types.START})
	dests, err := g.entryRouter(rt, state)
	if err != nil {
		return nil, err
	}
	return resolveDestinations(dests)
}

func (g *CompiledGraph) staticNext(ctx context.Context, nodeName string, state map[string]any) ([]task, error) {
	if router, ok := g.conditional[nodeName]; ok {
		rt := g.buildRuntime(ctx, task{node: nodeName})
		dests, err := router(rt, state)
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
// attempt loop. The effective policy is the node's own (installed via
// StateGraph.AddNodeWithPolicies) or, when the node has none, the graph-level
// default installed via WithDefaultRetryPolicy (already defaults-resolved at
// Compile). Nodes with neither take exactly one attempt, preserving the
// pre-policy behavior.
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
//   - A terminal failure of a node with an installed ErrorHandlerPolicy is
//     wrapped into a *NodeError so the run loop's task wrapper can run the
//     handler (see errorHandlerPolicy); nodes without a handler keep the
//     bare error, byte-for-byte identical to the pre-handler behavior.
//
// Events: the RawNodeStart/RawNodeEnd pair (in run's task wrapper) and the
// debug task_result emission bracket the whole attempt loop, so exactly one
// of each appears per task regardless of attempt count.
func (g *CompiledGraph) runTask(ctx context.Context, t task, state map[string]any, resumeQueue []any) (update map[string]any, cmd *types.Command, interrupts []types.Interrupt, consumed []any, err error) {
	var retry *RetryPolicy
	if policies, ok := g.policies[t.node]; ok && policies.Retry != nil {
		p := policies.Retry.withDefaults()
		retry = &p
	} else if g.defaultRetry != nil {
		retry = g.defaultRetry // resolved once at Compile (WithDefaultRetryPolicy)
	}
	for attempt := 1; ; attempt++ {
		result, intr, cons, rerr := g.runNode(ctx, t, state, resumeQueue, attempt)
		if len(intr) > 0 {
			return nil, nil, intr, cons, nil
		}
		if rerr == nil {
			update, cmd, nerr := normalizeNodeResult(result)
			return update, cmd, nil, nil, nerr
		}
		if retry == nil || attempt >= retry.MaxAttempts || !retry.RetryOn(rerr) {
			if g.errorHandlerPolicy(t.node) != nil {
				rerr = &NodeError{Node: t.node, Attempt: attempt, Err: rerr}
			}
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

// errorHandlerPolicy returns the node's installed error handler policy, or
// nil when the node has none (the only case in which task failures keep
// their bare, unwrapped shape).
func (g *CompiledGraph) errorHandlerPolicy(node string) *ErrorHandlerPolicy {
	if policies, ok := g.policies[node]; ok {
		return policies.ErrorHandler
	}
	return nil
}

// tracePolicy returns the node's installed trace policy, or nil when the node
// has none (nodes without a policy emit events untouched).
func (g *CompiledGraph) tracePolicy(node string) *TracePolicy {
	if policies, ok := g.policies[node]; ok {
		return policies.Trace
	}
	return nil
}

// normalizeErrorHandlerResult applies node-result normalization to a
// handler's return, wrapping handler failures so both the handler error and
// the original NodeError stay matchable via errors.Is.
func normalizeErrorHandlerResult(result any, herr error, ne *NodeError) (map[string]any, *types.Command, error) {
	if herr != nil {
		return nil, nil, fmt.Errorf("graph: error handler for node %q: %w (original: %w)", ne.Node, herr, ne.Err)
	}
	return normalizeNodeResult(result)
}

// runNode runs one node invocation. On an interrupt it additionally reports
// consumed — a copy of the resume-queue prefix the invocation consumed
// before panicking (ist.resumeQueue[:ist.idx]; at the panic idx ==
// len(resumeQueue), i.e. the full ordered prefix consumed so far, including
// values carried from earlier pause/resume cycles) — so the pause path can
// persist it as a single full-list ReservedResume write. Retry/error paths discard it: an
// errored task produces no pause checkpoint, so the prefix has nowhere to
// land.
//
// interrupts is a slice so a single node invocation can surface more than one
// interrupt (today: one per GraphInterrupt panic; a subgraph node's paused
// child run propagates its whole interrupt list the same way).
func (g *CompiledGraph) runNode(ctx context.Context, t task, state map[string]any, resumeQueue []any, attempt int) (result any, interrupts []types.Interrupt, consumed []any, err error) {
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
	// Refresh the execution meta's attempt for this invocation so
	// buildRuntime's ExecutionInfo reflects the current retry attempt (the
	// run loop publishes attempt=0 as a placeholder; runNode owns the real
	// value because it runs once per attempt). The same meta carries the
	// run's checkpoint namespace, which together with the planned task ID
	// published by the run loop yields this task's checkpoint namespace —
	// the NS every Interrupt() raised by this invocation is stamped with.
	if m, ok := nodeCtx.Value(executionMetaKey{}).(executionMeta); ok {
		m.attempt = attempt
		nodeCtx = context.WithValue(nodeCtx, executionMetaKey{}, m)
		taskID, _ := ctx.Value(plannedTaskIDKey{}).(string)
		ist.ns = taskCheckpointNS(m.opts.checkpointNS, t.node, taskID)
	}

	defer func() {
		if r := recover(); r != nil {
			if gi, ok := r.(*types.GraphInterrupt); ok {
				interrupts = []types.Interrupt{gi.Interrupt}
				ist.mu.Lock()
				consumed = append([]any{}, ist.resumeQueue[:ist.idx]...)
				ist.mu.Unlock()
				result = nil
				err = nil
				return
			}
			if sig, ok := r.(*subgraphInterruptSignal); ok {
				// A paused child run propagated its interrupts (see
				// invokeSubgraph): they become this task's interrupts verbatim.
				// consumed stays nil — the wrapper never consumes the
				// parent-level resume queue, so no ReservedResume prefix is
				// persisted for the subgraph task (the child namespace holds
				// the real prefix).
				interrupts = sig.interrupts
				result = nil
				err = nil
				return
			}
			panic(r)
		}
	}()

	var timeout *TimeoutPolicy
	if policies, ok := g.policies[t.node]; ok && policies.Timeout != nil {
		timeout = policies.Timeout
	}
	rt, cancelTimeout := g.buildTimedRuntime(nodeCtx, t, timeout)
	defer cancelTimeout()

	result, err = fn(rt, input)
	return
}

// buildTimedRuntime builds the node's Runtime and, when a non-nil, non-zero
// TimeoutPolicy is installed, layers the timeout onto the Runtime's backing
// context so a node that respects rt.Done()/rt.Err() aborts on expiry.
//
// RunTimeout is a hard context deadline (context.WithTimeout). IdleTimeout
// runs a watchdog goroutine that cancels the context if no rt.Heartbeat()
// arrives within IdleTimeout; rt.Heartbeat is wired to reset the timer
// (composing with any Heartbeat buildRuntime set). The watchdog exits when
// the context ends (attempt finished or run deadline), so it does not leak.
//
// The returned cancel func MUST be deferred by the caller; it is a no-op when
// no timeout policy applies. Cooperative cancellation only: Go cannot kill a
// goroutine, so a node blocking without checking rt.Done() overruns (mirrors
// Python's sync limitation under asyncio).
func (g *CompiledGraph) buildTimedRuntime(nodeCtx context.Context, t task, timeout *TimeoutPolicy) (runtime.Runtime, func()) {
	if timeout == nil || (timeout.RunTimeout == 0 && timeout.IdleTimeout == 0) {
		return g.buildRuntime(nodeCtx, t), func() {}
	}
	ctx, cancel := context.WithCancel(nodeCtx)
	if timeout.RunTimeout > 0 {
		var runCancel func()
		ctx, runCancel = context.WithTimeout(ctx, timeout.RunTimeout)
		base := cancel
		cancel = func() { base(); runCancel() }
	}
	rt := g.buildRuntime(ctx, t)
	if timeout.IdleTimeout > 0 {
		heartbeat := make(chan struct{}, 1)
		prev := rt.Heartbeat
		rt.Heartbeat = func() {
			if prev != nil {
				prev()
			}
			select {
			case heartbeat <- struct{}{}:
			default:
			}
		}
		go func() {
			idle := time.NewTimer(timeout.IdleTimeout)
			defer idle.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-heartbeat:
					idle.Reset(timeout.IdleTimeout)
				case <-idle.C:
					cancel()
					return
				}
			}
		}()
	}
	return rt, cancel
}

// buildRuntime constructs the runtime.Runtime passed to node functions and
// conditional edge routers. It is the single place the executor materializes
// a Runtime, mirroring Python's task-prep step that populates Runtime from
// the config + task context.
//
// Wiring:
//
//   - The backing context.Context (for Deadline/Done/Err/Value delegation)
//     is ctx, which already carries this node's interrupt state (see
//     runNode), event sink (see InvokeStream), stream writer (see
//     nodeContext under StreamCustom), checkpoint/subgraph keys, and any
//     context_schema values attached by agents.WithContextValues (which
//     delegates to runtime.ContextWithValues).
//   - rt.Context is the context_schema values bag (runtime.contextSchemaKey),
//     populated when present, mirroring Python's Runtime.context.
//   - rt.StreamWriter is the per-node writer installed by nodeContext when
//     StreamCustom is active (nil otherwise; consumers nil-check).
//   - rt.Store is g.store when a store was installed via WithStore (nil
//     otherwise; consumers nil-check before use).
//   - rt.ExecutionInfo is populated from the task's identity and the
//     executor's current checkpoint position when available; nil when no
//     checkpointer is in use (Python: nil before task prep populates it).
//   - rt.Control and rt.ServerInfo are nil for now (no server/control plane
//     in this port yet).
//
// The returned Runtime shares ctx by reference (via the unexported ctx
// field), so a node reading rt.Value(interruptCtxKey{}) still reaches the
// per-node interrupt state, and existing graph.Interrupt(ctx, ...) calls
// continue to work because Runtime satisfies context.Context.
func (g *CompiledGraph) buildRuntime(ctx context.Context, t task) runtime.Runtime {
	rt := runtime.NewRuntime(ctx)
	// Mirror Python: Runtime.context is the context_schema values bag. The
	// values are attached by agents.WithContextValues (which delegates to
	// runtime.ContextWithValues) at invoke time; buildRuntime surfaces them
	// on rt.Context so nodes that prefer the typed field (Python parity)
	// can use it directly. rt.Value(runtime's key) still reaches the same
	// bag via context.Context delegation, so agents.ContextValue stays
	// functional on a Runtime.
	if v := runtime.ValuesFromContext(ctx); v != nil {
		rt.Context = v
	}
	// StreamWriter was attached to ctx by nodeContext under StreamCustom
	// (see stream.go); surface it on rt.StreamWriter so nodes that prefer
	// the typed field (Python parity) can use it directly. Nil-check applies.
	if w := StreamWriterFromContext(ctx); w != nil {
		rt.StreamWriter = runtime.StreamWriter(w)
	}
	// Store: surface the compile-option store (WithStore) so nodes/tasks that
	// need cross-thread persistence can read rt.Store directly. Guarded nil
	// so a graph without a store keeps rt.Store nil (consumers nil-check).
	if g.store != nil {
		rt.Store = g.store
	}
	// ExecutionInfo: populate from the task and the current checkpoint
	// config when one is available. Python populates ExecutionInfo during
	// task prep; here we read the checkpoint config the run loop published
	// on ctx (see CompiledGraph.run).
	if ei := executionInfoFromContext(ctx, t); ei != nil {
		rt.ExecutionInfo = ei
	}
	return rt
}

// executionMetaKey is the context.Context key under which the run loop
// publishes the current checkpoint position + run options, so buildRuntime
// can populate ExecutionInfo without threading them through runTask/runNode
// signatures.
type executionMetaKey struct{}

// executionMeta is the run-scoped checkpoint position + options published on
// runCtx by CompiledGraph.run. Fields are read-only after publication; the
// pointer to currentCfg is shared with the run loop, which advances it
// between supersteps (the ExecutionInfo snapshot reads CheckpointID at
// buildRuntime time, so it tracks the latest save).
type executionMeta struct {
	cfg     *checkpoint.Config
	opts    Options
	attempt int // 1-indexed node attempt, set per runNode invocation
	// resuming reports whether the run entered via one of the run loop's
	// resume branches (explicit Options.Resume, a wrapper-forwarded
	// forceResume, or nil input over an existing checkpoint). The AddSubgraph
	// wrapper reads it to forward the resume payload into child runs
	// (invokeSubgraph), mirroring Python's CONFIG_KEY_RESUMING propagation.
	resuming bool
}

// executionInfoFromContext assembles an ExecutionInfo from the run's published
// execution meta and the task's identity. Returns nil when no meta is
// published (the plain context.Background() -> Invoke path), matching
// Python's "nil before task prep populates it."
func executionInfoFromContext(ctx context.Context, t task) *runtime.ExecutionInfo {
	m, ok := ctx.Value(executionMetaKey{}).(executionMeta)
	if !ok {
		return nil
	}
	ei := &runtime.ExecutionInfo{
		TaskID:      t.id,
		ThreadID:    m.opts.ThreadID,
		NodeAttempt: m.attempt,
	}
	if m.cfg != nil {
		ei.CheckpointID = m.cfg.CheckpointID
		ei.CheckpointNS = m.cfg.CheckpointNS
	}
	if ei.NodeAttempt == 0 {
		ei.NodeAttempt = 1
	}
	return ei
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
	// ns is this task's checkpoint namespace ("<parentNS>/<node>:<taskID>",
	// computed by runNode from the run's checkpoint namespace and the planned
	// task ID). Every Interrupt() raised by the invocation is stamped with it
	// so resume maps can address interrupts by namespace.
	ns string
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
		NS:    st.ns,
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
