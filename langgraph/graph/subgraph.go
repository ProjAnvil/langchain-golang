package graph

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// ParentCommandError aborts a graph run when a direct node returns a
// *types.Command whose Graph is types.ParentGraph, carrying the command up to
// the closest parent graph (D6). It is an internal control-flow signal: the
// node wrapper installed by StateGraph.AddSubgraph recovers it from the child
// run and returns the command (Graph cleared) as the subgraph node's normal
// result, so the parent executor applies the update and goto at its own
// level. Recursion through nested subgraphs yields correct multi-level
// semantics: each wrapper strips exactly one level of bubbling.
//
// A ParentCommandError surfacing from a TOP-level graph (one that is not a
// node of any parent) means the command has no graph to apply to;
// CompiledGraph's public entry points return it to the caller wrapped in a
// descriptive error (errors.As still finds the *ParentCommandError).
type ParentCommandError struct {
	// Command is the parent-targeted command as returned by the node,
	// Graph still set to types.ParentGraph.
	Command *types.Command
}

func (e *ParentCommandError) Error() string {
	return "graph: node returned a Command targeting the parent graph"
}

// subgraphCheckpoint carries a running graph's checkpoint identity down to
// subgraph nodes via the context: the run's checkpointer, thread, and the
// running graph's OWN checkpoint namespace, so a subgraph run can checkpoint
// into the same thread under <parentNS>/<name>. Installed by run only when
// the run is checkpointing (checkpointer + ThreadID present); nested runs
// shadow it with their own namespace, so names accumulate along the nesting
// path ("a", then "a/b").
type subgraphCheckpoint struct {
	saver    checkpoint.Saver
	threadID string
	ns       string
	// parents is the Metadata.Parents of the checkpoint the run started
	// from (nil for a fresh run): on re-entry, a subgraph node whose
	// namespace it names pins the child's GetTuple to that recorded
	// checkpoint ID instead of the namespace's latest (Python pins via
	// CONFIG_KEY_CHECKPOINT_MAP in PregelLoop init).
	parents map[string]string
	// current tracks the run's own checkpoint position, shared with child
	// runs so their checkpoints record Parents[ns] = the parent's position
	// at the time the child ran. It is advanced only between supersteps —
	// never while node tasks execute — and read by subgraph nodes during a
	// superstep.
	current *checkpoint.Config
	// children accumulates the latest checkpoint of each child namespace
	// used during the run, so the run's own checkpoints name each child's
	// position in Metadata.Parents.
	children *childCheckpoints
}

type subgraphCheckpointKey struct{}

// childCheckpoints records, per child checkpoint namespace, the latest
// checkpoint ID produced by subgraph runs of the current run, so the run's
// checkpoints can name each child's position in Metadata.Parents. Safe for
// concurrent use by sibling subgraph nodes in the same superstep.
type childCheckpoints struct {
	mu     sync.Mutex
	latest map[string]string
}

func (c *childCheckpoints) set(ns, id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.latest == nil {
		c.latest = map[string]string{}
	}
	c.latest[ns] = id
}

func (c *childCheckpoints) snapshot() map[string]string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.latest) == 0 {
		return nil
	}
	return maps.Clone(c.latest)
}

// joinCheckpointNS composes a namespace-relative name: parentNS/name, with
// the root namespace collapsing to just name. It serves both checkpoint
// namespaces (via taskCheckpointNS) and stream-emission namespaces (the node
// path; see streamEmitter.child).
func joinCheckpointNS(parentNS, name string) string {
	if parentNS == "" {
		return name
	}
	return parentNS + "/" + name
}

// nsTaskSep separates a subgraph node's name from its task ID within one
// checkpoint-namespace segment, mirroring Python's NS_END (":",
// _internal/_constants.py:89).
const nsTaskSep = ":"

// taskCheckpointNS composes a subgraph TASK's checkpoint namespace:
// <parentNS>/<node>:<taskID>, mirroring Python's task_checkpoint_ns
// (pregel/_algo.py:615-624 — Python's NS_SEP "|" becomes "/" in this port,
// matching its existing namespace separator). Every subgraph task — including
// each task of a Send fan-in, and each re-execution of a looped subgraph node
// — gets its own namespace, so concurrent or repeated invocations never share
// (and never fork) one checkpoint history. An empty taskID (contexts that
// carry no planned-task identity) degrades to the node-only namespace.
//
// Breaking change vs earlier Go releases, which namespaced by node only
// (<parentNS>/<name>): checkpoints persisted under the old format are not
// migrated. Reads are dual-format: the Metadata.Parents pin lookup accepts
// both the per-task key and the legacy node-only key (see invokeSubgraph).
func taskCheckpointNS(parentNS, name, taskID string) string {
	ns := joinCheckpointNS(parentNS, name)
	if taskID == "" {
		return ns
	}
	return ns + nsTaskSep + taskID
}

// AddSubgraph registers child as a node of the graph: the node runs child
// with the parent state map as its input and merges the child's final values
// back as the node's state update, so keys shared between parent and child
// flow in and out through each side's channels. It mirrors embedding a
// compiled graph via Python's `add_node(name, subgraph)`.
//
// Parent-targeted commands (D6): when a node of the child returns
// Command{Graph: types.ParentGraph}, the child's run aborts with a
// *ParentCommandError, which this wrapper recovers and returns — Graph
// cleared — as the subgraph node's normal result. The parent executor then
// applies the command's update and goto at its own level.
//
// Checkpointing: when the PARENT run is checkpointing (checkpointer installed
// via WithCheckpointer and Options.ThreadID set), the child runs against the
// parent's checkpointer and thread, namespaced under CheckpointNS =
// <parentNS>/<name>:<taskID> — one namespace per subgraph TASK, Python's
// task_checkpoint_ns (see taskCheckpointNS for the format and the breaking
// change vs the older node-only namespacing); the child's own compile-time
// checkpointer is ignored in that case, matching Python's rule that a
// subgraph without its own checkpointer shares the parent's. Without a
// checkpointing parent the child simply runs uncheckpointed (its own
// checkpointer, if any, is not consulted because no ThreadID is available).
//
// Parent and child checkpoints cross-record their positions in
// Metadata.Parents: child checkpoints name the parent's checkpoint position
// when the child ran, and parent checkpoints saved after the child ran name
// the child's position (keyed by the per-task namespace). When the parent run
// was pinned to or resumed from a checkpoint whose Metadata.Parents names the
// child's namespace, the re-entered child resumes from that recorded
// checkpoint instead of its namespace's latest; a legacy node-only Parents
// key (data written before per-task namespacing) pins the same way, with the
// pinned checkpoint loaded from the legacy namespace while new checkpoints
// still land in the per-task namespace. Because the namespace embeds the task
// ID, the pin applies only when the re-entered task's ID matches the recorded
// one (Python parity: a fresh-input re-run mints new task IDs and starts a
// fresh child namespace).
//
// A child that interrupts (in-node Interrupt or interrupt boundaries) is out
// of scope: the wrapper surfaces a descriptive error rather than silently
// treating the paused child as complete.
func (g *StateGraph) AddSubgraph(name string, child *CompiledGraph) *StateGraph {
	if child == nil {
		g.setErr(fmt.Errorf("graph: subgraph %q must not be nil", name))
		return g
	}
	prev := g.err
	g.AddNode(name, func(rt runtime.Runtime, state map[string]any) (any, error) {
		return invokeSubgraph(rt, name, child, state)
	})
	if g.err == prev {
		// Register the child for graph export (GetGraph with WithXrayDepth
		// expands the node into the child's own graph). The wrapper NodeFunc
		// captures the same child, so the registry adds no runtime behavior.
		g.subgraphs[name] = child
	}
	return g
}

// invokeSubgraph runs child as the subgraph node name with state as input and
// translates the outcome into a node result: final values become the update,
// a *ParentCommandError becomes the cleared command, and anything else is an
// error.
func invokeSubgraph(ctx context.Context, name string, child *CompiledGraph, state map[string]any) (any, error) {
	runner := child
	var opts Options
	var sc subgraphCheckpoint
	checkpointing := false
	if v, ok := ctx.Value(subgraphCheckpointKey{}).(subgraphCheckpoint); ok {
		sc = v
		checkpointing = true
		// Checkpoint the child into the parent run's thread, namespaced per
		// subgraph TASK under <parentNS>/<name>:<taskID> (the planned task ID
		// published by the parent's run loop; see plannedTaskIDKey). The copy
		// only swaps the checkpointer; the maps shared with child are
		// read-only during a run.
		cp := *child
		cp.checkpointer = sc.saver
		runner = &cp
		taskID, _ := ctx.Value(plannedTaskIDKey{}).(string)
		childNS := taskCheckpointNS(sc.ns, name, taskID)
		legacyNS := joinCheckpointNS(sc.ns, name)
		opts = Options{ThreadID: sc.threadID, checkpointNS: childNS}
		// Time-travel pin: when the parent run started from a checkpoint whose
		// Metadata.Parents names this child's namespace (a pinned or resumed
		// parent), the child resumes from that recorded position instead of
		// the namespace's latest checkpoint. Dual-format read: the per-task
		// key matches when the re-entered task's ID equals the recorded one;
		// the legacy node-only key (data written before per-task namespacing)
		// pins the child to a checkpoint stored under the legacy namespace —
		// pinNS loads it from there while this run writes into childNS.
		if pinned := sc.parents[childNS]; pinned != "" {
			opts.CheckpointID = pinned
		} else if pinned := sc.parents[legacyNS]; pinned != "" {
			opts.CheckpointID = pinned
			opts.pinNS = legacyNS
		}
	}

	// Propagate the parent run's per-invoke RecursionLimit override into the
	// child Options, mirroring Python's config propagation into subgraphs.
	// The parent's override is published on the context by the run loop (see
	// executionMeta); 0 keeps the child's own compiled default. Applies with
	// or without checkpointing, so the rebuilt Options above carry it too.
	if m, ok := ctx.Value(executionMetaKey{}).(executionMeta); ok {
		opts.RecursionLimit = m.opts.RecursionLimit
		// Likewise propagate a per-run durability override (Python sets
		// CONFIG_KEY_DURABILITY for subgraphs, main.py:2862-2864): an
		// explicitly overridden parent run makes the child (and, transitively,
		// grandchildren) run under the same mode. The child's own compiled
		// WithDurability value applies when the parent run was not
		// overridden.
		opts.Durability = m.opts.Durability
	}

	// Streaming: propagate the emission layer to the child run. The child's
	// emission namespace derives from the node path (see StreamChunk), NOT
	// from checkpoint config, so subgraph chunks are namespaced with or
	// without a checkpointer. When the stream did not request subgraphs, the
	// emitter is stripped so the child neither emits nor pays for hooks.
	if em := emitterFromContext(ctx); em != nil {
		if em.subgraphs {
			ctx = contextWithEmitter(ctx, em.child(name))
		} else {
			ctx = stripStreamCarriers(ctx)
		}
	}

	res, err := runner.run(ctx, state, opts, nil)
	if err != nil {
		var pce *ParentCommandError
		if errors.As(err, &pce) {
			cmd := *pce.Command
			cmd.Graph = ""
			return &cmd, nil
		}
		return nil, fmt.Errorf("graph: subgraph %q: %w", name, err)
	}
	if len(res.Interrupts) > 0 {
		return nil, fmt.Errorf("graph: subgraph %q interrupted (%v); resuming interrupted subgraphs is not supported", name, res.Interrupts)
	}
	if checkpointing && sc.children != nil {
		// Record the child's final position so the parent's subsequent
		// checkpoints name this namespace in Metadata.Parents.
		tup, terr := sc.saver.GetTuple(ctx, checkpoint.Config{ThreadID: sc.threadID, CheckpointNS: opts.checkpointNS})
		if terr != nil {
			return nil, fmt.Errorf("graph: subgraph %q: loading child checkpoint: %w", name, terr)
		}
		if tup != nil {
			sc.children.set(opts.checkpointNS, tup.Config.CheckpointID)
		}
	}
	return res.Values, nil
}
