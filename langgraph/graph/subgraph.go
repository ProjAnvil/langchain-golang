package graph

import (
	"context"
	"errors"
	"fmt"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
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
}

type subgraphCheckpointKey struct{}

// joinCheckpointNS composes a subgraph's checkpoint namespace from the parent
// graph's namespace and the subgraph's node name.
//
// Divergence from Python: Python derives subgraph namespaces from the parent
// namespace plus the task ID (`<ns>/<node>:<task_id>`), giving every subgraph
// TASK its own namespace. This port namespaces by node name only
// (`<parentNS>/<name>`), so repeated invocations of the same subgraph node
// within one thread share a namespace (later runs fork new turns off the
// previous child checkpoint, per D2) — task-ID-precision replay of subgraph
// tasks is not needed by the edge-driven executor.
func joinCheckpointNS(parentNS, name string) string {
	if parentNS == "" {
		return name
	}
	return parentNS + "/" + name
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
// <parentNS>/<name> (see joinCheckpointNS for the divergence from Python's
// ns+task_id namespacing); the child's own compile-time checkpointer is
// ignored in that case, matching Python's rule that a subgraph without its
// own checkpointer shares the parent's. Without a checkpointing parent the
// child simply runs uncheckpointed (its own checkpointer, if any, is not
// consulted because no ThreadID is available).
//
// A child that interrupts (in-node Interrupt or interrupt boundaries) is out
// of scope: the wrapper surfaces a descriptive error rather than silently
// treating the paused child as complete.
func (g *StateGraph) AddSubgraph(name string, child *CompiledGraph) *StateGraph {
	if child == nil {
		g.setErr(fmt.Errorf("graph: subgraph %q must not be nil", name))
		return g
	}
	return g.AddNode(name, func(ctx context.Context, state map[string]any) (any, error) {
		return invokeSubgraph(ctx, name, child, state)
	})
}

// invokeSubgraph runs child as the subgraph node name with state as input and
// translates the outcome into a node result: final values become the update,
// a *ParentCommandError becomes the cleared command, and anything else is an
// error.
func invokeSubgraph(ctx context.Context, name string, child *CompiledGraph, state map[string]any) (any, error) {
	runner := child
	var opts Options
	if sc, ok := ctx.Value(subgraphCheckpointKey{}).(subgraphCheckpoint); ok {
		// Checkpoint the child into the parent run's thread, namespaced under
		// <parentNS>/<name>. The copy only swaps the checkpointer; the maps
		// shared with child are read-only during a run.
		cp := *child
		cp.checkpointer = sc.saver
		runner = &cp
		opts = Options{ThreadID: sc.threadID, checkpointNS: joinCheckpointNS(sc.ns, name)}
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
	return res.Values, nil
}
