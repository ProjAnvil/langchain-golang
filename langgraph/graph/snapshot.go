package graph

import (
	"context"
	"fmt"
	"maps"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// StateSnapshot is a read-only view of one checkpoint in a thread's history,
// mirroring Python's `langgraph.pregel.StateSnapshot`.
type StateSnapshot struct {
	// Values is the graph state at this checkpoint: every channel's value.
	Values map[string]any
	// Next names the nodes planned for the superstep after this checkpoint;
	// empty when the run had reached END at this point.
	Next []string
	// Config selects exactly this checkpoint (ThreadID + CheckpointID).
	Config checkpoint.Config
	// Metadata describes how the checkpoint was produced (Source/Step).
	Metadata checkpoint.Metadata
	// CreatedAt is the checkpoint's wall-clock creation time.
	CreatedAt time.Time
	// ParentConfig selects this checkpoint's put-time parent (D3); nil for a
	// thread's first checkpoint.
	ParentConfig *checkpoint.Config
	// Interrupts holds the interrupts pending against this checkpoint when it
	// records a paused run; empty otherwise.
	Interrupts []types.Interrupt
}

// GetState returns the snapshot of the checkpoint identified by cfg — the
// checkpoint pinned by cfg.CheckpointID, or the thread's latest when it is
// empty — mirroring Python's `graph.get_state(config)`. It requires a
// checkpointer (see WithCheckpointer).
func (g *CompiledGraph) GetState(ctx context.Context, cfg checkpoint.Config) (StateSnapshot, error) {
	if g.checkpointer == nil {
		return StateSnapshot{}, fmt.Errorf("graph: GetState requires a checkpointer (see WithCheckpointer)")
	}
	tup, err := g.checkpointer.GetTuple(ctx, cfg)
	if err != nil {
		return StateSnapshot{}, fmt.Errorf("graph: GetState: %w", err)
	}
	if tup == nil {
		return StateSnapshot{}, fmt.Errorf("graph: no checkpoint found for thread %q", cfg.ThreadID)
	}
	return snapshotFromTuple(tup), nil
}

// GetStateHistory returns the thread's checkpoint history as snapshots,
// newest first, filtered by opts, mirroring Python's
// `graph.get_state_history(config)`. It requires a checkpointer.
//
// Limitation (D3): after a time-travel fork, the history mixes both branches
// (Saver.List is ID-ordered, not branch-aware); each snapshot's ParentConfig
// chain still identifies its actual branch.
func (g *CompiledGraph) GetStateHistory(ctx context.Context, cfg checkpoint.Config, opts checkpoint.ListOptions) ([]StateSnapshot, error) {
	if g.checkpointer == nil {
		return nil, fmt.Errorf("graph: GetStateHistory requires a checkpointer (see WithCheckpointer)")
	}
	tuples, err := g.checkpointer.List(ctx, cfg, opts)
	if err != nil {
		return nil, fmt.Errorf("graph: GetStateHistory: %w", err)
	}
	snaps := make([]StateSnapshot, len(tuples))
	for i := range tuples {
		snaps[i] = snapshotFromTuple(&tuples[i])
	}
	return snaps, nil
}

// UpdateState applies values to the checkpoint identified by cfg as a single
// write batch attributed to asNode, then saves the result as a new checkpoint
// with Metadata.Source "update", mirroring Python's
// `graph.update_state(config, values, as_node)`. The new checkpoint's Next is
// re-resolved from asNode's static/conditional edges against the updated
// state, and its ParentConfig points at the checkpoint it builds on (D3). The
// returned Config selects the new checkpoint, which is now the thread's
// latest (a subsequent nil-input invoke resumes from it).
//
// asNode must name a registered node; it may be left empty only when the
// graph has exactly one node, in which case the update is attributed to it.
// UpdateState requires a checkpointer (see WithCheckpointer).
func (g *CompiledGraph) UpdateState(ctx context.Context, cfg checkpoint.Config, values map[string]any, asNode string) (checkpoint.Config, error) {
	if g.checkpointer == nil {
		return checkpoint.Config{}, fmt.Errorf("graph: UpdateState requires a checkpointer (see WithCheckpointer)")
	}
	if asNode == "" {
		if len(g.nodes) != 1 {
			return checkpoint.Config{}, fmt.Errorf("graph: UpdateState requires asNode (graph has %d nodes)", len(g.nodes))
		}
		for name := range g.nodes {
			asNode = name
		}
	}
	if _, ok := g.nodes[asNode]; !ok {
		return checkpoint.Config{}, fmt.Errorf("graph: UpdateState: unknown node %q", asNode)
	}
	tup, err := g.checkpointer.GetTuple(ctx, cfg)
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("graph: UpdateState: %w", err)
	}
	if tup == nil {
		return checkpoint.Config{}, fmt.Errorf("graph: no checkpoint found for thread %q", cfg.ThreadID)
	}

	rs := newRunState(g.channelProtos)
	rs.restore(tup.Checkpoint)
	rs.step = tup.Metadata.Step
	if err := rs.applyWrites([]taskWrites{{node: asNode, update: values}}); err != nil {
		return checkpoint.Config{}, err
	}
	dests, err := g.staticNext(ctx, asNode, rs.snapshot())
	if err != nil {
		return checkpoint.Config{}, err
	}
	return g.saveCheckpoint(ctx, Options{ThreadID: cfg.ThreadID}, rs, tup.Config,
		checkpoint.Metadata{Source: "update", Step: tup.Metadata.Step}, plannedTasks(dests))
}

// snapshotFromTuple projects a checkpoint tuple into its StateSnapshot view:
// the channel values, the planned next nodes, and any pending interrupts
// (reconstructed from ReservedInterrupt pending writes).
func snapshotFromTuple(tup *checkpoint.Tuple) StateSnapshot {
	next := make([]string, 0, len(tup.Checkpoint.Next))
	for _, pt := range tup.Checkpoint.Next {
		next = append(next, pt.Node)
	}
	return StateSnapshot{
		Values:       maps.Clone(tup.Checkpoint.ChannelValues),
		Next:         next,
		Config:       tup.Config,
		Metadata:     tup.Metadata,
		CreatedAt:    tup.Checkpoint.TS,
		ParentConfig: tup.ParentConfig,
		Interrupts:   interruptsFromWrites(tup.PendingWrites),
	}
}
