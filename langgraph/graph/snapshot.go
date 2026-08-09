package graph

import (
	"context"
	"fmt"
	"maps"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
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
		if cfg.CheckpointID != "" {
			return StateSnapshot{}, fmt.Errorf("graph: no checkpoint %q found for thread %q", cfg.CheckpointID, cfg.ThreadID)
		}
		return StateSnapshot{}, fmt.Errorf("graph: no checkpoint found for thread %q", cfg.ThreadID)
	}
	snap := g.snapshotFromTuple(tup)
	// Reconstruct delta channels that used sentinel-only storage (absent from
	// ChannelValues) by walking the parent chain for the nearest snapshot blob
	// and ancestor writes.
	g.reconstructDeltaChannels(ctx, tup, snap.Values)
	return snap, nil
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
		snaps[i] = g.snapshotFromTuple(&tuples[i])
		g.reconstructDeltaChannels(ctx, &tuples[i], snaps[i].Values)
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
		if cfg.CheckpointID != "" {
			return checkpoint.Config{}, fmt.Errorf("graph: no checkpoint %q found for thread %q", cfg.CheckpointID, cfg.ThreadID)
		}
		return checkpoint.Config{}, fmt.Errorf("graph: no checkpoint found for thread %q", cfg.ThreadID)
	}

	rs := newRunState(g.channelProtos)
	rs.restore(tup.Checkpoint)
	rs.step = tup.Metadata.Step
	// Seed deltaCounters from the loaded checkpoint so the update checkpoint's
	// counter advancement continues the per-channel cadence (S3). (For
	// Source=="update" saveCheckpoint force-snapshots all available delta
	// channels, so the advanced counters end up zeroed in the saved metadata —
	// but the advancement still runs for parity with the loop path.)
	rs.deltaCounters = maps.Clone(tup.Metadata.CountersSinceDeltaSnapshot)
	if _, err := rs.applyWrites([]taskWrites{{node: asNode, update: values}}); err != nil {
		return checkpoint.Config{}, err
	}
	dests, err := g.staticNext(ctx, asNode, rs.snapshot())
	if err != nil {
		return checkpoint.Config{}, err
	}
	// Save into the SAME namespace the tuple was read from (a subgraph-
	// namespace update must not leak into the root namespace), and carry
	// Metadata.Parents forward so the update checkpoint keeps the cross-graph
	// position links of the checkpoint it builds on. S6: the update checkpoint
	// steps past the checkpoint it builds on (Python main.py:1734).
	md := checkpoint.Metadata{Source: "update", Step: tup.Metadata.Step + 1, Parents: tup.Metadata.Parents}
	return g.saveCheckpoint(ctx,
		Options{ThreadID: cfg.ThreadID, checkpointNS: cfg.CheckpointNS}, rs, tup.Config,
		md, plannedTasks(dests))
}

// snapshotFromTuple projects a checkpoint tuple into its StateSnapshot view:
// the channel values (minus join barrier channels — control plane, excluded
// from Python's output_keys as well), the planned next nodes, and any pending
// interrupts (reconstructed from ReservedInterrupt pending writes). Delta
// snapshot blobs stored in ChannelValues are unwrapped to their inner values.
func (g *CompiledGraph) snapshotFromTuple(tup *checkpoint.Tuple) StateSnapshot {
	values := maps.Clone(tup.Checkpoint.ChannelValues)
	for key, val := range values {
		if isJoinKey(g.channelProtos, key) {
			delete(values, key)
			continue
		}
		// Unwrap delta snapshot blobs into their plain values.
		if unwrapped, ok := channels.UnwrapDeltaSnapshot(val); ok {
			values[key] = unwrapped
		}
	}
	next := make([]string, 0, len(tup.Checkpoint.Next))
	for _, pt := range tup.Checkpoint.Next {
		next = append(next, pt.Node)
	}
	return StateSnapshot{
		Values:       values,
		Next:         next,
		Config:       tup.Config,
		Metadata:     tup.Metadata,
		CreatedAt:    tup.Checkpoint.TS,
		ParentConfig: tup.ParentConfig,
		Interrupts:   interruptsFromWrites(tup.PendingWrites),
	}
}

// reconstructDeltaChannels fills in delta channels that used sentinel-only
// storage (absent from ChannelValues) by walking the checkpoint parent chain.
// For each registered delta channel absent from values, it walks ancestors
// (following ParentConfig) collecting PendingWrites for that channel and
// looking for the nearest snapshot blob. The snapshot blob (or plain value)
// becomes the seed; collected writes are replayed on top via the channel's
// ReplayWrites method.
//
// This mirrors Python's BaseCheckpointSaver.get_delta_channel_history ancestor
// walk, but lives in the read path (GetState/GetStateHistory) because the Go
// executor does not yet persist per-task writes in the normal (non-interrupt)
// checkpoint flow — the ancestor walk primarily finds snapshot blobs.
func (g *CompiledGraph) reconstructDeltaChannels(ctx context.Context, tup *checkpoint.Tuple, values map[string]any) {
	// Collect the delta channel keys that are absent from values.
	var deltaKeys []string
	for key, proto := range g.channelProtos {
		if !channels.IsDelta(proto) {
			continue
		}
		if _, present := values[key]; present {
			continue
		}
		deltaKeys = append(deltaKeys, key)
	}
	if len(deltaKeys) == 0 {
		return
	}

	// Walk the parent chain. For each delta channel, collect pending writes
	// (reversed — oldest first) and look for a snapshot blob seed.
	collected := make(map[string][]any, len(deltaKeys))
	seeded := make(map[string]any, len(deltaKeys))
	remaining := make(map[string]bool, len(deltaKeys))
	for _, k := range deltaKeys {
		remaining[k] = true
	}

	// Also check the current tuple's pending writes (interrupt-path writes).
	collectDeltaWrites(tup.PendingWrites, deltaKeys, collected, remaining)

	cursor := tup.ParentConfig
	for cursor != nil && cursor.CheckpointID != "" && len(remaining) > 0 {
		select {
		case <-ctx.Done():
			return
		default:
		}
		ancestor, err := g.checkpointer.GetTuple(ctx, *cursor)
		if err != nil || ancestor == nil {
			break
		}
		// Collect writes from this ancestor (reversed: pending writes are
		// stored oldest-first; the walk is newest-to-oldest, so prepend).
		collectDeltaWrites(ancestor.PendingWrites, deltaKeys, collected, remaining)
		// Check for a snapshot blob or plain value in this ancestor's
		// ChannelValues.
		for _, key := range deltaKeys {
			if !remaining[key] {
				continue
			}
			if raw, ok := ancestor.Checkpoint.ChannelValues[key]; ok {
				if unwrapped, isSnap := channels.UnwrapDeltaSnapshot(raw); isSnap {
					seeded[key] = unwrapped
				} else {
					seeded[key] = raw // plain-value seed (migration)
				}
				delete(remaining, key)
			}
		}
		cursor = ancestor.ParentConfig
	}

	// Reconstruct each delta channel from seed + collected writes.
	for _, key := range deltaKeys {
		proto := g.channelProtos[key]
		delta, ok := channels.AsDelta(proto.FromCheckpoint(nil))
		if !ok {
			continue
		}
		// Apply the seed if one was found.
		if seed, hasSeed := seeded[key]; hasSeed {
			deltaCh := proto.FromCheckpoint(seed)
			if d, ok := channels.AsDelta(deltaCh); ok {
				delta = d
			}
		}
		// Replay collected writes (oldest-to-newest). collectDeltaWrites
		// already stored them in arrival order (oldest first within each
		// checkpoint, but across checkpoints they are newest-checkpoint-first
		// since we walk backward — reverse to get oldest-first).
		writes := collected[key]
		if len(writes) > 0 {
			// Reverse: we walked newest-to-oldest, so writes are in reverse
			// chronological order. ReplayWrites expects oldest-first.
			for i, j := 0, len(writes)-1; i < j; i, j = i+1, j-1 {
				writes[i], writes[j] = writes[j], writes[i]
			}
			delta.ReplayWrites(writes)
		}
		if delta.IsAvailable() {
			if v, err := delta.Get(); err == nil {
				values[key] = v
			}
		}
	}
}

// collectDeltaWrites scans writes for entries targeting any of the delta
// channel keys, appending their values to the corresponding collected slice.
func collectDeltaWrites(writes []checkpoint.Write, deltaKeys []string, collected map[string][]any, remaining map[string]bool) {
	for _, key := range deltaKeys {
		if !remaining[key] {
			continue
		}
		for i := len(writes) - 1; i >= 0; i-- {
			w := writes[i]
			if w.Channel == key {
				collected[key] = append(collected[key], w.Value)
			}
		}
	}
}
