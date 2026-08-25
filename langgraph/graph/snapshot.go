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
	// Tasks describes the tasks planned for the superstep after this
	// checkpoint, mirroring Python's StateSnapshot.tasks (tuple of
	// PregelTask, types.py:597). Empty when the run had reached END.
	Tasks []SnapshotTask
}

// SnapshotTask describes one task planned for the superstep after a
// checkpoint — the scoped Go subset of Python's PregelTask (types.py:597):
// id, name, path, interrupts. (Python's error/result/subgraph-state fields
// are out of scope for this port; pending state writes of completed sibling
// tasks are replayed via resume, not surfaced here.)
type SnapshotTask struct {
	// ID is the task's deterministic planned ID (graph.TaskID over the
	// owning checkpoint, step, node, and Send arg).
	ID string
	// Name is the node the task will invoke (PregelTask.name).
	Name string
	// Path is the task path recorded on the task's pending writes
	// (checkpoint.Write.TaskPath); empty at all current call sites, kept for
	// parity with PregelTask.path.
	Path string
	// Interrupts are the interrupts pending against this task
	// (PregelTask.interrupts).
	Interrupts []types.Interrupt
}

// BulkUpdate is a single state update within a bulk superstep, mirroring one
// `(values, as_node, task_id)` tuple of Python's `StateUpdate` passed to
// `graph.bulk_update_state`.
type BulkUpdate struct {
	// Values are the state writes to apply, attributed to AsNode.
	Values map[string]any
	// AsNode names the registered node the update is attributed to; it may be
	// left empty only when the graph has exactly one node (see UpdateState).
	AsNode string
	// TaskID is reserved for pending-write task-path addressing, mirroring
	// Python's optional `task_id` (which is secondary to the values/as_node
	// pair). The current single-update UpdateState path does not support
	// task-path pending writes, so TaskID is accepted but not used yet.
	TaskID string
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
	// UpdateState always persists synchronously (not subject to async/exit
	// durability — it's a manual operation outside the invoke loop).
	updateSink := newCheckpointSink(g.checkpointer, DurabilitySync, ctx, tup)
	return g.saveCheckpoint(ctx,
		updateSink,
		Options{ThreadID: cfg.ThreadID, checkpointNS: cfg.CheckpointNS}, rs, tup.Config,
		md, plannedTasks(dests))
}

// BulkUpdateState applies a sequence of state-update supersteps to the
// checkpoint identified by cfg and returns the Config of the final resulting
// checkpoint, mirroring Python's `graph.bulk_update_state(config, supersteps)`.
// Each superstep is a list of BulkUpdate entries applied sequentially: every
// update delegates to UpdateState with its Values/AsNode pair, and the Config
// returned by one update feeds the next, so the checkpoint is stepped forward
// once per update.
//
// BulkUpdate.TaskID is reserved (mirroring Python's optional `task_id`, which
// is secondary to the values/as_node pair): the single-update UpdateState path
// does not yet support task-path pending writes, so TaskID is accepted but not
// used.
//
// BulkUpdateState requires a checkpointer (see WithCheckpointer) and at least
// one non-empty superstep.
func (g *CompiledGraph) BulkUpdateState(ctx context.Context, cfg checkpoint.Config, supersteps [][]BulkUpdate) (checkpoint.Config, error) {
	if g.checkpointer == nil {
		return checkpoint.Config{}, fmt.Errorf("graph: BulkUpdateState requires a checkpointer (see WithCheckpointer)")
	}
	if len(supersteps) == 0 {
		return checkpoint.Config{}, fmt.Errorf("graph: BulkUpdateState: no supersteps provided")
	}
	for _, superstep := range supersteps {
		if len(superstep) == 0 {
			return checkpoint.Config{}, fmt.Errorf("graph: BulkUpdateState: no updates provided")
		}
	}

	curCfg := cfg
	for _, superstep := range supersteps {
		for _, u := range superstep {
			next, err := g.UpdateState(ctx, curCfg, u.Values, u.AsNode)
			if err != nil {
				return checkpoint.Config{}, err
			}
			curCfg = next
		}
	}
	return curCfg, nil
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
	tasks := make([]SnapshotTask, 0, len(tup.Checkpoint.Next))
	for _, pt := range tup.Checkpoint.Next {
		next = append(next, pt.Node)
		tasks = append(tasks, snapshotTask(pt, tup.PendingWrites))
	}
	return StateSnapshot{
		Values:       values,
		Next:         next,
		Config:       tup.Config,
		Metadata:     tup.Metadata,
		CreatedAt:    tup.Checkpoint.TS,
		ParentConfig: tup.ParentConfig,
		Interrupts:   interruptsFromWrites(tup.PendingWrites),
		Tasks:        tasks,
	}
}

// snapshotTask projects one planned task plus its pending-write attachments
// into a SnapshotTask, mirroring the per-task interrupt collection of
// Python's tasks_w_writes (pregel/debug.py:231-236). Interrupt writes match
// the task by planned ID (in-node interrupts, persistInterruptAndResume) or
// — for interrupt_before/after pauses — by NODE NAME, which the boundary
// path uses as the write's task ID (persistInterrupts, resume.go:57-66).
func snapshotTask(pt checkpoint.PlannedTask, writes []checkpoint.Write) SnapshotTask {
	st := SnapshotTask{ID: pt.ID, Name: pt.Node}
	for _, w := range writes {
		if w.TaskID != pt.ID && w.TaskID != pt.Node {
			continue
		}
		if st.Path == "" {
			st.Path = w.TaskPath
		}
		if w.Channel == checkpoint.ReservedInterrupt {
			if intr, ok := w.Value.(types.Interrupt); ok {
				st.Interrupts = append(st.Interrupts, intr)
			}
		}
	}
	return st
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
