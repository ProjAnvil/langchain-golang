package graph

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Interrupt persistence and resume replay machinery for the pause/resume
// paths.
//
// A paused run persists, as pending writes against the pause checkpoint
// (Saver.PutWrites), keyed by each task's planned ID (D5):
//   - each interrupted task's interrupts as ReservedInterrupt writes;
//   - each interrupted IN-NODE task's already-consumed resume values as
//     ReservedResume writes (one per value, in consumption order), so a
//     later resume rebuilds the task's full ordered resume queue instead of
//     misaligning the new value onto an already-answered interrupt;
//   - each COMPLETED sibling task's state writes as plain channel writes,
//     plus one ReservedTasks write per Command.Goto destination, plain node
//     names normalized to types.Send (D4).
//
// A resuming run replays completed siblings' writes instead of re-running
// them (their sends rejoin the task queue) and re-executes only interrupted
// or never-run tasks, feeding interrupted tasks their resume queues — each
// queue rebuilt from the persisted ReservedResume prefix followed by the
// values matched from THIS resume call (see resumeValuesFor).

// interruptsFromWrites reconstructs the pending interrupts recorded against
// a checkpoint from its ReservedInterrupt pending writes, in write order.
func interruptsFromWrites(writes []checkpoint.Write) []types.Interrupt {
	var out []types.Interrupt
	for _, w := range writes {
		if w.Channel != checkpoint.ReservedInterrupt {
			continue
		}
		if intr, ok := w.Value.(types.Interrupt); ok {
			out = append(out, intr)
		}
	}
	return out
}

// persistInterrupts records each pending interrupt as a ReservedInterrupt
// pending write against the checkpoint identified by cfg, stamped with
// taskID (the interrupted task's planned ID, or the boundary node name for
// interrupt_before/interrupt_after pauses).
func persistInterrupts(ctx context.Context, saver checkpoint.Saver, cfg checkpoint.Config, taskID string, interrupts []types.Interrupt) error {
	if len(interrupts) == 0 {
		return nil
	}
	writes := make([]checkpoint.Write, len(interrupts))
	for i, intr := range interrupts {
		writes[i] = checkpoint.Write{Channel: checkpoint.ReservedInterrupt, Value: intr}
	}
	return saver.PutWrites(ctx, cfg, writes, taskID, "")
}

// persistInterruptAndResume records a paused in-node task's pending
// interrupts plus the ordered prefix of resume values it has already
// consumed (one ReservedResume write per value, in consumption order), so
// the next resume rebuilds the full ordered queue (see resumeValuesFor).
// Boundary interrupts keep using persistInterrupts — no node ran, so there
// is no consumed prefix.
func persistInterruptAndResume(ctx context.Context, saver checkpoint.Saver, cfg checkpoint.Config, taskID string, interrupts []types.Interrupt, consumed []any) error {
	writes := make([]checkpoint.Write, 0, len(interrupts)+len(consumed))
	for _, intr := range interrupts {
		writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedInterrupt, Value: intr})
	}
	for _, v := range consumed {
		writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedResume, Value: v})
	}
	return saver.PutWrites(ctx, cfg, writes, taskID, "")
}

// completedTaskWrites builds the pending writes persisting a completed
// sibling task's work against an interrupted superstep's pause checkpoint:
// one write per state-update key (sorted for determinism), plus one
// ReservedTasks write per Command.Goto destination (normalized per D4).
// Saver.PutWrites stamps each write with the task's planned ID.
func completedTaskWrites(update map[string]any, cmd *types.Command) ([]checkpoint.Write, error) {
	keys := make([]string, 0, len(update))
	for k := range update {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	writes := make([]checkpoint.Write, 0, len(keys))
	for _, k := range keys {
		writes = append(writes, checkpoint.Write{Channel: k, Value: update[k]})
	}
	if cmd != nil && len(cmd.Goto) > 0 {
		sends, err := gotoSends(cmd.Goto)
		if err != nil {
			return nil, err
		}
		for _, s := range sends {
			writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedTasks, Value: s})
		}
	}
	return writes, nil
}

// outcomeFromCachedWrites rebuilds a task outcome from cached writes using
// the same channel classification planResume uses: ReservedTasks writes are
// routing (rebuilt as Command.Goto), every other channel a state-update
// entry. It is the inverse of completedTaskWrites, used when a cache hit
// injects a task's stored writes as its outcome instead of executing the
// node (see the cache lookup pass in CompiledGraph.run).
func outcomeFromCachedWrites(writes []checkpoint.Write) (map[string]any, *types.Command) {
	var update map[string]any
	var dests []any
	for _, w := range writes {
		if w.Channel == checkpoint.ReservedTasks {
			if send, ok := w.Value.(types.Send); ok {
				dests = append(dests, send)
			}
			continue
		}
		if update == nil {
			update = map[string]any{}
		}
		update[w.Channel] = w.Value
	}
	if len(dests) == 0 {
		return update, nil
	}
	return update, &types.Command{Update: update, Goto: dests}
}

// gotoSends normalizes routing destinations to types.Send values (D4): plain
// node names become Send{Node: name} with a nil Arg.
func gotoSends(dests []any) ([]types.Send, error) {
	sends := make([]types.Send, 0, len(dests))
	for _, d := range dests {
		switch v := d.(type) {
		case string:
			sends = append(sends, types.Send{Node: v})
		case *types.Send:
			sends = append(sends, types.Send{Node: v.Node, Arg: v.Arg})
		case types.Send:
			sends = append(sends, v)
		default:
			return nil, fmt.Errorf("graph: unsupported routing destination %T (want string or *types.Send)", d)
		}
	}
	return sends, nil
}

// resumePlan is everything a resuming run reconstructs from a checkpoint
// tuple: the tasks to dispatch in the resumed superstep, the per-task resume
// queues, the interrupt_before skip node, and completed siblings' state
// writes to replay before dispatching.
type resumePlan struct {
	tasks        []task
	resumeValues map[string][]any // keyed by PlannedTask.ID (D5)
	skipNode     string
	replayWrites []taskWrites
}

// resumeFromTuple restores rs from a paused (or completed) checkpoint and
// derives the tasks to dispatch, the resume value queues, and the
// interrupt_before skip node for the resumed run. Completed siblings' writes
// are replayed into rs (without re-running their tasks) before the resumed
// superstep starts; replay exposes those writes (nil when there are none) so
// the stream emission layer can re-emit them as `updates` chunks (Python
// parity: cached writes are re-streamed on resume, `_loop.py:676-679`).
func resumeFromTuple(rs *runState, tup *checkpoint.Tuple, resume any) (tasks []task, resumeValues map[string][]any, skipNode string, replay []taskWrites, err error) {
	rs.restore(tup.Checkpoint)
	rs.step = tup.Metadata.Step
	plan, err := planResume(tup, resume)
	if err != nil {
		return nil, nil, "", nil, err
	}
	if len(plan.replayWrites) > 0 {
		if _, err := rs.applyWrites(plan.replayWrites); err != nil {
			return nil, nil, "", nil, err
		}
	}
	return plan.tasks, plan.resumeValues, plan.skipNode, plan.replayWrites, nil
}

// planResume classifies each planned task of a checkpoint's Next by its
// pending writes, keyed by PlannedTask.ID (D5):
//   - tasks carrying a ReservedInterrupt write re-execute with their resume
//     queue, rebuilt from the task's persisted ReservedResume prefix (values
//     consumed in earlier pause/resume cycles, in consumption order)
//     followed by the values matched from THIS resume call;
//   - tasks with completed-work writes (state keys and/or ReservedTasks) are
//     NOT re-run: their state writes replay via applyWrites and their sends
//     rejoin the resumed superstep's task queue;
//   - tasks without pending writes never ran (e.g. an interrupt_before
//     pause) and dispatch normally.
//
// A non-map resume value with more than one pending interrupt across the
// checkpoint is an error, mirroring Python's requirement that multiple
// pending interrupts be resumed with an interrupt-ID map.
func planResume(tup *checkpoint.Tuple, resume any) (resumePlan, error) {
	pending := interruptsFromWrites(tup.PendingWrites)
	if resume != nil {
		if _, isMap := resume.(map[string]any); !isMap && len(pending) > 1 {
			return resumePlan{}, fmt.Errorf(
				"graph: resume with %d pending interrupts requires a map[string]any keyed by interrupt ID (a scalar resume only supports a single pending interrupt)",
				len(pending))
		}
	}

	byTask := map[string][]checkpoint.Write{}
	for _, w := range tup.PendingWrites {
		byTask[w.TaskID] = append(byTask[w.TaskID], w)
	}

	plan := resumePlan{resumeValues: map[string][]any{}}
	var replaySends []task
	for _, pt := range tup.Checkpoint.Next {
		if pt.Node == types.END {
			continue
		}
		var interrupts []types.Interrupt
		var resumePrefix []any
		var sends []types.Send
		update := map[string]any{}
		for _, w := range byTask[pt.ID] {
			switch w.Channel {
			case checkpoint.ReservedInterrupt:
				if intr, ok := w.Value.(types.Interrupt); ok {
					interrupts = append(interrupts, intr)
				}
			case checkpoint.ReservedResume:
				resumePrefix = append(resumePrefix, w.Value)
			case checkpoint.ReservedTasks:
				if send, ok := w.Value.(types.Send); ok {
					sends = append(sends, send)
				}
			default:
				update[w.Channel] = w.Value
			}
		}
		switch {
		case len(interrupts) > 0:
			plan.tasks = append(plan.tasks, task{id: pt.ID, node: pt.Node, arg: pt.Arg})
			plan.resumeValues[pt.ID] = resumeValuesFor(interrupts, resumePrefix, resume)
		case len(update) > 0 || len(sends) > 0:
			plan.replayWrites = append(plan.replayWrites, taskWrites{node: pt.Node, update: update})
			for _, s := range sends {
				replaySends = append(replaySends, task{node: s.Node, arg: s.Arg})
			}
		default:
			plan.tasks = append(plan.tasks, task{id: pt.ID, node: pt.Node, arg: pt.Arg})
		}
	}
	plan.tasks = append(plan.tasks, replaySends...)
	plan.skipNode = resumeSkipNode(pending)
	return plan, nil
}

// resumeValuesFor computes the resume queue for a re-run task: the persisted
// prefix of already-consumed resume values (ReservedResume writes, in
// consumption order) followed by the values matched from THIS resume call,
// mirroring Python's accumulated (RESUME, ...) scratchpad list
// (`types.py:905-925`). Matching rules for the new value are unchanged: a
// map[string]any addresses pending interrupts by ID (unmatched ones re-fire
// on re-run), a nil resume appends nothing (the pending interrupt re-fires,
// the run re-pauses), any other scalar feeds the first pending interrupt.
// Values for interrupts already answered in earlier cycles are carried by
// prefix, so a map entry naming an already-answered interrupt ID is ignored.
//
// Boundary interrupts (interrupt_before/interrupt_after) never reach this
// function: their pending writes are stamped with the node name, not a
// PlannedTask.ID, so their resume path never consults resume queues and a
// nil-resume boundary resume keeps working unchanged.
func resumeValuesFor(pending []types.Interrupt, prefix []any, resume any) []any {
	queue := append([]any{}, prefix...)
	if len(pending) == 0 || resume == nil {
		return queue
	}
	if byID, ok := resume.(map[string]any); ok {
		for _, p := range pending {
			if v, ok := byID[p.ID]; ok {
				queue = append(queue, v)
			}
		}
		return queue
	}
	return append(queue, resume)
}

// resumeSkipNode returns the node whose interrupt_before check should be
// skipped on the first superstep of a resume, reconstructed from the
// checkpoint's pending interrupts (D5): the node named by a pending
// interrupt-before-<node> interrupt ID, or "" when the pause was produced by
// interrupt_after or an in-node interrupt.
func resumeSkipNode(pending []types.Interrupt) string {
	for _, p := range pending {
		if node, ok := strings.CutPrefix(p.ID, interruptBeforeID); ok {
			return node
		}
	}
	return ""
}
