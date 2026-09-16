package graph

import (
	"fmt"
	"maps"
	"slices"
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
//   - each interrupted IN-NODE task's already-consumed resume values as ONE
//     ReservedResume write whose value is the whole ordered prefix list
//     (mirroring Python, whose every RESUME write carries the full
//     accumulated scratchpad list, types.py:905-925), so a later resume
//     rebuilds the task's full ordered resume queue instead of misaligning
//     the new value onto an already-answered interrupt — and so savers that
//     give __resume__ a single reserved write slot (sqlite/postgres idx -4,
//     last-write-wins) lose nothing;
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
		out = append(out, interruptsFromWrite(w.Value)...)
	}
	return out
}

// interruptsFromWrite unpacks one ReservedInterrupt write's value, which two
// writers shape differently:
//   - a single types.Interrupt — every root-level task surfaces at most one
//     interrupt per pause, and boundary interrupts (interrupt_before/after)
//     write one;
//   - a whole []types.Interrupt — a subgraph task propagating its paused
//     child's SEVERAL pending interrupts packs them into one write (see
//     interruptAndResumeWrites), mirroring Python's single
//     `(INTERRUPT, interrupts_tuple)` pending write (_runner.py:583-592):
//     savers give __interrupt__ ONE reserved slot per task (Python's
//     WRITES_IDX_MAP -3), so per-interrupt writes would collapse to the last.
func interruptsFromWrite(v any) []types.Interrupt {
	switch x := v.(type) {
	case types.Interrupt:
		return []types.Interrupt{x}
	case []types.Interrupt:
		return x
	}
	return nil
}

// interruptWrites builds the ReservedInterrupt pending writes recording each
// pending interrupt, to be persisted against the pause checkpoint via
// checkpointSink.putPauseWrites (stamped with the task's planned ID, or the
// boundary node name for interrupt_before/interrupt_after pauses). Returns nil
// for an empty list.
func interruptWrites(interrupts []types.Interrupt) []checkpoint.Write {
	if len(interrupts) == 0 {
		return nil
	}
	writes := make([]checkpoint.Write, len(interrupts))
	for i, intr := range interrupts {
		writes[i] = checkpoint.Write{Channel: checkpoint.ReservedInterrupt, Value: intr}
	}
	return writes
}

// errorWrites builds the ReservedError pending write recording a task
// failure's message, committed before the node's error handler runs
// (mirrors Python's ERROR channel task write, langgraph 1.2.0 error_handler).
// The write carries the message only — no attempt count — so a resumed
// handler reconstructs a NodeError with Attempt 0.
func errorWrites(ne *NodeError) []checkpoint.Write {
	return []checkpoint.Write{{Channel: checkpoint.ReservedError, Value: ne.Err.Error()}}
}

// interruptAndResumeWrites builds a paused in-node task's pending writes:
// each pending interrupt as a ReservedInterrupt write, plus — when the task
// has already consumed resume values — ONE ReservedResume write whose value
// is the whole ordered consumed prefix, mirroring Python, where every RESUME
// write carries the full accumulated scratchpad list (types.py:905-925). A
// single full-list write keeps the prefix intact under savers that assign
// __resume__ one reserved write slot (sqlite/postgres idx -4, INSERT OR
// REPLACE last-write-wins): one write, nothing to collapse. The next resume
// rebuilds the full ordered queue from this prefix (see resumeValuesFor).
// Boundary interrupts keep using interruptWrites — no node ran, so there is
// no consumed prefix.
func interruptAndResumeWrites(interrupts []types.Interrupt, consumed []any) []checkpoint.Write {
	if len(interrupts) == 0 {
		return nil
	}
	writes := make([]checkpoint.Write, 0, 2)
	if len(interrupts) == 1 {
		writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedInterrupt, Value: interrupts[0]})
	} else {
		// Several interrupts under one task (a subgraph task whose child
		// paused with multiple pending) travel as ONE write carrying the
		// whole list — Python's (INTERRUPT, interrupts_tuple) shape. Savers
		// give __interrupt__ a single reserved slot per task (Python's
		// WRITES_IDX_MAP -3; MemorySaver replaces in place), so one write per
		// interrupt would collapse to the last. Readers unpack either shape
		// (interruptsFromWrite); single-interrupt writes keep the exact
		// pre-T15 form.
		writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedInterrupt, Value: interrupts})
	}
	if len(consumed) > 0 {
		writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedResume, Value: consumed})
	}
	return writes
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
	slices.Sort(keys)
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
//
// graphNS optionally restricts/validates resume matching to pending
// interrupts whose NS it hits (strict NS addressing). All current entry
// points pass "" (match by interrupt ID, or by interrupt NS for map resumes
// — see resumeValuesFor).
func (g *CompiledGraph) resumeFromTuple(rs *runState, tup *checkpoint.Tuple, resume any, graphNS string) (tasks []task, resumeValues map[string][]any, skipNode string, replay []taskWrites, err error) {
	rs.restore(tup.Checkpoint)
	rs.step = tup.Metadata.Step
	// Seed deltaCounters from the loaded checkpoint so resume continues the
	// per-channel cadence (S3). Cloned so rs.deltaCounters is independent of
	// the (shared) loaded metadata map.
	rs.deltaCounters = maps.Clone(tup.Metadata.CountersSinceDeltaSnapshot)
	plan, err := g.planResume(tup, resume, graphNS)
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
//     rejoin the resumed superstep's task queue — completed writes also win
//     over a ReservedError write the same task may carry (a handler that
//     already produced this superstep's outcome);
//   - tasks whose ONLY write is a ReservedError write failed and had their
//     error handler scheduled (Python langgraph 1.2.0 error_handler): when
//     this build still configures a handler for the node, the resumed task
//     re-runs the HANDLER with the persisted message (crash recovery — the
//     node itself never re-executes); when the handler was removed, the
//     stored failure surfaces as a resume error instead of silently
//     re-running the node;
//   - tasks without pending writes never ran (e.g. an interrupt_before
//     pause) and dispatch normally.
//
// A non-map resume value with more than one pending interrupt across the
// checkpoint is an error, mirroring Python's requirement that multiple
// pending interrupts be resumed with an interrupt-ID map.
//
// graphNS (Options.Graph) restricts/validates resume matching to the pending
// interrupts whose NS lies within it (see nsCovers): the pending set is
// filtered BEFORE the scalar-count rule, so a scalar resume is allowed when
// exactly one pending interrupt hits even if others remain unmatched (they
// re-fire and the run re-pauses). A graphNS hitting nothing is a descriptive
// error listing the available NS values.
func (g *CompiledGraph) planResume(tup *checkpoint.Tuple, resume any, graphNS string) (resumePlan, error) {
	pending := interruptsFromWrites(tup.PendingWrites)
	if graphNS != "" {
		var hits []types.Interrupt
		var avail []string
		for _, p := range pending {
			if p.NS != "" {
				avail = append(avail, p.NS)
			}
			if nsCovers(graphNS, p.NS) {
				hits = append(hits, p)
			}
		}
		if len(hits) == 0 {
			if len(avail) == 0 {
				return resumePlan{}, fmt.Errorf(
					"graph: Options.Graph %q matches no pending interrupt (none of the %d pending interrupts carry an addressable NS)", graphNS, len(pending))
			}
			return resumePlan{}, fmt.Errorf(
				"graph: Options.Graph %q matches no pending interrupt (available NS: %s)", graphNS, strings.Join(avail, ", "))
		}
		pending = hits
	}
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
		var taskErrMsg string
		for _, w := range byTask[pt.ID] {
			switch w.Channel {
			case checkpoint.ReservedInterrupt:
				interrupts = append(interrupts, interruptsFromWrite(w.Value)...)
			case checkpoint.ReservedResume:
				// One write carries the WHOLE consumed prefix as a []any
				// (Python parity: types.py:905-925). A non-slice value is
				// malformed — ignore it, matching the type-assertion style
				// of the other reserved channels above.
				if prefix, ok := w.Value.([]any); ok {
					resumePrefix = prefix
				}
			case checkpoint.ReservedTasks:
				if send, ok := w.Value.(types.Send); ok {
					sends = append(sends, send)
				}
			case checkpoint.ReservedError:
				// A task failure whose handler was scheduled but never
				// completed (crash recovery). The write carries the message
				// only; a non-string value is malformed — ignore it, matching
				// the other reserved channels.
				if msg, ok := w.Value.(string); ok {
					taskErrMsg = msg
				}
			default:
				update[w.Channel] = w.Value
			}
		}
		switch {
		case len(interrupts) > 0:
			plan.tasks = append(plan.tasks, task{id: pt.ID, node: pt.Node, arg: pt.Arg})
			plan.resumeValues[pt.ID] = resumeValuesFor(interrupts, resumePrefix, resume, graphNS)
		case len(update) > 0 || len(sends) > 0:
			plan.replayWrites = append(plan.replayWrites, taskWrites{node: pt.Node, update: update})
			for _, s := range sends {
				replaySends = append(replaySends, task{node: s.Node, arg: s.Arg})
			}
		case taskErrMsg != "":
			if g.errorHandlerPolicy(pt.Node) != nil {
				// Re-run the handler with the persisted message — the node
				// itself never re-executes (Python langgraph 1.2.0 crash
				// recovery semantics for error_handler).
				plan.tasks = append(plan.tasks, task{id: pt.ID, node: pt.Node, arg: pt.Arg,
					runErrorHandler: true, resumeErrMsg: taskErrMsg})
			} else {
				// The write was produced by a graph that had a handler;
				// this build removed it — surface the stored failure
				// instead of silently re-running the node.
				return plan, fmt.Errorf("graph: task %s failed (persisted error): %s", pt.ID, taskErrMsg)
			}
		default:
			plan.tasks = append(plan.tasks, task{id: pt.ID, node: pt.Node, arg: pt.Arg})
		}
	}
	plan.tasks = append(plan.tasks, replaySends...)
	// Only boundary interrupts OWNED by this run's namespace may drive the
	// first-superstep skip: a subgraph's internal interrupt-before-<node>
	// must not make the parent skip its own same-named node (see
	// interruptOwnedBy). ownNS is the namespace of the checkpoint being
	// resumed, which is the resuming run's own namespace.
	plan.skipNode = resumeSkipNode(pending, tup.Config.CheckpointNS)
	return plan, nil
}

// resumeValuesFor computes the resume queue for a re-run task: the persisted
// prefix of already-consumed resume values (the single ReservedResume write's
// full list, in consumption order) followed by the values matched from THIS
// resume call, mirroring Python's accumulated (RESUME, ...) scratchpad list
// (`types.py:905-925`). Matching rules for the new value: a map[string]any
// addresses pending interrupts by NS first and by ID second (an interrupt
// whose NS is "" — persisted before NS stamping — matches by ID only;
// unmatched ones re-fire on re-run), a nil resume appends nothing (the
// pending interrupt re-fires, the run re-pauses), any other scalar feeds the
// first pending interrupt. Values for interrupts already answered in earlier
// cycles are carried by prefix, so a map entry naming an already-answered
// interrupt is ignored.
//
// graphNS (Options.Graph, forwarded verbatim into subgraph runs) filters the
// task's pending interrupts to those lying within the named namespace before
// matching: a scalar then feeds only the addressed task's queue (never a
// sibling's), and a task whose interrupts all fall outside the namespace
// keeps its prefix alone — its interrupts re-fire and the run re-pauses.
//
// Boundary interrupts (interrupt_before/interrupt_after) never reach this
// function: their pending writes are stamped with the node name, not a
// PlannedTask.ID, so their resume path never consults resume queues and a
// nil-resume boundary resume keeps working unchanged.
func resumeValuesFor(pending []types.Interrupt, prefix []any, resume any, graphNS string) []any {
	queue := append([]any{}, prefix...)
	if len(pending) == 0 || resume == nil {
		return queue
	}
	if graphNS != "" {
		filtered := make([]types.Interrupt, 0, len(pending))
		for _, p := range pending {
			if nsCovers(graphNS, p.NS) {
				filtered = append(filtered, p)
			}
		}
		if len(filtered) == 0 {
			return queue
		}
		pending = filtered
	}
	if byKey, ok := resume.(map[string]any); ok {
		for _, p := range pending {
			var v any
			matched := false
			if p.NS != "" {
				v, matched = byKey[p.NS] // NS addressing takes precedence
			}
			if !matched {
				v, matched = byKey[p.ID]
			}
			if matched {
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
// interrupt-before-<node> interrupt ID OWNED by this run (see
// interruptOwnedBy — a subgraph's internal boundary interrupt must not skip
// the parent's same-named node), or "" when the pause was produced by
// interrupt_after, an in-node interrupt, or a foreign-namespace boundary
// interrupt.
func resumeSkipNode(pending []types.Interrupt, ownNS string) string {
	for _, p := range pending {
		if node, ok := strings.CutPrefix(p.ID, interruptBeforeID); ok && interruptOwnedBy(ownNS, p.NS) {
			return node
		}
	}
	return ""
}

// nsCovers reports whether an interrupt stamped with checkpoint namespace
// interruptNS lies within the graph namespace ns: either exactly ns or
// nested under it (ns + "/"). It is the addressability test behind
// Options.Graph — a resume addressed to ns may feed any interrupt raised by
// the graph run that checkpointed under ns (including tasks of its nested
// subgraphs), but nothing above or beside it. A legacy interrupt persisted
// before NS stamping (NS == "") is covered only by the root namespace "",
// where every interrupt is addressable by construction.
func nsCovers(ns, interruptNS string) bool {
	if interruptNS == "" {
		return ns == ""
	}
	return interruptNS == ns || strings.HasPrefix(interruptNS, ns+"/")
}

// interruptOwnedBy reports whether an interrupt stamped with checkpoint
// namespace interruptNS belongs to the graph run whose own checkpoint
// namespace is ownNS: stripping the ownNS prefix (and its "/" separator, for
// a non-root ownNS) from interruptNS leaves a remainder with no further "/",
// meaning the interrupting task ran directly in this graph rather than in a
// nested subgraph (the root run has ownNS == "", where any interrupt NS
// without a "/" is owned). Legacy interrupts persisted before NS stamping
// carry NS == "" and count as owned: pre-NS data can only contain
// same-level interrupts, since subgraph interrupts never paused a parent.
func interruptOwnedBy(ownNS, interruptNS string) bool {
	if interruptNS == "" {
		return true
	}
	rest := interruptNS
	if ownNS != "" {
		var ok bool
		rest, ok = strings.CutPrefix(interruptNS, ownNS+"/")
		if !ok {
			return false
		}
	}
	return !strings.Contains(rest, "/")
}
