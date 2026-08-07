package graph

import (
	"context"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Interrupt persistence helpers for the pause/resume paths.
//
// A paused run persists its pending interrupts as ReservedInterrupt pending
// writes against the pause checkpoint (Saver.PutWrites); a resuming run
// reconstructs them from the checkpoint tuple. This replaces the M1
// single-checkpoint PendingInterrupts field. Full sibling-write replay
// (persisting completed tasks' writes and skipping them on resume) is M2
// Task 4; here a pause still re-runs only the recorded node(s).

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
// taskID (the node that raised them, or the boundary node).
func persistInterrupts(ctx context.Context, saver checkpoint.Saver, cfg checkpoint.Config, taskID string, interrupts []types.Interrupt) error {
	if len(interrupts) == 0 {
		return nil
	}
	writes := make([]checkpoint.Write, len(interrupts))
	for i, intr := range interrupts {
		writes[i] = checkpoint.Write{Channel: checkpoint.ReservedInterrupt, Value: intr}
	}
	return saver.PutWrites(ctx, cfg, writes, taskID)
}

// resumeValuesFor computes the resume queue for a re-run node from the
// pending interrupts and the caller-supplied resume value, mirroring Python:
// a map[string]any resume addresses interrupts by ID; any other non-nil
// resume feeds the first pending interrupt.
func resumeValuesFor(pending []types.Interrupt, resume any) []any {
	if len(pending) == 0 {
		return nil
	}
	if byID, ok := resume.(map[string]any); ok {
		values := make([]any, len(pending))
		for i, p := range pending {
			values[i] = byID[p.ID]
		}
		return values
	}
	values := make([]any, len(pending))
	values[0] = resume
	return values
}

// resumeSkipNode returns the node whose interrupt_before check should be
// skipped on the first superstep of a resume, or "" if not applicable. This
// is non-empty only for checkpoints produced by interrupt_before(N) (whose
// single pending interrupt ID is interruptBeforeID+N and whose next node is
// N); interrupt_after and in-node interrupts return "".
func resumeSkipNode(next string, pending []types.Interrupt) string {
	if len(pending) != 1 {
		return ""
	}
	if pending[0].ID != interruptBeforeID+next {
		return ""
	}
	return next
}
