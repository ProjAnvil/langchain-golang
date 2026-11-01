package graph

import "fmt"

// dispatchPath renders a task's position within its superstep as the task
// path persisted in checkpoint.Write.TaskPath. The Go executor has no
// Python-style task-path tuples; within one checkpoint every pending write
// belongs to a single superstep, so the dispatch index IS the live order,
// and savers replay a checkpoint's writes sorted by (task_path, task_id,
// idx) — the writes_sort_key Python introduced for the same reason (#8544).
// The fixed-width zero padding mirrors Python's int path element
// (`task_path_str`, pregel/_algo.py:1411-1419: f"{tup:010d}"), so the string
// order is the numeric order. Pathless writes (graph input, update_state)
// use "" and sort before every dispatch path.
func dispatchPath(i int) string {
	return fmt.Sprintf("%010d", i)
}

// exitDeltaTaskID creates the synthetic task ID for exit-mode delta writes of
// the run's FIRST (anchor) superstep. Embeds the superstep as a zero-padded
// prefix so the ID sorts before real task IDs (Go TaskIDs are 16-char hex
// strings, so a "0000000X" prefix sorts first with overwhelming probability)
// and orders by step among exit writes: when such a write shares a dispatch
// path with a real-id write of a LATER superstep on the same checkpoint (the
// exit pause path), the older write must win the tie. Mirrors Python's
// exit_delta_task_id (_checkpoint.py:65-77), which embeds the step in the
// first UUID group for the same ordering. Never the real task ID: under it,
// a run whose final checkpoint fails to save would leave the resumed task
// looking done to the next resume (Python #9114).
func exitDeltaTaskID(step int, taskID string) string {
	return fmt.Sprintf("%08d%s", step, taskID)
}

// exitDeltaLateTaskID creates the synthetic task ID for exit-mode delta
// writes of every superstep AFTER the anchor superstep. The ffffffff prefix
// sorts after every real task ID and the embedded step (big-endian hex)
// keeps superstep order, so replay preserves chronology whether a saver
// orders by task path or only by task ID. Mirrors Python's
// exit_delta_late_task_id (_checkpoint.py:80-86).
func exitDeltaLateTaskID(step int, taskID string) string {
	return fmt.Sprintf("ffffffff%08x%s", step, taskID)
}

// exitDeltaLatePath creates the task path stored with exitDeltaLateTaskID
// writes: it sorts after every real dispatch path (which are digit-only) in
// step order, mirroring Python's f"~~{step:010d}{path}"
// (_loop.py's _put_exit_delta_writes).
func exitDeltaLatePath(step int, path string) string {
	return fmt.Sprintf("~~%010d%s", step, path)
}

// exitDeltaEarlyPath creates the task path for exit-mode writes that share
// their checkpoint with real-id writes of a LATER superstep — the exit PAUSE
// path (putPauseCheckpointExit), where accumulated writes of earlier
// supersteps land on the same checkpoint as the pause superstep's direct
// writes. It sorts before every real dispatch path (digit-only) in step
// order — the mirror image of exitDeltaLatePath. Go-only: Python anchors
// pause-time exit writes on the run's start checkpoint instead, so it has no
// counterpart upstream.
func exitDeltaEarlyPath(step int, path string) string {
	return fmt.Sprintf("!%010d%s", step, path)
}
