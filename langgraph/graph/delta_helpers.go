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

// exitDeltaTaskID creates a synthetic task ID for exit-mode delta writes.
// Embeds the superstep as a zero-padded prefix so ORDER BY task_id preserves
// chronological order. Go TaskIDs are 16-char hex strings (no hyphens, unlike
// Python UUIDs), so we prefix with the step rather than injecting into UUID
// groups. Mirrors the intent of Python exit_delta_task_id (_checkpoint.py:39-47).
func exitDeltaTaskID(step int, taskID string) string {
	return fmt.Sprintf("%08d%s", step, taskID)
}
