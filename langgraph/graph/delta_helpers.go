package graph

import "fmt"

// exitDeltaTaskID creates a synthetic task ID for exit-mode delta writes.
// Embeds the superstep as a zero-padded prefix so ORDER BY task_id preserves
// chronological order. Go TaskIDs are 16-char hex strings (no hyphens, unlike
// Python UUIDs), so we prefix with the step rather than injecting into UUID
// groups. Mirrors the intent of Python exit_delta_task_id (_checkpoint.py:39-47).
func exitDeltaTaskID(step int, taskID string) string {
	return fmt.Sprintf("%08d%s", step, taskID)
}
