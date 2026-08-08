package graph

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
)

// TaskID computes the deterministic identity of a planned task: an fnv-1a
// 64-bit hash over the owning checkpoint's ID, the superstep the task is
// planned for, the node name, and the JSON encoding of the task's Send
// argument (falling back to a Go-syntax dump when the argument is not
// JSON-marshalable), hex-encoded. Same-node fan-out tasks with different
// arguments hash differently, so resume bookkeeping (M2 Task 4) can key on
// task identity rather than node name.
func TaskID(cpID string, step int, node string, arg map[string]any) string {
	h := fnv.New64a()
	fmt.Fprintf(h, "%s\x00%d\x00%s\x00", cpID, step, node)
	if data, err := json.Marshal(arg); err == nil {
		_, _ = h.Write(data)
	} else {
		fmt.Fprintf(h, "%#v", arg)
	}
	return fmt.Sprintf("%016x", h.Sum64())
}

// FnTaskID computes the deterministic identity of a functional-API task
// invocation: an fnv-1a 64-bit hash over the base checkpoint's ID, the
// checkpoint namespace, the run's step, the task name, the parent call path,
// and the per-path call index, hex-encoded. It is the Go analogue of
// Python's PUSH/Call task ID (`pregel/_algo.py:834-842`:
// task_id_func(checkpoint_id, checkpoint_ns, step, name, PUSH, parent_path,
// call_idx)). Same-task calls from a loop hash differently by callIdx, so
// checkpoint replay keys on call identity rather than task name.
func FnTaskID(cpID, ns string, step int, name, parentPath string, callIdx int) string {
	h := fnv.New64a()
	fmt.Fprintf(h, "%s\x00%s\x00%d\x00%s\x00%s\x00%d\x00", cpID, ns, step, name, parentPath, callIdx)
	return fmt.Sprintf("%016x", h.Sum64())
}
