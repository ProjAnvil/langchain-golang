package prebuilt

// ToolCallStream is a scoped view of a single tool call's lifecycle, mirroring
// Python's `langgraph.prebuilt._tool_call_stream.ToolCallStream`. A
// ToolCallTransformer creates one handle per "tool-started" event and mutates
// it as further events for the same tool_call_id arrive:
//
//   - ToolCallID, ToolName, Input are stable from the start event.
//   - Deltas accumulates each "tool-output-delta" payload in arrival order.
//   - Output is the terminal payload from "tool-finished" (nil on failure or
//     while in flight), with ToolMessage outputs unwrapped to their content.
//   - Err is the terminal error string from "tool-error" ("" otherwise).
//   - Completed is true once a terminal event has been observed (or the
//     transformer was finalized/failed with the call still in flight).
//
// Divergence: Python exposes `output_deltas` as a live StreamChannel for
// push-based sync/async iteration, because it lives inside the stream mux.
// The Go port has no stream-mux subsystem (langgraph/graph/events.go is a
// simpler NodeEventSink model), so Deltas is a plain slice consumers read
// after (or while) feeding events.
type ToolCallStream struct {
	ToolCallID string
	ToolName   string
	Input      map[string]any
	Deltas     []any
	Output     any
	Err        string
	Completed  bool
}
