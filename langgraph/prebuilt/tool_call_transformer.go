package prebuilt

import (
	"slices"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Tools-channel event names, mirroring the `data.event` values of Python's
// `tools` stream-channel protocol events consumed by ToolCallTransformer
// (_tool_call_transformer.py:127-146).
const (
	ToolEventStarted     = "tool-started"
	ToolEventOutputDelta = "tool-output-delta"
	ToolEventFinished    = "tool-finished"
	ToolEventError       = "tool-error"
)

// ToolEvent is the Go projection of one Python `tools`-channel ProtocolEvent's
// `params` payload (tests/test_tool_call_transformer.py:38-68): Namespace is
// `params.namespace`, Event/ToolCallID/ToolName/Input/Delta/Output/Message are
// the `params.data` fields (only the fields relevant to each event kind are
// populated, mirroring `_tool_event`).
type ToolEvent struct {
	// Namespace is the subgraph scope the event was emitted at; empty is the
	// root graph.
	Namespace []string
	// Event is one of the ToolEvent* constants.
	Event string
	// ToolCallID identifies the tool call; events without one are ignored.
	ToolCallID string
	// ToolName and Input are set on ToolEventStarted.
	ToolName string
	Input    map[string]any
	// Delta is set on ToolEventOutputDelta.
	Delta any
	// Output is set on ToolEventFinished.
	Output any
	// Message is set on ToolEventError.
	Message string
}

// ToolCallTransformer projects `tools`-channel events into ToolCallStream
// handles, mirroring Python's
// `langgraph.prebuilt._tool_call_transformer.ToolCallTransformer`
// (_tool_call_transformer.py:44). Each "tool-started" event spawns a handle
// appended to Streams(); "tool-output-delta" events append to that handle's
// Deltas; "tool-finished"/"tool-error" complete it.
//
// Divergence: Python's transformer is a native StreamTransformer plug-in for
// the langgraph stream mux (with pump wiring and a StreamChannel log). The Go
// port has no stream-mux subsystem, so this is a standalone event processor:
// callers feed ToolEvents (e.g. reconstructed from the wire protocol) and read
// the handles from Streams(). Process always returns true, preserving Python's
// pass-through contract (raw `tools` events flow to wire consumers untouched).
// Python's end-to-end `graph.stream_events(transformers=[ToolCallTransformer])`
// wiring (TestToolCallTransformerEndToEnd) has no Go equivalent yet.
//
// ToolCallTransformer is not goroutine-safe; feed events from a single
// goroutine (matching the single-consumer mux loop in Python).
type ToolCallTransformer struct {
	scope  []string
	log    []*ToolCallStream
	active map[string]*ToolCallStream
}

// NewToolCallTransformer creates a transformer for the given scope (empty =
// root graph). Only events whose Namespace exactly equals scope are projected
// (_tool_call_transformer.py:113-118); all events pass through regardless.
func NewToolCallTransformer(scope ...string) *ToolCallTransformer {
	return &ToolCallTransformer{
		scope:  append([]string(nil), scope...),
		active: make(map[string]*ToolCallStream),
	}
}

// Process folds one event into the projection, returning true (pass-through),
// mirroring `process` (_tool_call_transformer.py:109-150).
func (t *ToolCallTransformer) Process(event ToolEvent) bool {
	if !slices.Equal(event.Namespace, t.scope) {
		return true
	}
	if event.ToolCallID == "" {
		return true
	}
	switch event.Event {
	case ToolEventStarted:
		stream := &ToolCallStream{
			ToolCallID: event.ToolCallID,
			ToolName:   event.ToolName,
			Input:      event.Input,
		}
		t.active[event.ToolCallID] = stream
		t.log = append(t.log, stream)
	case ToolEventOutputDelta:
		if stream, ok := t.active[event.ToolCallID]; ok {
			stream.Deltas = append(stream.Deltas, event.Delta)
		}
	case ToolEventFinished:
		if stream, ok := t.active[event.ToolCallID]; ok {
			delete(t.active, event.ToolCallID)
			stream.Output = normalizeToolOutput(event.Output)
			stream.Completed = true
		}
	case ToolEventError:
		if stream, ok := t.active[event.ToolCallID]; ok {
			delete(t.active, event.ToolCallID)
			stream.Err = event.Message
			stream.Completed = true
		}
	}
	return true
}

// Streams returns the projected handles in start order (the contents of
// Python's `tool_calls` log). The slice is a copy; the handles are shared and
// keep mutating as further events are processed.
func (t *ToolCallTransformer) Streams() []*ToolCallStream {
	return append([]*ToolCallStream(nil), t.log...)
}

// Finalize completes any still-active streams with a nil output, mirroring
// `finalize` (_tool_call_transformer.py:152-157).
func (t *ToolCallTransformer) Finalize() {
	for id, stream := range t.active {
		if !stream.Completed {
			stream.Completed = true
		}
		delete(t.active, id)
	}
}

// Fail fails any still-active streams with err's message, mirroring `fail`
// (_tool_call_transformer.py:159-165). A nil err is treated as an empty
// message.
func (t *ToolCallTransformer) Fail(err error) {
	message := ""
	if err != nil {
		message = err.Error()
	}
	for id, stream := range t.active {
		if !stream.Completed {
			stream.Err = message
			stream.Completed = true
		}
		delete(t.active, id)
	}
}

// normalizeToolOutput mirrors `_normalize_tool_output`
// (_tool_call_transformer.py:34-41): a ToolMessage unwraps to its content, as
// does a serialized ToolMessage payload
// ({"type": "constructor", "id": [..., "ToolMessage"], "kwargs": {...}});
// anything else passes through untouched.
func normalizeToolOutput(output any) any {
	switch out := output.(type) {
	case messages.Message:
		if out.Role == messages.RoleTool {
			return out.Content
		}
		return output
	case map[string]any:
		typ, _ := out["type"].(string)
		id, _ := out["id"].([]any)
		if typ == "constructor" && len(id) > 0 && id[len(id)-1] == "ToolMessage" {
			if kwargs, ok := out["kwargs"].(map[string]any); ok {
				return kwargs["content"]
			}
		}
		return output
	default:
		return output
	}
}
