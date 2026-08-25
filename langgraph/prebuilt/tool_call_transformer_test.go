package prebuilt

import (
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// toolEvent builds a ToolEvent the way tests/test_tool_call_transformer.py's
// _tool_event helper (:38-68) builds ProtocolEvents.
func toolEvent(event, toolCallID string, mutate func(*ToolEvent)) ToolEvent {
	e := ToolEvent{Event: event, ToolCallID: toolCallID}
	if mutate != nil {
		mutate(&e)
	}
	return e
}

// Mirrors test_tool_started_yields_handle (:93-110).
func TestToolCallTransformerStartedYieldsHandle(t *testing.T) {
	tr := NewToolCallTransformer()
	if !tr.Process(toolEvent(ToolEventStarted, "tc1", func(e *ToolEvent) {
		e.ToolName = "echo"
		e.Input = map[string]any{"text": "hi"}
	})) {
		t.Fatal("Process returned false, want pass-through true")
	}
	streams := tr.Streams()
	if len(streams) != 1 {
		t.Fatalf("len(Streams()) = %d, want 1", len(streams))
	}
	h := streams[0]
	if h.ToolCallID != "tc1" || h.ToolName != "echo" || h.Input["text"] != "hi" || h.Completed {
		t.Errorf("handle = %+v, want tc1/echo/{text:hi}, not completed", h)
	}
}

// Mirrors test_delta_accumulates_on_active_stream (:112-119).
func TestToolCallTransformerDeltaAccumulates(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "tc1", func(e *ToolEvent) { e.ToolName = "echo" }))
	tr.Process(toolEvent(ToolEventOutputDelta, "tc1", func(e *ToolEvent) { e.Delta = "a" }))
	tr.Process(toolEvent(ToolEventOutputDelta, "tc1", func(e *ToolEvent) { e.Delta = "b" }))
	h := tr.Streams()[0]
	if len(h.Deltas) != 2 || h.Deltas[0] != "a" || h.Deltas[1] != "b" {
		t.Fatalf("Deltas = %v, want [a b]", h.Deltas)
	}
}

// Mirrors test_finish_closes_stream (:121-129).
func TestToolCallTransformerFinishClosesStream(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "tc1", func(e *ToolEvent) { e.ToolName = "echo" }))
	h := tr.Streams()[0]
	tr.Process(toolEvent(ToolEventFinished, "tc1", func(e *ToolEvent) { e.Output = "done" }))
	if !h.Completed || h.Output != "done" || h.Err != "" {
		t.Fatalf("handle = %+v, want completed with output \"done\"", h)
	}
	// A second finish for the same id is a no-op (the stream left `active`).
	tr.Process(toolEvent(ToolEventFinished, "tc1", func(e *ToolEvent) { e.Output = "other" }))
	if h.Output != "done" {
		t.Fatalf("handle.Output = %v after duplicate finish, want \"done\"", h.Output)
	}
}

// Mirrors test_finish_unwraps_tool_message_output (:131-143).
func TestToolCallTransformerFinishUnwrapsToolMessage(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "tc1", func(e *ToolEvent) { e.ToolName = "echo" }))
	h := tr.Streams()[0]
	tr.Process(toolEvent(ToolEventFinished, "tc1", func(e *ToolEvent) {
		e.Output = messages.Tool("tc1", "done")
	}))
	if !h.Completed || h.Output != "done" {
		t.Fatalf("handle = %+v, want completed with unwrapped output \"done\"", h)
	}
}

// Mirrors test_finish_unwraps_serialized_tool_message_output (:145-165).
func TestToolCallTransformerFinishUnwrapsSerializedToolMessage(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "tc1", func(e *ToolEvent) { e.ToolName = "echo" }))
	h := tr.Streams()[0]
	tr.Process(toolEvent(ToolEventFinished, "tc1", func(e *ToolEvent) {
		e.Output = map[string]any{
			"lc":   1,
			"type": "constructor",
			"id":   []any{"langchain_core", "messages", "ToolMessage"},
			"kwargs": map[string]any{
				"content":      "serialized done",
				"tool_call_id": "tc1",
			},
		}
	}))
	if !h.Completed || h.Output != "serialized done" {
		t.Fatalf("handle = %+v, want completed with unwrapped output \"serialized done\"", h)
	}
}

// A non-ToolMessage output passes through untouched (_normalize_tool_output's
// fallthrough, :41). The extra cases cover normalizeToolOutput's remaining
// arms for the coverage gate: a non-tool messages.Message, a map that is not a
// serialized constructor payload, and a constructor payload missing kwargs.
func TestToolCallTransformerFinishPassesThroughPlainOutput(t *testing.T) {
	cases := map[string]any{
		"plain scalar":        42,
		"non-tool message":    messages.AI("not a tool message"),
		"non-constructor map": map[string]any{"type": "not-a-constructor", "data": 1},
		"constructor without kwargs": map[string]any{
			"type": "constructor",
			"id":   []any{"langchain_core", "messages", "ToolMessage"},
		},
	}
	for name, output := range cases {
		t.Run(name, func(t *testing.T) {
			tr := NewToolCallTransformer()
			tr.Process(toolEvent(ToolEventStarted, "tc1", nil))
			h := tr.Streams()[0]
			tr.Process(toolEvent(ToolEventFinished, "tc1", func(e *ToolEvent) { e.Output = output }))
			if !reflect.DeepEqual(h.Output, output) {
				t.Fatalf("handle.Output = %#v, want %#v (pass-through)", h.Output, output)
			}
		})
	}
}

// Mirrors test_error_closes_stream (:167-175).
func TestToolCallTransformerErrorClosesStream(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "tc1", func(e *ToolEvent) { e.ToolName = "boom" }))
	h := tr.Streams()[0]
	tr.Process(toolEvent(ToolEventError, "tc1", func(e *ToolEvent) { e.Message = "nope" }))
	if !h.Completed || h.Output != nil || h.Err != "nope" {
		t.Fatalf("handle = %+v, want completed with error \"nope\"", h)
	}
}

// Mirrors test_concurrent_tool_calls_do_not_bleed (:177-190).
func TestToolCallTransformerConcurrentCallsDoNotBleed(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "a", func(e *ToolEvent) { e.ToolName = "t" }))
	tr.Process(toolEvent(ToolEventStarted, "b", func(e *ToolEvent) { e.ToolName = "t" }))
	tr.Process(toolEvent(ToolEventOutputDelta, "a", func(e *ToolEvent) { e.Delta = "A1" }))
	tr.Process(toolEvent(ToolEventOutputDelta, "b", func(e *ToolEvent) { e.Delta = "B1" }))
	tr.Process(toolEvent(ToolEventOutputDelta, "a", func(e *ToolEvent) { e.Delta = "A2" }))
	streams := tr.Streams()
	if len(streams) != 2 {
		t.Fatalf("len(Streams()) = %d, want 2", len(streams))
	}
	a, b := streams[0], streams[1]
	if len(a.Deltas) != 2 || a.Deltas[0] != "A1" || a.Deltas[1] != "A2" {
		t.Errorf("a.Deltas = %v, want [A1 A2]", a.Deltas)
	}
	if len(b.Deltas) != 1 || b.Deltas[0] != "B1" {
		t.Errorf("b.Deltas = %v, want [B1]", b.Deltas)
	}
}

// Mirrors test_out_of_scope_event_skipped (:199-226): a subgraph-scoped event
// is not projected by the root transformer, but still passes through.
func TestToolCallTransformerOutOfScopeSkipped(t *testing.T) {
	tr := NewToolCallTransformer()
	if !tr.Process(toolEvent(ToolEventStarted, "tc1", func(e *ToolEvent) {
		e.ToolName = "inner_echo"
		e.Namespace = []string{"child:abc"}
	})) {
		t.Fatal("Process returned false, want pass-through true")
	}
	if got := tr.Streams(); len(got) != 0 {
		t.Fatalf("Streams() = %v, want none (out-of-scope event)", got)
	}
}

// Mirrors test_in_scope_event_projected_when_scope_set (:228-267).
func TestToolCallTransformerScopedProjection(t *testing.T) {
	tr := NewToolCallTransformer("child:abc")
	atScope := toolEvent(ToolEventStarted, "tc1", func(e *ToolEvent) {
		e.ToolName = "echo"
		e.Namespace = []string{"child:abc"}
	})
	tr.Process(atScope)
	deeper := toolEvent(ToolEventStarted, "tc2", func(e *ToolEvent) {
		e.ToolName = "grandchild"
		e.Namespace = []string{"child:abc", "grand:xyz"}
	})
	tr.Process(deeper)
	root := toolEvent(ToolEventStarted, "tc3", func(e *ToolEvent) {
		e.ToolName = "root_tool"
	})
	tr.Process(root)
	streams := tr.Streams()
	if len(streams) != 1 || streams[0].ToolCallID != "tc1" {
		t.Fatalf("Streams() = %+v, want exactly the tc1 handle", streams)
	}
}

// Events without a tool_call_id are ignored (process:121-123).
func TestToolCallTransformerNoToolCallIDIgnored(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "", func(e *ToolEvent) { e.ToolName = "echo" }))
	if got := tr.Streams(); len(got) != 0 {
		t.Fatalf("Streams() = %v, want none", got)
	}
}

// Deltas/finishes/errors for unknown ids are no-ops (process:135-146).
func TestToolCallTransformerUnknownIDNoOp(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventOutputDelta, "ghost", func(e *ToolEvent) { e.Delta = "x" }))
	tr.Process(toolEvent(ToolEventFinished, "ghost", func(e *ToolEvent) { e.Output = "x" }))
	tr.Process(toolEvent(ToolEventError, "ghost", func(e *ToolEvent) { e.Message = "x" }))
	if got := tr.Streams(); len(got) != 0 {
		t.Fatalf("Streams() = %v, want none", got)
	}
}

// Mirrors finalize() (:152-157): still-active streams complete with nil output.
func TestToolCallTransformerFinalize(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "tc1", nil))
	h := tr.Streams()[0]
	tr.Finalize()
	if !h.Completed || h.Output != nil || h.Err != "" {
		t.Fatalf("handle = %+v, want completed with nil output after Finalize", h)
	}
	// Finalize clears active state: a late finish is a no-op.
	tr.Process(toolEvent(ToolEventFinished, "tc1", func(e *ToolEvent) { e.Output = "late" }))
	if h.Output != nil {
		t.Fatalf("handle.Output = %v after late finish, want nil", h.Output)
	}
}

// Mirrors fail() (:159-165): still-active streams fail with the error text.
func TestToolCallTransformerFail(t *testing.T) {
	tr := NewToolCallTransformer()
	tr.Process(toolEvent(ToolEventStarted, "tc1", nil))
	tr.Process(toolEvent(ToolEventStarted, "tc2", nil))
	tr.Process(toolEvent(ToolEventFinished, "tc2", func(e *ToolEvent) { e.Output = "ok" }))
	tr.Fail(errors.New("run exploded"))
	streams := tr.Streams()
	if streams[0].Err != "run exploded" || !streams[0].Completed {
		t.Errorf("streams[0] = %+v, want failed with \"run exploded\"", streams[0])
	}
	if streams[1].Err != "" || streams[1].Output != "ok" {
		t.Errorf("streams[1] = %+v, want untouched completed stream", streams[1])
	}
}
