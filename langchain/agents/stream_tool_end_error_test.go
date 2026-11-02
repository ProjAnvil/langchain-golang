package agents

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// Tests pinning the tool_end stream-event payload for tool errors. The v0.12.3
// layering move (handleToolErrors applied at the runOne level, outside the
// wrap chain) left the tool_end emit inside the wrap chain, so default-converted
// tool errors emitted a nil ToolResult. The fix emits tool_end from the tools
// node's outcome consumption, after conversion. These tests lock the contract.

// slowEchoTool sleeps before answering so concurrent sibling calls finish in a
// deterministic order (fast call would otherwise complete first).
func slowEchoTool(t *testing.T, name string, delay time.Duration) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewSimple(name, "echoes its input slowly",
		func(ctx context.Context, input string) (coretools.Result, error) {
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return coretools.Result{}, ctx.Err()
			}
			return coretools.Result{Content: name + ":" + input}, nil
		})
	if err != nil {
		t.Fatalf("new %s: %v", name, err)
	}
	return tool
}

// failingToolStreamModel scripts one failing_tool call followed by a final
// answer, with per-response stream chunks (StreamEvents needs the Stream path).
func failingToolStreamModel() *streamSequenceModel {
	call := messages.ToolCall{ID: "1", Name: "failing_tool", Args: map[string]any{"value": "x"}}
	return &streamSequenceModel{
		responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{call}},
			messages.AI("done"),
		},
		streamChunks: [][]messages.Message{
			{{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{call}}},
			{messages.AI("do"), messages.AI("ne")},
		},
	}
}

// toolEndEvents returns the StreamToolEnd events in emission order.
func toolEndEvents(events []StreamEvent) []StreamEvent {
	var out []StreamEvent
	for _, ev := range events {
		if ev.Type == StreamToolEnd {
			out = append(out, ev)
		}
	}
	return out
}

// RED case 1: a default-converted tool error must reach tool_end with the
// converted content and the error status — not a nil ToolResult.
func TestStreamEventsToolEndCarriesDefaultConvertedError(t *testing.T) {
	agentStream, err := CreateAgent(failingToolStreamModel(), []coretools.Tool{newFailingTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	stream, err := agentStream.StreamEvents(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)

	ends := toolEndEvents(events)
	if len(ends) != 1 {
		t.Fatalf("expected exactly one tool_end, got %d (types=%v)", len(ends), eventTypes(events))
	}
	got := ends[0].ToolResult
	if got == nil {
		t.Fatalf("tool_end ToolResult is nil; want the default-converted error content")
	}
	// DefaultHandleToolErrors template, verbatim.
	if want := "Error: secret detail: x\n Please fix your mistakes."; got["content"] != want {
		t.Fatalf("tool_end content = %q, want %q", got["content"], want)
	}
	if got["status"] != "error" {
		t.Fatalf("tool_end status = %v, want error", got["status"])
	}
}

// RED case 2: a declined error (ToolErrorMiddleware returns not-handled) fails
// the run and emits NO tool_end — Python emits no on_tool_end for failing
// tools, and a declined error never converts to a message.
func TestStreamEventsDeclinedToolErrorOmitsToolEnd(t *testing.T) {
	agentStream, err := CreateAgent(failingToolStreamModel(),
		[]coretools.Tool{newFailingTool(t)},
		WithAgentMiddleware(decliningToolErrorMiddleware(t)),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	stream, err := agentStream.StreamEvents(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)

	if ends := toolEndEvents(events); len(ends) != 0 {
		t.Fatalf("declined error must not emit tool_end, got %d (result=%v)", len(ends), ends[0].ToolResult)
	}
	if countType(events, StreamToolStart) != 1 {
		t.Fatalf("expected the paired tool_start to remain, types=%v", eventTypes(events))
	}
}

// RED case 3: with concurrent sibling calls, tool_end events follow the CALL
// order (the tools node emits them from the ordered outcomes), not the
// completion order.
func TestStreamEventsMultiToolEndsFollowCallOrder(t *testing.T) {
	calls := []messages.ToolCall{
		{ID: "slow", Name: "slow_echo", Args: map[string]any{"input": "a"}},
		{ID: "fast", Name: "fast_echo", Args: map[string]any{"input": "b"}},
	}
	model := &streamSequenceModel{
		responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: calls},
			messages.AI("done"),
		},
		streamChunks: [][]messages.Message{
			{{Role: messages.RoleAI, ToolCalls: calls}},
			{messages.AI("done")},
		},
	}
	agentStream, err := CreateAgent(model, []coretools.Tool{
		slowEchoTool(t, "slow_echo", 80*time.Millisecond),
		slowEchoTool(t, "fast_echo", 0),
	})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	stream, err := agentStream.StreamEvents(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)

	ends := toolEndEvents(events)
	if len(ends) != 2 {
		t.Fatalf("expected two tool_end events, got %d (types=%v)", len(ends), eventTypes(events))
	}
	// Completion order is fast-first (slow sleeps 80ms); call order is slow-first.
	if !strings.Contains(ends[0].ToolResult["content"].(string), "slow_echo:") {
		t.Fatalf("tool_end order must follow call order: first end = %v, want slow_echo", ends[0].ToolResult)
	}
	if !strings.Contains(ends[1].ToolResult["content"].(string), "fast_echo:") {
		t.Fatalf("tool_end order must follow call order: second end = %v, want fast_echo", ends[1].ToolResult)
	}
}

// REGRESSION PIN: the lifecycle skeleton around a default-converted error
// stays intact (tool_start < tool_end < tools node_end < post-tool model
// node_start) wherever the emit physically happens.
func TestStreamEventsToolEndErrorLifecycleOrdering(t *testing.T) {
	agentStream, err := CreateAgent(failingToolStreamModel(), []coretools.Tool{newFailingTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	stream, err := agentStream.StreamEvents(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)
	types := eventTypes(events)

	idxOf := func(t StreamEventType) int {
		for i, ty := range types {
			if ty == t {
				return i
			}
		}
		return -1
	}
	ts, te := idxOf(StreamToolStart), idxOf(StreamToolEnd)
	secondNodeStart := idxFrom(types, StreamNodeStart, 2)
	if ts < 0 || te < 0 || !(ts < te) {
		t.Fatalf("tool_start/tool_end ordering wrong: start(%d) end(%d), types=%v", ts, te, types)
	}
	if secondNodeStart < 0 || te >= secondNodeStart {
		t.Fatalf("tool_end(%d) must precede the post-tool model node_start(%d), types=%v", te, secondNodeStart, types)
	}
	// The run itself must still complete (default conversion keeps it alive):
	// the terminal StreamEnd event carries the final state.
	var ended bool
	for _, ev := range events {
		if ev.Type == StreamEnd {
			ended = true
		}
	}
	if !ended {
		t.Fatalf("default-converted error must not fail the run, missing StreamEnd, types=%v", types)
	}
}

// REGRESSION PIN: ToolErrorMiddleware-converted errors keep their content on
// tool_end (this path converted inside the wrap chain even before the fix).
func TestStreamEventsToolEndAfterMiddlewareConversion(t *testing.T) {
	agentStream, err := CreateAgent(failingToolStreamModel(),
		[]coretools.Tool{newFailingTool(t)},
		WithAgentMiddleware(handlingToolErrorMiddleware(t)),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	stream, err := agentStream.StreamEvents(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)

	ends := toolEndEvents(events)
	if len(ends) != 1 {
		t.Fatalf("expected exactly one tool_end, got %d (types=%v)", len(ends), eventTypes(events))
	}
	if got := ends[0].ToolResult["content"]; got != "handled: failing_tool" {
		t.Fatalf("tool_end content = %v, want handled: failing_tool", got)
	}
	if ends[0].ToolResult["status"] != "error" {
		t.Fatalf("tool_end status = %v, want error", ends[0].ToolResult["status"])
	}
}

// decliningToolErrorMiddleware declines every error (propagation path).
func decliningToolErrorMiddleware(t *testing.T) *middleware.ToolErrorMiddleware {
	t.Helper()
	mw, err := middleware.NewToolErrorMiddleware(func(error, middleware.ToolCallRequest) (any, bool) {
		return nil, false
	})
	if err != nil {
		t.Fatalf("new declining middleware: %v", err)
	}
	return mw
}

// handlingToolErrorMiddleware converts every error to named content.
func handlingToolErrorMiddleware(t *testing.T) *middleware.ToolErrorMiddleware {
	t.Helper()
	mw, err := middleware.NewToolErrorMiddleware(func(_ error, req middleware.ToolCallRequest) (any, bool) {
		return "handled: " + req.ToolCall.Name, true
	})
	if err != nil {
		t.Fatalf("new handling middleware: %v", err)
	}
	return mw
}
