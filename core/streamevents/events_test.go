package streamevents

import (
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestChatModelStreamTextDeltas(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": "Hi"}},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": " there"}},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	if got := stream.TextDeltas(); !reflect.DeepEqual(got, []string{"Hi", " there"}) {
		t.Fatalf("text deltas: got %#v", got)
	}
	if got := stream.Text(); got != "Hi there" {
		t.Fatalf("text: got %q", got)
	}
	if !stream.Done() {
		t.Fatal("stream should be done")
	}
}

func TestChatModelStreamReasoningDeltas(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "reasoning-delta", Value: map[string]any{"reasoning": "think"}},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	if got := stream.ReasoningDeltas(); !reflect.DeepEqual(got, []string{"think"}) {
		t.Fatalf("reasoning deltas: got %#v", got)
	}
	if got := stream.Reasoning(); got != "think" {
		t.Fatalf("reasoning: got %q", got)
	}
}

func TestChatModelStreamToolCallFinish(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockFinish,
		Index: 0,
		Content: messages.ToolCallBlock{
			ID:   "tc1",
			Name: "search",
			Args: map[string]any{"q": "test"},
		},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	calls := stream.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "search" || calls[0].Args["q"] != "test" {
		t.Fatalf("tool calls: %+v", calls)
	}
	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if len(output.ToolCalls) != 1 || output.ToolCalls[0].ID != "tc1" {
		t.Fatalf("output tool calls: %+v", output.ToolCalls)
	}
}

func TestChatModelStreamFinishReconcilesTextPerBlock(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": "aaa"}},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 1,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": "bb"}},
	})
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   0,
		Content: messages.TextBlock{Text: "XXX"},
	})
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   1,
		Content: messages.TextBlock{Text: "bb"},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	if got := stream.Text(); got != "XXXbb" {
		t.Fatalf("text: got %q", got)
	}
	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if got := []any{messages.BlockToMap(output.ContentBlocks[0])["text"], messages.BlockToMap(output.ContentBlocks[1])["text"]}; !reflect.DeepEqual(got, []any{"XXX", "bb"}) {
		t.Fatalf("blocks: %+v", output.ContentBlocks)
	}
}

func TestChatModelStreamInterleavesTextToolAndReasoningBlocks(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": "before"}},
	})
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   0,
		Content: messages.TextBlock{Text: "before"},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockFinish,
		Index: 1,
		Content: messages.ToolCallBlock{
			ID:   "tc1",
			Name: "search",
			Args: map[string]any{"q": "x"},
		},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 2,
		Delta: messages.NonStandardContentBlock{Type: "reasoning-delta", Value: map[string]any{"reasoning": "think"}},
	})
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   2,
		Content: messages.ReasoningBlock{Reasoning: "thinking"},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	gotTypes := []any{
		messages.BlockToMap(output.ContentBlocks[0])["type"],
		messages.BlockToMap(output.ContentBlocks[1])["type"],
		messages.BlockToMap(output.ContentBlocks[2])["type"],
	}
	if !reflect.DeepEqual(gotTypes, []any{"text", "tool_call", "reasoning"}) {
		t.Fatalf("block types: %+v", output.ContentBlocks)
	}
	if stream.Reasoning() != "thinking" {
		t.Fatalf("reasoning: %q", stream.Reasoning())
	}
}

func TestChatModelStreamSweepsMalformedToolCallChunk(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{
			Type: "tool_call_chunk",
			Value: map[string]any{
				"id":   "call_1",
				"name": "search",
				"args": `{"q": `,
			},
		},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if len(output.ToolCalls) != 0 || len(output.InvalidToolCalls) != 1 {
		t.Fatalf("output calls: tool=%+v invalid=%+v", output.ToolCalls, output.InvalidToolCalls)
	}
	if output.InvalidToolCalls[0].Name != "search" {
		t.Fatalf("invalid call: %+v", output.InvalidToolCalls[0])
	}
}

func TestChatModelStreamFailPropagatesToOutput(t *testing.T) {
	stream := NewChatModelStream()
	stream.Fail(errors.New("boom"))

	if !stream.Done() {
		t.Fatal("stream should be done")
	}
	if _, err := stream.Output(); err == nil || err.Error() != "boom" {
		t.Fatalf("output error: %v", err)
	}
}

func TestChatModelStreamEventsReplayIsIsolated(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event:   EventContentBlockDelta,
		Index:   0,
		Delta:   messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": "hi"}},
		Content: messages.TextBlock{Text: "ignored"},
		Extra:   map[string]any{"k": "v"},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	replay := stream.Events()
	if len(replay) != 2 {
		t.Fatalf("replay length: %d", len(replay))
	}
	if replay[0].Event != EventContentBlockDelta || replay[0].Extra["k"] != "v" {
		t.Fatalf("replay event: %+v", replay[0])
	}

	// Mutating the replayed copy must not affect the recorded events.
	replay[0].Extra["k"] = "mutated"
	fresh := stream.Events()
	if fresh[0].Extra["k"] != "v" {
		t.Fatalf("replay not isolated: %+v", fresh[0].Extra)
	}
}

func TestChatModelStreamDispatchIgnoredAfterDone(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": "hi"}},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": "late"}},
	})

	if got := stream.Text(); got != "hi" {
		t.Fatalf("text after late dispatch: %q", got)
	}
	if got := stream.Events(); len(got) != 2 {
		t.Fatalf("events after late dispatch: %d", len(got))
	}
}

func TestChatModelStreamOutputBeforeDone(t *testing.T) {
	stream := NewChatModelStream()
	if _, err := stream.Output(); err == nil || err.Error() != "stream is not done" {
		t.Fatalf("expected not-done error, got %v", err)
	}
}

func TestChatModelStreamEmptyProjections(t *testing.T) {
	stream := NewChatModelStream()
	if got := stream.Text(); got != "" {
		t.Fatalf("text: %q", got)
	}
	if got := stream.Reasoning(); got != "" {
		t.Fatalf("reasoning: %q", got)
	}
	if got := stream.ToolCalls(); got != nil {
		t.Fatalf("tool calls: %+v", got)
	}
	if got := stream.InvalidToolCalls(); got != nil {
		t.Fatalf("invalid tool calls: %+v", got)
	}
}

func TestChatModelStreamPushDeltaIgnoresEmptyAndUnknown(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{Event: EventContentBlockDelta, Index: 0, Delta: nil})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": ""}},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "reasoning-delta", Value: map[string]any{"reasoning": ""}},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "unknown-delta", Value: map[string]any{"x": "y"}},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	if got := stream.TextDeltas(); len(got) != 0 {
		t.Fatalf("text deltas: %#v", got)
	}
	if got := stream.ReasoningDeltas(); len(got) != 0 {
		t.Fatalf("reasoning deltas: %#v", got)
	}
	if got := stream.Text(); got != "" {
		t.Fatalf("text: %q", got)
	}
}

func TestChatModelStreamToolCallChunksAccumulateAndSweep(t *testing.T) {
	stream := NewChatModelStream()
	chunk := func(index int, value map[string]any) Event {
		return Event{
			Event: EventContentBlockDelta,
			Index: index,
			Delta: messages.NonStandardContentBlock{Type: "tool_call_chunk", Value: value},
		}
	}
	// Higher index first, to exercise index ordering in the sweep.
	stream.Dispatch(chunk(2, map[string]any{"name": "noop"}))
	stream.Dispatch(chunk(0, map[string]any{"id": "call_1", "name": "search", "args": `{"q":`}))
	// Later chunks must not overwrite the already-set id/name; args append.
	stream.Dispatch(chunk(0, map[string]any{"id": "other", "name": "other", "args": `"x"}`}))
	stream.Dispatch(Event{Event: EventMessageFinish})

	calls := stream.ToolCalls()
	if len(calls) != 2 {
		t.Fatalf("tool calls: %+v", calls)
	}
	if calls[0].ID != "call_1" || calls[0].Name != "search" || calls[0].Args["q"] != "x" {
		t.Fatalf("swept call[0]: %+v", calls[0])
	}
	if calls[1].Name != "noop" || calls[1].Args != nil {
		t.Fatalf("swept call[1]: %+v", calls[1])
	}

	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	gotTypes := []any{
		messages.BlockToMap(output.ContentBlocks[0])["type"],
		messages.BlockToMap(output.ContentBlocks[1])["type"],
	}
	if !reflect.DeepEqual(gotTypes, []any{"tool_call", "tool_call"}) {
		t.Fatalf("block types: %+v", output.ContentBlocks)
	}
}

func TestChatModelStreamInvalidToolCallFinish(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockFinish,
		Index: 0,
		Content: messages.InvalidToolCallBlock{
			ID:    "itc1",
			Name:  "search",
			Args:  "{bad",
			Error: "parse error",
		},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	invalid := stream.InvalidToolCalls()
	if len(invalid) != 1 || invalid[0].ID != "itc1" || invalid[0].Name != "search" {
		t.Fatalf("invalid tool calls: %+v", invalid)
	}
	if got := stream.ToolCalls(); len(got) != 0 {
		t.Fatalf("tool calls: %+v", got)
	}

	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if len(output.InvalidToolCalls) != 1 || output.InvalidToolCalls[0].ID != "itc1" {
		t.Fatalf("output invalid calls: %+v", output.InvalidToolCalls)
	}
	if got := messages.BlockToMap(output.ContentBlocks[0])["type"]; got != "invalid_tool_call" {
		t.Fatalf("block type: %v", got)
	}
}

func TestChatModelStreamFinishBlockEdgeCases(t *testing.T) {
	stream := NewChatModelStream()
	// Nil content finish is a no-op.
	stream.Dispatch(Event{Event: EventContentBlockFinish, Index: 3, Content: nil})
	// Empty text finish falls back to accumulated deltas for the block.
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{Type: "text-delta", Value: map[string]any{"text": "abc"}},
	})
	stream.Dispatch(Event{Event: EventContentBlockFinish, Index: 0, Content: messages.TextBlock{Text: ""}})
	// Empty reasoning finish with no deltas still records the block.
	stream.Dispatch(Event{Event: EventContentBlockFinish, Index: 1, Content: messages.ReasoningBlock{Reasoning: ""}})
	// Unknown block types are recorded as-is via the default branch.
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   2,
		Content: messages.ServerToolCall{ID: "stc1", Name: "web", Args: map[string]any{"q": "x"}},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	if got := stream.Text(); got != "abc" {
		t.Fatalf("text: %q", got)
	}
	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if len(output.ContentBlocks) != 3 {
		t.Fatalf("blocks: %+v", output.ContentBlocks)
	}
	if got := messages.BlockToMap(output.ContentBlocks[0])["text"]; got != "abc" {
		t.Fatalf("text block: %v", got)
	}
	reasoningMap := messages.BlockToMap(output.ContentBlocks[1])
	if got := reasoningMap["type"]; got != "reasoning" {
		t.Fatalf("reasoning block type: %v", got)
	}
	if _, ok := reasoningMap["reasoning"]; ok {
		t.Fatalf("reasoning block should have no reasoning key: %+v", reasoningMap)
	}
	if got := messages.BlockToMap(output.ContentBlocks[2])["type"]; got != "server_tool_call" {
		t.Fatalf("server tool call block type: %v", got)
	}
}

// toolCallDelta emits a #41257 tool-call-delta event carrying only an args
// slice.
func toolCallDelta(index int, args string) Event {
	return Event{
		Event: EventContentBlockDelta,
		Index: index,
		Delta: messages.NonStandardContentBlock{
			Type:  ToolCallDeltaType,
			Value: map[string]any{"args": args},
		},
	}
}

// TestChatModelStreamToolCallDeltaSeedsIDNameFromStart covers the #41246
// consumer shape: the content-block-start skeleton carries the call's id and
// name, tool-call-delta events carry only args slices, and the assembled call
// keeps the start-seeded identity.
func TestChatModelStreamToolCallDeltaSeedsIDNameFromStart(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{Event: EventMessageStart})
	stream.Dispatch(Event{
		Event: EventContentBlockStart,
		Index: 0,
		Content: messages.ToolCallChunkBlock{
			ID:    "call_1",
			Name:  "search",
			Args:  "",
			Index: 0,
		},
	})
	stream.Dispatch(toolCallDelta(0, `{"q":`))
	stream.Dispatch(toolCallDelta(0, `"x"}`))
	stream.Dispatch(Event{
		Event: EventContentBlockFinish,
		Index: 0,
		Content: messages.ToolCallBlock{
			ID:   "call_1",
			Name: "search",
			Args: map[string]any{"q": "x"},
		},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	calls := stream.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls: %+v", calls)
	}
	if calls[0].ID != "call_1" || calls[0].Name != "search" || calls[0].Args["q"] != "x" {
		t.Fatalf("assembled call: %+v", calls[0])
	}

	// Deltas without a finish are swept at message-finish and keep the
	// start-seeded id/name plus the accumulated args.
	stream = NewChatModelStream()
	stream.Dispatch(Event{
		Event:   EventContentBlockStart,
		Index:   0,
		Content: messages.ToolCallChunkBlock{ID: "call_2", Name: "write"},
	})
	stream.Dispatch(toolCallDelta(0, `{"path": `))
	stream.Dispatch(toolCallDelta(0, `"a.py"}`))
	stream.Dispatch(Event{Event: EventMessageFinish})

	calls = stream.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("swept tool calls: %+v", calls)
	}
	if calls[0].ID != "call_2" || calls[0].Name != "write" || calls[0].Args["path"] != "a.py" {
		t.Fatalf("swept call: %+v", calls[0])
	}
}

// TestChatModelStreamToolCallDeltaWithoutStartLeftToFinish mirrors Python: a
// tool-call-delta whose content-block-start this stream never saw is left to
// content-block-finish, which carries the call.
func TestChatModelStreamToolCallDeltaWithoutStartLeftToFinish(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(toolCallDelta(0, `{"q":"x"}`))
	stream.Dispatch(Event{Event: EventMessageFinish})

	// The orphan slice is dropped rather than fabricated into a call.
	if got := stream.ToolCalls(); len(got) != 0 {
		t.Fatalf("tool calls from orphan delta: %+v", got)
	}
	if got := stream.InvalidToolCalls(); len(got) != 0 {
		t.Fatalf("invalid tool calls from orphan delta: %+v", got)
	}

	stream = NewChatModelStream()
	stream.Dispatch(toolCallDelta(0, `{"q":"x"}`))
	stream.Dispatch(Event{
		Event: EventContentBlockFinish,
		Index: 0,
		Content: messages.ToolCallBlock{
			ID:   "call_1",
			Name: "search",
			Args: map[string]any{"q": "x"},
		},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	calls := stream.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Args["q"] != "x" {
		t.Fatalf("finish-carried call: %+v", calls)
	}
}

// TestChatModelStreamToolCallDeltaLegacyFallback keeps the legacy
// tool_call_chunk delta shape working: id/name ride the first delta when no
// start skeleton was emitted, and a start-seeded identity is sticky.
func TestChatModelStreamToolCallDeltaLegacyFallback(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{
			Type: "tool_call_chunk",
			Value: map[string]any{
				"id":   "call_1",
				"name": "search",
				"args": `{"q":`,
			},
		},
	})
	// Legacy producers repeat (or change) id/name on every delta; the first
	// observation wins.
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{
			Type: "tool_call_chunk",
			Value: map[string]any{
				"id":   "other",
				"name": "other",
				"args": `"x"}`,
			},
		},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	calls := stream.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls: %+v", calls)
	}
	if calls[0].ID != "call_1" || calls[0].Name != "search" || calls[0].Args["q"] != "x" {
		t.Fatalf("legacy call: %+v", calls[0])
	}

	// A start-seeded call can also be extended by legacy deltas: the start's
	// id/name stick, args append across both shapes.
	stream = NewChatModelStream()
	stream.Dispatch(Event{
		Event:   EventContentBlockStart,
		Index:   0,
		Content: messages.ToolCallChunkBlock{ID: "call_2", Name: "write"},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.NonStandardContentBlock{
			Type: "tool_call_chunk",
			Value: map[string]any{
				"id":   "ignored",
				"name": "ignored",
				"args": `{"p":`,
			},
		},
	})
	stream.Dispatch(toolCallDelta(0, `"a"}`))
	stream.Dispatch(Event{Event: EventMessageFinish})

	calls = stream.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("mixed tool calls: %+v", calls)
	}
	if calls[0].ID != "call_2" || calls[0].Name != "write" || calls[0].Args["p"] != "a" {
		t.Fatalf("mixed call: %+v", calls[0])
	}
}

// TestChatModelStreamToolCallDeltaPartialJSONMidStream locks mid-stream
// tolerance: args slices routinely form invalid JSON until the final slice,
// so Dispatch must never parse them eagerly and projections must be safe to
// query mid-stream.
func TestChatModelStreamToolCallDeltaPartialJSONMidStream(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event:   EventContentBlockStart,
		Index:   0,
		Content: messages.ToolCallChunkBlock{ID: "call_1", Name: "write"},
	})
	for _, slice := range []string{`{"pat`, `h": "a`, `.py", "`, `x": 1}`} {
		stream.Dispatch(toolCallDelta(0, slice))
		// Mid-stream projections must not blow up on the partial JSON seen
		// so far, even though it cannot parse yet.
		if got := stream.Text(); got != "" {
			t.Fatalf("mid-stream text: %q", got)
		}
		if got := stream.ToolCalls(); len(got) != 0 {
			t.Fatalf("mid-stream tool calls: %+v", got)
		}
		if got := stream.Events(); len(got) == 0 {
			t.Fatal("mid-stream events should be recorded")
		}
	}
	stream.Dispatch(Event{Event: EventMessageFinish})

	calls := stream.ToolCalls()
	if len(calls) != 1 || calls[0].Args["path"] != "a.py" || calls[0].Args["x"] != float64(1) {
		t.Fatalf("assembled partial-stream call: %+v", calls)
	}

	// A stream that ends with args that never parse becomes an invalid call
	// at sweep time, not a mid-stream failure.
	stream = NewChatModelStream()
	stream.Dispatch(Event{
		Event:   EventContentBlockStart,
		Index:   0,
		Content: messages.ToolCallChunkBlock{ID: "call_2", Name: "broken"},
	})
	stream.Dispatch(toolCallDelta(0, `{"oops": `))
	stream.Dispatch(Event{Event: EventMessageFinish})

	if got := stream.ToolCalls(); len(got) != 0 {
		t.Fatalf("unparseable call should be invalid, got %+v", got)
	}
	if got := stream.InvalidToolCalls(); len(got) != 1 || got[0].Name != "broken" {
		t.Fatalf("invalid calls: %+v", got)
	}
}

func TestChatModelStreamUsageProjection(t *testing.T) {
	stream := NewChatModelStream()
	// Empty usage before message-finish.
	if got := stream.Usage(); got.TotalTokens != 0 {
		t.Fatalf("expected zero usage before finish, got %+v", got)
	}

	stream.Dispatch(Event{
		Event: EventMessageFinish,
		Output: messages.Message{
			Role: messages.RoleAI,
			UsageMetadata: messages.UsageMetadata{
				InputTokens:  10,
				OutputTokens: 20,
				TotalTokens:  30,
			},
		},
	})

	if !stream.Done() {
		t.Fatal("stream should be done")
	}
	usage := stream.Usage()
	if usage.InputTokens != 10 || usage.OutputTokens != 20 || usage.TotalTokens != 30 {
		t.Fatalf("unexpected usage: %+v", usage)
	}

	// Output should also carry the usage.
	out, err := stream.Output()
	if err != nil {
		t.Fatalf("output error: %v", err)
	}
	if out.UsageMetadata != usage {
		t.Fatalf("output usage mismatch: %+v vs %+v", out.UsageMetadata, usage)
	}
}
