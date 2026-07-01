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
		Delta: messages.ContentBlock{"type": "text-delta", "text": "Hi"},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.ContentBlock{"type": "text-delta", "text": " there"},
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
		Delta: messages.ContentBlock{"type": "reasoning-delta", "reasoning": "think"},
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
		Content: messages.ContentBlock{
			"type": "tool_call",
			"id":   "tc1",
			"name": "search",
			"args": map[string]any{"q": "test"},
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
		Delta: messages.ContentBlock{"type": "text-delta", "text": "aaa"},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 1,
		Delta: messages.ContentBlock{"type": "text-delta", "text": "bb"},
	})
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   0,
		Content: messages.ContentBlock{"type": "text", "text": "XXX"},
	})
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   1,
		Content: messages.ContentBlock{"type": "text", "text": "bb"},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	if got := stream.Text(); got != "XXXbb" {
		t.Fatalf("text: got %q", got)
	}
	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if got := []any{output.ContentBlocks[0]["text"], output.ContentBlocks[1]["text"]}; !reflect.DeepEqual(got, []any{"XXX", "bb"}) {
		t.Fatalf("blocks: %+v", output.ContentBlocks)
	}
}

func TestChatModelStreamInterleavesTextToolAndReasoningBlocks(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 0,
		Delta: messages.ContentBlock{"type": "text-delta", "text": "before"},
	})
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   0,
		Content: messages.ContentBlock{"type": "text", "text": "before"},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockFinish,
		Index: 1,
		Content: messages.ContentBlock{
			"type": "tool_call",
			"id":   "tc1",
			"name": "search",
			"args": map[string]any{"q": "x"},
		},
	})
	stream.Dispatch(Event{
		Event: EventContentBlockDelta,
		Index: 2,
		Delta: messages.ContentBlock{"type": "reasoning-delta", "reasoning": "think"},
	})
	stream.Dispatch(Event{
		Event:   EventContentBlockFinish,
		Index:   2,
		Content: messages.ContentBlock{"type": "reasoning", "reasoning": "thinking"},
	})
	stream.Dispatch(Event{Event: EventMessageFinish})

	output, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	gotTypes := []any{
		output.ContentBlocks[0]["type"],
		output.ContentBlocks[1]["type"],
		output.ContentBlocks[2]["type"],
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
		Delta: messages.ContentBlock{
			"type": "tool_call_chunk",
			"id":   "call_1",
			"name": "search",
			"args": `{"q": `,
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
