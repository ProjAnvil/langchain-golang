package streamevents

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// TestFinishReasonFields pins the core-side finish-field corrections
// (Python langchain-core 1.8.4 _finish_reason.py, #41926): read the three
// keys a provider set, apply the refusal and forced-tool-choice stop
// corrections, and stay empty when the provider set none.
func TestFinishReasonFields(t *testing.T) {
	cases := []struct {
		name       string
		metadata   map[string]any
		hasTools   bool
		hasRefusal bool
		want       map[string]any
	}{
		{
			name:     "none set",
			metadata: map[string]any{"model": "m"},
			want:     map[string]any{},
		},
		{
			name:     "passthrough without normalized",
			metadata: map[string]any{"finish_reason": "tool_use"},
			want:     map[string]any{"finish_reason": "tool_use"},
		},
		{
			name:     "stop with tool calls becomes tool_call",
			metadata: map[string]any{"finish_reason": "stop", "normalized_finish_reason": "stop"},
			hasTools: true,
			want:     map[string]any{"finish_reason": "stop", "normalized_finish_reason": "tool_call"},
		},
		{
			name:       "refusal wins over any normalized reason",
			metadata:   map[string]any{"normalized_finish_reason": "stop", "finish_state": "complete"},
			hasRefusal: true,
			want:       map[string]any{"normalized_finish_reason": "refusal", "finish_state": "complete"},
		},
		{
			name:     "non-stop normalized untouched",
			metadata: map[string]any{"normalized_finish_reason": "length"},
			hasTools: true,
			want:     map[string]any{"normalized_finish_reason": "length"},
		},
	}
	for _, tc := range cases {
		got := FinishReasonFields(tc.metadata, tc.hasTools, tc.hasRefusal)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %#v, want %#v", tc.name, got, tc.want)
		}
	}
}

// TestFinishAppliesFinishReasonCorrections drives a content-block-finish
// tool call plus a provider message-finish through ChatModelStream.Dispatch:
// the provider metadata's normalized stop must become tool_call once the
// tool call is part of the assembled message (#41926).
func TestFinishAppliesFinishReasonCorrections(t *testing.T) {
	stream := NewChatModelStream()
	stream.Dispatch(Event{Event: EventMessageStart})
	stream.Dispatch(Event{
		Event: EventContentBlockFinish,
		Index: 0,
		Content: messages.ToolCallBlock{
			ID:   "call_1",
			Name: "search",
			Args: map[string]any{"q": "x"},
		},
	})
	output := messages.AI("")
	output.ResponseMetadata = map[string]any{
		"finish_reason":            "stop",
		"normalized_finish_reason": "stop",
	}
	stream.Dispatch(Event{Event: EventMessageFinish, Output: output})
	final, err := stream.Output()
	if err != nil {
		t.Fatalf("output: %v", err)
	}
	if final.ResponseMetadata["normalized_finish_reason"] != "tool_call" {
		t.Fatalf("corrected normalized: %#v", final.ResponseMetadata)
	}
	if len(final.ToolCalls) != 1 || final.ToolCalls[0].Name != "search" {
		t.Fatalf("tool calls: %#v", final.ToolCalls)
	}
}

// TestHasRefusalBlock covers both refusal spellings: a RefusalBlock in the
// content blocks and additional_kwargs refusal.
func TestHasRefusalBlock(t *testing.T) {
	if hasRefusalBlock(messages.AI("hi")) {
		t.Fatal("plain message must not report refusal")
	}
	withBlock := messages.AI("")
	withBlock.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "no"}}
	if !hasRefusalBlock(withBlock) {
		t.Fatal("RefusalBlock not detected")
	}
	withKwargs := messages.AI("")
	withKwargs.AdditionalKwargs = map[string]any{"refusal": "no"}
	if !hasRefusalBlock(withKwargs) {
		t.Fatal("additional_kwargs refusal not detected")
	}
}

// TestFinishReasonFieldsNilMetadataAndRawOnly port the upstream
// test_finish_fields_empty_without_provider_fields and
// test_finish_fields_raw_only: a nil map yields nothing, and a raw-only
// field set is passed through uncorrected even with a refusal — there is
// no normalized value to correct.
func TestFinishReasonFieldsNilMetadataAndRawOnly(t *testing.T) {
	if got := FinishReasonFields(nil, false, false); len(got) != 0 {
		t.Errorf("nil metadata: %#v", got)
	}
	got := FinishReasonFields(map[string]any{"stop_reason": "end_turn"}, false, false)
	if len(got) != 0 {
		t.Errorf("non-shared keys only: %#v", got)
	}
	rawOnly := FinishReasonFields(map[string]any{"finish_reason": "stop"}, false, true)
	if !reflect.DeepEqual(rawOnly, map[string]any{"finish_reason": "stop"}) {
		t.Errorf("raw-only with refusal: %#v", rawOnly)
	}
}
