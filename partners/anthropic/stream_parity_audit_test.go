package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// TestReasoningBlockRoundTripsAsThinking pins the request-side mapping
// (chat_models.py:882-890): a reasoning block goes back to the Messages API
// as a thinking block keeping its signature, with the text defaulting to an
// explicit empty string when only a signature came back.
func TestReasoningBlockRoundTripsAsThinking(t *testing.T) {
	cases := []struct {
		name         string
		block        messages.ContentBlock
		wantThinking string // "" means the key may be absent
		wantLiteral  string // exact JSON for the thinking field
	}{
		{
			name: "text and signature",
			block: messages.ReasoningBlock{
				Reasoning: "chain of thought",
				Extras:    map[string]any{"signature": "sig_1"},
			},
			wantThinking: "chain of thought",
		},
		{
			name: "signature only travels an explicit empty thinking",
			block: messages.ReasoningBlock{
				Extras: map[string]any{"signature": "sig_2"},
			},
			wantLiteral: `"thinking":""`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			converted, err := formatContentBlock(tc.block)
			if err != nil {
				t.Fatalf("format: %v", err)
			}
			if converted.Type != "thinking" || converted.Signature == "" {
				t.Fatalf("thinking block: %+v", converted)
			}
			if tc.wantLiteral != "" {
				encoded, err := json.Marshal(converted)
				if err != nil {
					t.Fatalf("marshal: %v", err)
				}
				if !jsonContains(encoded, tc.wantLiteral) {
					t.Fatalf("wire form %s must contain %s", encoded, tc.wantLiteral)
				}
			}
		})
	}
}

// jsonContains reports whether the compact JSON contains the literal.
func jsonContains(encoded []byte, literal string) bool {
	return strings.Contains(string(encoded), literal)
}

// TestTextStartCarriesOpeningText pins #41907's neighboring behavior
// (chat_models.py:2473-2483): a content_block_start whose text block
// already carries text seeds the block and streams that opening text as a
// chunk.
func TestTextStartCarriesOpeningText(t *testing.T) {
	server := streamServer(
		sse("message_start", streamUsageMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Opening"}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" tail"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("go")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var sawOpening bool
	for {
		chunk, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		for _, block := range chunk.ContentBlocks {
			if text, isText := block.(messages.TextBlock); isText && text.Text == "Opening" {
				sawOpening = true
			}
		}
	}
	if !sawOpening {
		t.Fatal("the opening text must stream as its own chunk")
	}
	// The assembled message keeps both the opening text and the delta.
	if end := streamEndMessage(t, recorder); end.Content != "Opening tail" {
		t.Fatalf("assembled text: %q", end.Content)
	}
}

// TestRedactedThinkingStartEmitsChunk pins the redacted-start chunk
// (chat_models.py:2413-2458): the start surfaces as a chunk carrying the
// index and the opaque data.
func TestRedactedThinkingStartEmitsChunk(t *testing.T) {
	server := streamServer(
		sse("message_start", streamUsageMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"ZW5j","id":"rb_1"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var sawRedacted bool
	for {
		chunk, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		for _, block := range chunk.ContentBlocks {
			if reasoning, isReasoning := block.(messages.ReasoningBlock); isReasoning && reasoning.Index == 0 {
				if data, _ := reasoning.Extras["data"].(string); data == "ZW5j" {
					sawRedacted = true
				}
			}
		}
	}
	if !sawRedacted {
		t.Fatal("redacted thinking start must emit a chunk with its data")
	}
}
