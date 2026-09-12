package anthropic

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
)

const streamUsageMessageStart = `{"type":"message_start","message":{"id":"msg_u","type":"message","role":"assistant","model":"claude","content":[],"usage":{"input_tokens":25,"cache_read_input_tokens":100,"cache_creation_input_tokens":50,"output_tokens":1}}}`

// streamUsageServer yields one text block plus a message_delta carrying final
// usage, mirroring the wire shape Python's _stream tests rely on.
func streamUsageServer(t *testing.T, deltaFrame string) *httptest.Server {
	t.Helper()
	return streamServer(
		sse("message_start", streamUsageMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", deltaFrame),
		sse("message_stop", `{"type":"message_stop"}`),
	)
}

func usageCarryingChunks(chunks []messages.Message) []messages.Message {
	var out []messages.Message
	for _, c := range chunks {
		u := c.UsageMetadata
		if u.InputTokens > 0 || u.OutputTokens > 0 || u.TotalTokens > 0 {
			out = append(out, c)
		}
	}
	return out
}

// TestStreamYieldsUsageChunk mirrors Python's _stream (chat_models.py:1702-1725):
// the message_delta event yields a terminal empty-content chunk carrying
// usage_metadata and stop_reason, so consumers aggregating over chunks see the
// token counts without touching callbacks.
func TestStreamYieldsUsageChunk(t *testing.T) {
	server := streamUsageServer(t, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)
	defer server.Close()

	stream, _ := streamWithRecorder(t, server)
	chunks := drainStream(t, stream)

	usageChunks := usageCarryingChunks(chunks)
	if len(usageChunks) != 1 {
		t.Fatalf("usage chunks: got %d want 1 (chunks: %+v)", len(usageChunks), chunks)
	}
	u := usageChunks[0].UsageMetadata
	// message_start input 25 + cache_read 100 + cache_creation 50 = 175;
	// message_delta merges output 1 -> 5.
	if u.InputTokens != 175 || u.OutputTokens != 5 {
		t.Fatalf("usage: input=%d output=%d", u.InputTokens, u.OutputTokens)
	}
	if u.InputTokenDetails == nil ||
		u.InputTokenDetails.CacheReadInputTokens != 100 ||
		u.InputTokenDetails.CacheCreationInputTokens != 50 {
		t.Fatalf("input token details: %+v", u.InputTokenDetails)
	}
	if usageChunks[0].Content != "" {
		t.Fatalf("usage chunk content: %q", usageChunks[0].Content)
	}
	if got := usageChunks[0].ResponseMetadata["stop_reason"]; got != "end_turn" {
		t.Fatalf("usage chunk stop_reason: %v", got)
	}
	// Text deltas keep zero usage so input tokens appear in at most one chunk.
	for _, c := range chunks {
		if c.Content == "Hi" && (c.UsageMetadata.InputTokens != 0 || c.UsageMetadata.OutputTokens != 0) {
			t.Fatalf("text chunk carries usage: %+v", c.UsageMetadata)
		}
	}
}

// TestStreamUsageChunkDisabled verifies WithStreamUsage(false) suppresses the
// usage chunk, mirroring Python's stream_usage=False.
func TestStreamUsageChunkDisabled(t *testing.T) {
	server := streamUsageServer(t, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
		WithStreamUsage(false),
	)
	stream, err := model.Stream(context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(callbacks.NewRecorder())),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunks := drainStream(t, stream)
	if got := usageCarryingChunks(chunks); len(got) != 0 {
		t.Fatalf("usage chunks with stream usage disabled: %+v", got)
	}
}
