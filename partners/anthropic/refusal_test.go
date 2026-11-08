package anthropic

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// Ports partners/anthropic/tests/unit_tests/test_refusal.py（144 行，
// anthropic 1.7.10 + 未发布 #41715）。

// --- stop_reason=refusal → RefusalBlock，category/explanation 进 extras/文本 ---

func TestRefusalStopReasonProducesBlock(t *testing.T) {
	raw := `{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
		"content": [{"type": "text", "text": ""}],
		"stop_reason": "refusal",
		"stop_details": {"category": "safety", "explanation": "Cannot help with weapons."},
		"usage": {"input_tokens": 3, "output_tokens": 1}
	}`
	var payload messagePayload
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		t.Fatal(err)
	}
	msg := payload.toMessage()
	var rb *messages.RefusalBlock
	for _, b := range msg.ContentBlocks {
		if r, ok := b.(messages.RefusalBlock); ok {
			cpy := r
			rb = &cpy
		}
	}
	if rb == nil {
		t.Fatalf("no RefusalBlock in %v", msg.ContentBlocks)
	}
	if rb.Refusal != "Cannot help with weapons." {
		t.Fatalf("refusal text = %q", rb.Refusal)
	}
	if rb.Extras["category"] != "safety" {
		t.Fatalf("extras = %v", rb.Extras)
	}
	// 不设固定 index（#41658 后上游 translator 产出端无 index）。
	if rb.Index != nil {
		t.Fatalf("index must be nil, got %v", rb.Index)
	}
	if msg.ResponseMetadata["stop_reason"] != "refusal" {
		t.Fatalf("metadata = %v", msg.ResponseMetadata)
	}
	if msg.ResponseMetadata["stop_details"] == nil {
		t.Fatal("stop_details must be captured in metadata")
	}
}

// --- 输入方向：refusal 块不发往 API ---

func TestRefusalBlockDroppedOnInput(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.TextBlock{Text: "ok"},
		messages.RefusalBlock{Refusal: "no"},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range blocks {
		if b.Type == "refusal" {
			t.Fatalf("refusal must not be sent to the Messages API: %+v", blocks)
		}
	}
	if len(blocks) != 1 || blocks[0].Type != "text" {
		t.Fatalf("want only the text block, got %+v", blocks)
	}
}

// --- #41715：结构化输出 no-tool_call + refusal → ModelRefusalError ---

func TestInvokeStructuredRefusalNotParseError(t *testing.T) {
	// 构造 stop_reason=refusal 且无 tool_use 的响应，走 InvokeStructured 的
	// 报错分支。无 HTTP fake 时直接测试判定函数（见 Step 3 提取的
	// refusalOrNoToolCallError）。
	payload := messagePayload{StopReason: "refusal"}
	msg := payload.toMessage()
	err := noToolCallError(msg)
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}

	payload2 := messagePayload{StopReason: "end_turn"}
	err2 := noToolCallError(payload2.toMessage())
	if err2 == nil || errors.As(err2, &mre) {
		t.Fatalf("non-refusal no-tool_call must stay a plain error, got %v", err2)
	}
}

// --- 流式：message_delta 的 stop chunk 不受 streamUsage 门控 ---

// TestStreamStopChunkEmittedWithoutStreamUsage drives messageStream with the
// stream_usage_test.go SSE fixture pattern: message_start(usage) →
// message_delta(stop_reason="end_turn", usage) → message_stop, with
// WithStreamUsage(false). The consumer must still receive the terminal chunk
// carrying stop_reason, and that chunk's UsageMetadata must be zero (usage
// stays gated).
func TestStreamStopChunkEmittedWithoutStreamUsage(t *testing.T) {
	server := streamUsageServer(t, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
		WithStreamUsage(false),
	)
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunks := drainStream(t, stream)

	var stopChunks []messages.Message
	for _, c := range chunks {
		if c.ResponseMetadata["stop_reason"] != nil {
			stopChunks = append(stopChunks, c)
		}
	}
	if len(stopChunks) != 1 {
		t.Fatalf("stop chunks: got %d want 1 (chunks: %+v)", len(stopChunks), chunks)
	}
	if got := stopChunks[0].ResponseMetadata["stop_reason"]; got != "end_turn" {
		t.Fatalf("stop chunk stop_reason: %v", got)
	}
	if stopChunks[0].UsageMetadata != (messages.UsageMetadata{}) {
		t.Fatalf("stop chunk must carry no usage with stream usage disabled: %+v", stopChunks[0].UsageMetadata)
	}
}

// TestStreamStopChunkWithStreamUsage is the streamUsage=true contrast: the
// terminal chunk carries both the stop reason and the merged usage.
func TestStreamStopChunkWithStreamUsage(t *testing.T) {
	server := streamUsageServer(t, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)
	defer server.Close()

	stream, _ := streamWithRecorder(t, server)
	chunks := drainStream(t, stream)

	var stopChunks []messages.Message
	for _, c := range chunks {
		if c.ResponseMetadata["stop_reason"] != nil {
			stopChunks = append(stopChunks, c)
		}
	}
	if len(stopChunks) != 1 {
		t.Fatalf("stop chunks: got %d want 1 (chunks: %+v)", len(stopChunks), chunks)
	}
	u := stopChunks[0].UsageMetadata
	if u.InputTokens != 175 || u.OutputTokens != 5 {
		t.Fatalf("stop chunk usage: input=%d output=%d", u.InputTokens, u.OutputTokens)
	}
}

// --- 流式：message_delta 的 refusal → RefusalBlock 进最终消息 ---

func TestStreamRefusalProducesBlock(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"category":"safety","explanation":"Cannot help."}},"usage":{"output_tokens":1}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	_ = drainStream(t, stream)
	out := finalOutput(t, recorder)

	var rb *messages.RefusalBlock
	for _, b := range out.ContentBlocks {
		if r, ok := b.(messages.RefusalBlock); ok {
			cpy := r
			rb = &cpy
		}
	}
	if rb == nil {
		t.Fatalf("no RefusalBlock in %v", out.ContentBlocks)
	}
	if rb.Refusal != "Cannot help." {
		t.Fatalf("refusal text = %q", rb.Refusal)
	}
	if rb.Extras["category"] != "safety" {
		t.Fatalf("extras = %v", rb.Extras)
	}
	if out.ResponseMetadata["stop_reason"] != "refusal" {
		t.Fatalf("metadata = %v", out.ResponseMetadata)
	}
	if out.ResponseMetadata["stop_details"] == nil {
		t.Fatal("stop_details must be captured in stream metadata")
	}
}
