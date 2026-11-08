package gemini

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"google.golang.org/genai"
)

// Ports core/tests/unit_tests/messages/test_refusal.py 的 gemini 段
//（test_refusal.py:210-284，#41658）。

func TestRefusalFromSafetyFinishReason(t *testing.T) {
	meta := map[string]any{
		"finish_reason":  "SAFETY",
		"finish_message": "Blocked for safety.",
	}
	rb := refusalFromMetadata(meta)
	if rb == nil {
		t.Fatal("SAFETY finish reason must produce a refusal block")
	}
	if rb.Refusal != "Blocked for safety." {
		t.Fatalf("refusal text = %q", rb.Refusal)
	}
	if rb.Extras["finish_reason"] != "SAFETY" {
		t.Fatalf("extras = %v", rb.Extras)
	}
}

func TestRefusalFromPromptFeedback(t *testing.T) {
	meta := map[string]any{
		"prompt_feedback": map[string]any{"block_reason": "PROHIBITED_CONTENT"},
	}
	rb := refusalFromMetadata(meta)
	if rb == nil {
		t.Fatal("prompt block reason must produce a refusal block")
	}
	if rb.Refusal != "" {
		t.Fatalf("refusal text must be empty without finish_message, got %q", rb.Refusal)
	}
	if rb.Extras["block_reason"] != "PROHIBITED_CONTENT" {
		t.Fatalf("extras = %v", rb.Extras)
	}
}

func TestNoRefusalOnCleanFinish(t *testing.T) {
	for _, meta := range []map[string]any{
		{"finish_reason": "STOP"},
		{"finish_reason": "MAX_TOKENS"},
		{"prompt_feedback": map[string]any{"block_reason": "BLOCKED_REASON_UNSPECIFIED"}},
		{},
	} {
		if rb := refusalFromMetadata(meta); rb != nil {
			t.Fatalf("meta %v must not produce a refusal block", meta)
		}
	}
}

func TestRefusalReasonEnumTail(t *testing.T) {
	// _reason_name 对齐：枚举串取尾段（"FinishReason.SAFETY" → "SAFETY"）。
	meta := map[string]any{"finish_reason": "FinishReason.SAFETY"}
	rb := refusalFromMetadata(meta)
	if rb == nil || rb.Extras["finish_reason"] != "SAFETY" {
		t.Fatalf("enum-qualified reason must be normalized: %+v", rb)
	}
}

// responseToMessage 端到端：SAFETY finish 的响应须在内容块尾部带 refusal 块。
func TestResponseToMessageAppendsSafetyRefusal(t *testing.T) {
	response := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content:       &genai.Content{Role: "model"},
			FinishReason:  genai.FinishReasonSafety,
			FinishMessage: "Blocked for safety.",
		}},
	}
	message, err := responseToMessage(response)
	if err != nil {
		t.Fatalf("responseToMessage: %v", err)
	}
	if len(message.ContentBlocks) == 0 {
		t.Fatal("expected a refusal block")
	}
	last, ok := message.ContentBlocks[len(message.ContentBlocks)-1].(messages.RefusalBlock)
	if !ok {
		t.Fatalf("last block = %T, want messages.RefusalBlock", message.ContentBlocks[len(message.ContentBlocks)-1])
	}
	if last.Refusal != "Blocked for safety." {
		t.Fatalf("refusal text = %q", last.Refusal)
	}
	if last.Extras["finish_reason"] != "SAFETY" {
		t.Fatalf("extras = %v", last.Extras)
	}
}
