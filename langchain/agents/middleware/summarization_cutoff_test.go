package middleware

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// TestFindSafeCutoffPoint_SnapsToAIMessage mirrors Python's
// _find_safe_cutoff_point (summarization.py:762-796): when the cutoff lands on
// a ToolMessage, it snaps BACK to the AIMessage whose tool_calls produced it,
// so the kept suffix never starts with orphaned tool responses.
func TestFindSafeCutoffPoint_SnapsToAIMessage(t *testing.T) {
	msgs := []messages.Message{
		messages.Human("q1"),                                                                                                    // 0
		messages.AI("a1"),                                                                                                       // 1
		messages.Human("q2"),                                                                                                    // 2
		{Role: messages.RoleAI, Content: "", ToolCalls: []messages.ToolCall{{ID: "t1", Name: "search"}}},                        // 3
		{Role: messages.RoleTool, ToolCallID: "t1", Content: "result"},                                                          // 4
		messages.Human("q3"),                                                                                                    // 5
	}
	// Cutoff at 4 (a ToolMessage) must snap back to 3 (the AI that issued t1).
	if got := findSafeCutoffPoint(msgs, 4); got != 3 {
		t.Fatalf("cutoff on ToolMessage: got %d, want 3", got)
	}
	// Cutoff NOT on a ToolMessage is returned unchanged.
	if got := findSafeCutoffPoint(msgs, 2); got != 2 {
		t.Fatalf("cutoff on HumanMessage: got %d, want 2", got)
	}
	// Cutoff beyond slice is returned unchanged.
	if got := findSafeCutoffPoint(msgs, len(msgs)); got != len(msgs) {
		t.Fatalf("cutoff > len: got %d, want %d", got, len(msgs))
	}
}

// TestKeepStart_MessageCountPathSnapsToSafeCutoff asserts the message-count
// keepStart path applies findSafeCutoffPoint (Python applies it at :760).
func TestKeepStart_MessageCountPathSnapsToSafeCutoff(t *testing.T) {
	mw := &SummarizationMiddleware{Keep: KeepPolicy{Messages: 2}} // keep last 2
	msgs := []messages.Message{
		messages.Human("q1"),                                                                                                    // 0
		{Role: messages.RoleAI, Content: "", ToolCalls: []messages.ToolCall{{ID: "t1", Name: "s"}}},                             // 1
		{Role: messages.RoleTool, ToolCallID: "t1", Content: "r"},                                                               // 2
		messages.Human("q3"),                                                                                                    // 3
	}
	// Naive keep-last-2 → start=2 (ToolMessage). Safe cutoff snaps to 1 (the AI).
	got := mw.keepStart(msgs)
	if got != 1 {
		t.Fatalf("keepStart message-count path: got %d, want 1 (snapped to AI)", got)
	}
}
