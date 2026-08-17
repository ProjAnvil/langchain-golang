package middleware

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestNewClearToolUsesEditDefaults(t *testing.T) {
	edit := NewClearToolUsesEdit()
	if edit.Trigger != 100000 || edit.Keep != 3 || edit.Placeholder != DefaultToolPlaceholder {
		t.Fatalf("defaults mismatch: %#v", edit)
	}
}

func toolConversation() []messages.Message {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{
		{ID: "1", Name: "search", Args: map[string]any{"q": "a"}},
		{ID: "2", Name: "calc", Args: map[string]any{"e": "b"}},
		{ID: "3", Name: "search", Args: map[string]any{"q": "c"}},
	}
	m1 := messages.Tool("1", "result one")
	m1.Name = "search"
	m2 := messages.Tool("2", "result two")
	m2.Name = "calc"
	m3 := messages.Tool("3", "result three")
	m3.Name = "search"
	return []messages.Message{messages.Human("hi"), ai, m1, m2, m3}
}

func TestClearToolUsesEditBelowTriggerKeepsEverything(t *testing.T) {
	msgs := toolConversation()
	edit := ClearToolUsesEdit{Trigger: 1000, Keep: 0, Placeholder: "[x]"}
	edited := edit.Apply(msgs, ApproximateTokenCount)
	if edited[2].Content != "result one" {
		t.Fatalf("below trigger should not clear: %#v", edited[2])
	}
}

func TestClearToolUsesEditKeepCoversAllCandidates(t *testing.T) {
	msgs := toolConversation()
	edit := ClearToolUsesEdit{Trigger: 0, Keep: 3, Placeholder: "[x]"}
	edited := edit.Apply(msgs, ApproximateTokenCount)
	for i := 2; i <= 4; i++ {
		if edited[i].Content == "[x]" {
			t.Fatalf("keep >= candidates should clear nothing: %#v", edited[i])
		}
	}
}

func TestClearToolUsesEditExcludeTools(t *testing.T) {
	msgs := toolConversation()
	edit := ClearToolUsesEdit{Trigger: 0, Keep: 0, Placeholder: "[x]", ExcludeTools: []string{"search"}}
	edited := edit.Apply(msgs, ApproximateTokenCount)
	if edited[2].Content != "result one" || edited[4].Content != "result three" {
		t.Fatalf("excluded tools should be kept: %#v %#v", edited[2], edited[4])
	}
	if edited[3].Content != "[x]" {
		t.Fatalf("non-excluded tool should be cleared: %#v", edited[3])
	}
}

func TestClearToolUsesEditExclusionFallsBackToCallName(t *testing.T) {
	msgs := toolConversation()
	msgs[2].Name = "" // no name on the tool message: fall back to the call name
	edit := ClearToolUsesEdit{Trigger: 0, Keep: 0, Placeholder: "[x]", ExcludeTools: []string{"search"}}
	edited := edit.Apply(msgs, ApproximateTokenCount)
	if edited[2].Content != "result one" {
		t.Fatalf("exclusion should use the originating call name: %#v", edited[2])
	}
}

func TestClearToolUsesEditStopsAfterClearAtLeast(t *testing.T) {
	msgs := toolConversation()
	edit := ClearToolUsesEdit{Trigger: 0, Keep: 0, ClearAtLeast: 1, Placeholder: "[x]"}
	edited := edit.Apply(msgs, ApproximateTokenCount)
	if edited[2].Content != "[x]" {
		t.Fatalf("first candidate should be cleared: %#v", edited[2])
	}
	if edited[3].Content == "[x]" || edited[4].Content == "[x]" {
		t.Fatalf("clearing should stop once ClearAtLeast is satisfied: %#v %#v", edited[3], edited[4])
	}
}

func TestClearToolUsesEditSkipsWithoutOriginatingCall(t *testing.T) {
	orphan := messages.Tool("missing", "orphan result")
	msgs := []messages.Message{messages.Human("hi"), orphan}
	edit := ClearToolUsesEdit{Trigger: 0, Keep: 0, Placeholder: "[x]"}
	edited := edit.Apply(msgs, ApproximateTokenCount)
	if edited[1].Content != "orphan result" {
		t.Fatalf("orphan tool message should be skipped: %#v", edited[1])
	}
}

func TestClearToolUsesEditSkipsAlreadyCleared(t *testing.T) {
	msgs := toolConversation()
	msgs[2].ResponseMetadata = map[string]any{"context_editing": map[string]any{"cleared": true}}
	edit := ClearToolUsesEdit{Trigger: 0, Keep: 0, Placeholder: "[x]"}
	edited := edit.Apply(msgs, ApproximateTokenCount)
	if edited[2].Content != "result one" {
		t.Fatalf("already-cleared message should be skipped: %#v", edited[2])
	}
	if edited[3].Content != "[x]" {
		t.Fatalf("other candidates should still clear: %#v", edited[3])
	}
}

func TestApproximateTokenCountIncludesBlocksAndToolCalls(t *testing.T) {
	ai := messages.AI("")
	ai.ContentBlocks = []messages.ContentBlock{messages.TextBlock{Text: "hello world"}}
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search", Args: map[string]any{"q": "two words", "n": 42}}}
	got := ApproximateTokenCount([]messages.Message{ai})
	// Block map strings: "text" (the type) + "hello world" (2); then "search"
	// (1) + "two words" (2); the int arg is skipped.
	if got != 6 {
		t.Fatalf("approximate token count mismatch: %d", got)
	}
}

func TestApproximateTextTokensWhitespaceOnly(t *testing.T) {
	if got := approximateTextTokens("   \n\t "); got != 1 {
		t.Fatalf("whitespace-only text should count as 1 token, got %d", got)
	}
}

func TestApproximateTokenCountCharsPerToken(t *testing.T) {
	ai := messages.AI("abcd")
	ai.ContentBlocks = []messages.ContentBlock{messages.TextBlock{Text: "efgh"}}
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "tool", Args: map[string]any{"q": "ijkl"}}}
	got := ApproximateTokenCountCharsPerToken([]messages.Message{ai}, 4)
	// 4 (content) + 4 ("text" type) + 4 (block text) + 4 (name) + 4 (arg) =
	// 20 chars / 4 = 5 tokens.
	if got != 5 {
		t.Fatalf("chars-per-token count mismatch: %d", got)
	}
	if got := ApproximateTokenCountCharsPerToken(nil, 4); got != 0 {
		t.Fatalf("empty input should count 0, got %d", got)
	}
	// Non-positive charsPerToken falls back to 4.
	if got := ApproximateTokenCountCharsPerToken([]messages.Message{messages.Human("abcdefgh")}, 0); got != 2 {
		t.Fatalf("fallback chars-per-token mismatch: %d", got)
	}
}

func TestContextEditingCleared(t *testing.T) {
	msg := messages.Tool("1", "x")
	if contextEditingCleared(msg) {
		t.Fatal("message without metadata should not be cleared")
	}
	msg.ResponseMetadata = map[string]any{"context_editing": map[string]any{"cleared": true}}
	if !contextEditingCleared(msg) {
		t.Fatal("expected cleared metadata to be detected")
	}
	msg.ResponseMetadata = map[string]any{"context_editing": "not-a-map"}
	if contextEditingCleared(msg) {
		t.Fatal("malformed metadata should not be cleared")
	}
}

func TestFindOriginatingToolCall(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	msgs := []messages.Message{messages.Human("hi"), ai, messages.Tool("1", "r")}

	idx, call, ok := findOriginatingToolCall(msgs, "1")
	if !ok || idx != 1 || call.Name != "search" {
		t.Fatalf("expected to find call: %d %#v %v", idx, call, ok)
	}
	// The most recent AI message without a matching call ends the search.
	if _, _, ok := findOriginatingToolCall(msgs, "nope"); ok {
		t.Fatal("expected no match for unknown id")
	}
	if _, _, ok := findOriginatingToolCall([]messages.Message{messages.Human("hi")}, "1"); ok {
		t.Fatal("expected no match without AI message")
	}
}

func TestClearToolInputMetadata(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{
		{ID: "1", Name: "search", Args: map[string]any{"q": "x"}},
		{ID: "2", Name: "calc", Args: map[string]any{"e": "y"}},
	}
	cleared := clearToolInput(ai, "1")
	if len(cleared.ToolCalls[0].Args) != 0 {
		t.Fatalf("expected args cleared: %#v", cleared.ToolCalls[0].Args)
	}
	if len(cleared.ToolCalls[1].Args) != 1 {
		t.Fatalf("unrelated call args should be kept: %#v", cleared.ToolCalls[1].Args)
	}
	entry, ok := cleared.ResponseMetadata["context_editing"].(map[string]any)
	if !ok {
		t.Fatalf("expected context_editing metadata: %#v", cleared.ResponseMetadata)
	}
	ids, ok := entry["cleared_tool_inputs"].([]string)
	if !ok || len(ids) != 1 || ids[0] != "1" {
		t.Fatalf("cleared_tool_inputs mismatch: %#v", entry)
	}

	// A second clear on the same message accumulates ids.
	again := clearToolInput(cleared, "2")
	entry = again.ResponseMetadata["context_editing"].(map[string]any)
	ids = entry["cleared_tool_inputs"].([]string)
	if len(ids) != 2 {
		t.Fatalf("expected accumulated cleared ids: %#v", ids)
	}

	// Unknown id: no metadata added.
	untouched := clearToolInput(ai, "unknown")
	if untouched.ResponseMetadata["context_editing"] != nil {
		t.Fatalf("no metadata expected for unknown id: %#v", untouched.ResponseMetadata)
	}
}

func TestAppendClearedID(t *testing.T) {
	got := appendClearedID([]string{"a", "a", "b"}, "c")
	if len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Fatalf("appendClearedID mismatch: %#v", got)
	}
	got = appendClearedID([]string{"a"}, "a")
	if len(got) != 1 {
		t.Fatalf("duplicate id should not be appended: %#v", got)
	}
	got = appendClearedID("not-a-slice", "x")
	if len(got) != 1 || got[0] != "x" {
		t.Fatalf("unexpected existing value should be replaced: %#v", got)
	}
}

func TestContextEditingMiddlewareEmptyMessagesPassThrough(t *testing.T) {
	request, err := NewModelRequest(ModelRequest{Model: "model"})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	middleware := NewContextEditingMiddleware()
	called := false
	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		called = true
		if len(req.Messages) != 0 {
			t.Fatalf("expected no messages: %#v", req.Messages)
		}
		return ModelResponse{}, nil
	})
	if err != nil || !called {
		t.Fatalf("wrap model call: %v called=%v", err, called)
	}
}

func TestContextEditingMiddlewareNilCounterUsesDefault(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	request, err := NewModelRequest(ModelRequest{
		Model:    "model",
		Messages: []messages.Message{ai, messages.Tool("1", "old result")},
	})
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	middleware := NewContextEditingMiddleware(ClearToolUsesEdit{Trigger: 0, Keep: 0})
	middleware.CountTokens = nil
	_, err = middleware.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if req.Messages[1].Content != DefaultToolPlaceholder {
			t.Fatalf("expected default placeholder: %#v", req.Messages[1])
		}
		return ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
}
