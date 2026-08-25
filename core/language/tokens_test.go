package language

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Mirrors Python's base get_num_tokens = len(get_token_ids)
// (language_models/base.py:433). The default approximation splits text into
// 4-rune chunks (the chars-per-token heuristic of
// messages.CountTokensApproximately), so "hello world" (11 runes) is 3 tokens.
func TestGetNumTokensIsTokenIDCount(t *testing.T) {
	if got := GetNumTokens(nil, "hello world"); got != 3 {
		t.Fatalf("GetNumTokens = %d, want 3", got)
	}
	if got := GetNumTokens(nil, ""); got != 0 {
		t.Fatalf("GetNumTokens(empty) = %d, want 0", got)
	}
	if got := GetNumTokens(NewFakeChatModel(), "hello world"); got != 3 {
		t.Fatalf("GetNumTokens(FakeChatModel) = %d, want 3", got)
	}
}

// Default token IDs are deterministic, non-negative, and chunk-sized.
func TestDefaultGetTokenIDsDeterministic(t *testing.T) {
	first := DefaultGetTokenIDs("表情符号是\n🦜🔗")
	second := DefaultGetTokenIDs("表情符号是\n🦜🔗")
	if len(first) != len(second) || len(first) == 0 {
		t.Fatalf("length mismatch: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("id %d differs: %d vs %d", i, first[i], second[i])
		}
		if first[i] < 0 {
			t.Fatalf("id %d negative: %d", i, first[i])
		}
	}
	if got := DefaultGetTokenIDs(""); len(got) != 0 {
		t.Fatalf("DefaultGetTokenIDs(empty) = %v, want empty", got)
	}
}

// Mirrors Python's base get_num_tokens_from_messages
// (language_models/base.py:450-485): sum of get_num_tokens(get_buffer_string([m])).
// Tool schemas are not counted (Python warns and ignores them).
func TestGetNumTokensFromMessagesSumsBufferStrings(t *testing.T) {
	msgs := []messages.Message{
		messages.Human("Hello"),
		messages.AI("Hi there"),
	}
	got, err := GetNumTokensFromMessages(nil, msgs)
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	want := 0
	for _, m := range msgs {
		want += GetNumTokens(nil, messages.BufferString([]messages.Message{m}))
	}
	if got != want {
		t.Fatalf("GetNumTokensFromMessages = %d, want %d", got, want)
	}
	contentOnly := GetNumTokens(nil, "Hello") + GetNumTokens(nil, "Hi there")
	if got <= contentOnly {
		t.Fatalf("expected role prefixes to add tokens, got %d (content-only %d)", got, contentOnly)
	}
}

// A model implementing TokenCounter/MessageTokenCounter overrides the default.
type tokenCountingModel struct{}

func (tokenCountingModel) GetTokenIDs(text string) []int {
	ids := make([]int, 0, len(text))
	for i := range text {
		ids = append(ids, i)
	}
	return ids
}

func (tokenCountingModel) GetNumTokensFromMessages(_ []messages.Message) (int, error) {
	return 42, nil
}

func TestTokenCountingDispatchesToModel(t *testing.T) {
	ids := GetTokenIDs(tokenCountingModel{}, "abc")
	if len(ids) != 3 || ids[0] != 0 || ids[2] != 2 {
		t.Fatalf("model token ids: %v", ids)
	}
	if got := GetNumTokens(tokenCountingModel{}, "abcd"); got != 4 {
		t.Fatalf("GetNumTokens via model = %d, want 4", got)
	}
	got, err := GetNumTokensFromMessages(tokenCountingModel{}, []messages.Message{messages.Human("x")})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 42 {
		t.Fatalf("GetNumTokensFromMessages = %d, want 42", got)
	}
}
