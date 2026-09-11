package language

import (
	"testing"

	"github.com/tiktoken-go/tokenizer"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Mirrors Python's base get_num_tokens = len(get_token_ids)
// (language_models/base.py:433). The default fallback is the real GPT-2 BPE
// tokenizer (language_models/base.py:98-104), so "hello world" is 2 tokens.
func TestGetNumTokensIsTokenIDCount(t *testing.T) {
	if got := GetNumTokens(nil, "hello world"); got != 2 {
		t.Fatalf("GetNumTokens = %d, want 2", got)
	}
	if got := GetNumTokens(nil, ""); got != 0 {
		t.Fatalf("GetNumTokens(empty) = %d, want 0", got)
	}
	if got := GetNumTokens(NewFakeChatModel(), "hello world"); got != 2 {
		t.Fatalf("GetNumTokens(FakeChatModel) = %d, want 2", got)
	}
}

// Default token IDs are real GPT-2 BPE IDs: deterministic, non-negative, and
// within the r50k_base vocabulary size (50257).
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
		if first[i] >= 50257 {
			t.Fatalf("id %d outside r50k_base vocabulary: %d", i, first[i])
		}
	}
	if got := DefaultGetTokenIDs(""); len(got) != 0 {
		t.Fatalf("DefaultGetTokenIDs(empty) = %v, want empty", got)
	}
}

// DefaultGetTokenIDs encodes with the real GPT-2 BPE tokenizer (gpt2 =
// r50k_base), mirroring Python's fallback get_token_ids
// (language_models/base.py:98-104): "hello world" is 2 tokens with the true
// GPT-2 token IDs, and the output matches a direct tokenizer.Encode call.
func TestDefaultGetTokenIDsUsesGPT2BPE(t *testing.T) {
	ids := DefaultGetTokenIDs("hello world")
	if len(ids) != 2 {
		t.Fatalf("DefaultGetTokenIDs(\"hello world\") = %v, want 2 ids", ids)
	}
	codec, err := tokenizer.Get(tokenizer.R50kBase)
	if err != nil {
		t.Fatalf("tokenizer.Get(GPT2Enc): %v", err)
	}
	want, _, err := codec.Encode("hello world")
	if err != nil {
		t.Fatalf("codec.Encode: %v", err)
	}
	if len(ids) != len(want) {
		t.Fatalf("len = %d, want %d (direct encode)", len(ids), len(want))
	}
	for i := range ids {
		if ids[i] != int(want[i]) {
			t.Fatalf("id %d = %d, want %d (direct encode)", i, ids[i], want[i])
		}
	}
	// Two calls return equal IDs (cached tokenizer, deterministic encoding).
	again := DefaultGetTokenIDs("hello world")
	for i := range ids {
		if ids[i] != again[i] {
			t.Fatalf("second call id %d = %d, want %d", i, again[i], ids[i])
		}
	}
}

// Multi-byte text tokenizes by UTF-8 bytes under BPE, so CJK is not collapsed
// into one fake chunk: r50k_base's byte-level vocabulary expands "你好" to 4
// tokens (each CJK character spans multiple byte-level tokens), and multi-byte
// emoji likewise count more than one token each.
func TestDefaultGetTokenIDsMultibyte(t *testing.T) {
	if got := DefaultGetTokenIDs("你好"); len(got) != 4 {
		t.Fatalf("DefaultGetTokenIDs(\"你好\") = %v, want 4 ids", got)
	}
	emoji := DefaultGetTokenIDs("🦜🔗")
	if len(emoji) != 6 { // three byte-level tokens per 4-byte emoji
		t.Fatalf("DefaultGetTokenIDs(\"🦜🔗\") = %v, want 6 ids", emoji)
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
