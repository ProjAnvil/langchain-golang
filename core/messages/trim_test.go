package messages

import (
	"strings"
	"testing"
)

func TestCountTokensApproximately(t *testing.T) {
	msgs := []Message{Human("hello world")}
	got := CountTokensApproximately(msgs)
	// extra 3 + ceil((11 content + 4 role "user")/4)=4 → 7
	if got != 7 {
		t.Fatalf("CountTokensApproximately = %d, want 7", got)
	}
}

func TestTrimMessagesLastDropsOldest(t *testing.T) {
	msgs := []Message{
		Human(strings.Repeat("a", 40)),
		AI(strings.Repeat("b", 40)),
		Human("recent"),
	}
	out, err := TrimMessages(msgs, TrimMessagesOptions{MaxTokens: 10, Strategy: TrimStrategyLast})
	if err != nil {
		t.Fatalf("TrimMessages: %v", err)
	}
	if len(out) != 1 || out[0].Content != "recent" {
		t.Fatalf("trimmed = %#v, want only the recent message", out)
	}
}

func TestTrimMessagesFirstDropsNewest(t *testing.T) {
	msgs := []Message{
		Human("first"),
		AI(strings.Repeat("b", 40)),
		Human(strings.Repeat("c", 40)),
	}
	out, err := TrimMessages(msgs, TrimMessagesOptions{MaxTokens: 10, Strategy: TrimStrategyFirst})
	if err != nil {
		t.Fatalf("TrimMessages: %v", err)
	}
	if len(out) != 1 || out[0].Content != "first" {
		t.Fatalf("trimmed = %#v, want only the first message", out)
	}
}

func TestTrimMessagesIncludeSystem(t *testing.T) {
	msgs := []Message{
		System("you are helpful"),
		Human(strings.Repeat("a", 40)),
		Human("recent"),
	}
	out, err := TrimMessages(msgs, TrimMessagesOptions{MaxTokens: 10, Strategy: TrimStrategyLast, IncludeSystem: true})
	if err != nil {
		t.Fatalf("TrimMessages: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2 (system + recent)", len(out))
	}
	if out[0].Role != RoleSystem || out[1].Content != "recent" {
		t.Fatalf("trimmed = %#v", out)
	}
}

func TestTrimMessagesInvalidStrategy(t *testing.T) {
	if _, err := TrimMessages([]Message{Human("x")}, TrimMessagesOptions{MaxTokens: 10, Strategy: "bogus"}); err == nil {
		t.Fatal("expected error for invalid strategy")
	}
}

func TestTrimMessagesNegativeMaxTokens(t *testing.T) {
	if _, err := TrimMessages([]Message{Human("x")}, TrimMessagesOptions{MaxTokens: -1}); err == nil {
		t.Fatal("expected error for negative max_tokens")
	}
}

func TestTrimMessagesStartOn(t *testing.T) {
	msgs := []Message{
		Human("old a"),
		AI("old b"),
		Human("keep from here"),
		AI("tail"),
	}
	out, err := TrimMessages(msgs, TrimMessagesOptions{
		MaxTokens: 1000,
		Strategy:  TrimStrategyLast,
		StartOn:   []Role{RoleAI},
	})
	if err != nil {
		t.Fatalf("TrimMessages: %v", err)
	}
	if len(out) != 3 || out[0].Role != RoleAI || out[0].Content != "old b" {
		t.Fatalf("start_on trim = %#v, want to start at first AI (index 1)", out)
	}
}

func TestTrimMessagesEndOn(t *testing.T) {
	msgs := []Message{
		Human("a"),
		AI("b"),
		Human("c"),
	}
	out, err := TrimMessages(msgs, TrimMessagesOptions{MaxTokens: 1000, EndOn: []Role{RoleHuman}})
	if err != nil {
		t.Fatalf("TrimMessages: %v", err)
	}
	// end_on=human → keep through the LAST human (index 2), so all 3 stay.
	if len(out) != 3 {
		t.Fatalf("end_on trim = %#v, want 3", out)
	}
}

func TestCountTokensApproximatelyToolCallsAndImages(t *testing.T) {
	ai := AI("")
	ai.ToolCalls = []ToolCall{{Name: "search", Args: map[string]any{"q": "x"}}}
	tool := Tool("call-1", "")
	withImage := Human("")
	withImage.ContentBlocks = []ContentBlock{ImageBlock{URL: "http://x/i.png"}}
	msgs := []Message{
		{Role: RoleHuman, Content: "hello world", Name: "alice"},
		ai,
		tool,
		withImage,
	}

	// Defaults (charsPerToken=4, extra=3, countName=true, tokensPerImage=85):
	//   human: 3 + ceil((11 content + 4 role + 5 name)/4)=5  = 8
	//   ai:    3 + ceil((36 tool calls JSON + 9 role)/4)=12  = 15
	//   tool:  3 + ceil((6 tool_call_id + 4 role)/4)=3       = 6
	//   image: 85 + 3 + ceil(4 role/4)=1                     = 89
	if got := CountTokensApproximately(msgs); got != 118 {
		t.Fatalf("CountTokensApproximately = %d, want 118", got)
	}

	// Custom options: charsPerToken=2, extra=10, countName=true.
	//   human: 10 + ceil(20/2)=10 = 20
	//   ai:    10 + ceil(45/2)=23 = 33
	//   tool:  10 + ceil(10/2)=5  = 15
	//   image: 85 + 10 + ceil(4/2)=2 = 97
	got := CountTokensApproximately(msgs,
		WithCharsPerToken(2),
		WithExtraTokensPerMessage(10),
		WithCountName(true),
	)
	if got != 165 {
		t.Fatalf("CountTokensApproximately with options = %d, want 165", got)
	}
}

// scalingTestMessages builds a three-message conversation whose unscaled
// estimate is 21 tokens (defaults: charsPerToken=4, extra=3, roles counted):
//
//	human "hello world":    3 + ceil((11+4)/4)=4  = 7
//	ai "hi":                3 + ceil((2+9)/4)=3   = 6  (running total 13)
//	human "more text here": 3 + ceil((14+4)/4)=5  = 8  (total 21)
//
// The AI message reports totalTokens and provider "anthropic".
func scalingTestMessages(totalTokens int) []Message {
	ai := AI("hi")
	ai.UsageMetadata = UsageMetadata{TotalTokens: totalTokens}
	ai.ResponseMetadata = map[string]any{"model_provider": "anthropic"}
	return []Message{
		Human("hello world"),
		ai,
		Human("more text here"),
	}
}

// TestCountTokensApproximatelyUsageMetadataScaling mirrors Python's
// count_tokens_approximately(use_usage_metadata_scaling=True): the estimate is
// scaled by last_ai_total_tokens / approx_at_last_ai, clamped to [1.0, 1.25],
// with a single final ceil.
func TestCountTokensApproximatelyUsageMetadataScaling(t *testing.T) {
	// Baseline: scaling disabled (default) → 21.
	if got := CountTokensApproximately(scalingTestMessages(20)); got != 21 {
		t.Fatalf("unscaled count = %d, want 21", got)
	}

	// Scaling on, factor = 20/13 ≈ 1.54 → clamped to 1.25 → ceil(21*1.25) = 27.
	if got := CountTokensApproximately(scalingTestMessages(20), WithUsageMetadataScaling(true)); got != 27 {
		t.Fatalf("scaled count (clamped up) = %d, want 27", got)
	}

	// In-range factor: 15/13 ≈ 1.15 → ceil(21*15/13) = ceil(24.23) = 25.
	if got := CountTokensApproximately(scalingTestMessages(15), WithUsageMetadataScaling(true)); got != 25 {
		t.Fatalf("scaled count (in range) = %d, want 25", got)
	}

	// Lower clamp: 5/13 ≈ 0.38 → clamped to 1.0 → unchanged 21.
	if got := CountTokensApproximately(scalingTestMessages(5), WithUsageMetadataScaling(true)); got != 21 {
		t.Fatalf("scaled count (clamped down) = %d, want 21", got)
	}
}

// TestCountTokensApproximatelyUsageMetadataScalingNotApplied covers the
// conditions under which Python skips scaling.
func TestCountTokensApproximatelyUsageMetadataScalingNotApplied(t *testing.T) {
	scaling := WithUsageMetadataScaling(true)

	// AI message without usage metadata → no scale.
	noUsage := scalingTestMessages(0)
	if got := CountTokensApproximately(noUsage, scaling); got != 21 {
		t.Fatalf("missing usage_metadata: got %d, want 21", got)
	}

	// Inconsistent model_provider across AI messages → no scale.
	mismatch := scalingTestMessages(20)
	other := AI("x")
	other.UsageMetadata = UsageMetadata{TotalTokens: 50}
	other.ResponseMetadata = map[string]any{"model_provider": "openai"}
	mismatch = append(mismatch, other)
	// unscaled: 21 + (3 + ceil((1+9)/4)=3) = 27
	if got := CountTokensApproximately(mismatch, scaling); got != 27 {
		t.Fatalf("provider mismatch: got %d, want 27 (unscaled)", got)
	}

	// No model_provider on any AI message → no scale.
	noProvider := scalingTestMessages(20)
	noProvider[1].ResponseMetadata = nil
	if got := CountTokensApproximately(noProvider, scaling); got != 21 {
		t.Fatalf("missing provider: got %d, want 21", got)
	}

	// Fewer than two messages → no scale (Python requires len(messages) > 1).
	single := scalingTestMessages(20)[1:2]
	if got := CountTokensApproximately(single, scaling); got != 6 {
		t.Fatalf("single message: got %d, want 6", got)
	}
}
