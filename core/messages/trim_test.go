package messages

import (
	"strings"
	"testing"
)

func TestCountTokensApproximately(t *testing.T) {
	msgs := []Message{Human("hello world")}
	got := CountTokensApproximately(msgs)
	// extra 3 + ceil(11/4)=3 → 6
	if got != 6 {
		t.Fatalf("CountTokensApproximately = %d, want 6", got)
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

	// Defaults (charsPerToken=4, extra=3, countName=false, tokensPerImage=85):
	//   human: 3 + ceil(11/4)=3                = 6
	//   ai:    3 + ceil(6/4)=2 + ceil(9/4)=3   = 8   (tool call name + args JSON)
	//   tool:  3 + ceil(6/4)=2                 = 5   (tool_call_id)
	//   image: 3 + 85                          = 88
	if got := CountTokensApproximately(msgs); got != 107 {
		t.Fatalf("CountTokensApproximately = %d, want 107", got)
	}

	// Custom options: charsPerToken=2, extra=10, countName=true.
	//   human: 10 + ceil(11/2)=6 + ceil(5/2)=3 = 19
	//   ai:    10 + ceil(6/2)=3 + ceil(9/2)=5  = 18
	//   tool:  10 + ceil(6/2)=3                = 13
	//   image: 10 + 85                         = 95
	got := CountTokensApproximately(msgs,
		WithCharsPerToken(2),
		WithExtraTokensPerMessage(10),
		WithCountName(true),
	)
	if got != 145 {
		t.Fatalf("CountTokensApproximately with options = %d, want 145", got)
	}
}
