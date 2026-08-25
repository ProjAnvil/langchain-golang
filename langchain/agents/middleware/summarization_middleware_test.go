package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/modelprofiles"
)

func TestSummarizationMiddlewareNoMessages(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		t.Fatal("summarizer should not be called")
		return "", nil
	})
	update, err := middleware.BeforeModel(context.Background(), map[string]any{})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without messages: %#v %v", update, err)
	}
}

func TestSummarizationMiddlewareRequiresSummarizer(t *testing.T) {
	middleware := &SummarizationMiddleware{
		Trigger: []TriggerClause{{Messages: 1}},
		Keep:    KeepPolicy{Messages: 1},
	}
	_, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), messages.Human("two"),
	}})
	if err == nil || !strings.Contains(err.Error(), "summarizer function") {
		t.Fatalf("expected missing summarizer error, got %v", err)
	}
}

func TestSummarizationMiddlewareKeepCoversAllMessages(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		t.Fatal("summarizer should not be called")
		return "", nil
	})
	middleware.Trigger = []TriggerClause{{Messages: 1}}
	middleware.Keep = KeepPolicy{Messages: 10}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), messages.Human("two"),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update when keep covers everything: %#v %v", update, err)
	}
}

func TestSummarizationMiddlewareSummarizeErrorPropagates(t *testing.T) {
	wantErr := errors.New("summarizer down")
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		return "", wantErr
	})
	middleware.Trigger = []TriggerClause{{Messages: 2}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TrimTokensToSummarize = 0
	_, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), messages.Human("two"),
	}})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected summarizer error, got %v", err)
	}
}

func TestSummarizationMiddlewareEmptyAfterTrim(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		t.Fatal("summarizer should not be called for an empty trim")
		return "", nil
	})
	middleware.Trigger = []TriggerClause{{Messages: 2}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TrimTokensToSummarize = 1
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one two three four five"),
		messages.Human("keep me"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if msgs[0].Content != "Previous conversation was too long to summarize." {
		t.Fatalf("placeholder summary mismatch: %q", msgs[0].Content)
	}
}

func TestSummarizationMiddlewareDefaultTrigger(t *testing.T) {
	called := false
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		called = true
		return "summary", nil
	})
	middleware.Trigger = nil // resolves to the default {Messages: 50}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TrimTokensToSummarize = 0

	below := make([]messages.Message, 49)
	for i := range below {
		below[i] = messages.Human("msg")
	}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": below})
	if err != nil || update != nil || called {
		t.Fatalf("default trigger should not fire at 49 messages: %#v %v", update, err)
	}

	at := append(below, messages.Human("msg"))
	update, err = middleware.BeforeModel(context.Background(), map[string]any{"messages": at})
	if err != nil || update == nil || !called {
		t.Fatalf("default trigger should fire at 50 messages: %#v %v", update, err)
	}
}

func TestSummarizationMiddlewareTokenTrigger(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		return "summary", nil
	})
	middleware.Trigger = []TriggerClause{{Tokens: 10}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TrimTokensToSummarize = 0
	middleware.TokenCounter = func(msgs []messages.Message) int { return len(msgs) * 5 }

	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"),
	}})
	if err != nil || update != nil {
		t.Fatalf("token trigger should not fire below threshold: %#v %v", update, err)
	}

	update, err = middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), messages.Human("two"), messages.Human("three"),
	}})
	if err != nil || update == nil {
		t.Fatalf("token trigger should fire at threshold: %#v %v", update, err)
	}
}

func TestSummarizationMiddlewareReportedTokensTrigger(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		return "summary", nil
	})
	middleware.Trigger = []TriggerClause{{Tokens: 100}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TrimTokensToSummarize = 0
	middleware.TokenCounter = func(msgs []messages.Message) int { return 1 } // estimate below threshold
	middleware.Model = fakeLLMTypeModel{llmType: "anthropic-chat"}

	ai := messages.AI("answer")
	ai.UsageMetadata.TotalTokens = 150
	ai.ResponseMetadata = map[string]any{"model_provider": "anthropic"}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), ai, messages.Human("two"),
	}})
	if err != nil || update == nil {
		t.Fatalf("reported tokens should trigger summarization: %#v %v", update, err)
	}

	// Provider mismatch: no trigger.
	ai.ResponseMetadata = map[string]any{"model_provider": "openai"}
	update, err = middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), ai, messages.Human("two"),
	}})
	if err != nil || update != nil {
		t.Fatalf("provider mismatch should not trigger: %#v %v", update, err)
	}
}

func TestShouldSummarizeBasedOnReportedTokensEdgeCases(t *testing.T) {
	middleware := &SummarizationMiddleware{Model: fakeLLMTypeModel{llmType: "anthropic-chat"}}

	// No AI message.
	if middleware.shouldSummarizeBasedOnReportedTokens([]messages.Message{messages.Human("hi")}, 10) {
		t.Fatal("no AI message should not trigger")
	}
	// No usage metadata.
	if middleware.shouldSummarizeBasedOnReportedTokens([]messages.Message{messages.AI("hi")}, 10) {
		t.Fatal("missing usage metadata should not trigger")
	}
	// Below threshold.
	ai := messages.AI("hi")
	ai.UsageMetadata.TotalTokens = 5
	if middleware.shouldSummarizeBasedOnReportedTokens([]messages.Message{ai}, 10) {
		t.Fatal("below-threshold usage should not trigger")
	}
	// Missing provider metadata.
	ai.UsageMetadata.TotalTokens = 50
	if middleware.shouldSummarizeBasedOnReportedTokens([]messages.Message{ai}, 10) {
		t.Fatal("missing provider should not trigger")
	}
	// Model without LLMType.
	ai.ResponseMetadata = map[string]any{"model_provider": "anthropic"}
	bare := &SummarizationMiddleware{}
	if bare.shouldSummarizeBasedOnReportedTokens([]messages.Message{ai}, 10) {
		t.Fatal("model without LLMType should not trigger")
	}
}

func TestProviderMatches(t *testing.T) {
	cases := []struct {
		provider string
		llmType  string
		want     bool
	}{
		{"anthropic", "anthropic-chat", true},
		{"openai", "openai-chat", true},
		{"anthropic", "anthropic", true},
		{"openai", "anthropic-chat", false},
		{"anthropic", "", false},
	}
	for _, tt := range cases {
		if got := providerMatches(tt.provider, tt.llmType); got != tt.want {
			t.Fatalf("providerMatches(%q, %q) = %v, want %v", tt.provider, tt.llmType, got, tt.want)
		}
	}
}

func TestSummarizationMiddlewareTokenKeepPolicy(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		return "summary", nil
	})
	middleware.Trigger = []TriggerClause{{Messages: 4}}
	middleware.Keep = KeepPolicy{Tokens: 10}
	middleware.TrimTokensToSummarize = 0
	middleware.TokenCounter = func(msgs []messages.Message) int { return len(msgs) * 5 }

	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), messages.Human("two"), messages.Human("three"), messages.Human("four"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 3 || msgs[1].Content != "three" || msgs[2].Content != "four" {
		t.Fatalf("token keep mismatch: %#v", msgs)
	}
}

func TestSummarizationMiddlewareFractionKeepWithoutProfile(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		return "summary", nil
	})
	middleware.Trigger = []TriggerClause{{Messages: 1}}
	middleware.Keep = KeepPolicy{Fraction: 0.5} // no model: falls back to DefaultMessagesToKeep
	middleware.TrimTokensToSummarize = 0

	msgs := make([]messages.Message, 25)
	for i := range msgs {
		msgs[i] = messages.Human("msg")
	}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": msgs})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	got := update["messages"].([]messages.Message)
	// summary + DefaultMessagesToKeep trailing messages.
	if len(got) != 1+DefaultMessagesToKeep {
		t.Fatalf("fallback keep mismatch: %d messages", len(got))
	}
}

func TestFindSafeCutoffPoint(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}
	msgs := []messages.Message{
		messages.Human("hi"),
		ai,
		messages.Tool("1", "result"),
		messages.Human("next"),
	}

	// Cutoff on the tool message snaps back to the originating AI message.
	if got := findSafeCutoffPoint(msgs, 2); got != 1 {
		t.Fatalf("cutoff should snap to the AI message, got %d", got)
	}
	// Cutoff on a non-tool message is unchanged.
	if got := findSafeCutoffPoint(msgs, 3); got != 3 {
		t.Fatalf("non-tool cutoff mismatch: %d", got)
	}
	// Out-of-range cutoff is unchanged.
	if got := findSafeCutoffPoint(msgs, 4); got != 4 {
		t.Fatalf("out-of-range cutoff mismatch: %d", got)
	}

	// No matching AI message: advance past the tool messages.
	orphan := []messages.Message{
		messages.Human("hi"),
		messages.Tool("missing", "result"),
		messages.Human("next"),
	}
	if got := findSafeCutoffPoint(orphan, 1); got != 2 {
		t.Fatalf("orphan tool cutoff should advance, got %d", got)
	}
}

func TestFractionThresholdClampsToOne(t *testing.T) {
	if got := fractionThreshold(10, 0.01); got != 1 {
		t.Fatalf("fraction threshold should clamp to 1, got %d", got)
	}
	if got := fractionThreshold(100, 0.5); got != 50 {
		t.Fatalf("fraction threshold mismatch: %d", got)
	}
}

func TestMaxInputTokensResolution(t *testing.T) {
	cases := []struct {
		name  string
		model any
		want  int
		ok    bool
	}{
		{"nil model", nil, 0, false},
		{"nil profile", fakeProfileModel{profile: nil}, 0, false},
		{"missing field", fakeProfileModel{profile: modelprofiles.Profile{}}, 0, false},
		{"int", fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 100}}, 100, true},
		{"int64", fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": int64(200)}}, 200, true},
		{"float64", fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 300.0}}, 300, true},
		{"wrong type", fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": "100"}}, 0, false},
		{"zero", fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 0}}, 0, false},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			middleware := &SummarizationMiddleware{Model: tt.model}
			got, ok := middleware.maxInputTokens()
			if got != tt.want || ok != tt.ok {
				t.Fatalf("maxInputTokens = (%d, %v), want (%d, %v)", got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestResolveTokenCounter(t *testing.T) {
	msgs := []messages.Message{messages.Human("abcd")}

	// User override always wins.
	override := resolveTokenCounter(fakeLLMTypeModel{llmType: "anthropic-chat"}, func([]messages.Message) int { return 42 })
	if got := override(msgs); got != 42 {
		t.Fatalf("override mismatch: %d", got)
	}
	// Anthropic models use the 3.3 chars/token counter:
	// 3 (per-message extra) + ceil((4 content + 4 role "user")/3.3) = 6.
	anthropic := resolveTokenCounter(fakeLLMTypeModel{llmType: "anthropic-chat"}, nil)
	if got := anthropic(msgs); got != 6 {
		t.Fatalf("anthropic counter mismatch: %d", got)
	}
	// Other/nil models use the default 4.0 chars/token counter:
	// 3 + ceil(8/4) = 5.
	if got := resolveTokenCounter(fakeLLMTypeModel{llmType: "openai-chat"}, nil)(msgs); got != 5 {
		t.Fatalf("default counter mismatch: %d", got)
	}
	if got := resolveTokenCounter(nil, nil)(msgs); got != 5 {
		t.Fatalf("nil model counter mismatch: %d", got)
	}
}

func TestSummarizationMiddlewareDefaultPromptUsed(t *testing.T) {
	var prompt string
	middleware := NewSummarizationMiddleware(func(p string, _ []messages.Message) (string, error) {
		prompt = p
		return "summary", nil
	})
	middleware.SummaryPrompt = "" // falls back to DefaultSummaryPrompt
	middleware.Trigger = []TriggerClause{{Messages: 2}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TrimTokensToSummarize = 0

	_, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), messages.Human("two"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if !strings.Contains(prompt, "Context Extraction Assistant") || !strings.Contains(prompt, "human: one") {
		t.Fatalf("default prompt mismatch: %q", prompt)
	}
}

func TestTokenBudgetTrimFromEnd(t *testing.T) {
	middleware := &SummarizationMiddleware{
		TokenCounter: func(msgs []messages.Message) int { return len(msgs) * 5 },
	}
	if got := middleware.tokenBudgetTrimFromEnd(nil, 10); len(got) != 0 {
		t.Fatalf("empty input mismatch: %#v", got)
	}

	msgs := []messages.Message{
		messages.System("sys"),
		messages.Human("one"),
		messages.Human("two"),
		messages.Human("three"),
	}
	// Budget fits only the trailing message, but the system message is kept.
	got := middleware.tokenBudgetTrimFromEnd(msgs, 5)
	if len(got) != 2 || got[0].Role != messages.RoleSystem || got[1].Content != "three" {
		t.Fatalf("system-preserving trim mismatch: %#v", got)
	}
}

func TestDropUntilHuman(t *testing.T) {
	if got := dropUntilHuman(nil); got != nil {
		t.Fatalf("empty input mismatch: %#v", got)
	}

	system := messages.System("sys")
	human := messages.Human("hi")
	ai := messages.AI("ai")

	got := dropUntilHuman([]messages.Message{system, ai, human})
	if len(got) != 2 || got[0].Role != messages.RoleSystem || got[1].Role != messages.RoleHuman {
		t.Fatalf("system+human mismatch: %#v", got)
	}
	got = dropUntilHuman([]messages.Message{ai, human})
	if len(got) != 1 || got[0].Role != messages.RoleHuman {
		t.Fatalf("human-only mismatch: %#v", got)
	}
	got = dropUntilHuman([]messages.Message{system, ai})
	if len(got) != 1 || got[0].Role != messages.RoleSystem {
		t.Fatalf("system-only mismatch: %#v", got)
	}
	if got := dropUntilHuman([]messages.Message{ai}); got != nil {
		t.Fatalf("no human should yield nil: %#v", got)
	}
}

func TestBufferString(t *testing.T) {
	got := bufferString([]messages.Message{messages.Human("hi"), messages.AI("yo")})
	if got != "human: hi\nai: yo\n" {
		t.Fatalf("buffer string mismatch: %q", got)
	}
}
