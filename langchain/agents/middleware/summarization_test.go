package middleware

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/modelprofiles"
)

// fakeProfileModel implements ModelProfileProvider for tests exercising
// fraction-based trigger/keep resolution.
type fakeProfileModel struct {
	profile modelprofiles.Profile
}

func (f fakeProfileModel) ModelProfile() modelprofiles.Profile {
	return f.profile
}

func TestSummarizationMiddlewareSummarizesWhenTriggered(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(prompt string, msgs []messages.Message) (string, error) {
		if len(msgs) != 2 {
			t.Fatalf("summarized message count mismatch: %#v", msgs)
		}
		if !strings.Contains(prompt, "human: one") {
			t.Fatalf("prompt missing buffer: %q", prompt)
		}
		return "summary text", nil
	})
	middleware.Trigger = []TriggerClause{{Messages: 3}}
	middleware.Keep = KeepPolicy{Messages: 1}

	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"),
		messages.AI("two"),
		messages.Human("three"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 2 || msgs[0].Content != "summary text" || msgs[1].Content != "three" {
		t.Fatalf("summary messages mismatch: %#v", msgs)
	}
	if msgs[0].ResponseMetadata["summary"] != true {
		t.Fatalf("summary metadata missing: %#v", msgs[0].ResponseMetadata)
	}
}

func TestSummarizationMiddlewareDoesNotTriggerBelowThreshold(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		t.Fatal("summarizer should not be called")
		return "", nil
	})
	middleware.Trigger = []TriggerClause{{Messages: 5}}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{messages.Human("one")}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if update != nil {
		t.Fatalf("expected nil update, got %#v", update)
	}
}

func TestSummarizationMiddlewareFractionTrigger(t *testing.T) {
	called := false
	middleware := NewSummarizationMiddleware(func(prompt string, msgs []messages.Message) (string, error) {
		called = true
		return "summary", nil
	})
	middleware.Model = fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 20}}
	middleware.TokenCounter = func(msgs []messages.Message) int { return len(msgs) * 5 }
	middleware.Trigger = []TriggerClause{{Fraction: 0.5}} // threshold = 10 tokens
	middleware.Keep = KeepPolicy{Messages: 1}

	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"),
		messages.Human("two"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if !called {
		t.Fatalf("expected summarizer to be called")
	}
	if update == nil {
		t.Fatalf("expected update")
	}
}

func TestSummarizationMiddlewareFractionTriggerBelowThreshold(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		t.Fatal("summarizer should not be called")
		return "", nil
	})
	middleware.Model = fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 20}}
	middleware.TokenCounter = func(msgs []messages.Message) int { return len(msgs) * 5 }
	middleware.Trigger = []TriggerClause{{Fraction: 0.5}} // threshold = 10 tokens

	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if update != nil {
		t.Fatalf("expected nil update, got %#v", update)
	}
}

func TestSummarizationMiddlewareFractionTriggerWithoutProfileNeverFires(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		t.Fatal("summarizer should not be called")
		return "", nil
	})
	// No Model set: fraction clauses can't be resolved and must not fire,
	// rather than erroring or falling back to some other metric.
	middleware.Trigger = []TriggerClause{{Fraction: 0.01}}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), messages.Human("two"), messages.Human("three"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if update != nil {
		t.Fatalf("expected nil update, got %#v", update)
	}
}

func TestSummarizationMiddlewareFractionKeep(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(prompt string, msgs []messages.Message) (string, error) {
		return "summary", nil
	})
	middleware.Model = fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 20}}
	middleware.TokenCounter = func(msgs []messages.Message) int { return len(msgs) * 5 }
	middleware.Trigger = []TriggerClause{{Messages: 4}}
	middleware.Keep = KeepPolicy{Fraction: 0.5} // budget = 10 tokens -> keep last 2 messages

	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"),
		messages.Human("two"),
		messages.Human("three"),
		messages.Human("four"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 3 || msgs[1].Content != "three" || msgs[2].Content != "four" {
		t.Fatalf("unexpected keep result: %#v", msgs)
	}
}

func TestSummarizationMiddlewareTrimTokensToSummarize(t *testing.T) {
	var received []messages.Message
	middleware := NewSummarizationMiddleware(func(prompt string, msgs []messages.Message) (string, error) {
		received = msgs
		return "summary", nil
	})
	middleware.TokenCounter = func(msgs []messages.Message) int { return len(msgs) * 5 }
	middleware.Trigger = []TriggerClause{{Messages: 5}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TrimTokensToSummarize = 10 // budget only fits the trailing 2 of the 4 summarized messages

	_, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("h1"),
		messages.AI("a1"),
		messages.Human("h2"),
		messages.AI("a2"),
		messages.Human("h3"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if len(received) != 2 || received[0].Content != "h2" || received[1].Content != "a2" {
		t.Fatalf("unexpected trimmed messages passed to summarizer: %#v", received)
	}
}

func TestSummarizationMiddlewareTrimDisabled(t *testing.T) {
	var received []messages.Message
	middleware := NewSummarizationMiddleware(func(prompt string, msgs []messages.Message) (string, error) {
		received = msgs
		return "summary", nil
	})
	middleware.TokenCounter = func(msgs []messages.Message) int { return len(msgs) * 5 }
	middleware.Trigger = []TriggerClause{{Messages: 5}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TrimTokensToSummarize = 0

	_, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.AI("a1"), // would be dropped by dropUntilHuman if trimming were enabled
		messages.Human("h1"),
		messages.AI("a2"),
		messages.Human("h2"),
		messages.Human("h3"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if len(received) != 4 || received[0].Content != "a1" {
		t.Fatalf("expected untrimmed messages when trimming disabled, got %#v", received)
	}
}
