package middleware

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
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

	update, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": []messages.Message{
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
	update, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": []messages.Message{messages.Human("one")}})
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

	update, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": []messages.Message{
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

	update, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": []messages.Message{
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
	update, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": []messages.Message{
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

	update, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": []messages.Message{
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

	_, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": []messages.Message{
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

	_, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": []messages.Message{
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

// overflowTestMessages yields trimmed == [Human("abcdef"), AI("ghij")] under a
// Keep{Messages: 1} policy, i.e. bufferString(trimmed) ==
// "human: abcdef\nai: ghij\n" (23 chars).
func overflowTestMessages() []messages.Message {
	return []messages.Message{
		messages.Human("abcdef"),
		messages.AI("ghij"),
		messages.Human("final"),
	}
}

func TestSummarizationMiddlewareRetriesOnceOnContextOverflow(t *testing.T) {
	var prompts []string
	middleware := NewSummarizationMiddleware(func(prompt string, msgs []messages.Message) (string, error) {
		prompts = append(prompts, prompt)
		if len(prompts) == 1 {
			return "", fmt.Errorf("wrap: %w", lcerrors.ErrModelContextOverflow)
		}
		if len(msgs) != 2 {
			t.Fatalf("retry must receive the original trimmed messages, got %#v", msgs)
		}
		return "recovered summary", nil
	})
	middleware.Trigger = []TriggerClause{{Messages: 3}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TokenCounter = func([]messages.Message) int { return 23 }
	// Trailing whitespace makes the retry prompt's rstrip observable.
	middleware.SummaryPrompt = "Summary:\n{messages}\n  \n"

	update, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": overflowTestMessages()})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	if len(prompts) != 2 {
		t.Fatalf("expected exactly 2 summarizer calls, got %d", len(prompts))
	}
	if !strings.Contains(prompts[0], "abcdef") || !strings.Contains(prompts[0], "ghij") {
		t.Fatalf("first prompt should carry the full history: %q", prompts[0])
	}
	// tokens = 23, budget = 23/2 = 11 (no model profile), keepChars = 11,
	// head = 2, tail = 9; the splice's and template's trailing whitespace are
	// trimmed like Python's .rstrip().
	want := "Summary:\nhu\n\n[... middle of conversation omitted ...]\n\nai: ghij"
	if prompts[1] != want {
		t.Fatalf("retry prompt mismatch:\n got: %q\nwant: %q", prompts[1], want)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 2 || msgs[0].Content != "recovered summary" || msgs[1].Content != "final" {
		t.Fatalf("unexpected summary result: %#v", msgs)
	}
}

func TestSummarizationMiddlewareSecondOverflowPropagates(t *testing.T) {
	var calls int
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		calls++
		if calls == 1 {
			return "", lcerrors.NewProviderError("openai", "/responses", 400, "maximum context length is 4097 tokens", 0)
		}
		return "", lcerrors.NewProviderError("openai", "/responses", 400, "context_length_exceeded on retry", 0)
	})
	middleware.Trigger = []TriggerClause{{Messages: 3}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TokenCounter = func([]messages.Message) int { return 23 }

	_, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": overflowTestMessages()})
	if err == nil {
		t.Fatal("expected the second overflow error to propagate")
	}
	if calls != 2 {
		t.Fatalf("expected exactly 2 summarizer calls, got %d", calls)
	}
	if !errors.Is(err, lcerrors.ErrModelContextOverflow) {
		t.Fatalf("expected context overflow sentinel, got %v", err)
	}
	var providerErr *lcerrors.ProviderError
	if !errors.As(err, &providerErr) || !strings.Contains(providerErr.Body, "on retry") {
		t.Fatalf("expected the second error verbatim, got %v", err)
	}
}

func TestSummarizationMiddlewareOverflowWithoutRetryBudgetPropagates(t *testing.T) {
	var calls int
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		calls++
		return "", fmt.Errorf("wrap: %w", lcerrors.ErrModelContextOverflow)
	})
	middleware.Trigger = []TriggerClause{{Messages: 3}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TokenCounter = func([]messages.Message) int { return 23 }
	// max_input_tokens*2/5 == 0, so keepChars == 0 and no retry prompt exists.
	middleware.Model = fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 1}}

	_, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": overflowTestMessages()})
	if err == nil {
		t.Fatal("expected the first overflow error to propagate unchanged")
	}
	if calls != 1 {
		t.Fatalf("expected exactly 1 summarizer call, got %d", calls)
	}
	if !errors.Is(err, lcerrors.ErrModelContextOverflow) {
		t.Fatalf("expected context overflow sentinel, got %v", err)
	}
}

func TestSummarizationMiddlewareDoesNotRetryNonOverflowErrors(t *testing.T) {
	var calls int
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		calls++
		return "", errors.New("boom")
	})
	middleware.Trigger = []TriggerClause{{Messages: 3}}
	middleware.Keep = KeepPolicy{Messages: 1}
	middleware.TokenCounter = func([]messages.Message) int { return 23 }

	_, err := middleware.BeforeModel(t.Context(), map[string]any{"messages": overflowTestMessages()})
	if err == nil || err.Error() != "boom" {
		t.Fatalf("expected the original error verbatim, got %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected exactly 1 summarizer call, got %d", calls)
	}
}

func TestSummarizationMiddlewareOverflowRetryPromptBudgetPaths(t *testing.T) {
	// bufferString(trimmed) == "human: abcdef\nai: ghij\n" (23 chars); the
	// token counter reports a constant 23 tokens for the formatted history.
	trimmed := []messages.Message{messages.Human("abcdef"), messages.AI("ghij")}
	newMiddleware := func() *SummarizationMiddleware {
		m := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
			return "unused", nil
		})
		m.TokenCounter = func([]messages.Message) int { return 23 }
		m.SummaryPrompt = "Summary:\n{messages}\n  \n"
		return m
	}

	t.Run("no profile halves the token count", func(t *testing.T) {
		prompt, ok := newMiddleware().overflowRetryPrompt(trimmed)
		if !ok {
			t.Fatal("expected a retry prompt")
		}
		// budget = 23/2 = 11; keepChars = 23*11/23 = 11; head = 2; tail = 9.
		want := "Summary:\nhu\n\n[... middle of conversation omitted ...]\n\nai: ghij"
		if prompt != want {
			t.Fatalf("retry prompt mismatch:\n got: %q\nwant: %q", prompt, want)
		}
	})

	t.Run("profile caps the budget at 40 percent", func(t *testing.T) {
		m := newMiddleware()
		m.Model = fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 20}}
		prompt, ok := m.overflowRetryPrompt(trimmed)
		if !ok {
			t.Fatal("expected a retry prompt")
		}
		// budget = min(11, 20*2/5 = 8) = 8; keepChars = 8; head = 2; tail = 6.
		want := "Summary:\nhu\n\n[... middle of conversation omitted ...]\n\n ghij"
		if prompt != want {
			t.Fatalf("retry prompt mismatch:\n got: %q\nwant: %q", prompt, want)
		}
	})

	t.Run("large profile leaves the half-budget path intact", func(t *testing.T) {
		m := newMiddleware()
		m.Model = fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 100}}
		prompt, ok := m.overflowRetryPrompt(trimmed)
		if !ok {
			t.Fatal("expected a retry prompt")
		}
		// budget = min(11, 100*2/5 = 40) = 11, identical to the no-profile path.
		want := "Summary:\nhu\n\n[... middle of conversation omitted ...]\n\nai: ghij"
		if prompt != want {
			t.Fatalf("retry prompt mismatch:\n got: %q\nwant: %q", prompt, want)
		}
	})

	t.Run("zero budget yields no retry prompt", func(t *testing.T) {
		m := newMiddleware()
		m.Model = fakeProfileModel{profile: modelprofiles.Profile{"max_input_tokens": 1}}
		if prompt, ok := m.overflowRetryPrompt(trimmed); ok {
			t.Fatalf("expected no retry prompt, got %q", prompt)
		}
	})
}
