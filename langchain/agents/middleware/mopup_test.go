package middleware

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
)

// FakeOpenAIChatModel exercises provider inference by class name: its reflect
// type name contains "OpenAI".
type FakeOpenAIChatModel struct{}

func TestInferProviderClassNameVariants(t *testing.T) {
	if got := InferProvider(FakeOpenAIChatModel{}, nil); got != "openai" {
		t.Fatalf("openai class name mismatch: %q", got)
	}
	if got := InferProvider(struct{}{}, nil); got != "" {
		t.Fatalf("unknown class name should yield empty provider, got %q", got)
	}
}

func TestDetectCreditCardLuhnDoublingAboveNine(t *testing.T) {
	// Digits >= 5 in doubled positions exercise the value > 9 branch of the
	// Luhn checksum (5555...4444 is a standard Luhn-valid test number).
	matches := DetectCreditCard("pay 5555-5555-5555-4444")
	if len(matches) != 1 {
		t.Fatalf("expected valid card to be detected: %#v", matches)
	}
}

func TestDetectURLSkipsBareDomainWithoutPath(t *testing.T) {
	matches := DetectURL("the domain example.com is not linked")
	if len(matches) != 0 {
		t.Fatalf("bare domain without www/path should not match: %#v", matches)
	}
}

func TestSummarizationMiddlewareDefaultKeepBelowRetention(t *testing.T) {
	middleware := NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		t.Fatal("summarizer should not be called")
		return "", nil
	})
	middleware.Trigger = []TriggerClause{{Messages: 1}}
	middleware.Keep = KeepPolicy{} // no policy: DefaultMessagesToKeep fallback
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("one"), messages.Human("two"),
	}})
	if err != nil || update != nil {
		t.Fatalf("fewer messages than the retention default should not summarize: %#v %v", update, err)
	}
}

func TestToolCallLimitMiddlewareNoMatchingCalls(t *testing.T) {
	limit := 1
	middleware, err := NewToolCallLimitMiddleware("search", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "calc"}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update when no calls match: %#v %v", update, err)
	}
}

func TestToolCallLimitMiddlewareMissingMessagesKey(t *testing.T) {
	limit := 1
	middleware, err := NewToolCallLimitMiddleware("", &limit, nil, "continue")
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"other": 1})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without messages key: %#v %v", update, err)
	}
}

func TestIntFromAnyNumericTypes(t *testing.T) {
	cases := []struct {
		value any
		want  int
	}{
		{1, 1},
		{int64(2), 2},
		{float64(3), 3},
		{float32(4), 4},
		{"nope", 0},
		{nil, 0},
	}
	for _, tt := range cases {
		if got := intFromAny(tt.value); got != tt.want {
			t.Fatalf("intFromAny(%v) = %d, want %d", tt.value, got, tt.want)
		}
	}
	// Nil state yields an empty count map.
	if got := mapStringIntFromState(nil, "key"); len(got) != 0 {
		t.Fatalf("nil state mismatch: %#v", got)
	}
}

func TestPIIMiddlewareBeforeModelSkipsEmptyToolContent(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIApplyToInput(false), WithPIIApplyToToolResults(true))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "lookup"}}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		ai,
		messages.Tool("1", ""),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update for empty tool content: %#v %v", update, err)
	}
}

func TestPIIMiddlewareAfterModelBlockErrorOnInvalidToolCallArgs(t *testing.T) {
	middleware, err := NewPIIMiddleware("email",
		WithPIIApplyToInput(false),
		WithPIIApplyToOutput(true),
		WithPIIStrategy(RedactionBlock),
	)
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("")
	ai.InvalidToolCalls = []messages.ToolCall{{ID: "1", Name: "send", Args: map[string]any{"to": "user@example.com"}}}
	if _, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}}); err == nil {
		t.Fatal("expected PIIDetectionError from invalid tool call args")
	}
}

func TestShellExecutionPolicyRunnerOption(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellExecutionPolicyRunner(HostExecutionPolicy{}))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "printf ok", false)
	if err != nil || result.Output != "ok" {
		t.Fatalf("run mismatch: %#v %v", result, err)
	}
}

func TestShellRunWithStateRestartSpawnError(t *testing.T) {
	middleware, err := NewShellToolMiddleware(
		t.TempDir(),
		WithShellCommand("/nonexistent-lg-shell", "-c"),
		WithShellStartupCommands("printf hi"),
	)
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	state := map[string]any{}
	state[ShellSessionResourcesKey] = &ShellSessionResources{WorkspaceRoot: middleware.WorkspaceRoot}
	if _, err := middleware.RunWithState(context.Background(), state, "", true); err == nil ||
		!strings.Contains(err.Error(), "startup command") {
		t.Fatalf("expected restart spawn error, got %v", err)
	}
}

func TestShellPersistentSessionStartFailure(t *testing.T) {
	middleware, err := NewShellToolMiddleware(
		t.TempDir(),
		WithShellPersistentSession(),
		WithShellCommand("/nonexistent-lg-shell"),
	)
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	if _, err := middleware.Run(context.Background(), "printf hi", false); err == nil ||
		!strings.Contains(err.Error(), "start persistent shell") {
		t.Fatalf("expected persistent start error, got %v", err)
	}
}

func TestShellPersistentSessionRedactionError(t *testing.T) {
	rule, err := (RedactionRule{PIIType: "email", Strategy: RedactionBlock}).Resolve()
	if err != nil {
		t.Fatalf("resolve rule: %v", err)
	}
	middleware, err := NewShellToolMiddleware(
		t.TempDir(),
		WithShellPersistentSession(),
		WithShellRedactionRules(rule),
	)
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	if _, err := middleware.Run(context.Background(), "echo user@example.com", false); err == nil {
		t.Fatal("expected persistent redaction block error")
	}
	middleware.stopPersistentSession(5 * time.Second)
}

func TestShellSessionExecuteDefaultTimeout(t *testing.T) {
	s := newStartedSession(t)
	// A non-positive timeout falls back to the 30s default.
	r, err := s.Execute(context.Background(), "echo ok", 0)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if r.Output != "ok\n" {
		t.Fatalf("default timeout mismatch: %#v", r)
	}
}

func TestShellSessionExecuteOutputBeforeEOF(t *testing.T) {
	s := newStartedSession(t)
	// The shell prints output and then exits without a done marker; the
	// buffered output is still returned.
	r, err := s.Execute(context.Background(), "echo partial; exit", 10*time.Second)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(r.Output, "partial") {
		t.Fatalf("buffered output mismatch: %#v", r)
	}
}
