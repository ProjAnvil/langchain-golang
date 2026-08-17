package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestPIIMiddlewareName(t *testing.T) {
	middleware, err := NewPIIMiddleware("email")
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	if middleware.Name() != "PIIMiddleware[email]" {
		t.Fatalf("name mismatch: %q", middleware.Name())
	}
}

func TestPIIMiddlewareCustomDetector(t *testing.T) {
	detector := func(content string) []PIIMatch {
		if idx := strings.Index(content, "SECRET"); idx >= 0 {
			return []PIIMatch{{Type: "custom", Value: "SECRET", Start: idx, End: idx + 6}}
		}
		return nil
	}
	middleware, err := NewPIIMiddleware("custom", WithPIIDetector(detector))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("the SECRET is out"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if msgs[0].Content != "the [REDACTED_CUSTOM] is out" {
		t.Fatalf("custom detector redaction mismatch: %q", msgs[0].Content)
	}
}

func TestPIIMiddlewareBeforeModelNoopPaths(t *testing.T) {
	// Both input and tool-result application disabled.
	middleware, err := NewPIIMiddleware("email", WithPIIApplyToInput(false))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("user@example.com"),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update when disabled: %#v %v", update, err)
	}

	// No messages in state.
	middleware, err = NewPIIMiddleware("email")
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	update, err = middleware.BeforeModel(context.Background(), nil)
	if err != nil || update != nil {
		t.Fatalf("expected nil update for nil state: %#v %v", update, err)
	}
	update, err = middleware.BeforeModel(context.Background(), map[string]any{"messages": "nope"})
	if err != nil || update != nil {
		t.Fatalf("expected nil update for wrong message type: %#v %v", update, err)
	}

	// Human message without PII: unmodified.
	update, err = middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("nothing sensitive"),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update when nothing matched: %#v %v", update, err)
	}
}

func TestPIIMiddlewareBeforeModelToolResultBlockError(t *testing.T) {
	middleware, err := NewPIIMiddleware("email",
		WithPIIApplyToInput(false),
		WithPIIApplyToToolResults(true),
		WithPIIStrategy(RedactionBlock),
	)
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "lookup"}}
	_, err = middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("hi"),
		ai,
		messages.Tool("1", "saw user@example.com"),
	}})
	var piiErr PIIDetectionError
	if !errors.As(err, &piiErr) {
		t.Fatalf("expected PIIDetectionError from tool result, got %v", err)
	}
}

func TestPIIMiddlewareBeforeModelToolResultsWithoutAI(t *testing.T) {
	middleware, err := NewPIIMiddleware("email",
		WithPIIApplyToInput(false),
		WithPIIApplyToToolResults(true),
	)
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	// No AI message: tool results are not scanned.
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Tool("1", "saw user@example.com"),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without AI message: %#v %v", update, err)
	}
}

func TestPIIMiddlewareAfterModelNoopPaths(t *testing.T) {
	// ApplyToOutput disabled.
	middleware, err := NewPIIMiddleware("email")
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.AI("contact user@example.com"),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update when output disabled: %#v %v", update, err)
	}

	middleware, err = NewPIIMiddleware("email", WithPIIApplyToInput(false), WithPIIApplyToOutput(true))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	// No messages.
	update, err = middleware.AfterModel(context.Background(), map[string]any{})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without messages: %#v %v", update, err)
	}
	// No AI message.
	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("user@example.com"),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without AI message: %#v %v", update, err)
	}
	// AI message without PII: unchanged.
	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.AI("nothing sensitive"),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update when nothing matched: %#v %v", update, err)
	}
}

func TestPIIMiddlewareAfterModelBlockErrorOnContent(t *testing.T) {
	middleware, err := NewPIIMiddleware("email",
		WithPIIApplyToInput(false),
		WithPIIApplyToOutput(true),
		WithPIIStrategy(RedactionBlock),
	)
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	_, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.AI("write to user@example.com"),
	}})
	var piiErr PIIDetectionError
	if !errors.As(err, &piiErr) {
		t.Fatalf("expected PIIDetectionError, got %v", err)
	}
}

func TestPIIMiddlewareAfterModelBlockErrorOnToolArgs(t *testing.T) {
	middleware, err := NewPIIMiddleware("email",
		WithPIIApplyToInput(false),
		WithPIIApplyToOutput(true),
		WithPIIStrategy(RedactionBlock),
	)
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "send", Args: map[string]any{"to": "user@example.com"}}}
	_, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	var piiErr PIIDetectionError
	if !errors.As(err, &piiErr) {
		t.Fatalf("expected PIIDetectionError from tool args, got %v", err)
	}
}

func TestPIIMiddlewareAfterModelRedactsInvalidToolCalls(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIApplyToInput(false), WithPIIApplyToOutput(true))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("")
	ai.InvalidToolCalls = []messages.ToolCall{{ID: "1", Name: "send", Args: map[string]any{
		"to":    "user@example.com",
		"count": 3,
	}}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	args := msgs[0].InvalidToolCalls[0].Args
	if args["to"] != "[REDACTED_EMAIL]" {
		t.Fatalf("invalid tool call args not redacted: %#v", args)
	}
	if args["count"] != 3 {
		t.Fatalf("non-string args must be preserved: %#v", args)
	}
}

func TestPIIMiddlewareAfterModelNilToolArgs(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIApplyToInput(false), WithPIIApplyToOutput(true))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "send"}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update for nil args: %#v %v", update, err)
	}
}

func TestPIIStreamTransformerDefaults(t *testing.T) {
	xform := NewPIIStreamTransformer(nil)
	if xform.Lookback() != defaultStreamLookback {
		t.Fatalf("empty patterns should use the default lookback, got %d", xform.Lookback())
	}
	if len(xform.Patterns()) != 0 {
		t.Fatalf("expected no patterns, got %#v", xform.Patterns())
	}
	// Flush without any transform call is a no-op.
	if got := xform.Flush(); got != "" {
		t.Fatalf("flush without state should be empty, got %q", got)
	}
	// Empty text deltas pass through untouched.
	tf := xform.TransformModelStream(func(s string) string { return s })
	if got := tf(""); got != "" {
		t.Fatalf("empty delta mismatch: %q", got)
	}
	// No patterns: nothing is redacted, deltas flow through with the lookback
	// tail held back, and Flush releases the remainder.
	out := tf("hello world") + xform.Flush()
	if out != "hello world" {
		t.Fatalf("no-pattern stream mismatch: %q", out)
	}
}

func TestPIIStreamTransformerPatternsReturnsCopy(t *testing.T) {
	xform := NewPIIStreamTransformer([]string{`SSN-\d+`})
	patterns := xform.Patterns()
	if len(patterns) != 1 {
		t.Fatalf("patterns mismatch: %#v", patterns)
	}
	patterns[0] = nil
	if xform.Patterns()[0] == nil {
		t.Fatal("Patterns must return a copy")
	}
}
