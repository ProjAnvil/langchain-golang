package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestRedactionDetectsAndAppliesStrategies(t *testing.T) {
	emailText := "Contact me at a@example.com"
	emails := DetectEmail(emailText)
	if len(emails) != 1 || emails[0].Value != "a@example.com" {
		t.Fatalf("email matches: %#v", emails)
	}
	redacted, err := ApplyRedactionStrategy(emailText, emails, RedactionRedact)
	if err != nil {
		t.Fatalf("redact: %v", err)
	}
	if redacted != "Contact me at [REDACTED_EMAIL]" {
		t.Fatalf("redacted mismatch: %q", redacted)
	}

	cards := DetectCreditCard("card 4111-1111-1111-1111")
	if len(cards) != 1 {
		t.Fatalf("credit card matches: %#v", cards)
	}
	masked, err := ApplyRedactionStrategy("card 4111-1111-1111-1111", cards, RedactionMask)
	if err != nil {
		t.Fatalf("mask: %v", err)
	}
	if masked != "card ****-****-****-1111" {
		t.Fatalf("masked mismatch: %q", masked)
	}

	hashed, err := ApplyRedactionStrategy(emailText, emails, RedactionHash)
	if err != nil {
		t.Fatalf("hash: %v", err)
	}
	if !strings.Contains(hashed, "<email_hash:") {
		t.Fatalf("hash mismatch: %q", hashed)
	}
}

func TestPIIMiddlewareBeforeModelRedactsInputAndToolResults(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIApplyToToolResults(true))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "lookup"}}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("email me at user@example.com"),
		ai,
		messages.Tool("1", "tool saw tool@example.com"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if !strings.Contains(msgs[0].Content, "[REDACTED_EMAIL]") {
		t.Fatalf("human not redacted: %#v", msgs[0])
	}
	if !strings.Contains(msgs[2].Content, "[REDACTED_EMAIL]") {
		t.Fatalf("tool not redacted: %#v", msgs[2])
	}
}

func TestPIIMiddlewareAfterModelRedactsOutputAndToolArgs(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIApplyToInput(false), WithPIIApplyToOutput(true))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("send to bot@example.com")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "send", Args: map[string]any{"to": "user@example.com"}}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{messages.Human("hi"), ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if !strings.Contains(msgs[1].Content, "[REDACTED_EMAIL]") {
		t.Fatalf("ai content not redacted: %#v", msgs[1])
	}
	if msgs[1].ToolCalls[0].Args["to"] != "[REDACTED_EMAIL]" {
		t.Fatalf("tool args not redacted: %#v", msgs[1].ToolCalls[0].Args)
	}
}

func TestPIIMiddlewareBlockRaisesDetectionError(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIStrategy(RedactionBlock))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	_, err = middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{messages.Human("user@example.com")}})
	var piiErr PIIDetectionError
	if !errors.As(err, &piiErr) {
		t.Fatalf("expected PIIDetectionError, got %v", err)
	}
}

func TestPIIMiddlewareUnknownTypeRequiresDetector(t *testing.T) {
	_, err := NewPIIMiddleware("api_key")
	if err == nil {
		t.Fatal("expected unknown type error")
	}
}
