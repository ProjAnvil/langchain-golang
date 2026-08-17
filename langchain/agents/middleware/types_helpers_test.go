package middleware

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestToolCallRequestOverrideRuntimeAndStore(t *testing.T) {
	base := ToolCallRequest{
		ToolCall: ToolCall{Name: "search", Args: map[string]any{"q": "x"}, ID: "1"},
	}
	next := base.Override(
		WithToolCallRuntime("runtime-value"),
		WithStore(nil),
	)
	if next.Runtime != "runtime-value" {
		t.Fatalf("runtime not overridden: %#v", next.Runtime)
	}
	if base.Runtime != nil {
		t.Fatalf("original runtime mutated: %#v", base.Runtime)
	}
}

func TestCommandValidateForWrapModelCall(t *testing.T) {
	if err := (Command{}).ValidateForWrapModelCall(); err != nil {
		t.Fatalf("empty command should validate: %v", err)
	}
	if err := (Command{Goto: "end"}).ValidateForWrapModelCall(); err == nil ||
		!strings.Contains(err.Error(), "goto") {
		t.Fatalf("expected goto error, got %v", err)
	}
	if err := (Command{Resume: "x"}).ValidateForWrapModelCall(); err == nil ||
		!strings.Contains(err.Error(), "resume") {
		t.Fatalf("expected resume error, got %v", err)
	}
	if err := (Command{Graph: "sub"}).ValidateForWrapModelCall(); err == nil ||
		!strings.Contains(err.Error(), "graph") {
		t.Fatalf("expected graph error, got %v", err)
	}
}

func TestModelRequestSystemPromptTextFromBlocks(t *testing.T) {
	request := ModelRequest{}
	if got := request.SystemPromptText(); got != "" {
		t.Fatalf("nil system message should yield empty prompt, got %q", got)
	}

	system := messages.System("")
	system.ContentBlocks = []messages.ContentBlock{
		messages.TextBlock{Text: "first "},
		messages.TextBlock{Text: "second"},
	}
	request.SystemMessage = &system
	if got := request.SystemPromptText(); got != "first second" {
		t.Fatalf("block-derived prompt mismatch: %q", got)
	}

	withContent := messages.System("direct")
	request.SystemMessage = &withContent
	if got := request.SystemPromptText(); got != "direct" {
		t.Fatalf("content should take precedence over blocks: %q", got)
	}
}

func TestCloneMessagesNil(t *testing.T) {
	if got := cloneMessages(nil); got != nil {
		t.Fatalf("cloneMessages(nil) should be nil, got %#v", got)
	}
}
