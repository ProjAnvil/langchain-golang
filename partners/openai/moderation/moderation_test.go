package moderation

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func flaggedServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"self_harm":true,"hate":false},"category_scores":{"self_harm":0.9,"hate":0.05}}]}`))
	}))
}

func cleanServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"flagged":false,"categories":{},"category_scores":{}}]}`))
	}))
}

func TestModerateFlagged(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	c := NewClient(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("omni-moderation-latest"))
	res, err := c.Moderate(context.Background(), "bad text")
	if err != nil {
		t.Fatalf("Moderate: %v", err)
	}
	if !res.Flagged || res.Categories["self_harm"] != true {
		t.Fatalf("result = %#v", res)
	}
}

func TestModerateClean(t *testing.T) {
	server := cleanServer()
	defer server.Close()

	c := NewClient(modelconfig.WithBaseURL(server.URL))
	res, err := c.Moderate(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Moderate: %v", err)
	}
	if res.Flagged {
		t.Fatalf("expected clean result")
	}
}

// error behavior

func TestBeforeModelErrorBehavior(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.ExitBehavior = ExitError
	state := map[string]any{"messages": []messages.Message{messages.Human("flagged")}}

	_, err := m.BeforeModel(context.Background(), state)
	var ve *ViolationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected ViolationError, got %v", err)
	}
	if ve.Stage != "input" || len(ve.Categories) == 0 {
		t.Fatalf("violation = %#v", ve)
	}
}

// end behavior (default)

func TestBeforeModelEndBehavior(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	state := map[string]any{"messages": []messages.Message{messages.Human("flagged")}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	if out == nil || out["jump_to"] != "end" {
		t.Fatalf("update = %#v, want jump_to=end", out)
	}
	msgs := out["messages"].([]messages.Message)
	if len(msgs) != 1 || msgs[0].Role != messages.RoleAI || !strings.Contains(msgs[0].Content, "self harm") {
		t.Fatalf("violation message = %#v", msgs)
	}
}

// replace behavior

func TestBeforeModelReplaceBehavior(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.ExitBehavior = ExitReplace
	state := map[string]any{"messages": []messages.Message{
		messages.Human("clean"),
		messages.Human("flagged"),
	}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	msgs := out["messages"].([]messages.Message)
	// The last human message is the one that got flagged and replaced.
	if !strings.Contains(msgs[len(msgs)-1].Content, "flagged") || msgs[len(msgs)-1].Role != messages.RoleHuman {
		t.Fatalf("replaced messages = %#v", msgs)
	}
}

// custom violation message

func TestCustomViolationMessage(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.ViolationMessage = "Policy block: {categories}"
	state := map[string]any{"messages": []messages.Message{messages.Human("flagged")}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	msgs := out["messages"].([]messages.Message)
	if msgs[0].Content != "Policy block: self harm" {
		t.Fatalf("violation message = %q", msgs[0].Content)
	}
}

// tool messages (replace behavior, mirroring Python's
// test_tool_messages_are_moderated_when_enabled)

func TestToolMessagesModeratedWhenEnabled(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.CheckToolResults = true
	m.ExitBehavior = ExitReplace
	state := map[string]any{"messages": []messages.Message{
		messages.Human("question"),
		messages.AI("call tool"),
		messages.Tool("tool-1", "dangerous"),
	}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	msgs := out["messages"].([]messages.Message)
	toolMsg := msgs[len(msgs)-1]
	if toolMsg.Role != messages.RoleTool || toolMsg.ToolCallID != "tool-1" {
		t.Fatalf("tool message = %#v", toolMsg)
	}
	if !strings.Contains(toolMsg.Content, "self harm") {
		t.Fatalf("tool message content = %q, want flagged replacement", toolMsg.Content)
	}
}

// after_model replace (mirroring test_after_model_replaces_flagged_message)

func TestAfterModelReplacesFlaggedMessage(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.ExitBehavior = ExitReplace
	ai := messages.AI("unsafe")
	ai.ID = "ai-1"
	state := map[string]any{"messages": []messages.Message{ai}}

	out, err := m.AfterModel(context.Background(), state)
	if err != nil {
		t.Fatalf("AfterModel: %v", err)
	}
	msgs := out["messages"].([]messages.Message)
	if msgs[len(msgs)-1].ID != "ai-1" || !strings.Contains(msgs[len(msgs)-1].Content, "self harm") {
		t.Fatalf("replaced AI message = %#v", msgs[len(msgs)-1])
	}
}

// clean pass-through + tool-skip default

func TestBeforeModelCleanPasses(t *testing.T) {
	server := cleanServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	state := map[string]any{"messages": []messages.Message{messages.Human("clean")}}
	out, err := m.BeforeModel(context.Background(), state)
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	if out != nil {
		t.Fatalf("expected no state update for clean input, got %#v", out)
	}
}

func TestToolMessagesSkippedByDefault(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"self_harm":true},"category_scores":{"self_harm":0.9}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	state := map[string]any{"messages": []messages.Message{
		messages.AI("assistant"),
		messages.Tool("call-1", "tool output"),
	}}
	if _, err := m.BeforeModel(context.Background(), state); err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	if calls != 0 {
		t.Fatalf("expected no moderation calls, got %d", calls)
	}
}

// client defaults + error paths

func TestNewClientDefaults(t *testing.T) {
	c := NewClient()
	if c.config.BaseURL != defaultBaseURL {
		t.Fatalf("BaseURL = %q, want %q", c.config.BaseURL, defaultBaseURL)
	}
	if c.config.Model != "omni-moderation-latest" {
		t.Fatalf("Model = %q, want omni-moderation-latest", c.config.Model)
	}
	if c.config.Timeout == 0 {
		t.Fatalf("expected non-zero default timeout")
	}
}

func TestModerateEmptyResults(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[]}`))
	}))
	defer server.Close()

	c := NewClient(modelconfig.WithBaseURL(server.URL))
	_, err := c.Moderate(context.Background(), "text")
	if err == nil || !strings.Contains(err.Error(), "empty results") {
		t.Fatalf("err = %v, want empty results error", err)
	}
}

func TestModerateServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer server.Close()

	c := NewClient(modelconfig.WithBaseURL(server.URL), modelconfig.WithMaxRetries(0))
	_, err := c.Moderate(context.Background(), "text")
	if err == nil {
		t.Fatalf("expected error for 500 response")
	}
}

func TestModerateSendsAuthAndCustomHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", got)
		}
		if got := r.Header.Get("X-Custom"); got != "custom-value" {
			t.Errorf("X-Custom = %q, want custom-value", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", got)
		}
		_, _ = w.Write([]byte(`{"results":[{"flagged":false,"categories":{},"category_scores":{}}]}`))
	}))
	defer server.Close()

	c := NewClient(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
		modelconfig.WithHeader("X-Custom", "custom-value"),
	)
	if _, err := c.Moderate(context.Background(), "text"); err != nil {
		t.Fatalf("Moderate: %v", err)
	}
}

// ViolationError.Error

func TestViolationErrorMessage(t *testing.T) {
	withStage := &ViolationError{Stage: "output", Categories: []string{"hate", "self_harm"}}
	if got := withStage.Error(); got != "openai moderation flagged output: hate, self_harm" {
		t.Fatalf("Error() = %q", got)
	}
	noStage := &ViolationError{Categories: []string{"hate"}}
	if got := noStage.Error(); got != "openai moderation flagged content: hate" {
		t.Fatalf("Error() = %q, want default stage 'content'", got)
	}
}

// before/after model guard clauses

func TestBeforeModelNoMessages(t *testing.T) {
	m := NewMiddleware(NewClient())

	out, err := m.BeforeModel(context.Background(), map[string]any{})
	if err != nil || out != nil {
		t.Fatalf("missing messages: out=%#v err=%v", out, err)
	}

	out, err = m.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{}})
	if err != nil || out != nil {
		t.Fatalf("empty messages: out=%#v err=%v", out, err)
	}

	out, err = m.BeforeModel(context.Background(), map[string]any{"messages": "not-messages"})
	if err != nil || out != nil {
		t.Fatalf("wrong messages type: out=%#v err=%v", out, err)
	}
}

func TestBeforeModelSkipsEmptyContent(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"hate":true},"category_scores":{}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	state := map[string]any{"messages": []messages.Message{messages.Human("")}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil || out != nil {
		t.Fatalf("out=%#v err=%v", out, err)
	}
	if calls != 0 {
		t.Fatalf("expected no moderation calls for empty content, got %d", calls)
	}
}

func TestBeforeModelModerationErrorPropagates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL), modelconfig.WithMaxRetries(0)))
	state := map[string]any{"messages": []messages.Message{messages.Human("text")}}

	_, err := m.BeforeModel(context.Background(), state)
	if err == nil {
		t.Fatalf("expected moderation error to propagate")
	}
}

func TestBeforeModelUnknownExitBehavior(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.ExitBehavior = ExitBehavior("bogus")
	state := map[string]any{"messages": []messages.Message{messages.Human("flagged")}}

	_, err := m.BeforeModel(context.Background(), state)
	var ve *ViolationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected ViolationError for unknown exit behavior, got %v", err)
	}
}

func TestAfterModelCheckOutputDisabled(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"hate":true},"category_scores":{}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.CheckOutput = false
	state := map[string]any{"messages": []messages.Message{messages.AI("flagged")}}

	out, err := m.AfterModel(context.Background(), state)
	if err != nil || out != nil {
		t.Fatalf("out=%#v err=%v", out, err)
	}
	if calls != 0 {
		t.Fatalf("expected no moderation calls, got %d", calls)
	}
}

func TestAfterModelNoMessagesOrNoAI(t *testing.T) {
	m := NewMiddleware(NewClient())

	out, err := m.AfterModel(context.Background(), map[string]any{})
	if err != nil || out != nil {
		t.Fatalf("missing messages: out=%#v err=%v", out, err)
	}

	state := map[string]any{"messages": []messages.Message{messages.Human("only human")}}
	out, err = m.AfterModel(context.Background(), state)
	if err != nil || out != nil {
		t.Fatalf("no AI message: out=%#v err=%v", out, err)
	}
}

func TestAfterModelErrorBehavior(t *testing.T) {
	server := flaggedServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.ExitBehavior = ExitError
	state := map[string]any{"messages": []messages.Message{
		messages.Human("question"),
		messages.AI("unsafe answer"),
	}}

	_, err := m.AfterModel(context.Background(), state)
	var ve *ViolationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected ViolationError, got %v", err)
	}
	if ve.Stage != "output" {
		t.Fatalf("stage = %q, want output", ve.Stage)
	}
}

func TestAfterModelCleanPasses(t *testing.T) {
	server := cleanServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	state := map[string]any{"messages": []messages.Message{messages.AI("safe answer")}}

	out, err := m.AfterModel(context.Background(), state)
	if err != nil || out != nil {
		t.Fatalf("out=%#v err=%v", out, err)
	}
}

// tool-message edge cases

func TestToolMessagesNoAIMessage(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"hate":true},"category_scores":{}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.CheckToolResults = true
	m.CheckInput = false
	state := map[string]any{"messages": []messages.Message{
		messages.Human("question"),
		messages.Tool("call-1", "tool output"),
	}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil || out != nil {
		t.Fatalf("out=%#v err=%v", out, err)
	}
	if calls != 0 {
		t.Fatalf("expected no moderation calls without an AI message, got %d", calls)
	}
}

func TestToolMessageEmptyContentSkipped(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"hate":true},"category_scores":{}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.CheckToolResults = true
	m.CheckInput = false
	state := map[string]any{"messages": []messages.Message{
		messages.AI("call tool"),
		messages.Tool("call-1", ""),
	}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil || out != nil {
		t.Fatalf("out=%#v err=%v", out, err)
	}
	if calls != 0 {
		t.Fatalf("expected no moderation calls for empty tool content, got %d", calls)
	}
}

func TestToolMessagesCleanPassThrough(t *testing.T) {
	server := cleanServer()
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.CheckToolResults = true
	m.CheckInput = false
	m.CheckOutput = false
	state := map[string]any{"messages": []messages.Message{
		messages.Human("question"),
		messages.AI("call tool"),
		messages.Tool("call-1", "harmless output"),
	}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil || out != nil {
		t.Fatalf("out=%#v err=%v", out, err)
	}
}

// violation-message templating

func TestViolationMessageScoresAndOriginalContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"self_harm":true},"category_scores":{"self_harm":0.9}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	m.ViolationMessage = "blocked [{categories}] scores={category_scores} original={original_content}"
	state := map[string]any{"messages": []messages.Message{messages.Human("bad stuff")}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	msgs := out["messages"].([]messages.Message)
	want := `blocked [self harm] scores={"self_harm":0.9} original=bad stuff`
	if msgs[0].Content != want {
		t.Fatalf("violation message = %q, want %q", msgs[0].Content, want)
	}
}

func TestViolationMessageNoCategoriesFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{},"category_scores":{}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	state := map[string]any{"messages": []messages.Message{messages.Human("flagged")}}

	out, err := m.BeforeModel(context.Background(), state)
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	msgs := out["messages"].([]messages.Message)
	if !strings.Contains(msgs[0].Content, "OpenAI's safety policies") {
		t.Fatalf("violation message = %q, want safety-policies fallback", msgs[0].Content)
	}
}
