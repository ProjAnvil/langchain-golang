package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/decisions"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/tools"
)

// noulClassifier builds a fake decision model answering the is_risky Noul
// question with the given probability.
func noulClassifier(probability float64) *fakeDecisionModel {
	return &fakeDecisionModel{response: decisions.Response{
		Model: "fake",
		Answers: map[string]decisions.Answer{
			AutoModeQuestionID: decisions.NoulAnswer{Noul: probability},
		},
	}}
}

func autoModeRequest(name string, msgs []messages.Message, tool tools.Tool) ToolCallRequest {
	return ToolCallRequest{
		ToolCall: ToolCall{ID: "call_1", Name: name, Args: map[string]any{}},
		State:    map[string]any{"messages": msgs},
		Tool:     tool,
	}
}

// deleteFileTool builds a simple guarded tool with a description.
func deleteFileTool(t *testing.T) tools.Simple {
	t.Helper()
	tool, err := tools.NewSimple("delete_file", "Deletes a file.", func(context.Context, string) (tools.Result, error) {
		return tools.Result{Content: "deleted"}, nil
	})
	if err != nil {
		t.Fatalf("new simple tool: %v", err)
	}
	return tool
}

func TestGenericAutoModeAllowsLowRiskCalls(t *testing.T) {
	m, err := NewAutoModeMiddleware(noulClassifier(0.2), []any{"delete_file"})
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	ran := false
	out, err := m.WrapToolCall(t.Context(), autoModeRequest("delete_file", nil, nil), func(_ context.Context, _ ToolCallRequest) (messages.Message, error) {
		ran = true
		return messages.Tool("call_1", "deleted"), nil
	})
	if err != nil || !ran || out.Content != "deleted" {
		t.Fatalf("allow path: ran=%v err=%v out=%#v", ran, err, out)
	}
}

func TestGenericAutoModeBlocksRiskyCalls(t *testing.T) {
	m, err := NewAutoModeMiddleware(noulClassifier(0.87), []any{"delete_file"})
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	out, err := m.WrapToolCall(t.Context(), autoModeRequest("delete_file", nil, nil), func(_ context.Context, _ ToolCallRequest) (messages.Message, error) {
		t.Fatal("handler must not run for a blocked call")
		return messages.Message{}, nil
	})
	if err != nil {
		t.Fatalf("block path returns the error ToolMessage, not an error: %v", err)
	}
	if out.AdditionalKwargs["status"] != "error" || out.Name != "delete_file" {
		t.Fatalf("blocked message: %#v", out)
	}
	if !strings.Contains(out.Content, "probability: 0.87") || !strings.Contains(out.Content, "The tool was not executed.") {
		t.Fatalf("blocked content: %q", out.Content)
	}
}

func TestGenericAutoModeUnlistedToolsBypass(t *testing.T) {
	m, err := NewAutoModeMiddleware(noulClassifier(0.99), []any{"delete_file"})
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	ran := false
	_, err = m.WrapToolCall(t.Context(), autoModeRequest("read_file", nil, nil), func(_ context.Context, _ ToolCallRequest) (messages.Message, error) {
		ran = true
		return messages.Tool("call_1", "data"), nil
	})
	if err != nil || !ran {
		t.Fatalf("unlisted tool must bypass: ran=%v err=%v", ran, err)
	}
}

func TestGenericAutoModeClassificationFailurePropagates(t *testing.T) {
	boom := errors.New("classifier down")
	m, err := NewAutoModeMiddleware(&fakeDecisionModel{err: boom}, []any{"delete_file"})
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if _, err := m.WrapToolCall(t.Context(), autoModeRequest("delete_file", nil, nil), func(_ context.Context, _ ToolCallRequest) (messages.Message, error) {
		t.Fatal("handler must not run when classification fails")
		return messages.Message{}, nil
	}); !errors.Is(err, boom) {
		t.Fatalf("expected classifier error, got %v", err)
	}
}

func TestGenericAutoModeMissingAnswerFailsClosed(t *testing.T) {
	m, err := NewAutoModeMiddleware(&fakeDecisionModel{}, []any{"delete_file"})
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if _, err := m.WrapToolCall(t.Context(), autoModeRequest("delete_file", nil, nil), func(_ context.Context, _ ToolCallRequest) (messages.Message, error) {
		t.Fatal("handler must not run without an answer")
		return messages.Message{}, nil
	}); err == nil {
		t.Fatal("expected missing-answer error")
	}
}

func TestGenericAutoModeClassificationStateShape(t *testing.T) {
	classifier := noulClassifier(0.1)
	tool := deleteFileTool(t)
	msgs := make([]messages.Message, 40)
	for i := range msgs {
		msgs[i] = messages.Human("m")
	}
	m, err := NewAutoModeMiddleware(classifier, []any{tool})
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if _, err := m.WrapToolCall(t.Context(), autoModeRequest("delete_file", msgs, tool), func(_ context.Context, _ ToolCallRequest) (messages.Message, error) {
		return messages.Tool("call_1", "ok"), nil
	}); err != nil {
		t.Fatalf("wrap: %v", err)
	}
	state, ok := classifier.requests[0].State.(map[string]any)
	if !ok {
		t.Fatalf("state: %#v", classifier.requests[0].State)
	}
	window, _ := state["messages"].([]messages.Message)
	if len(window) != autoModeStateWindow {
		t.Fatalf("message window: %d, want %d", len(window), autoModeStateWindow)
	}
	call, _ := state["tool_call"].(map[string]any)
	if call["name"] != "delete_file" || call["id"] != "call_1" {
		t.Fatalf("tool call state: %#v", call)
	}
	if state["tool_description"] != "Deletes a file." {
		t.Fatalf("tool description: %#v", state["tool_description"])
	}
	question, ok := classifier.requests[0].Questions[AutoModeQuestionID].(decisions.Noul)
	if !ok {
		t.Fatalf("question: %#v", classifier.requests[0].Questions[AutoModeQuestionID])
	}
	if question.Instructions != autoModeInstructions {
		t.Fatalf("default instructions: %.60q...", question.Instructions)
	}
	// Criteria default to nil (upstream test_criteria_default_to_none).
	if question.Criteria != nil {
		t.Fatalf("default criteria: %#v, want nil", question.Criteria)
	}
}

func TestNewGenericAutoModeValidation(t *testing.T) {
	if _, err := NewAutoModeMiddleware(nil, []any{"t"}); err == nil {
		t.Fatal("nil classifier must fail")
	}
	if _, err := NewAutoModeMiddleware(noulClassifier(0), nil); err == nil {
		t.Fatal("empty tools must fail")
	}
	if _, err := NewAutoModeMiddleware(noulClassifier(0), []any{42}); err == nil {
		t.Fatal("non-tool entry must fail")
	}
}

// TestGenericAutoModeConfiguredQuestion pins the upstream
// test_risk_question_uses_configured_instructions_and_criteria: custom
// instructions and criteria reach the Noul question verbatim.
func TestGenericAutoModeConfiguredQuestion(t *testing.T) {
	classifier := noulClassifier(0.9)
	criteria := decisions.NoulCriteria{True: "The call modifies data.", False: "The call reads data."}
	m, err := NewAutoModeMiddleware(classifier, []any{"delete_file"},
		WithAutoModeInstructions("Assess the impact."),
		WithAutoModeCriteria(&criteria))
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if _, err := m.WrapToolCall(t.Context(), autoModeRequest("delete_file", nil, nil), func(_ context.Context, _ ToolCallRequest) (messages.Message, error) {
		return messages.Tool("call_1", "ok"), nil
	}); err != nil {
		t.Fatalf("wrap: %v", err)
	}
	question, ok := classifier.requests[0].Questions[AutoModeQuestionID].(decisions.Noul)
	if !ok {
		t.Fatalf("question: %#v", classifier.requests[0].Questions[AutoModeQuestionID])
	}
	if question.Instructions != "Assess the impact." {
		t.Fatalf("instructions: %q", question.Instructions)
	}
	if question.Criteria == nil || question.Criteria.True != "The call modifies data." ||
		question.Criteria.False != "The call reads data." {
		t.Fatalf("criteria: %#v", question.Criteria)
	}
}

// TestGenericAutoModeTracePolicy covers the trace policy accessor.
func TestGenericAutoModeTracePolicy(t *testing.T) {
	m, err := NewAutoModeMiddleware(noulClassifier(0.1), []any{"t"})
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if m.TracePolicy().ProcessInputs == nil {
		t.Fatal("process-inputs policy must be set")
	}
}
