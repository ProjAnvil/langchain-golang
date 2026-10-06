package middleware

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestHumanInTheLoopMiddlewareProcessesDecisions(t *testing.T) {
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionEdit, DecisionReject}},
	}, func(request HITLRequest) ([]Decision, error) {
		if len(request.ActionRequests) != 1 || request.ActionRequests[0].Name != "search" {
			t.Fatalf("hitl request mismatch: %#v", request)
		}
		return []Decision{{Type: DecisionEdit, EditedAction: &ToolCall{Name: "lookup", Args: map[string]any{"q": "edited"}}}}, nil
	})
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search", Args: map[string]any{"q": "old"}}, {ID: "2", Name: "calc"}}

	update, err := middleware.AfterModel(t.Context(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 1 || len(msgs[0].ToolCalls) != 2 {
		t.Fatalf("messages mismatch: %#v", msgs)
	}
	// The AIMessage keeps the model's original call; the edit is recorded in
	// state for execution-time substitution (Python #40463).
	if msgs[0].ToolCalls[0].ID != "1" || msgs[0].ToolCalls[0].Name != "search" {
		t.Fatalf("original call must stay recorded: %#v", msgs[0].ToolCalls[0])
	}
	if msgs[0].ToolCalls[1].Name != "calc" {
		t.Fatalf("auto-approved call missing: %#v", msgs[0].ToolCalls)
	}
	edited, ok := update[EditedToolCallsStateKey].(map[string]ToolCall)
	if !ok || edited["1"].Name != "lookup" || edited["1"].Args["q"] != "edited" {
		t.Fatalf("edited action not recorded: %#v", update[EditedToolCallsStateKey])
	}
}

func TestHumanInTheLoopMiddlewareDescriptionFunc(t *testing.T) {
	var captured ToolCallRequest
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {
			AllowedDecisions: []DecisionType{DecisionApprove},
			Description:      "should be ignored in favor of DescriptionFunc",
			DescriptionFunc: func(req ToolCallRequest) string {
				captured = req
				return "dynamic: " + req.ToolCall.Name + " " + req.ToolCall.Args["q"].(string)
			},
		},
	}, func(request HITLRequest) ([]Decision, error) {
		if len(request.ActionRequests) != 1 {
			t.Fatalf("expected one action request, got %#v", request.ActionRequests)
		}
		if got, want := request.ActionRequests[0].Description, "dynamic: search old"; got != want {
			t.Fatalf("description = %q, want %q", got, want)
		}
		return []Decision{{Type: DecisionApprove}}, nil
	})
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search", Args: map[string]any{"q": "old"}}}

	if _, err := middleware.AfterModel(t.Context(), map[string]any{"messages": []messages.Message{ai}}); err != nil {
		t.Fatalf("after model: %v", err)
	}
	if captured.ToolCall.Name != "search" {
		t.Fatalf("expected DescriptionFunc to receive the pending tool call, got %#v", captured)
	}
}

func TestHumanInTheLoopMiddlewareRejectCreatesToolMessage(t *testing.T) {
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"delete": {AllowedDecisions: []DecisionType{DecisionReject}},
	}, func(HITLRequest) ([]Decision, error) {
		return []Decision{{Type: DecisionReject}}, nil
	})
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "delete"}}

	update, err := middleware.AfterModel(t.Context(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 2 || msgs[1].Role != messages.RoleTool || msgs[1].ResponseMetadata["status"] != "error" {
		t.Fatalf("reject messages mismatch: %#v", msgs)
	}
	if !strings.Contains(msgs[1].Content, "User rejected") {
		t.Fatalf("reject content mismatch: %q", msgs[1].Content)
	}
}

func TestHumanInTheLoopMiddlewareDecisionCountMismatch(t *testing.T) {
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	}, func(HITLRequest) ([]Decision, error) {
		return nil, nil
	})
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}}

	_, err := middleware.AfterModel(t.Context(), map[string]any{"messages": []messages.Message{ai}})
	if err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("expected mismatch error, got %v", err)
	}
}

// TestProcessHumanDecisionRejectReasonFraming verifies a custom rejection
// reason is framed as a user rejection when sent to the model, rather than
// passed through raw (Python langchain #39773).
func TestProcessHumanDecisionRejectReasonFraming(t *testing.T) {
	call := messages.ToolCall{ID: "call_9", Name: "test_tool"}
	config := InterruptConfig{AllowedDecisions: []DecisionType{DecisionApprove, DecisionEdit, DecisionReject}}

	_, msg, err := processHumanDecision(Decision{Type: DecisionReject, Message: "Custom response message"}, call, config)
	if err != nil {
		t.Fatalf("processHumanDecision: %v", err)
	}
	want := "User rejected the tool call for `test_tool` with reason: Custom response message"
	if msg.Content != want {
		t.Fatalf("content = %q, want %q", msg.Content, want)
	}
	if msg.ResponseMetadata["status"] != "error" {
		t.Fatalf("status = %v, want error", msg.ResponseMetadata["status"])
	}

	// No reason keeps the id-bearing default text.
	_, msg, err = processHumanDecision(Decision{Type: DecisionReject}, call, config)
	if err != nil {
		t.Fatalf("processHumanDecision default: %v", err)
	}
	wantDefault := "User rejected the tool call for `test_tool` with id call_9. The tool was not executed. Do not retry this tool call unless the user explicitly requests it."
	if msg.Content != wantDefault {
		t.Fatalf("default content = %q, want %q", msg.Content, wantDefault)
	}
}

const testEditNotice = "Note: a human reviewer replaced this tool call before it ran. The call recorded in your message is the one you produced, not the one that executed. This was intentional and authorized. Do not re-issue your original call."

// TestHITLEditPreservesOriginalCallAndRecordsEdit verifies Python #40463
// semantics: the AI message keeps the model's original tool call; the edited
// action is recorded in state under hitl_edited_tool_calls for substitution
// at execution time.
func TestHITLEditPreservesOriginalCallAndRecordsEdit(t *testing.T) {
	mw := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionEdit}},
	}, func(request HITLRequest) ([]Decision, error) {
		return []Decision{{Type: DecisionEdit, EditedAction: &ToolCall{Name: "lookup", Args: map[string]any{"q": "edited"}}}}, nil
	})
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search", Args: map[string]any{"q": "old"}}}

	update, err := mw.AfterModel(t.Context(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 1 || msgs[0].ToolCalls[0].Name != "search" || msgs[0].ToolCalls[0].ID != "1" {
		t.Fatalf("AI message must keep the model's original call: %#v", msgs)
	}
	edited, ok := update[EditedToolCallsStateKey].(map[string]ToolCall)
	if !ok || len(edited) != 1 {
		t.Fatalf("update missing edited-tool-calls state: %#v", update)
	}
	if got := edited["1"]; got.Name != "lookup" || got.Args["q"] != "edited" {
		t.Fatalf("edited action = %#v, want lookup/edited", got)
	}
}

// TestHITLEditWrapToolCallSubstitutesAndNotices verifies the WrapToolCall
// hook substitutes the edited action before dispatch and prepends the edit
// notice to the resulting ToolMessage.
func TestHITLEditWrapToolCallSubstitutesAndNotices(t *testing.T) {
	mw := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{}, nil)
	var invoked ToolCall
	msg, err := mw.WrapToolCall(t.Context(), ToolCallRequest{
		ToolCall: ToolCall{ID: "1", Name: "search", Args: map[string]any{"q": "old"}},
		State: map[string]any{EditedToolCallsStateKey: map[string]ToolCall{
			"1": {Name: "lookup", Args: map[string]any{"q": "edited"}},
		}},
	}, func(_ context.Context, req ToolCallRequest) (messages.Message, error) {
		invoked = req.ToolCall
		msg := messages.Tool(req.ToolCall.ID, "result: "+req.ToolCall.Name)
		msg.Name = req.ToolCall.Name
		return msg, nil
	})
	if err != nil {
		t.Fatalf("wrap tool call: %v", err)
	}
	if invoked.Name != "lookup" || invoked.Args["q"] != "edited" || invoked.ID != "1" {
		t.Fatalf("dispatched call = %#v, want the substituted lookup call", invoked)
	}
	want := testEditNotice + "\n\n" + "result: lookup"
	if msg.Content != want {
		t.Fatalf("content = %q, want %q", msg.Content, want)
	}
}

// TestHITLEditNoticeDisabledAndCustom covers WithEditNotice("") disabling the
// notice and a custom notice text replacing the default.
func TestHITLEditNoticeDisabledAndCustom(t *testing.T) {
	state := map[string]any{EditedToolCallsStateKey: map[string]ToolCall{
		"1": {Name: "lookup", Args: nil},
	}}
	dispatch := func(_ context.Context, req ToolCallRequest) (messages.Message, error) {
		msg := messages.Tool(req.ToolCall.ID, "raw")
		return msg, nil
	}

	off := NewInterruptHumanInTheLoopMiddleware(nil, WithEditNotice(""))
	msg, err := off.WrapToolCall(t.Context(), ToolCallRequest{
		ToolCall: ToolCall{ID: "1", Name: "search"}, State: state,
	}, dispatch)
	if err != nil {
		t.Fatalf("wrap: %v", err)
	}
	if msg.Content != "raw" {
		t.Fatalf("disabled notice: content = %q, want raw", msg.Content)
	}

	custom := NewInterruptHumanInTheLoopMiddleware(nil, WithEditNotice("custom notice"))
	msg, err = custom.WrapToolCall(t.Context(), ToolCallRequest{
		ToolCall: ToolCall{ID: "1", Name: "search"}, State: state,
	}, dispatch)
	if err != nil {
		t.Fatalf("wrap custom: %v", err)
	}
	if msg.Content != "custom notice\n\nraw" {
		t.Fatalf("custom notice: content = %q", msg.Content)
	}

	// Non-edited calls pass through untouched.
	plain := NewInterruptHumanInTheLoopMiddleware(nil)
	msg, err = plain.WrapToolCall(t.Context(), ToolCallRequest{
		ToolCall: ToolCall{ID: "2", Name: "search"}, State: state,
	}, dispatch)
	if err != nil {
		t.Fatalf("wrap plain: %v", err)
	}
	if msg.Content != "raw" {
		t.Fatalf("plain call: content = %q, want raw", msg.Content)
	}
}
