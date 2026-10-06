package middleware

import (
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
	if msgs[0].ToolCalls[0].ID != "1" || msgs[0].ToolCalls[0].Name != "lookup" {
		t.Fatalf("edited call mismatch: %#v", msgs[0].ToolCalls[0])
	}
	if msgs[0].ToolCalls[1].Name != "calc" {
		t.Fatalf("auto-approved call missing: %#v", msgs[0].ToolCalls)
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
