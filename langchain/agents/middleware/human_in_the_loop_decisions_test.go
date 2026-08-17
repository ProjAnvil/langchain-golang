package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestNewHumanInTheLoopMiddlewareFiltersEmptyConfigs(t *testing.T) {
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
		"calc":   {}, // no allowed decisions: dropped
	}, func(HITLRequest) ([]Decision, error) { return nil, nil })
	if len(middleware.InterruptOn) != 1 {
		t.Fatalf("expected only configs with allowed decisions, got %#v", middleware.InterruptOn)
	}
	if _, ok := middleware.InterruptOn["search"]; !ok {
		t.Fatal("search config missing")
	}
}

func aiWithCalls(calls ...messages.ToolCall) messages.Message {
	ai := messages.AI("")
	ai.ToolCalls = calls
	return ai
}

func TestHumanInTheLoopMiddlewareNoMessagesOrCalls(t *testing.T) {
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	}, func(HITLRequest) ([]Decision, error) {
		t.Fatal("decide should not be called")
		return nil, nil
	})

	update, err := middleware.AfterModel(context.Background(), map[string]any{})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without messages: %#v %v", update, err)
	}

	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{messages.Human("hi")}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without AI message: %#v %v", update, err)
	}

	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{messages.AI("no calls")}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without tool calls: %#v %v", update, err)
	}

	// Tool call not present in InterruptOn: nothing to review.
	update, err = middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{aiWithCalls(messages.ToolCall{ID: "1", Name: "other"})}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update for unconfigured tool: %#v %v", update, err)
	}
}

func TestHumanInTheLoopMiddlewareWhenPredicate(t *testing.T) {
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {
			AllowedDecisions: []DecisionType{DecisionApprove},
			When: func(req ToolCallRequest) bool {
				return req.ToolCall.Args["q"] == "review"
			},
		},
	}, func(HITLRequest) ([]Decision, error) {
		t.Fatal("decide should not be called when the predicate rejects the call")
		return nil, nil
	})

	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "1", Name: "search", Args: map[string]any{"q": "skip"}}),
	}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update when When rejects: %#v %v", update, err)
	}
}

func TestHumanInTheLoopMiddlewareDefaultDescription(t *testing.T) {
	var captured HITLRequest
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	}, func(request HITLRequest) ([]Decision, error) {
		captured = request
		return []Decision{{Type: DecisionApprove}}, nil
	})
	// An empty prefix falls back to the standard one.
	middleware.DescriptionPrefix = ""

	_, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "1", Name: "search", Args: map[string]any{"q": "x"}}),
	}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	desc := captured.ActionRequests[0].Description
	if !strings.Contains(desc, "Tool execution requires approval") || !strings.Contains(desc, "Tool: search") {
		t.Fatalf("default description mismatch: %q", desc)
	}
	if captured.ReviewConfigs[0].ActionName != "search" {
		t.Fatalf("review config mismatch: %#v", captured.ReviewConfigs)
	}
}

func TestHumanInTheLoopMiddlewareDecideRequired(t *testing.T) {
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	}, nil)
	_, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "1", Name: "search"}),
	}})
	if err == nil || !strings.Contains(err.Error(), "decision function is required") {
		t.Fatalf("expected missing decide error, got %v", err)
	}
}

func TestHumanInTheLoopMiddlewareDecideErrorPropagates(t *testing.T) {
	wantErr := errors.New("human unavailable")
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	}, func(HITLRequest) ([]Decision, error) { return nil, wantErr })
	_, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "1", Name: "search"}),
	}})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected decide error, got %v", err)
	}
}

func TestProcessHumanDecision(t *testing.T) {
	call := messages.ToolCall{ID: "1", Name: "search", Args: map[string]any{"q": "x"}}
	approveOnly := InterruptConfig{AllowedDecisions: []DecisionType{DecisionApprove}}

	// Disallowed decision type.
	if _, _, err := processHumanDecision(Decision{Type: DecisionReject}, call, approveOnly); err == nil ||
		!strings.Contains(err.Error(), "is not allowed") {
		t.Fatalf("expected not-allowed error, got %v", err)
	}

	// Approve keeps the original call.
	next, msg, err := processHumanDecision(Decision{Type: DecisionApprove}, call, approveOnly)
	if err != nil || msg != nil || next == nil || next.Name != "search" {
		t.Fatalf("approve mismatch: %#v %#v %v", next, msg, err)
	}

	// Edit without an edited action.
	editConfig := InterruptConfig{AllowedDecisions: []DecisionType{DecisionEdit}}
	if _, _, err := processHumanDecision(Decision{Type: DecisionEdit}, call, editConfig); err == nil ||
		!strings.Contains(err.Error(), "requires edited action") {
		t.Fatalf("expected edited action error, got %v", err)
	}

	// Respond builds a success tool message carrying the human's message.
	respondConfig := InterruptConfig{AllowedDecisions: []DecisionType{DecisionRespond}}
	next, msg, err = processHumanDecision(Decision{Type: DecisionRespond, Message: "human answer"}, call, respondConfig)
	if err != nil || next == nil || msg == nil {
		t.Fatalf("respond mismatch: %#v %#v %v", next, msg, err)
	}
	if msg.Content != "human answer" || msg.ResponseMetadata["status"] != "success" || msg.Name != "search" {
		t.Fatalf("respond message mismatch: %#v", msg)
	}

	// Unknown decision type.
	unknownConfig := InterruptConfig{AllowedDecisions: []DecisionType{DecisionType("bogus")}}
	if _, _, err := processHumanDecision(Decision{Type: DecisionType("bogus")}, call, unknownConfig); err == nil ||
		!strings.Contains(err.Error(), "unexpected human decision") {
		t.Fatalf("expected unexpected decision error, got %v", err)
	}
}

func TestHumanInTheLoopMiddlewareDecisionErrorPropagates(t *testing.T) {
	middleware := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	}, func(HITLRequest) ([]Decision, error) {
		return []Decision{{Type: DecisionReject}}, nil // not allowed for this config
	})
	_, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "1", Name: "search"}),
	}})
	if err == nil || !strings.Contains(err.Error(), "is not allowed") {
		t.Fatalf("expected decision processing error, got %v", err)
	}
}
