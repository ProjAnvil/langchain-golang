package middleware

// Interrupt-mode tests for HumanInTheLoopMiddleware (T16 PR1): the
// HITLResponse/HITLRequest data shapes, their dual-form decoders, the
// interrupt-mode constructor/options, the AfterModel no-op contract, and the
// Interrupt call point in AfterModelNode. The end-to-end pause/resume
// behavior is covered by the agents-package HITL node tests (T16 PR2).

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func TestNewInterruptHumanInTheLoopMiddleware(t *testing.T) {
	m := NewInterruptHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
		"calc":   {}, // no allowed decisions: dropped
	})
	if m.Decide != nil {
		t.Fatalf("interrupt-mode middleware must have a nil Decide, got %#v", m.Decide)
	}
	if !m.HitlInterruptEnabled() {
		t.Fatal("HitlInterruptEnabled must be true for a nil-Decide middleware")
	}
	if len(m.InterruptOn) != 1 {
		t.Fatalf("expected only configs with allowed decisions, got %#v", m.InterruptOn)
	}
	if m.DescriptionPrefix != "Tool execution requires approval" {
		t.Fatalf("default prefix mismatch: %q", m.DescriptionPrefix)
	}

	custom := NewInterruptHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	}, WithDescriptionPrefix("Custom prefix"))
	if custom.DescriptionPrefix != "Custom prefix" {
		t.Fatalf("WithDescriptionPrefix mismatch: %q", custom.DescriptionPrefix)
	}

	// A Decide-mode middleware is not interrupt-enabled (mutual exclusion).
	decide := NewHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	}, func(HITLRequest) ([]Decision, error) { return nil, nil })
	if decide.HitlInterruptEnabled() {
		t.Fatal("HitlInterruptEnabled must be false for a Decide-mode middleware")
	}
}

func TestInterruptModeAfterModelNoOp(t *testing.T) {
	m := NewInterruptHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	})
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search", Args: map[string]any{"q": "x"}}}

	// Inline AfterModel must be a no-op in interrupt mode: the dedicated HITL
	// graph node owns the review flow, so running it inline too would pause
	// before the model node's update commits (double-run guard).
	update, err := m.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	if err != nil {
		t.Fatalf("interrupt-mode AfterModel error: %v", err)
	}
	if update != nil {
		t.Fatalf("interrupt-mode AfterModel must return a nil update, got %#v", update)
	}
}

func TestAfterModelNodeInterruptCallPoint(t *testing.T) {
	m := NewInterruptHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	})

	// No reviewable calls: the node hook is a no-op and must not touch the
	// interrupt machinery at all.
	plain := messages.AI("no calls")
	update, err := m.AfterModelNode(context.Background(), map[string]any{"messages": []messages.Message{plain}})
	if err != nil || update != nil {
		t.Fatalf("expected nil update without reviewable calls, got %#v %v", update, err)
	}

	// Reviewable calls: the hook must reach graph.Interrupt, which (outside a
	// graph node execution) panics — proving the pause is routed through the
	// interrupt primitive rather than a local callback.
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search", Args: map[string]any{"q": "x"}}}
	panicked := false
	func() {
		defer func() {
			r := recover()
			if r == nil {
				return
			}
			panicked = true
			msg, _ := r.(string)
			if !strings.Contains(msg, "outside of a graph node execution") {
				t.Fatalf("expected the graph Interrupt panic, got %v", r)
			}
		}()
		_, _ = m.AfterModelNode(context.Background(), map[string]any{"messages": []messages.Message{ai}})
	}()
	if !panicked {
		t.Fatal("expected AfterModelNode to call graph.Interrupt for reviewable calls")
	}
}

func TestDecodeHITLResponse(t *testing.T) {
	edit := Decision{
		Type:         DecisionEdit,
		EditedAction: &ToolCall{Name: "lookup", Args: map[string]any{"q": "edited"}},
	}
	want := []Decision{
		{Type: DecisionApprove},
		edit,
		{Type: DecisionReject, Message: "no thanks"},
		{Type: DecisionRespond, Message: "human answer"},
	}

	// Struct direct (what a Go caller passes to Options.Resume).
	got, err := DecodeHITLResponse(HITLResponse{Decisions: want})
	if err != nil {
		t.Fatalf("decode struct: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("struct decode mismatch: %#v", got)
	}

	// Pointer form.
	got, err = DecodeHITLResponse(&HITLResponse{Decisions: want})
	if err != nil {
		t.Fatalf("decode pointer: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("pointer decode mismatch: %#v", got)
	}

	// Bare []Decision.
	got, err = DecodeHITLResponse(want)
	if err != nil {
		t.Fatalf("decode slice: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("slice decode mismatch: %#v", got)
	}

	// Map wire form (Python HITLResponse / JSON round-trip): decisions as maps
	// keyed by type/edited_action/message.
	got, err = DecodeHITLResponse(map[string]any{"decisions": []any{
		map[string]any{"type": "approve"},
		map[string]any{
			"type":          "edit",
			"edited_action": map[string]any{"name": "lookup", "args": map[string]any{"q": "edited"}},
		},
		map[string]any{"type": "reject", "message": "no thanks"},
		map[string]any{"type": "respond", "message": "human answer"},
	}})
	if err != nil {
		t.Fatalf("decode map form: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("map decode mismatch: %#v", got)
	}

	// Go-JSON spelling of the map form (json.Marshal of a HITLResponse has no
	// struct tags, so keys round-trip capitalized).
	got, err = DecodeHITLResponse(map[string]any{"Decisions": []any{
		map[string]any{"Type": "approve", "Message": "ignored-when-approve"},
	}})
	if err != nil {
		t.Fatalf("decode Go-JSON map form: %v", err)
	}
	if len(got) != 1 || got[0].Type != DecisionApprove || got[0].Message != "ignored-when-approve" {
		t.Fatalf("Go-JSON map decode mismatch: %#v", got)
	}

	// Rejected shapes.
	for _, bad := range []any{
		"decisions",                 // not a response at all
		42,                          // ditto
		map[string]any{"other": 1},  // no decisions key
		map[string]any{"decisions": "not-a-list"},
		[]any{"not-a-decision"},     // element not a map/Decision
		map[string]any{"decisions": []any{map[string]any{ /* missing type */ }}},
		map[string]any{"decisions": []any{map[string]any{"type": 7}}},
	} {
		if _, err := DecodeHITLResponse(bad); err == nil {
			t.Fatalf("expected decode error for %#v", bad)
		}
	}
}

func TestHITLRequestFromInterrupt(t *testing.T) {
	request := HITLRequest{
		ActionRequests: []ActionRequest{
			{Name: "search", Args: map[string]any{"q": "x"}, Description: "review me"},
		},
		ReviewConfigs: []ReviewConfig{
			{ActionName: "search", AllowedDecisions: []DecisionType{DecisionApprove, DecisionEdit}, ArgsSchema: map[string]any{"type": "object"}},
		},
	}

	// Struct direct: the live in-memory interrupt value form.
	got, err := HITLRequestFromInterrupt(types.Interrupt{Value: request})
	if err != nil {
		t.Fatalf("decode struct: %v", err)
	}
	if !reflect.DeepEqual(got, request) {
		t.Fatalf("struct decode mismatch: %#v", got)
	}

	// Map wire form: what a JSON-saver round-trip (or a Python producer)
	// leaves in Interrupt.Value.
	got, err = HITLRequestFromInterrupt(types.Interrupt{Value: map[string]any{
		"action_requests": []any{
			map[string]any{"name": "search", "args": map[string]any{"q": "x"}, "description": "review me"},
		},
		"review_configs": []any{
			map[string]any{
				"action_name":        "search",
				"allowed_decisions":  []any{"approve", "edit"},
				"args_schema":        map[string]any{"type": "object"},
			},
		},
	}})
	if err != nil {
		t.Fatalf("decode map: %v", err)
	}
	if !reflect.DeepEqual(got, request) {
		t.Fatalf("map decode mismatch: %#v", got)
	}

	if _, err := HITLRequestFromInterrupt(types.Interrupt{Value: "not-a-request"}); err == nil {
		t.Fatal("expected decode error for a non-request interrupt value")
	}
}

// TestApplyHITLDecisionsErrors covers the shared post-interrupt validations
// (design §6.8): the decision count must match the number of interrupted tool
// calls, and each decision type must be allowed for its tool.
func TestApplyHITLDecisionsErrors(t *testing.T) {
	m := NewInterruptHumanInTheLoopMiddleware(map[string]InterruptConfig{
		"search": {AllowedDecisions: []DecisionType{DecisionApprove}},
	})
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "search"}, {ID: "2", Name: "search"}}

	// Count mismatch: two interrupted calls, one decision.
	_, err := m.applyHITLDecisions(ai, []int{0, 1}, []Decision{{Type: DecisionApprove}})
	if err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("expected count mismatch error, got %v", err)
	}

	// Decision type not allowed for the tool.
	_, err = m.applyHITLDecisions(ai, []int{0}, []Decision{{Type: DecisionReject}})
	if err == nil || !strings.Contains(err.Error(), "is not allowed") {
		t.Fatalf("expected not-allowed error, got %v", err)
	}

	// Decisions decoded from the map wire form flow through the same path:
	// an approve decision revises the tool calls in original order.
	update, err := m.applyHITLDecisions(ai, []int{0}, []Decision{{Type: DecisionApprove}})
	if err != nil {
		t.Fatalf("apply approve: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if len(msgs) != 1 || len(msgs[0].ToolCalls) != 2 || msgs[0].ToolCalls[0].ID != "1" {
		t.Fatalf("approve revision mismatch: %#v", msgs)
	}
}
