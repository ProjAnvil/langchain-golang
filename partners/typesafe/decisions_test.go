package typesafe

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/decisions"
)

// TestQuestionToTypesafeMapping pins the core → TypeSafe question mapping:
// Noul/Choice/Score convert, Box errors.
func TestQuestionToTypesafeMapping(t *testing.T) {
	noul, err := questionToTypesafe(decisions.Noul{Instructions: "urgent?"})
	if err != nil || noul.(Noul).Instructions != "urgent?" {
		t.Fatalf("noul: %#v %v", noul, err)
	}
	choice, err := questionToTypesafe(decisions.Choice{
		Instructions: "route",
		Criteria:     map[string]decisions.Content{"billing": "payments", "other": nil},
	})
	if err != nil {
		t.Fatalf("choice: %v", err)
	}
	typed := choice.(Choice)
	if len(typed.Criteria) != 2 || typed.Criteria[0].Label != "billing" || typed.Criteria[0].Description != "payments" {
		t.Fatalf("choice criteria: %#v", typed.Criteria)
	}
	score, err := questionToTypesafe(decisions.Score{Instructions: "rate", Criteria: []decisions.Content{"low", "high"}})
	if err != nil || len(score.(Score).Criteria) != 2 {
		t.Fatalf("score: %#v %v", score, err)
	}
	if _, err := questionToTypesafe(decisions.Box{Instructions: "find the dog"}); err == nil {
		t.Fatal("box must error on TypeSafe")
	}
}

// TestTypesafeResponseToCore pins the TypeSafe → core response mapping,
// including usage completion.
func TestTypesafeResponseToCore(t *testing.T) {
	input, output := 3, 4
	core := responseToCore(Response{
		Model: "jev",
		Answers: map[string]Answer{
			"q": ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"billing": 0.9}, Confidence: 0.9},
		},
		Usage:     Usage{InputTokens: &input, OutputTokens: &output},
		RequestID: "req_1",
	})
	answer, ok := core.Answers["q"].(decisions.ChoiceAnswer)
	if !ok || answer.Choice != "billing" || answer.Confidence != 0.9 {
		t.Fatalf("answers: %#v", core.Answers)
	}
	if core.Usage == nil || core.Usage.TotalTokens != 7 {
		t.Fatalf("usage: %#v", core.Usage)
	}
	if core.RequestID != "req_1" {
		t.Fatalf("request id: %q", core.RequestID)
	}
}
