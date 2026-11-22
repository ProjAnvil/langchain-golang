package decisions

import (
	"testing"

	coredecisions "github.com/projanvil/langchain-golang/core/decisions"
)

// TestQuestionToOpenAIMapping pins the core → Decisions API question
// mapping: Noul→Predicate, Choice, Score convert; Box and non-string
// instructions/score levels error.
func TestQuestionToOpenAIMapping(t *testing.T) {
	predicate, err := questionToOpenAI(coredecisions.Noul{Instructions: "risky?"})
	if err != nil || predicate.(Predicate).Instructions != "risky?" {
		t.Fatalf("noul/predicate: %#v %v", predicate, err)
	}
	choice, err := questionToOpenAI(coredecisions.Choice{
		Instructions: "route",
		Criteria:     map[string]coredecisions.Content{"billing": "payments", "other": nil},
	})
	if err != nil {
		t.Fatalf("choice: %v", err)
	}
	typed := choice.(Choice)
	if len(typed.Choices) != 2 || typed.Choices[0].Value != "billing" || typed.Choices[0].Description == nil {
		t.Fatalf("choice options: %#v", typed.Choices)
	}
	score, err := questionToOpenAI(coredecisions.Score{Instructions: "rate", Criteria: []coredecisions.Content{"low", "high"}})
	if err != nil || len(score.(Score).Levels) != 2 {
		t.Fatalf("score: %#v %v", score, err)
	}
	if _, err := questionToOpenAI(coredecisions.Box{Instructions: "find"}); err == nil {
		t.Fatal("box must error on the Decisions API")
	}
	if _, err := questionToOpenAI(coredecisions.Noul{Instructions: map[string]any{"k": "v"}}); err == nil {
		t.Fatal("non-string instructions must error")
	}
	if _, err := questionToOpenAI(coredecisions.Score{Instructions: "r", Criteria: []coredecisions.Content{map[string]any{}, "high"}}); err == nil {
		t.Fatal("non-string score level must error")
	}
}

// TestOpenAIResponseToCore pins the Decisions API → core response mapping:
// Predicate→Noul, string choices keep their label, bool choices and
// refusals are dropped, and usage completes.
func TestOpenAIResponseToCore(t *testing.T) {
	input := 5
	core := responseToCore(Response{
		Model: "gpt-test",
		Answers: map[string]Answer{
			"risk":  PredicateAnswer{Probability: 0.75},
			"route": ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"billing": 1.0}, Confidence: 1.0},
			"bool":  ChoiceAnswer{Choice: true},
			"decl":  RefusalAnswer{},
		},
		Usage:     Usage{InputTokens: &input},
		RequestID: "req_2",
	})
	if noul := core.Answers["risk"].(coredecisions.NoulAnswer); noul.Noul != 0.75 {
		t.Fatalf("predicate mapping: %#v", noul)
	}
	if choice := core.Answers["route"].(coredecisions.ChoiceAnswer); choice.Choice != "billing" {
		t.Fatalf("choice mapping: %#v", choice)
	}
	if _, has := core.Answers["bool"]; has {
		t.Fatal("bool choice must be dropped")
	}
	if _, has := core.Answers["decl"]; has {
		t.Fatal("refusal answer must be dropped")
	}
	if core.Usage == nil || core.Usage.TotalTokens != 5 {
		t.Fatalf("usage: %#v", core.Usage)
	}
}
