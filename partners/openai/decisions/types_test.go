package decisions

import (
	"reflect"
	"testing"
)

func TestPredicateToAPI(t *testing.T) {
	p := Predicate{Instructions: "Is this request urgent?"}
	got := p.toAPI("urgent")
	want := map[string]any{
		"type":         "predicate",
		"name":         "urgent",
		"instructions": "Is this request urgent?",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Predicate.toAPI() = %#v, want %#v", got, want)
	}
}

func TestChoiceToAPI(t *testing.T) {
	c := Choice{
		Instructions: "Which department should handle this?",
		Choices: []ChoiceOption{
			{Value: "billing", Description: new("Payments, invoices, and refunds.")},
			{Value: "technical"},
			{Value: true},
			{Value: false, Description: new("Everything else.")},
		},
	}
	got := c.toAPI("department")
	want := map[string]any{
		"type":         "choice",
		"name":         "department",
		"instructions": "Which department should handle this?",
		"choices": []any{
			map[string]any{"value": "billing", "description": "Payments, invoices, and refunds."},
			map[string]any{"value": "technical"},
			map[string]any{"value": true},
			map[string]any{"value": false, "description": "Everything else."},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Choice.toAPI() = %#v, want %#v", got, want)
	}
}

func TestScoreToAPI(t *testing.T) {
	s := Score{
		Instructions: "How frustrated does the customer appear?",
		Levels: []ScoreLevel{
			L("Calm"),
			{Label: "Concerned", Description: new("Mild frustration is showing.")},
			{Label: "Angry"},
		},
	}
	got := s.toAPI("frustration")
	want := map[string]any{
		"type":         "score",
		"name":         "frustration",
		"instructions": "How frustrated does the customer appear?",
		"levels": []any{
			map[string]any{"label": "Calm"},
			map[string]any{"label": "Concerned", "description": "Mild frustration is showing."},
			map[string]any{"label": "Angry"},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Score.toAPI() = %#v, want %#v", got, want)
	}
}

func TestResponseViews(t *testing.T) {
	resp := Response{
		Model: "gpt-6-luna",
		Answers: map[string]Answer{
			"urgent":      PredicateAnswer{Probability: 0.9},
			"department":  ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"billing": 0.8}, Confidence: 0.8},
			"frustration": ScoreAnswer{Score: 1.1, Legend: map[int]string{0: "Calm"}, Probabilities: map[int]float64{0: 0.4}, Confidence: 0.6},
			"blocked_q":   RefusalAnswer{Extra: map[string]any{"reason": "safety"}},
		},
	}
	if got := resp.Predicates(); !reflect.DeepEqual(got, map[string]PredicateAnswer{"urgent": {Probability: 0.9}}) {
		t.Fatalf("Predicates() = %#v", got)
	}
	if got := resp.Choices(); !reflect.DeepEqual(got, map[string]ChoiceAnswer{
		"department": {Choice: "billing", Probabilities: map[string]float64{"billing": 0.8}, Confidence: 0.8},
	}) {
		t.Fatalf("Choices() = %#v", got)
	}
	if got := resp.Scores(); !reflect.DeepEqual(got, map[string]ScoreAnswer{
		"frustration": {Score: 1.1, Legend: map[int]string{0: "Calm"}, Probabilities: map[int]float64{0: 0.4}, Confidence: 0.6},
	}) {
		t.Fatalf("Scores() = %#v", got)
	}
	if got := resp.Refusals(); !reflect.DeepEqual(got, map[string]RefusalAnswer{
		"blocked_q": {Extra: map[string]any{"reason": "safety"}},
	}) {
		t.Fatalf("Refusals() = %#v", got)
	}
}

func TestSealedInterfaces(t *testing.T) {
	// The marker methods only run when values flow through the sealed
	// interfaces, so drive them directly once.
	for _, q := range []Question{
		Predicate{Instructions: "i"},
		Choice{Instructions: "i", Choices: []ChoiceOption{{Value: "v"}}},
		Score{Instructions: "i", Levels: []ScoreLevel{L("a"), L("b")}},
	} {
		if m, ok := q.(interface{ questionMarker() }); ok {
			m.questionMarker()
		}
		if got := q.toAPI("q"); got == nil {
			t.Fatalf("toAPI() returned a nil map for %T", q)
		}
	}
	for _, a := range []Answer{
		PredicateAnswer{},
		ChoiceAnswer{},
		ScoreAnswer{},
		RefusalAnswer{},
	} {
		if m, ok := a.(interface{ answerMarker() }); ok {
			m.answerMarker()
		}
	}
	var _ Question = Predicate{}
	var _ Question = Choice{}
	var _ Question = Score{}
	var _ Answer = PredicateAnswer{}
	var _ Answer = ChoiceAnswer{}
	var _ Answer = ScoreAnswer{}
	var _ Answer = RefusalAnswer{}
}
