package typesafe

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Wire golden tests mirror langchain_typesafe 0.0.1a3 (upstream #40542):
// pydantic marshals questions by field declaration order, so Noul emits
// type/instructions/criteria while Choice and Score emit
// type/criteria/instructions. exclude_none semantics drop nil keys
// recursively (classifier.py:447 model_dump(mode="json", exclude_none=True)).

func TestNoulMarshalFull(t *testing.T) {
	q := Noul{
		Instructions: "Does this message require an urgent response?",
		Criteria:     &NoulCriteria{True: "Harm is possible.", False: "All clear."},
	}
	got, err := json.Marshal(q)
	if err != nil {
		t.Fatalf("marshal noul: %v", err)
	}
	want := `{"type":"noul","instructions":"Does this message require an urgent response?","criteria":{"true":"Harm is possible.","false":"All clear."}}`
	if string(got) != want {
		t.Errorf("noul wire mismatch:\n got %s\nwant %s", got, want)
	}
}

func TestNoulMarshalExcludeNone(t *testing.T) {
	cases := []struct {
		name string
		q    Noul
		want string
	}{
		{
			name: "criteria only true",
			q:    Noul{Instructions: "i", Criteria: &NoulCriteria{True: "t"}},
			want: `{"type":"noul","instructions":"i","criteria":{"true":"t"}}`,
		},
		{
			name: "criteria only false",
			q:    Noul{Instructions: "i", Criteria: &NoulCriteria{False: "f"}},
			want: `{"type":"noul","instructions":"i","criteria":{"false":"f"}}`,
		},
		{
			name: "criteria present but empty",
			q:    Noul{Instructions: "i", Criteria: &NoulCriteria{}},
			want: `{"type":"noul","instructions":"i","criteria":{}}`,
		},
		{
			name: "nil criteria omitted",
			q:    Noul{Instructions: "i"},
			want: `{"type":"noul","instructions":"i"}`,
		},
		{
			name: "nil instructions omitted",
			q:    Noul{Criteria: &NoulCriteria{True: "t"}},
			want: `{"type":"noul","criteria":{"true":"t"}}`,
		},
		{
			name: "bare noul",
			q:    Noul{},
			want: `{"type":"noul"}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := json.Marshal(tc.q)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("wire mismatch:\n got %s\nwant %s", got, tc.want)
			}
		})
	}
}

func TestNoulMarshalStructuredInstructions(t *testing.T) {
	q := Noul{Instructions: map[string]any{"ask": "route"}}
	got, err := json.Marshal(q)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"type":"noul","instructions":{"ask":"route"}}`
	if string(got) != want {
		t.Errorf("wire mismatch:\n got %s\nwant %s", got, want)
	}
}

func TestChoiceMarshal(t *testing.T) {
	q := Choice{
		Instructions: "Which team should handle this request?",
		Criteria: []ChoiceCriterion{
			{Label: "billing", Description: "Payment."},
			{Label: "sales"}, // nil description: key omitted
			{Label: "technical", Description: "Bugs."},
		},
	}
	got, err := json.Marshal(q)
	if err != nil {
		t.Fatalf("marshal choice: %v", err)
	}
	want := `{"type":"choice","criteria":{"billing":"Payment.","technical":"Bugs."},"instructions":"Which team should handle this request?"}`
	if string(got) != want {
		t.Errorf("choice wire mismatch:\n got %s\nwant %s", got, want)
	}
}

func TestChoiceMarshalStructuredCriterion(t *testing.T) {
	q := Choice{
		Instructions: "route",
		Criteria:     []ChoiceCriterion{{Label: "a", Description: map[string]any{"k": 1}}},
	}
	got, err := json.Marshal(q)
	if err != nil {
		t.Fatalf("marshal choice: %v", err)
	}
	want := `{"type":"choice","criteria":{"a":{"k":1}},"instructions":"route"}`
	if string(got) != want {
		t.Errorf("choice wire mismatch:\n got %s\nwant %s", got, want)
	}
}

func TestChoiceMarshalEmptyCriteria(t *testing.T) {
	if _, err := json.Marshal(Choice{Instructions: "i"}); err == nil {
		t.Fatal("expected error for empty choice criteria")
	}
}

func TestScoreMarshal(t *testing.T) {
	q := Score{
		Instructions: "How frustrated does the customer appear?",
		Criteria:     []any{"Calm and neutral.", "Very angry."},
	}
	got, err := json.Marshal(q)
	if err != nil {
		t.Fatalf("marshal score: %v", err)
	}
	want := `{"type":"score","criteria":["Calm and neutral.","Very angry."],"instructions":"How frustrated does the customer appear?"}`
	if string(got) != want {
		t.Errorf("score wire mismatch:\n got %s\nwant %s", got, want)
	}
}

func TestScoreMarshalTooFewCriteria(t *testing.T) {
	if _, err := json.Marshal(Score{Instructions: "i", Criteria: []any{"only"}}); err == nil {
		t.Fatal("expected error for score with fewer than two criteria")
	}
}

func TestQuestionMapMarshalKeepsOrderedValues(t *testing.T) {
	questions := map[string]Question{
		"route": Choice{Instructions: "r", Criteria: []ChoiceCriterion{{Label: "a", Description: "d"}}},
	}
	got, err := json.Marshal(questions)
	if err != nil {
		t.Fatalf("marshal questions: %v", err)
	}
	want := `{"route":{"type":"choice","criteria":{"a":"d"},"instructions":"r"}}`
	if string(got) != want {
		t.Errorf("questions wire mismatch:\n got %s\nwant %s", got, want)
	}
}

func TestToAPIShape(t *testing.T) {
	noul := Noul{Instructions: "i", Criteria: &NoulCriteria{True: "t", False: "f"}}
	got, err := noul.toAPI()
	if err != nil {
		t.Fatalf("noul toAPI: %v", err)
	}
	want := map[string]any{
		"type":         "noul",
		"instructions": "i",
		"criteria":     map[string]any{"true": "t", "false": "f"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("noul toAPI mismatch:\n got %#v\nwant %#v", got, want)
	}

	choice := Choice{Instructions: "i", Criteria: []ChoiceCriterion{{Label: "a", Description: "d"}}}
	got, err = choice.toAPI()
	if err != nil {
		t.Fatalf("choice toAPI: %v", err)
	}
	want = map[string]any{
		"type":         "choice",
		"instructions": "i",
		"criteria":     map[string]any{"a": "d"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("choice toAPI mismatch:\n got %#v\nwant %#v", got, want)
	}

	score := Score{Instructions: "i", Criteria: []any{"a", "b"}}
	got, err = score.toAPI()
	if err != nil {
		t.Fatalf("score toAPI: %v", err)
	}
	want = map[string]any{
		"type":         "score",
		"instructions": "i",
		"criteria":     []any{"a", "b"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("score toAPI mismatch:\n got %#v\nwant %#v", got, want)
	}

	if _, err := (Score{Instructions: "i"}).toAPI(); err == nil {
		t.Error("expected error from score toAPI with too few criteria")
	}
	if _, err := (Choice{Instructions: "i"}).toAPI(); err == nil {
		t.Error("expected error from choice toAPI with empty criteria")
	}
}

func TestResponseViews(t *testing.T) {
	in, out := 7, 9
	response := Response{
		Model: "jev-latest",
		Answers: map[string]Answer{
			"urgent":      NoulAnswer{Noul: 0.75},
			"department":  ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"billing": 0.9}, Confidence: 0.9},
			"frustration": ScoreAnswer{Score: 1.5, Legend: map[int]any{0: "calm", 1: "angry"}, Probabilities: map[int]float64{0: 0.25, 1: 0.75}, Confidence: 0.5},
		},
		Usage:     Usage{InputTokens: &in, OutputTokens: &out},
		RequestID: "req-1",
	}

	nouls := response.Nouls()
	if len(nouls) != 1 || nouls["urgent"].Noul != 0.75 {
		t.Errorf("nouls view mismatch: %#v", nouls)
	}
	choices := response.Choices()
	if len(choices) != 1 || choices["department"].Choice != "billing" {
		t.Errorf("choices view mismatch: %#v", choices)
	}
	scores := response.Scores()
	if len(scores) != 1 || scores["frustration"].Score != 1.5 || scores["frustration"].Legend[1] != "angry" {
		t.Errorf("scores view mismatch: %#v", scores)
	}
}

func TestResponseViewsPointerAnswers(t *testing.T) {
	response := Response{
		Answers: map[string]Answer{
			"u": &NoulAnswer{Noul: 0.25},
			"d": &ChoiceAnswer{Choice: "x"},
			"s": &ScoreAnswer{Score: 2},
		},
	}
	if response.Nouls()["u"].Noul != 0.25 {
		t.Errorf("pointer noul answer not surfaced: %#v", response.Nouls())
	}
	if response.Choices()["d"].Choice != "x" {
		t.Errorf("pointer choice answer not surfaced: %#v", response.Choices())
	}
	if response.Scores()["s"].Score != 2 {
		t.Errorf("pointer score answer not surfaced: %#v", response.Scores())
	}
}

func TestSealedInterfaces(t *testing.T) {
	// Compile-time and runtime check of the sealed unions; calling the
	// markers directly keeps the no-op methods covered.
	var noul Question = Noul{}
	var choice Question = Choice{Criteria: []ChoiceCriterion{{Label: "a"}}}
	var score Question = Score{Criteria: []any{"a", "b"}}
	noul.questionMarker()
	choice.questionMarker()
	score.questionMarker()

	var noulAnswer Answer = NoulAnswer{}
	var choiceAnswer Answer = ChoiceAnswer{}
	var scoreAnswer Answer = ScoreAnswer{}
	noulAnswer.answerMarker()
	choiceAnswer.answerMarker()
	scoreAnswer.answerMarker()
}

func TestMarshalUnencodableContent(t *testing.T) {
	ch := make(chan int)
	if _, err := json.Marshal(Noul{Instructions: ch}); err == nil {
		t.Error("expected noul instructions marshal error")
	}
	if _, err := json.Marshal(Noul{Criteria: &NoulCriteria{True: ch}}); err == nil {
		t.Error("expected noul criteria marshal error")
	}
	if _, err := json.Marshal(Choice{
		Instructions: "i",
		Criteria:     []ChoiceCriterion{{Label: "a", Description: ch}},
	}); err == nil {
		t.Error("expected choice description marshal error")
	}
	if _, err := json.Marshal(Choice{
		Instructions: ch,
		Criteria:     []ChoiceCriterion{{Label: "a", Description: "d"}},
	}); err == nil {
		t.Error("expected choice instructions marshal error")
	}
	if _, err := json.Marshal(Score{Instructions: "i", Criteria: []any{ch, "b"}}); err == nil {
		t.Error("expected score criteria marshal error")
	}
	if _, err := json.Marshal(Score{Instructions: ch, Criteria: []any{"a", "b"}}); err == nil {
		t.Error("expected score instructions marshal error")
	}
}
