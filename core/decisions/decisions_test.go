package decisions

import (
	"encoding/json"
	"reflect"
	"testing"
)

// TestQuestionsRequireInstructions mirrors upstream
// test_questions_require_instructions: every question type rejects a nil
// instructions value.
func TestQuestionsRequireInstructions(t *testing.T) {
	questions := map[string]Question{
		"noul":   Noul{},
		"choice": Choice{Criteria: map[string]Content{"a": nil}},
		"score":  Score{Criteria: []Content{"low", "high"}},
		"box":    Box{},
	}
	for name, question := range questions {
		if err := question.Validate(); err == nil {
			t.Errorf("%s: expected instructions error", name)
		}
	}
}

// TestQuestionsValidateCriteria mirrors upstream
// test_questions_validate_their_criteria: Choice requires at least one
// label and Score at least two rubric levels.
func TestQuestionsValidateCriteria(t *testing.T) {
	invalid := map[string]Question{
		"choice empty":       Choice{Instructions: "c", Criteria: map[string]Content{}},
		"score empty":        Score{Instructions: "c", Criteria: []Content{}},
		"score one level":    Score{Instructions: "c", Criteria: []Content{"only"}},
		"choice no criteria": Choice{Instructions: "c"},
	}
	for name, question := range invalid {
		if err := question.Validate(); err == nil {
			t.Errorf("%s: expected criteria error", name)
		}
	}
	if err := (Choice{Instructions: "c", Criteria: map[string]Content{"a": nil}}).Validate(); err != nil {
		t.Errorf("valid choice rejected: %v", err)
	}
	if err := (Score{Instructions: "c", Criteria: []Content{"low", "high"}}).Validate(); err != nil {
		t.Errorf("valid score rejected: %v", err)
	}
}

// TestQuestionsAcceptStructuredContent mirrors upstream
// test_questions_accept_structured_content: instructions and criteria hold
// any JSON value, and the wire form keeps the type discriminator.
func TestQuestionsAcceptStructuredContent(t *testing.T) {
	question := Choice{
		Instructions: map[string]any{"task": "Route the request."},
		Criteria: map[string]Content{
			"billing": map[string]any{"examples": []any{"refund"}},
			"other":   nil,
		},
	}
	if err := question.Validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	encoded, err := json.Marshal(question)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded["type"] != "choice" {
		t.Fatalf("type discriminator: %v", decoded["type"])
	}
	criteria, _ := decoded["criteria"].(map[string]any)
	if criteria["billing"].(map[string]any)["examples"].([]any)[0] != "refund" {
		t.Fatalf("structured criteria: %#v", criteria)
	}
	if _, has := criteria["other"]; !has {
		t.Fatalf("explicit nil criterion must stay on the wire: %#v", criteria)
	}
	if decoded["instructions"].(map[string]any)["task"] != "Route the request." {
		t.Fatalf("structured instructions: %#v", decoded["instructions"])
	}
}

// TestResponseViewsFilterByAnswerType mirrors upstream
// test_response_views_filter_by_answer_type: the four views keep only their
// own answer types, keyed by question ID.
func TestResponseViewsFilterByAnswerType(t *testing.T) {
	response, err := UnmarshalResponse([]byte(`{
		"model": "fake-model",
		"answers": {
			"urgent": {"type": "noul", "noul": 0.9},
			"dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0}, "confidence": 1.0},
			"severity": {"type": "score", "score": 0.2, "legend": {"0": "low", "1": {"label": "high"}},
				"probabilities": {"0": 0.8, "1": 0.2}, "confidence": 0.5},
			"dog": {"type": "box", "box": {"top": 0.1, "bottom": 0.9, "left": 0.2, "right": 0.8}, "confidence": null}
		}
	}`))
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if nouls := response.Nouls(); len(nouls) != 1 || nouls["urgent"].Noul != 0.9 {
		t.Fatalf("nouls view: %#v", nouls)
	}
	if choices := response.Choices(); len(choices) != 1 || choices["dept"].Choice != "billing" ||
		choices["dept"].Confidence != 1.0 {
		t.Fatalf("choices view: %#v", choices)
	}
	scores := response.Scores()
	if len(scores) != 1 {
		t.Fatalf("scores view: %#v", scores)
	}
	severity := scores["severity"]
	if severity.Legend[0] != "low" || severity.Legend[1].(map[string]any)["label"] != "high" {
		t.Fatalf("score legend: %#v", severity.Legend)
	}
	if severity.Probabilities[0] != 0.8 || severity.Probabilities[1] != 0.2 {
		t.Fatalf("score probabilities: %#v", severity.Probabilities)
	}
	boxes := response.Boxes()
	if len(boxes) != 1 || boxes["dog"].Box != (BoxCoordinates{Top: 0.1, Bottom: 0.9, Left: 0.2, Right: 0.8}) {
		t.Fatalf("boxes view: %#v", boxes)
	}
	if boxes["dog"].Confidence != nil {
		t.Fatalf("box confidence: %#v", boxes["dog"].Confidence)
	}
}

// TestResponseRejectsUnknownAnswerTypes mirrors upstream
// test_response_rejects_unknown_answer_types.
func TestResponseRejectsUnknownAnswerTypes(t *testing.T) {
	if _, err := UnmarshalResponse([]byte(`{
		"model": "m",
		"answers": {"future": {"type": "UNKNOWN"}}
	}`)); err == nil {
		t.Fatal("expected unknown answer type error")
	}
}

// TestResponseIgnoresUnknownFields mirrors upstream
// test_response_ignores_unknown_fields.
func TestResponseIgnoresUnknownFields(t *testing.T) {
	response, err := UnmarshalResponse([]byte(`{
		"model": "fake-model",
		"answers": {},
		"prompt_feedback": null
	}`))
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(response.Answers) != 0 || response.Usage != nil {
		t.Fatalf("unexpected content: %#v", response)
	}
}

// TestPartialUsageIsCompleted mirrors upstream test_partial_usage_is_completed:
// a missing count is 0 and total_tokens defaults to input+output.
func TestPartialUsageIsCompleted(t *testing.T) {
	response, err := UnmarshalResponse([]byte(`{
		"model": "m",
		"answers": {},
		"usage": {"input_tokens": 12}
	}`))
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	want := int64(12)
	if response.Usage.InputTokens != 12 || response.Usage.OutputTokens != 0 ||
		response.Usage.TotalTokens != int(want) {
		t.Fatalf("completed usage: %#v", response.Usage)
	}
}

// TestRequestValidateChecksEveryQuestion wires Request.Validate over the
// question map.
func TestRequestValidateChecksEveryQuestion(t *testing.T) {
	if err := (Request{}).Validate(); err == nil {
		t.Fatal("empty request must fail")
	}
	request := Request{
		State:     "state",
		Questions: map[string]Question{"q": Choice{Instructions: "c"}},
	}
	if err := request.Validate(); err == nil {
		t.Fatal("choice without criteria must fail")
	}
	request.Questions = map[string]Question{"q": Noul{Instructions: "yes?"}}
	if err := request.Validate(); err != nil {
		t.Fatalf("valid request rejected: %v", err)
	}
}

// TestAnswerRoundTrip checks the typed answers survive a marshal/unmarshal
// round trip through the wire envelope.
func TestAnswerRoundTrip(t *testing.T) {
	confidence := 0.7
	answers := map[string]Answer{
		"n": NoulAnswer{Noul: 0.25},
		"c": ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"billing": 0.9}, Confidence: 0.9},
		"s": ScoreAnswer{Score: 1.35, Legend: map[int]Content{0: "low", 1: "high"},
			Probabilities: map[int]float64{0: 0.3, 1: 0.7}, Confidence: 0.6},
		"b": BoxAnswer{Box: BoxCoordinates{Top: 0.1, Bottom: 0.9, Left: 0.2, Right: 0.8}, Confidence: &confidence},
	}
	for id, answer := range answers {
		encoded, err := json.Marshal(answer)
		if err != nil {
			t.Fatalf("marshal %s: %v", id, err)
		}
		decoded, err := UnmarshalAnswer(encoded)
		if err != nil {
			t.Fatalf("unmarshal %s: %v", id, err)
		}
		if !reflect.DeepEqual(decoded, answer) {
			t.Errorf("%s round trip: got %#v, want %#v", id, decoded, answer)
		}
	}
}

// TestQuestionWireForms pins each question type's wire form: the type
// discriminator, field order, and nil-criteria omission.
func TestQuestionWireForms(t *testing.T) {
	noul, err := json.Marshal(Noul{Instructions: "yes?", Criteria: &NoulCriteria{True: "good"}})
	if err != nil || string(noul) != `{"type":"noul","instructions":"yes?","criteria":{"true":"good"}}` {
		t.Fatalf("noul wire: %s %v", noul, err)
	}
	bareNoul, err := json.Marshal(Noul{Instructions: "yes?"})
	if err != nil || string(bareNoul) != `{"type":"noul","instructions":"yes?"}` {
		t.Fatalf("bare noul wire: %s %v", bareNoul, err)
	}
	score, err := json.Marshal(Score{Instructions: "rate", Criteria: []Content{"low", "high"}})
	if err != nil || string(score) != `{"type":"score","criteria":["low","high"],"instructions":"rate"}` {
		t.Fatalf("score wire: %s %v", score, err)
	}
	box, err := json.Marshal(Box{Instructions: "find the dog"})
	if err != nil || string(box) != `{"type":"box","instructions":"find the dog"}` {
		t.Fatalf("box wire: %s %v", box, err)
	}
}

// TestUnmarshalAnswerErrorPaths covers the decode error branches: a broken
// envelope, missing per-type payload fields, and malformed maps.
func TestUnmarshalAnswerErrorPaths(t *testing.T) {
	broken := []string{
		`{"type":"noul"}`,   // missing noul
		`{"type":"choice"}`, // missing choice
		`{"type":"score"}`,  // missing score
		`{"type":"box"}`,    // missing box
		`not json`,          // broken envelope
		`{"type":"choice","choice":"x","probabilities":{"a":"high"}}`, // bad probabilities
		`{"type":"score","score":1,"legend":{"x":"low"}}`,             // bad legend
		`{"type":"score","score":1,"probabilities":{"x":1}}`,          // bad levels
	}
	for _, payload := range broken {
		if _, err := UnmarshalAnswer([]byte(payload)); err == nil {
			t.Errorf("expected decode error for %s", payload)
		}
	}
}

// TestMarshalAnswerErrorPaths covers marshal failure branches: unmarshalable
// map values cannot occur for float maps, but a ScoreAnswer legend holding a
// channel value must fail.
func TestMarshalAnswerErrorPaths(t *testing.T) {
	bad := ScoreAnswer{Legend: map[int]Content{0: make(chan int)}}
	if _, err := json.Marshal(bad); err == nil {
		t.Fatal("expected score marshal error")
	}
	badChoice := ChoiceAnswer{Probabilities: map[string]float64{"billing": 1}, Choice: "billing"}
	encoded, err := json.Marshal(badChoice)
	if err != nil || encoded == nil {
		t.Fatalf("choice marshal: %v", err)
	}
}

// TestResponseUnmarshalErrors covers the response decode error branches.
func TestResponseUnmarshalErrors(t *testing.T) {
	if _, err := UnmarshalResponse([]byte(`nope`)); err == nil {
		t.Fatal("broken envelope must fail")
	}
	if _, err := UnmarshalResponse([]byte(`{"model":"m","answers":{"q":"not-an-object"}}`)); err == nil {
		t.Fatal("broken answer must fail")
	}
	// Usage with explicit total keeps it.
	response, err := UnmarshalResponse([]byte(
		`{"model":"m","answers":{},"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":10}}`))
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if response.Usage.TotalTokens != 10 {
		t.Fatalf("explicit total must be kept: %#v", response.Usage)
	}
}

// TestDecodeMapEmptyInputs covers the empty-payload decode branches: an
// absent probabilities or legend decodes to a nil map, not an error.
func TestDecodeMapEmptyInputs(t *testing.T) {
	choice, err := UnmarshalAnswer([]byte(`{"type":"choice","choice":"x"}`))
	if err != nil {
		t.Fatalf("choice without probabilities: %v", err)
	}
	if typed := choice.(ChoiceAnswer); typed.Probabilities != nil {
		t.Fatalf("absent probabilities: %#v", typed.Probabilities)
	}
	score, err := UnmarshalAnswer([]byte(`{"type":"score","score":1}`))
	if err != nil {
		t.Fatalf("score without maps: %v", err)
	}
	if typed := score.(ScoreAnswer); typed.Legend != nil || typed.Probabilities != nil {
		t.Fatalf("absent maps: %#v", typed)
	}
}
