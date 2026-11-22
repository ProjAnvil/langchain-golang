package decisions

import (
	"encoding/json"
	"reflect"
	"strings"
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
	// An empty questions mapping is legal (Python's ClassifierRequest is a
	// bare TypedDict with no runtime validation).
	if err := (ClassifierRequest{}).Validate(); err != nil {
		t.Fatalf("empty request must pass: %v", err)
	}
	request := ClassifierRequest{
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
// discriminator, field order, and explicit nulls for unset fields.
func TestQuestionWireForms(t *testing.T) {
	noul, err := json.Marshal(Noul{Instructions: "yes?", Criteria: &NoulCriteria{True: "good"}})
	if err != nil || string(noul) != `{"type":"noul","instructions":"yes?","criteria":{"true":"good","false":null}}` {
		t.Fatalf("noul wire: %s %v", noul, err)
	}
	bareNoul, err := json.Marshal(Noul{Instructions: "yes?"})
	if err != nil || string(bareNoul) != `{"type":"noul","instructions":"yes?","criteria":null}` {
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

// TestDecodeMapEmptyInputs pins the required-field semantics (pydantic
// rejects missing probabilities/legend just like missing keys): absent
// maps are errors, and a nil map marshals back as an explicit null.
func TestDecodeMapEmptyInputs(t *testing.T) {
	if _, err := UnmarshalAnswer([]byte(`{"type":"choice","choice":"x"}`)); err == nil {
		t.Fatal("choice without probabilities must fail")
	}
	if _, err := UnmarshalAnswer([]byte(`{"type":"score","score":1}`)); err == nil {
		t.Fatal("score without legend/probabilities must fail")
	}
	if _, err := UnmarshalAnswer([]byte(`{"type":"choice","choice":"x","probabilities":{"a":1}}`)); err == nil {
		t.Fatal("choice without confidence must fail")
	}
}

// TestCriteriaAcceptAnyJsonValue ports the upstream
// test_criteria_accept_any_json_value: criteria hold any JSON-compatible
// content — numbers as Noul outcome descriptions, mixed-type rubrics.
func TestCriteriaAcceptAnyJsonValue(t *testing.T) {
	noul := Noul{Instructions: "Is this urgent?", Criteria: &NoulCriteria{True: 1, False: 0}}
	if noul.Criteria.True != 1 || noul.Criteria.False != 0 {
		t.Fatalf("noul criteria: %#v", noul.Criteria)
	}
	score := Score{Instructions: "Rate it.", Criteria: []Content{0.0, map[string]any{"label": "high"}}}
	if err := score.Validate(); err != nil {
		t.Fatalf("mixed-type rubric: %v", err)
	}
	encoded, err := json.Marshal(score)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(encoded), `"criteria":[0,{"label":"high"}]`) {
		t.Fatalf("wire form: %s", encoded)
	}
}

// TestDecodeMalformedMaps covers the map-decode failure branches: a
// probabilities or legend that is not a JSON object must error.
func TestDecodeMalformedMaps(t *testing.T) {
	if _, err := UnmarshalAnswer([]byte(`{"type":"choice","choice":"x","probabilities":"nope"}`)); err == nil {
		t.Fatal("malformed choice probabilities must fail")
	}
	if _, err := UnmarshalAnswer([]byte(`{"type":"score","score":1,"legend":"nope"}`)); err == nil {
		t.Fatal("malformed score legend must fail")
	}
	if _, err := UnmarshalAnswer([]byte(`{"type":"score","score":1,"probabilities":"nope"}`)); err == nil {
		t.Fatal("malformed score probabilities must fail")
	}
}

// TestScoreAnswerNilMapsMarshal covers the nil-map marshal paths: a score
// answer without legend or probabilities marshals with both keys omitted.
func TestScoreAnswerNilMapsMarshal(t *testing.T) {
	encoded, err := json.Marshal(ScoreAnswer{Score: 1.5})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// Nil maps marshal as explicit nulls on the envelope (no omitempty);
	// a round trip must decode them back to nil maps.
	decoded, err := UnmarshalAnswer(encoded)
	if err != nil {
		t.Fatalf("round trip: %v", err)
	}
	if typed := decoded.(ScoreAnswer); typed.Legend != nil || typed.Probabilities != nil {
		t.Fatalf("nil maps must round trip to nil: %#v", typed)
	}
}

// TestUnmarshalAnswerRequiredAndRangeErrors covers the audit-driven
// validation branches: noul and confidence range checks, and the required
// fields of every answer type.
func TestUnmarshalAnswerRequiredAndRangeErrors(t *testing.T) {
	bad := []string{
		`{"type":"noul","noul":1.5}`,
		`{"type":"noul","noul":-0.1}`,
		`{"type":"choice","choice":"x","probabilities":{"x":1}}`,
		`{"type":"choice","choice":"x","probabilities":{"x":1},"confidence":1.5}`,
		`{"type":"score","score":1,"legend":{"0":"l"},"probabilities":{"0":1}}`,
		`{"type":"score","score":1,"legend":{"0":"l"},"probabilities":{"0":1},"confidence":-1}`,
		`{"type":"box","box":{"top":0,"bottom":1,"left":0,"right":1},"confidence":2}`,
	}
	for _, payload := range bad {
		if _, err := UnmarshalAnswer([]byte(payload)); err == nil {
			t.Errorf("expected validation error for %s", payload)
		}
	}
}

// TestUnmarshalResponseRequiredKeys covers the response presence checks:
// missing or null model/answers are rejected (pydantic would).
func TestUnmarshalResponseRequiredKeys(t *testing.T) {
	bad := []string{
		`{"answers":{}}`,
		`{"model":"m"}`,
		`{"model":null,"answers":{}}`,
		`{"model":"m","answers":null}`,
	}
	for _, payload := range bad {
		if _, err := UnmarshalResponse([]byte(payload)); err == nil {
			t.Errorf("expected presence error for %s", payload)
		}
	}
}

// TestAnswerMarshalForms pins each answer type's wire form: only its own
// fields, in the core declaration order, with nil confidence on a box kept
// as an explicit null.
func TestAnswerMarshalForms(t *testing.T) {
	noul, _ := json.Marshal(NoulAnswer{Noul: 0.25})
	if string(noul) != `{"type":"noul","noul":0.25}` {
		t.Fatalf("noul wire: %s", noul)
	}
	choice, _ := json.Marshal(ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"billing": 1}, Confidence: 0.9})
	if string(choice) != `{"type":"choice","choice":"billing","probabilities":{"billing":1},"confidence":0.9}` {
		t.Fatalf("choice wire: %s", choice)
	}
	box, _ := json.Marshal(BoxAnswer{})
	if string(box) != `{"type":"box","box":{"top":0,"bottom":0,"left":0,"right":0},"confidence":null}` {
		t.Fatalf("box wire: %s", box)
	}
}

// TestScoreAnswerMarshalKeepsNilMapsAsNulls pins the score wire: nil legend
// and probabilities marshal as explicit nulls between the score and
// confidence keys.
func TestScoreAnswerMarshalKeepsNilMapsAsNulls(t *testing.T) {
	encoded, err := json.Marshal(ScoreAnswer{Score: 1.5, Confidence: 0.5})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(encoded) != `{"type":"score","score":1.5,"legend":null,"probabilities":null,"confidence":0.5}` {
		t.Fatalf("score wire: %s", encoded)
	}
}

// TestDecodeAndMarshalEdgeBranches covers the remaining decode/marshal
// error paths: integer-key maps with non-integer keys, malformed typed
// maps, and the error-wrapped decode context.
func TestDecodeAndMarshalEdgeBranches(t *testing.T) {
	// Non-integer level keys in score maps.
	if _, err := UnmarshalAnswer([]byte(
		`{"type":"score","score":1,"legend":{"x":"low"},"probabilities":{"0":1},"confidence":0.5}`)); err == nil {
		t.Error("non-integer legend key must fail")
	}
	if _, err := UnmarshalAnswer([]byte(
		`{"type":"score","score":1,"legend":{"0":"low"},"probabilities":{"x":1},"confidence":0.5}`)); err == nil {
		t.Error("non-integer probabilities key must fail")
	}
	// Typed-map decode failures reached through the answer envelope.
	if _, err := UnmarshalAnswer([]byte(
		`{"type":"score","score":1,"legend":[1,2],"probabilities":{"0":1},"confidence":0.5}`)); err == nil {
		t.Error("non-object legend must fail")
	}
	if _, err := UnmarshalAnswer([]byte(
		`{"type":"score","score":1,"legend":{"0":"low"},"probabilities":[1],"confidence":0.5}`)); err == nil {
		t.Error("non-object probabilities must fail")
	}
}

// TestQuestionMarshalErrorBranches covers the instructions/criteria marshal
// failure paths with unserializable content values.
func TestQuestionMarshalErrorBranches(t *testing.T) {
	if _, err := json.Marshal(Noul{Instructions: make(chan int)}); err == nil {
		t.Error("noul instructions marshal must fail")
	}
	if _, err := json.Marshal(Noul{Instructions: "x", Criteria: &NoulCriteria{True: make(chan int)}}); err == nil {
		t.Error("noul criteria marshal must fail")
	}
	if _, err := json.Marshal(Choice{Instructions: make(chan int), Criteria: map[string]Content{"a": nil}}); err == nil {
		t.Error("choice instructions marshal must fail")
	}
	if _, err := json.Marshal(Choice{Instructions: "x", Criteria: map[string]Content{"a": make(chan int)}}); err == nil {
		t.Error("choice criteria marshal must fail")
	}
	if _, err := json.Marshal(Score{Instructions: make(chan int), Criteria: []Content{"a", "b"}}); err == nil {
		t.Error("score instructions marshal must fail")
	}
	if _, err := json.Marshal(Score{Instructions: "x", Criteria: []Content{"a", make(chan int)}}); err == nil {
		t.Error("score criteria marshal must fail")
	}
	if _, err := json.Marshal(Box{Instructions: func() {}}); err == nil {
		t.Error("box instructions marshal must fail")
	}
}

// TestUnmarshalResponseEnvelopeErrors covers the outer decode failure: a
// model that is not a string fails the envelope decode.
func TestUnmarshalResponseEnvelopeErrors(t *testing.T) {
	if _, err := UnmarshalResponse([]byte(`{"model":5,"answers":{}}`)); err == nil {
		t.Error("non-string model must fail")
	}
}
