// Package decisions provides a Go client for the OpenAI Decisions API,
// mirroring langchain_openai.decisions (openai partner 1.7.0, PR #41125):
// typed Predicate/Choice/Score questions answered over shared input, with
// probabilities, confidence, and token usage preserved on the response.
package decisions

// Question is a sealed union of the question types the Decisions API
// accepts. Names are supplied by the caller's questions map and injected
// into the wire form by toAPI (Python's discriminated union keyed on
// "type", types.py:188).
type Question interface {
	questionMarker()
	toAPI(name string) map[string]any
}

// Predicate asks for the probability that a condition is true
// (Python Predicate, types.py:34).
type Predicate struct {
	Instructions string
}

func (Predicate) questionMarker() {}

func (p Predicate) toAPI(name string) map[string]any {
	return map[string]any{
		"type":         "predicate",
		"name":         name,
		"instructions": p.Instructions,
	}
}

// ChoiceOption is one candidate value with an optional description of when
// it applies (Python choices values, types.py:102).
type ChoiceOption struct {
	Value       any // string or bool
	Description *string
}

// Choice selects one value from a fixed, unordered set of options
// (Python Choice, types.py:66). The ordered slice preserves the wire array
// order; descriptions are omitted from the wire form when nil.
type Choice struct {
	Instructions string
	Choices      []ChoiceOption
}

func (Choice) questionMarker() {}

func (c Choice) toAPI(name string) map[string]any {
	choices := make([]any, 0, len(c.Choices))
	for _, option := range c.Choices {
		entry := map[string]any{"value": option.Value}
		if option.Description != nil {
			entry["description"] = *option.Description
		}
		choices = append(choices, entry)
	}
	return map[string]any{
		"type":         "choice",
		"name":         name,
		"instructions": c.Instructions,
		"choices":      choices,
	}
}

// ScoreLevel is one level of a Score rubric (Python Level, types.py:122).
type ScoreLevel struct {
	Label       string
	Description *string
}

// L is shorthand for a score level with only a label, mirroring Python's
// acceptance of plain strings in Score.levels (types.py:169-173).
func L(label string) ScoreLevel {
	return ScoreLevel{Label: label}
}

// Score rates the input against ordered levels, indexed from zero in the
// order supplied (Python Score, types.py:132).
type Score struct {
	Instructions string
	Levels       []ScoreLevel // two or more, ordered lowest to highest
}

func (Score) questionMarker() {}

func (s Score) toAPI(name string) map[string]any {
	levels := make([]any, 0, len(s.Levels))
	for _, level := range s.Levels {
		entry := map[string]any{"label": level.Label}
		if level.Description != nil {
			entry["description"] = *level.Description
		}
		levels = append(levels, entry)
	}
	return map[string]any{
		"type":         "score",
		"name":         name,
		"instructions": s.Instructions,
		"levels":       levels,
	}
}

// Answer is a sealed union of the answer types the Decisions API returns
// (Python Answer, types.py:264).
type Answer interface {
	answerMarker()
}

// PredicateAnswer is the probability that a Predicate condition is true
// (Python PredicateAnswer, types.py:206).
type PredicateAnswer struct {
	Probability float64
}

func (PredicateAnswer) answerMarker() {}

// ChoiceAnswer is the selected Choice value with its probability
// distribution and confidence (Python ChoiceAnswer, types.py:216).
// Probabilities keys are the stringified candidate values.
type ChoiceAnswer struct {
	Choice        any // string or bool
	Probabilities map[string]float64
	Confidence    float64
}

func (ChoiceAnswer) answerMarker() {}

// ScoreAnswer is the expected Score value with its levels, distribution,
// and confidence (Python ScoreAnswer, types.py:236). Legend maps zero-based
// level indices to labels; Probabilities maps indices to probabilities.
type ScoreAnswer struct {
	Score         float64
	Legend        map[int]string
	Probabilities map[int]float64
	Confidence    float64
}

func (ScoreAnswer) answerMarker() {}

// RefusalAnswer reports that the model declined to answer a question. Extra
// carries any additional fields the API returned, mirroring Python's
// extra="allow" model config (types.py:258).
type RefusalAnswer struct {
	Extra map[string]any
}

func (RefusalAnswer) answerMarker() {}

// Usage reports token usage for a Decisions request; nil fields mean the
// API did not report that count (Python Usage, types.py:271).
type Usage struct {
	InputTokens  *int
	OutputTokens *int
}

// Response is the typed answer set returned from one Decisions request
// (Python DecisionResponse, types.py:281). Answers is keyed by the question
// names supplied in Request.Questions.
type Response struct {
	Model     string
	Answers   map[string]Answer
	Usage     Usage
	RequestID string
}

// Predicates returns predicate answers keyed by question name.
func (r *Response) Predicates() map[string]PredicateAnswer {
	return answersOfType[PredicateAnswer](r.Answers)
}

// Choices returns choice answers keyed by question name.
func (r *Response) Choices() map[string]ChoiceAnswer {
	return answersOfType[ChoiceAnswer](r.Answers)
}

// Scores returns score answers keyed by question name.
func (r *Response) Scores() map[string]ScoreAnswer {
	return answersOfType[ScoreAnswer](r.Answers)
}

// Refusals returns refusal answers keyed by question name.
func (r *Response) Refusals() map[string]RefusalAnswer {
	return answersOfType[RefusalAnswer](r.Answers)
}

func answersOfType[T Answer](answers map[string]Answer) map[string]T {
	out := make(map[string]T)
	for name, answer := range answers {
		if typed, ok := answer.(T); ok {
			out[name] = typed
		}
	}
	return out
}

// Request is the complete input for one Client invocation: the input to
// evaluate and the questions to answer over it (Python DecisionRequest,
// types.py:192).
type Request struct {
	Input     any
	Questions map[string]Question
}
