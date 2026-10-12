// Package typesafe is the Go port of the langchain-typesafe partner for
// TypeSafe AI's Jev decision model (langchain_typesafe 0.0.1a3, upstream
// #40542/#40570). It mirrors the question/answer vocabulary of
// langchain_typesafe/types.py: Noul (binary), Choice (categorical), and
// Score (ordinal) questions, their typed answers, and the classifier
// request/response envelopes.
package typesafe

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Question is the sealed union of TypeSafe question types (Python's
// `Question` discriminated union on the "type" field, types.py:197).
type Question interface {
	questionMarker()
	// toAPI returns the wire object for this question as a plain map.
	// Map iteration order is not stable; the ordered wire form used on the
	// wire is produced by each type's MarshalJSON, which follows the pydantic
	// field declaration order (see the MarshalJSON methods).
	toAPI() (map[string]any, error)
}

// NoulCriteria optionally describes the two outcomes of a Noul question
// (types.py:34-48). True and False accept any JSON-compatible content; nil
// fields are omitted when marshaled, mirroring pydantic's recursive
// exclude_none semantics (classifier.py:447).
type NoulCriteria struct {
	True  any `json:"true,omitempty"`
	False any `json:"false,omitempty"`
}

// MarshalJSON writes {"true": ..., "false": ...} with nil fields omitted,
// following the pydantic declaration order.
func (c NoulCriteria) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	if c.True != nil {
		encoded, err := json.Marshal(c.True)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal noul criteria true: %w", err)
		}
		buf.WriteString(`"true":`)
		buf.Write(encoded)
	}
	if c.False != nil {
		if c.True != nil {
			buf.WriteByte(',')
		}
		encoded, err := json.Marshal(c.False)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal noul criteria false: %w", err)
		}
		buf.WriteString(`"false":`)
		buf.Write(encoded)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// Noul asks a binary question and reports the probability that the answer is
// yes (types.py:51-94).
type Noul struct {
	// Instructions is the complete yes/no judgment (string or structured
	// JSON content). A nil value omits the wire key, mirroring
	// model_dump(exclude_none=True).
	Instructions any
	// Criteria optionally describes the yes and no outcomes. Nil omits the
	// wire key entirely.
	Criteria *NoulCriteria
}

func (Noul) questionMarker() {}

// MarshalJSON writes the wire object in pydantic field declaration order:
// type, instructions, criteria (types.py:83-93).
func (q Noul) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteString(`{"type":"noul"`)
	if q.Instructions != nil {
		encoded, err := json.Marshal(q.Instructions)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal noul instructions: %w", err)
		}
		buf.WriteString(`,"instructions":`)
		buf.Write(encoded)
	}
	if q.Criteria != nil {
		encoded, err := json.Marshal(*q.Criteria)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal noul criteria: %w", err)
		}
		buf.WriteString(`,"criteria":`)
		buf.Write(encoded)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

func (q Noul) toAPI() (map[string]any, error) {
	out := map[string]any{"type": "noul"}
	if q.Instructions != nil {
		out["instructions"] = q.Instructions
	}
	if q.Criteria != nil {
		criteria := map[string]any{}
		if q.Criteria.True != nil {
			criteria["true"] = q.Criteria.True
		}
		if q.Criteria.False != nil {
			criteria["false"] = q.Criteria.False
		}
		out["criteria"] = criteria
	}
	return out, nil
}

// ChoiceCriterion is one candidate label with its optional description
// (Python's criteria dict entries, types.py:138-143). Descriptions may be
// text, structured JSON, or nil; nil descriptions are omitted from the wire
// object.
type ChoiceCriterion struct {
	Label       string
	Description any
}

// Choice selects one label from a fixed set of alternatives (types.py:97-146).
// The TypeSafe wire format carries criteria as an object mapping label to
// description, unlike OpenAI Decisions' array form.
type Choice struct {
	// Instructions is the complete categorical judgment (string or
	// structured JSON content). A nil value omits the wire key.
	Instructions any
	// Criteria holds the candidate labels in wire order; at least one is
	// required (Python Field(min_length=1)).
	Criteria []ChoiceCriterion
}

func (Choice) questionMarker() {}

// MarshalJSON writes the wire object in pydantic field declaration order:
// type, criteria, instructions (types.py:135-145). The criteria object
// preserves the slice order and omits nil descriptions.
func (q Choice) MarshalJSON() ([]byte, error) {
	if len(q.Criteria) == 0 {
		return nil, fmt.Errorf("typesafe: choice question requires at least one criterion")
	}
	var buf bytes.Buffer
	buf.WriteString(`{"type":"choice","criteria":{`)
	for i, criterion := range q.Criteria {
		if criterion.Description == nil {
			continue
		}
		if i > 0 {
			buf.WriteByte(',')
		}
		label, err := json.Marshal(criterion.Label)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal choice label: %w", err)
		}
		description, err := json.Marshal(criterion.Description)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal choice description: %w", err)
		}
		buf.Write(label)
		buf.WriteByte(':')
		buf.Write(description)
	}
	buf.WriteString(`}`)
	if q.Instructions != nil {
		encoded, err := json.Marshal(q.Instructions)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal choice instructions: %w", err)
		}
		buf.WriteString(`,"instructions":`)
		buf.Write(encoded)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

func (q Choice) toAPI() (map[string]any, error) {
	if len(q.Criteria) == 0 {
		return nil, fmt.Errorf("typesafe: choice question requires at least one criterion")
	}
	criteria := map[string]any{}
	for _, criterion := range q.Criteria {
		if criterion.Description != nil {
			criteria[criterion.Label] = criterion.Description
		}
	}
	out := map[string]any{"type": "choice", "criteria": criteria}
	if q.Instructions != nil {
		out["instructions"] = q.Instructions
	}
	return out, nil
}

// Score evaluates state against an ordered rubric of two or more levels
// (types.py:149-194).
type Score struct {
	// Instructions is the complete ordinal judgment (string or structured
	// JSON content). A nil value omits the wire key.
	Instructions any
	// Criteria holds the ordered level descriptions; at least two are
	// required (Python Field(min_length=2)).
	Criteria []any
}

func (Score) questionMarker() {}

// MarshalJSON writes the wire object in pydantic field declaration order:
// type, criteria, instructions (types.py:187-193).
func (q Score) MarshalJSON() ([]byte, error) {
	if len(q.Criteria) < 2 {
		return nil, fmt.Errorf("typesafe: score question requires at least two criteria")
	}
	var buf bytes.Buffer
	buf.WriteString(`{"type":"score"`)
	encoded, err := json.Marshal(q.Criteria)
	if err != nil {
		return nil, fmt.Errorf("typesafe: marshal score criteria: %w", err)
	}
	buf.WriteString(`,"criteria":`)
	buf.Write(encoded)
	if q.Instructions != nil {
		encoded, err := json.Marshal(q.Instructions)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal score instructions: %w", err)
		}
		buf.WriteString(`,"instructions":`)
		buf.Write(encoded)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

func (q Score) toAPI() (map[string]any, error) {
	if len(q.Criteria) < 2 {
		return nil, fmt.Errorf("typesafe: score question requires at least two criteria")
	}
	out := map[string]any{"type": "score", "criteria": q.Criteria}
	if q.Instructions != nil {
		out["instructions"] = q.Instructions
	}
	return out, nil
}

// Answer is the sealed union of TypeSafe answer types (types.py:273).
type Answer interface {
	answerMarker()
}

// NoulAnswer reports the probability that a Noul question's answer is yes
// (types.py:215-221). Noul answers carry no separate confidence value.
type NoulAnswer struct {
	Noul float64
}

func (NoulAnswer) answerMarker() {}

// ChoiceAnswer reports a selected label, its full probability distribution,
// and a derived confidence (types.py:225-247).
type ChoiceAnswer struct {
	Choice        string
	Probabilities map[string]float64
	Confidence    float64
}

func (ChoiceAnswer) answerMarker() {}

// ScoreAnswer reports the expected rubric position, the original zero-based
// legend, the level distribution, and a derived confidence (types.py:250-270).
type ScoreAnswer struct {
	Score         float64
	Legend        map[int]any
	Probabilities map[int]float64
	Confidence    float64
}

func (ScoreAnswer) answerMarker() {}

// Usage reports token usage for one classification request (types.py:277-284).
// Nil counters mean "not reported".
type Usage struct {
	InputTokens  *int
	OutputTokens *int
}

// Response is the typed envelope returned by one TypeSafe request
// (types.py:287-332). RequestID is empty when the provider did not report
// one.
type Response struct {
	Model     string
	Answers   map[string]Answer
	Usage     Usage
	RequestID string
}

// Nouls returns the binary answers keyed by their original question IDs.
func (r *Response) Nouls() map[string]NoulAnswer {
	out := make(map[string]NoulAnswer, len(r.Answers))
	for name, answer := range r.Answers {
		switch v := answer.(type) {
		case NoulAnswer:
			out[name] = v
		case *NoulAnswer:
			if v != nil {
				out[name] = *v
			}
		}
	}
	return out
}

// Choices returns the categorical answers keyed by their original question IDs.
func (r *Response) Choices() map[string]ChoiceAnswer {
	out := make(map[string]ChoiceAnswer, len(r.Answers))
	for name, answer := range r.Answers {
		switch v := answer.(type) {
		case ChoiceAnswer:
			out[name] = v
		case *ChoiceAnswer:
			if v != nil {
				out[name] = *v
			}
		}
	}
	return out
}

// Scores returns the ordinal answers keyed by their original question IDs.
func (r *Response) Scores() map[string]ScoreAnswer {
	out := make(map[string]ScoreAnswer, len(r.Answers))
	for name, answer := range r.Answers {
		switch v := answer.(type) {
		case ScoreAnswer:
			out[name] = v
		case *ScoreAnswer:
			if v != nil {
				out[name] = *v
			}
		}
	}
	return out
}

// Request is the complete input for one classifier invocation (Python's
// ClassifierRequest TypedDict, types.py:201-212). Questions are serialized
// keyed by name in stable sorted order (a deliberate divergence from
// Python's dict insertion order; see the client).
type Request struct {
	State     any
	Questions map[string]Question
}
