package decisions

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Question is the sealed union of question types accepted by decision
// models (Python's `Question` discriminated union on the "type" field):
// Noul, Choice, Score, and Box.
type Question interface {
	questionMarker()
	// Validate checks the question's invariants (required instructions,
	// criteria shape).
	Validate() error
}

// Content is the free-form instruction payload of a question: text, a JSON
// object, or a JSON array (Python's Content alias). LangChain standard
// media blocks may appear anywhere; providers document which media they
// support.
type Content = any

// State is the root state accepted by decision models (Python's State
// alias): a string, JSON object or array, LangChain messages, with
// standard media blocks anywhere in it. It shares Content's any shape —
// the alias exists so the exported vocabulary matches upstream's.
type State = Content

// NoulCriteria optionally describes the two outcomes of a Noul question.
// Both values accept any JSON-compatible content, so callers can provide a
// short description or structured examples. Unset fields stay on the wire
// as explicit nulls (Python's model_dump).
type NoulCriteria struct {
	True  Content `json:"true"`
	False Content `json:"false"`
}

// Noul asks a binary question and reports the probability that its answer
// is yes (Noul/noul.py). A value near 1 indicates strong support for yes,
// near 0 strong support for no, near 0.5 uncertainty. Noul answers carry no
// separate confidence.
type Noul struct {
	// Instructions is the complete yes/no judgment to make about the input
	// state. Required (Validate rejects nil).
	Instructions Content
	// Criteria optionally describes what the yes and no outcomes mean.
	Criteria *NoulCriteria
}

func (Noul) questionMarker() {}

// Validate checks the question's invariants: instructions are required.
func (q Noul) Validate() error {
	if q.Instructions == nil {
		return fmt.Errorf("decisions: Noul requires instructions")
	}
	return nil
}

// MarshalJSON writes the wire object in the core field order:
// type, instructions, criteria.
func (q Noul) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteString(`{"type":"noul","instructions":`)
	encoded, err := json.Marshal(q.Instructions)
	if err != nil {
		return nil, fmt.Errorf("decisions: marshal noul instructions: %w", err)
	}
	buf.Write(encoded)
	buf.WriteString(`,"criteria":`)
	if encoded, err = json.Marshal(q.Criteria); err != nil {
		return nil, fmt.Errorf("decisions: marshal noul criteria: %w", err)
	}
	buf.Write(encoded)
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// Choice selects one label from a fixed set of alternatives. The answer
// includes the selected label, a probability for every supplied label, and
// a confidence value.
type Choice struct {
	// Criteria maps candidate labels to their descriptions. At least one
	// entry is required; include an "other" or "none_of_the_above" label
	// when the alternatives may not cover every input. Descriptions may be
	// text, structured JSON, or nil.
	Criteria map[string]Content
	// Instructions is the complete categorical judgment. Required.
	Instructions Content
}

func (Choice) questionMarker() {}

// Validate checks the question's invariants: instructions are required and
// criteria non-empty.
func (q Choice) Validate() error {
	if q.Instructions == nil {
		return fmt.Errorf("decisions: Choice requires instructions")
	}
	if len(q.Criteria) == 0 {
		return fmt.Errorf("decisions: Choice requires at least one label in criteria")
	}
	return nil
}

// MarshalJSON writes the wire object in the core field order:
// type, criteria, instructions.
func (q Choice) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteString(`{"type":"choice","criteria":`)
	encoded, err := json.Marshal(q.Criteria)
	if err != nil {
		return nil, fmt.Errorf("decisions: marshal choice criteria: %w", err)
	}
	buf.Write(encoded)
	buf.WriteString(`,"instructions":`)
	if encoded, err = json.Marshal(q.Instructions); err != nil {
		return nil, fmt.Errorf("decisions: marshal choice instructions: %w", err)
	}
	buf.Write(encoded)
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// Score evaluates state against an ordered rubric. Criteria are numbered
// from zero in their supplied order; the returned score is an expected
// value and may fall between integer levels. The full probability
// distribution stays available for custom decision logic.
type Score struct {
	// Criteria holds two or more ordered descriptions for the score levels
	// starting at zero.
	Criteria []Content
	// Instructions is the complete ordinal judgment. Required.
	Instructions Content
}

func (Score) questionMarker() {}

// Validate checks the question's invariants: instructions are required and
// criteria hold at least two levels.
func (q Score) Validate() error {
	if q.Instructions == nil {
		return fmt.Errorf("decisions: Score requires instructions")
	}
	if len(q.Criteria) < 2 {
		return fmt.Errorf("decisions: Score requires at least two rubric levels in criteria")
	}
	return nil
}

// MarshalJSON writes the wire object in the core field order:
// type, criteria, instructions.
func (q Score) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteString(`{"type":"score","criteria":`)
	encoded, err := json.Marshal(q.Criteria)
	if err != nil {
		return nil, fmt.Errorf("decisions: marshal score criteria: %w", err)
	}
	buf.Write(encoded)
	buf.WriteString(`,"instructions":`)
	if encoded, err = json.Marshal(q.Instructions); err != nil {
		return nil, fmt.Errorf("decisions: marshal score instructions: %w", err)
	}
	buf.Write(encoded)
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// Box locates what the instructions describe in an image in the state. The
// answer gives the edges of a box as fractions of the image height and
// width.
type Box struct {
	// Instructions describes what to locate in the image. Required.
	Instructions Content
}

func (Box) questionMarker() {}

// Validate checks the question's invariants: instructions are required.
func (q Box) Validate() error {
	if q.Instructions == nil {
		return fmt.Errorf("decisions: Box requires instructions")
	}
	return nil
}

// MarshalJSON writes the wire object in the core field order:
// type, instructions.
func (q Box) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteString(`{"type":"box","instructions":`)
	encoded, err := json.Marshal(q.Instructions)
	if err != nil {
		return nil, fmt.Errorf("decisions: marshal box instructions: %w", err)
	}
	buf.Write(encoded)
	buf.WriteByte('}')
	return buf.Bytes(), nil
}
