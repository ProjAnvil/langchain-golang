package decisions

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Answer is the sealed union of typed answers returned by decision models
// (Python's `Answer` discriminated union on the "type" field): NoulAnswer,
// ChoiceAnswer, ScoreAnswer, and BoxAnswer.
type Answer interface {
	answerMarker()
}

// NoulAnswer reports the probability that a Noul question's answer is yes.
type NoulAnswer struct {
	// Noul is the probability of yes in the inclusive range 0..1.
	Noul float64
}

func (NoulAnswer) answerMarker() {}

// ChoiceAnswer reports a selected Choice label with its probability
// distribution and confidence.
type ChoiceAnswer struct {
	// Choice is the label selected from the options supplied in
	// Choice.Criteria.
	Choice string
	// Probabilities holds the probability assigned to each candidate label,
	// keyed by label name, so applications can use a confidence measure or
	// risk policy different from the provider's.
	Probabilities map[string]float64
	// Confidence is a scalar certainty in 0..1 for the selected label.
	Confidence float64
}

func (ChoiceAnswer) answerMarker() {}

// ScoreAnswer reports an expected Score value with its rubric,
// distribution, and confidence.
type ScoreAnswer struct {
	// Score is the expected position on the ordered rubric. It may be
	// fractional because it summarizes the probability distribution over
	// integer rubric levels rather than selecting one level.
	Score float64
	// Legend holds the original rubric descriptions keyed by their
	// zero-based integer levels.
	Legend map[int]Content
	// Probabilities is the distribution over the zero-based integer levels.
	Probabilities map[int]float64
	// Confidence is a scalar certainty in 0..1, derived from the level
	// distribution.
	Confidence float64
}

func (ScoreAnswer) answerMarker() {}

// BoxCoordinates gives the edges of a box as fractions of the image height
// and width, from 0 to 1.
type BoxCoordinates struct {
	Top    float64 `json:"top"`
	Bottom float64 `json:"bottom"`
	Left   float64 `json:"left"`
	Right  float64 `json:"right"`
}

// BoxAnswer locates what a Box question describes.
type BoxAnswer struct {
	// Box holds the edges of the located box.
	Box BoxCoordinates
	// Confidence is a scalar certainty in 0..1, or nil when the provider
	// does not report it.
	Confidence *float64
}

func (BoxAnswer) answerMarker() {}

// answerWire is the on-the-wire answer envelope: a "type" discriminator
// plus the per-type payload fields. Unknown fields are ignored, matching
// pydantic's default behavior.
type answerWire struct {
	Type          string          `json:"type"`
	Noul          *float64        `json:"noul"`
	Choice        *string         `json:"choice"`
	Probabilities json.RawMessage `json:"probabilities"`
	Confidence    *float64        `json:"confidence"`
	Score         *float64        `json:"score"`
	Legend        json.RawMessage `json:"legend"`
	Box           *BoxCoordinates `json:"box"`
}

// UnmarshalAnswer decodes one wire answer envelope into its typed form,
// rejecting unknown answer types (Python's discriminated-union validation).
func UnmarshalAnswer(data []byte) (Answer, error) {
	var wire answerWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return nil, fmt.Errorf("decisions: decode answer: %w", err)
	}
	switch wire.Type {
	case "noul":
		if wire.Noul == nil {
			return nil, fmt.Errorf("decisions: noul answer missing \"noul\"")
		}
		return NoulAnswer{Noul: *wire.Noul}, nil
	case "choice":
		if wire.Choice == nil {
			return nil, fmt.Errorf("decisions: choice answer missing \"choice\"")
		}
		probabilities, err := decodeStringFloatMap(wire.Probabilities)
		if err != nil {
			return nil, fmt.Errorf("decisions: decode choice probabilities: %w", err)
		}
		confidence := 0.0
		if wire.Confidence != nil {
			confidence = *wire.Confidence
		}
		return ChoiceAnswer{
			Choice:        *wire.Choice,
			Probabilities: probabilities,
			Confidence:    confidence,
		}, nil
	case "score":
		if wire.Score == nil {
			return nil, fmt.Errorf("decisions: score answer missing \"score\"")
		}
		legend, err := decodeIntContentMap(wire.Legend)
		if err != nil {
			return nil, fmt.Errorf("decisions: decode score legend: %w", err)
		}
		probabilities, err := decodeIntFloatMap(wire.Probabilities)
		if err != nil {
			return nil, fmt.Errorf("decisions: decode score probabilities: %w", err)
		}
		confidence := 0.0
		if wire.Confidence != nil {
			confidence = *wire.Confidence
		}
		return ScoreAnswer{
			Score:         *wire.Score,
			Legend:        legend,
			Probabilities: probabilities,
			Confidence:    confidence,
		}, nil
	case "box":
		if wire.Box == nil {
			return nil, fmt.Errorf("decisions: box answer missing \"box\"")
		}
		return BoxAnswer{Box: *wire.Box, Confidence: wire.Confidence}, nil
	default:
		return nil, fmt.Errorf("decisions: unknown answer type %q", wire.Type)
	}
}

// MarshalJSON writes one answer in its wire envelope.
func (a NoulAnswer) MarshalJSON() ([]byte, error) {
	return json.Marshal(answerWire{Type: "noul", Noul: &a.Noul})
}

// MarshalJSON writes one answer in its wire envelope.
func (a ChoiceAnswer) MarshalJSON() ([]byte, error) {
	probabilities, err := json.Marshal(a.Probabilities)
	if err != nil {
		return nil, fmt.Errorf("decisions: marshal choice probabilities: %w", err)
	}
	return json.Marshal(answerWire{
		Type:          "choice",
		Choice:        &a.Choice,
		Probabilities: probabilities,
		Confidence:    &a.Confidence,
	})
}

// MarshalJSON writes one answer in its wire envelope; integer map keys
// serialize as JSON string keys.
func (a ScoreAnswer) MarshalJSON() ([]byte, error) {
	legend, err := marshalIntKeyMap(a.Legend)
	if err != nil {
		return nil, fmt.Errorf("decisions: marshal score legend: %w", err)
	}
	probabilities, err := marshalIntKeyMap(a.Probabilities)
	if err != nil {
		return nil, fmt.Errorf("decisions: marshal score probabilities: %w", err)
	}
	return json.Marshal(answerWire{
		Type:          "score",
		Score:         &a.Score,
		Legend:        legend,
		Probabilities: probabilities,
		Confidence:    &a.Confidence,
	})
}

// MarshalJSON writes one answer in its wire envelope.
func (a BoxAnswer) MarshalJSON() ([]byte, error) {
	box := a.Box
	return json.Marshal(answerWire{Type: "box", Box: &box, Confidence: a.Confidence})
}

// decodeStringFloatMap decodes {"label": p} wire objects.
func decodeStringFloatMap(data []byte) (map[string]float64, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var decoded map[string]float64
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

// decodeIntFloatMap decodes {"0": p} wire objects whose keys are integer
// level numbers spelled as strings (JSON object keys are always strings).
func decodeIntFloatMap(data []byte) (map[int]float64, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var decoded map[string]float64
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, err
	}
	out := make(map[int]float64, len(decoded))
	for key, value := range decoded {
		level, err := strconv.Atoi(key)
		if err != nil {
			return nil, fmt.Errorf("non-integer level key %q", key)
		}
		out[level] = value
	}
	return out, nil
}

// decodeIntContentMap decodes {"0": <any json>} wire objects into
// integer-keyed rubric legends.
func decodeIntContentMap(data []byte) (map[int]Content, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var decoded map[string]Content
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, err
	}
	out := make(map[int]Content, len(decoded))
	for key, value := range decoded {
		level, err := strconv.Atoi(key)
		if err != nil {
			return nil, fmt.Errorf("non-integer level key %q", key)
		}
		out[level] = value
	}
	return out, nil
}

// marshalIntKeyMap serializes an integer-keyed map with string JSON keys.
func marshalIntKeyMap[V any](m map[int]V) ([]byte, error) {
	if m == nil {
		return nil, nil
	}
	out := make(map[string]V, len(m))
	for key, value := range m {
		out[strconv.Itoa(key)] = value
	}
	return json.Marshal(out)
}

// Request is the complete input for one decision model invocation: the
// state to classify and the questions keyed by answer ID (Python's
// ClassifierRequest). State is a string, JSON object, array, or LangChain
// messages; standard media blocks may appear anywhere in it.
type Request struct {
	State     Content             `json:"state"`
	Questions map[string]Question `json:"questions"`
}

// Validate checks every question's invariants and that at least one
// question is present.
func (r Request) Validate() error {
	if len(r.Questions) == 0 {
		return fmt.Errorf("decisions: request requires at least one question")
	}
	for id, question := range r.Questions {
		if err := question.Validate(); err != nil {
			return fmt.Errorf("question %q: %w", id, err)
		}
	}
	return nil
}

// Response is the typed answers and metadata returned from one
// classification request (Python's ClassifierResponse). Access every answer
// through Answers, or the Nouls/Choices/Scores/Boxes views filtered by
// answer type. Each mapping preserves the question IDs supplied to the
// request.
type Response struct {
	// Model names the model that answered the request.
	Model string `json:"model"`
	// Answers holds all answers keyed by their original question IDs.
	Answers map[string]Answer `json:"answers"`
	// Usage reports token usage for the request, or nil when the provider
	// did not report any. Partial usage is completed on decode: a missing
	// count is 0 and total_tokens defaults to input+output.
	Usage *messages.UsageMetadata `json:"usage,omitempty"`
	// RequestID is the provider request ID, useful when diagnosing a
	// request with provider support.
	RequestID string `json:"request_id,omitempty"`
}

// responseWire is the wire envelope; unknown fields are ignored.
type responseWire struct {
	Model     string                     `json:"model"`
	Answers   map[string]json.RawMessage `json:"answers"`
	Usage     *messages.UsageMetadata    `json:"usage"`
	RequestID string                     `json:"request_id"`
}

// UnmarshalResponse decodes a wire classifier response, rejecting unknown
// answer types and completing partial usage.
func UnmarshalResponse(data []byte) (Response, error) {
	var wire responseWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return Response{}, fmt.Errorf("decisions: decode response: %w", err)
	}
	answers := make(map[string]Answer, len(wire.Answers))
	for id, raw := range wire.Answers {
		answer, err := UnmarshalAnswer(raw)
		if err != nil {
			return Response{}, fmt.Errorf("answer %q: %w", id, err)
		}
		answers[id] = answer
	}
	completeUsage(wire.Usage)
	return Response{
		Model:     wire.Model,
		Answers:   answers,
		Usage:     wire.Usage,
		RequestID: wire.RequestID,
	}, nil
}

// completeUsage fills a missing output count with 0 and a missing total
// with input+output (Python's ClassifierResponse._complete_usage).
func completeUsage(usage *messages.UsageMetadata) {
	if usage == nil {
		return
	}
	total := usage.InputTokens + usage.OutputTokens
	if usage.TotalTokens == 0 {
		usage.TotalTokens = total
	}
}

// Nouls returns the binary answers keyed by their original question IDs.
func (r Response) Nouls() map[string]NoulAnswer {
	out := map[string]NoulAnswer{}
	for id, answer := range r.Answers {
		if typed, ok := answer.(NoulAnswer); ok {
			out[id] = typed
		}
	}
	return out
}

// Choices returns the categorical answers keyed by their original question
// IDs.
func (r Response) Choices() map[string]ChoiceAnswer {
	out := map[string]ChoiceAnswer{}
	for id, answer := range r.Answers {
		if typed, ok := answer.(ChoiceAnswer); ok {
			out[id] = typed
		}
	}
	return out
}

// Scores returns the ordinal answers keyed by their original question IDs.
func (r Response) Scores() map[string]ScoreAnswer {
	out := map[string]ScoreAnswer{}
	for id, answer := range r.Answers {
		if typed, ok := answer.(ScoreAnswer); ok {
			out[id] = typed
		}
	}
	return out
}

// Boxes returns the localization answers keyed by their original question
// IDs.
func (r Response) Boxes() map[string]BoxAnswer {
	out := map[string]BoxAnswer{}
	for id, answer := range r.Answers {
		if typed, ok := answer.(BoxAnswer); ok {
			out[id] = typed
		}
	}
	return out
}
