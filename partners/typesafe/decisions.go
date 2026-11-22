// Adapter from the TypeSafe classifier onto the core decisions.DecisionModel
// interface (langchain-core 1.8, #41765): core questions map onto the
// TypeSafe vocabulary and typed answers map back with request metadata, so
// the generic ModelRouterMiddleware and AutoModeMiddleware can be driven by
// a TypeSafe classifier.

package typesafe

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/projanvil/langchain-golang/core/decisions"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/partners/internal/providerutil"
)

// AsDecisionModel adapts the classifier onto the core
// decisions.DecisionModel interface. Core Noul/Choice/Score questions map
// onto the TypeSafe wire vocabulary; Box has no TypeSafe counterpart and
// returns an error at invoke time.
func (c *Classifier) AsDecisionModel() decisions.DecisionModel {
	return &classifierAdapter{classifier: c}
}

type classifierAdapter struct {
	classifier *Classifier
}

// Invoke converts the core request, classifies, and converts the response.
// The classification surfaces as a chat-model run carrying the TypeSafe
// provider identity and the request's token usage (#41765 base-run
// semantics; tracing failures never fail the classification).
func (a *classifierAdapter) Invoke(ctx context.Context, request decisions.ClassifierRequest, opts ...runnables.Option) (decisions.ClassifierResponse, error) {
	if err := request.Validate(); err != nil {
		return decisions.ClassifierResponse{}, err
	}
	questions := make(map[string]Question, len(request.Questions))
	for id, question := range request.Questions {
		converted, err := questionToTypesafe(question)
		if err != nil {
			return decisions.ClassifierResponse{}, fmt.Errorf("question %q: %w", id, err)
		}
		questions[id] = converted
	}
	cfg := runnables.NewConfig(opts...)
	providerutil.DecisionStart(ctx, cfg, provider, a.classifier.Model, request)
	response, err := a.classifier.Invoke(ctx, Request{State: request.State, Questions: questions})
	if err != nil {
		providerutil.DecisionEnd(ctx, cfg, provider, a.classifier.Model, nil, nil, err)
		return decisions.ClassifierResponse{}, err
	}
	core := responseToCore(response)
	providerutil.DecisionEnd(ctx, cfg, provider, a.classifier.Model, core, core.Usage, nil)
	return core, nil
}

// questionToTypesafe maps one core question onto the TypeSafe vocabulary.
func questionToTypesafe(question decisions.Question) (Question, error) {
	switch q := question.(type) {
	case decisions.Noul:
		var criteria *NoulCriteria
		if q.Criteria != nil {
			criteria = &NoulCriteria{True: q.Criteria.True, False: q.Criteria.False}
		}
		return Noul{Instructions: q.Instructions, Criteria: criteria}, nil
	case decisions.Choice:
		criteria := make([]ChoiceCriterion, 0, len(q.Criteria))
		for _, label := range slices.Sorted(maps.Keys(q.Criteria)) {
			criteria = append(criteria, ChoiceCriterion{Label: label, Description: q.Criteria[label]})
		}
		return Choice{Instructions: q.Instructions, Criteria: criteria}, nil
	case decisions.Score:
		return Score{Instructions: q.Instructions, Criteria: q.Criteria}, nil
	case decisions.Box:
		return Box{Instructions: q.Instructions}, nil
	default:
		return nil, fmt.Errorf("typesafe: unsupported question type %T", question)
	}
}

// responseToCore maps a TypeSafe response onto the core envelope, completing
// partial usage (total = input + output).
func responseToCore(response Response) decisions.ClassifierResponse {
	answers := make(map[string]decisions.Answer, len(response.Answers))
	for id, answer := range response.Answers {
		switch v := answer.(type) {
		case NoulAnswer:
			answers[id] = decisions.NoulAnswer{Noul: v.Noul}
		case ChoiceAnswer:
			answers[id] = decisions.ChoiceAnswer{
				Choice:        v.Choice,
				Probabilities: v.Probabilities,
				Confidence:    v.Confidence,
			}
		case ScoreAnswer:
			answers[id] = decisions.ScoreAnswer{
				Score:         v.Score,
				Legend:        v.Legend,
				Probabilities: v.Probabilities,
				Confidence:    v.Confidence,
			}
		case BoxAnswer:
			coreBox := decisions.BoxAnswer{
				Box: decisions.BoxCoordinates{
					Top:    v.Box.Top,
					Bottom: v.Box.Bottom,
					Left:   v.Box.Left,
					Right:  v.Box.Right,
				},
				Confidence: v.Confidence,
			}
			answers[id] = coreBox
		}
	}
	core := decisions.ClassifierResponse{
		Model:     response.Model,
		Answers:   answers,
		RequestID: response.RequestID,
	}
	if response.Usage.InputTokens != nil || response.Usage.OutputTokens != nil {
		input, output := 0, 0
		if response.Usage.InputTokens != nil {
			input = *response.Usage.InputTokens
		}
		if response.Usage.OutputTokens != nil {
			output = *response.Usage.OutputTokens
		}
		core.Usage = &messages.UsageMetadata{
			InputTokens:  input,
			OutputTokens: output,
			TotalTokens:  input + output,
		}
	}
	return core
}

// compile-time proof the adapter satisfies the core interface.
var _ decisions.DecisionModel = (*classifierAdapter)(nil)
