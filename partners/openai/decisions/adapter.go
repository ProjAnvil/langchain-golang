// Adapter from the OpenAI Decisions client onto the core
// decisions.DecisionModel interface (langchain-core 1.8, #41765): core
// questions map onto the Decisions API vocabulary (Predicate for Noul,
// Choice, Score) and typed answers map back, so the generic
// ModelRouterMiddleware and AutoModeMiddleware can be driven by the
// OpenAI Decisions API.

package decisions

import (
	"context"
	"fmt"
	"maps"
	"slices"

	coredecisions "github.com/projanvil/langchain-golang/core/decisions"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/partners/internal/providerutil"
)

// AsDecisionModel adapts the client onto the core decisions.DecisionModel
// interface. Core Noul maps onto the Decisions API's Predicate (the core
// vocabulary's name for the same binary-probability question); Choice and
// Score map directly. Box has no Decisions counterpart and returns an error
// at invoke time. Refusal answers are dropped: the core vocabulary has no
// refusal answer type.
func (c *Client) AsDecisionModel() coredecisions.DecisionModel {
	return &clientAdapter{client: c}
}

type clientAdapter struct {
	client *Client
}

// Invoke converts the core request, classifies, and converts the response.
// The classification surfaces as a chat-model run carrying the OpenAI
// provider identity and the request's token usage (#41765 base-run
// semantics; tracing failures never fail the classification).
func (a *clientAdapter) Invoke(ctx context.Context, request coredecisions.Request, opts ...runnables.Option) (coredecisions.Response, error) {
	if err := request.Validate(); err != nil {
		return coredecisions.Response{}, err
	}
	questions := make(map[string]Question, len(request.Questions))
	for id, question := range request.Questions {
		converted, err := questionToOpenAI(question)
		if err != nil {
			return coredecisions.Response{}, fmt.Errorf("question %q: %w", id, err)
		}
		questions[id] = converted
	}
	cfg := runnables.NewConfig(opts...)
	providerutil.DecisionStart(ctx, cfg, "openai", a.client.Model, request)
	response, err := a.client.Invoke(ctx, Request{Input: request.State, Questions: questions})
	if err != nil {
		providerutil.DecisionEnd(ctx, cfg, "openai", a.client.Model, nil, nil, err)
		return coredecisions.Response{}, err
	}
	core := responseToCore(response)
	providerutil.DecisionEnd(ctx, cfg, "openai", a.client.Model, core, core.Usage, nil)
	return core, nil
}

// questionToOpenAI maps one core question onto the Decisions API
// vocabulary. Instructions must be strings on this API.
func questionToOpenAI(question coredecisions.Question) (Question, error) {
	switch q := question.(type) {
	case coredecisions.Noul:
		instructions, err := stringInstructions(q.Instructions)
		if err != nil {
			return nil, err
		}
		// Noul's optional criteria have no Predicate counterpart; they are
		// folded into the instructions by the caller if needed.
		return Predicate{Instructions: instructions}, nil
	case coredecisions.Choice:
		instructions, err := stringInstructions(q.Instructions)
		if err != nil {
			return nil, err
		}
		options := make([]ChoiceOption, 0, len(q.Criteria))
		for _, label := range slices.Sorted(maps.Keys(q.Criteria)) {
			option := ChoiceOption{Value: label}
			if description, ok := q.Criteria[label].(string); ok {
				text := description
				option.Description = &text
			}
			options = append(options, option)
		}
		return Choice{Instructions: instructions, Choices: options}, nil
	case coredecisions.Score:
		instructions, err := stringInstructions(q.Instructions)
		if err != nil {
			return nil, err
		}
		levels := make([]ScoreLevel, 0, len(q.Criteria))
		for _, criterion := range q.Criteria {
			label, ok := criterion.(string)
			if !ok {
				return nil, fmt.Errorf("openai decisions: score levels must be strings, got %T", criterion)
			}
			levels = append(levels, L(label))
		}
		return Score{Instructions: instructions, Levels: levels}, nil
	default:
		return nil, fmt.Errorf("openai decisions: unsupported question type %T (no Box on the Decisions API)", question)
	}
}

// stringInstructions extracts a string instructions value.
func stringInstructions(instructions any) (string, error) {
	text, ok := instructions.(string)
	if !ok {
		return "", fmt.Errorf("openai decisions: instructions must be a string, got %T", instructions)
	}
	return text, nil
}

// responseToCore maps a Decisions response onto the core envelope.
func responseToCore(response Response) coredecisions.Response {
	answers := make(map[string]coredecisions.Answer, len(response.Answers))
	for id, answer := range response.Answers {
		switch v := answer.(type) {
		case PredicateAnswer:
			answers[id] = coredecisions.NoulAnswer{Noul: v.Probability}
		case ChoiceAnswer:
			label, ok := v.Choice.(string)
			if !ok {
				continue // bool-valued choices have no core vocabulary slot
			}
			answers[id] = coredecisions.ChoiceAnswer{
				Choice:        label,
				Probabilities: v.Probabilities,
				Confidence:    v.Confidence,
			}
		case ScoreAnswer:
			legend := make(map[int]coredecisions.Content, len(v.Legend))
			for level, label := range v.Legend {
				legend[level] = label
			}
			answers[id] = coredecisions.ScoreAnswer{
				Score:         v.Score,
				Legend:        legend,
				Probabilities: v.Probabilities,
				Confidence:    v.Confidence,
			}
		case RefusalAnswer:
			// Dropped: the core vocabulary has no refusal answer type.
		}
	}
	core := coredecisions.Response{
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
var _ coredecisions.DecisionModel = (*clientAdapter)(nil)
