// Adapter from the OpenAI Decisions client onto the core
// decisions.DecisionModel interface (langchain-core 1.8, #41765): core
// questions map onto the Decisions API vocabulary (Predicate for Noul,
// Choice, Score) and typed answers map back, so the generic
// ModelRouterMiddleware and AutoModeMiddleware can be driven by the
// OpenAI Decisions API.

package decisions

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	coredecisions "github.com/projanvil/langchain-golang/core/decisions"
	"github.com/projanvil/langchain-golang/core/lcerrors"
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
func (a *clientAdapter) Invoke(ctx context.Context, request coredecisions.ClassifierRequest, opts ...runnables.Option) (coredecisions.ClassifierResponse, error) {
	if err := request.Validate(); err != nil {
		return coredecisions.ClassifierResponse{}, err
	}
	questions := make(map[string]Question, len(request.Questions))
	for id, question := range request.Questions {
		converted, err := questionToOpenAI(question)
		if err != nil {
			return coredecisions.ClassifierResponse{}, fmt.Errorf("question %q: %w", id, err)
		}
		questions[id] = converted
	}
	cfg := runnables.NewConfig(opts...)
	providerutil.DecisionStart(ctx, cfg, "openai", a.client.Model, request)
	response, err := a.client.Invoke(ctx, Request{Input: request.State, Questions: questions})
	if err != nil {
		providerutil.DecisionEnd(ctx, cfg, "openai", a.client.Model, nil, nil, err)
		return coredecisions.ClassifierResponse{}, err
	}
	// A refused question fails the whole request; the other answers are
	// discarded (classifier.py:307-314).
	if refusals := response.Refusals(); len(refusals) > 0 {
		names := append([]string(nil), slices.Sorted(maps.Keys(refusals))...)
		err := &lcerrors.ModelRefusalError{Message: "The OpenAI Decisions API refused to answer: " + strings.Join(names, ", ") + "."}
		providerutil.DecisionEnd(ctx, cfg, "openai", a.client.Model, nil, nil, err)
		return coredecisions.ClassifierResponse{}, err
	}
	core, err := responseToCore(response, request.Questions)
	if err != nil {
		providerutil.DecisionEnd(ctx, cfg, "openai", a.client.Model, nil, nil, err)
		return coredecisions.ClassifierResponse{}, err
	}
	// One answer per question (classifier.py:307-310): a missing or
	// duplicated answer is an error, not a silent gap.
	if len(core.Answers) != len(request.Questions) {
		err := fmt.Errorf("openai decisions: expected one answer per question (got %d answers for %d questions)", len(core.Answers), len(request.Questions))
		providerutil.DecisionEnd(ctx, cfg, "openai", a.client.Model, nil, nil, err)
		return coredecisions.ClassifierResponse{}, err
	}
	providerutil.DecisionEnd(ctx, cfg, "openai", a.client.Model, core, core.Usage, nil)
	return core, nil
}

// toText renders a Content value the way upstream's _to_text does: a
// string stays as-is, anything else serializes to compact JSON text
// (_classifier_input.py:257-267).
func toText(content any) (string, error) {
	if text, ok := content.(string); ok {
		return text, nil
	}
	encoded, err := json.Marshal(content)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

// noulCriteriaText renders Noul outcome descriptions as instruction lines
// (_classifier_input.py:161-170): "Answer true if: <text>" per non-nil
// outcome, joined by newlines.
func noulCriteriaText(criteria *coredecisions.NoulCriteria) (string, error) {
	var lines []string
	for _, outcome := range []struct {
		name        string
		description any
	}{
		{"true", criteria.True},
		{"false", criteria.False},
	} {
		if outcome.description == nil {
			continue
		}
		text, err := toText(outcome.description)
		if err != nil {
			return "", err
		}
		lines = append(lines, fmt.Sprintf("Answer %s if: %s", outcome.name, text))
	}
	return strings.Join(lines, "\n"), nil
}

// questionToOpenAI maps one core question onto the Decisions API
// vocabulary. Instructions must be strings on this API.
func questionToOpenAI(question coredecisions.Question) (Question, error) {
	switch q := question.(type) {
	case coredecisions.Noul:
		instructions, err := toText(q.Instructions)
		if err != nil {
			return nil, fmt.Errorf("openai decisions: instructions: %w", err)
		}
		// The API's predicate question has no outcome descriptions, so
		// Noul criteria are appended to the instructions as text
		// (_classifier_input.py:173-178).
		if q.Criteria != nil {
			if extra, err := noulCriteriaText(q.Criteria); err == nil && extra != "" {
				instructions += "\n\n" + extra
			} else if err != nil {
				return nil, err
			}
		}
		return Predicate{Instructions: instructions}, nil
	case coredecisions.Choice:
		if len(q.Criteria) < 2 {
			return nil, fmt.Errorf("openai decisions: the Decisions API requires at least two Choice labels")
		}
		instructions, err := toText(q.Instructions)
		if err != nil {
			return nil, fmt.Errorf("openai decisions: instructions: %w", err)
		}
		options := make([]ChoiceOption, 0, len(q.Criteria))
		for _, label := range slices.Sorted(maps.Keys(q.Criteria)) {
			option := ChoiceOption{Value: label}
			if description := q.Criteria[label]; description != nil {
				text, err := toText(description)
				if err != nil {
					return nil, fmt.Errorf("openai decisions: choice description for %q: %w", label, err)
				}
				option.Description = &text
			}
			options = append(options, option)
		}
		return Choice{Instructions: instructions, Choices: options}, nil
	case coredecisions.Score:
		instructions, err := toText(q.Instructions)
		if err != nil {
			return nil, fmt.Errorf("openai decisions: instructions: %w", err)
		}
		levels := make([]ScoreLevel, 0, len(q.Criteria))
		for _, criterion := range q.Criteria {
			label, err := toText(criterion)
			if err != nil {
				return nil, fmt.Errorf("openai decisions: score level: %w", err)
			}
			levels = append(levels, L(label))
		}
		return Score{Instructions: instructions, Levels: levels}, nil
	default:
		return nil, fmt.Errorf("openai decisions: unsupported question type %T (no Box on the Decisions API)", question)
	}
}

// responseToCore maps a Decisions response onto the core envelope. The
// score legends are rebuilt from the questions' own criteria
// (classifier.py:349) rather than trusting the response echo.
func responseToCore(response Response, questions map[string]coredecisions.Question) (coredecisions.ClassifierResponse, error) {
	answers := make(map[string]coredecisions.Answer, len(response.Answers))
	failures := map[string]error{}
	for id, answer := range response.Answers {
		switch v := answer.(type) {
		case PredicateAnswer:
			answers[id] = coredecisions.NoulAnswer{Noul: v.Probability}
		case ChoiceAnswer:
			// Only string labels have a core vocabulary slot; a bool answer
			// (never produced for adapter-built questions) must fail loud
			// rather than vanish (upstream surfaces it as a validation
			// error).
			label, ok := v.Choice.(string)
			if !ok {
				failures[id] = fmt.Errorf("choice answer is not a string label: %T", v.Choice)
				continue
			}
			answers[id] = coredecisions.ChoiceAnswer{
				Choice:        label,
				Probabilities: v.Probabilities,
				Confidence:    v.Confidence,
			}
		case ScoreAnswer:
			legend := make(map[int]coredecisions.Content, len(v.Legend))
			if question, asked := questions[id]; asked {
				if score, isScore := question.(coredecisions.Score); isScore {
					for level, criterion := range score.Criteria {
						legend[level] = criterion
					}
				}
			}
			if len(legend) == 0 {
				for level, label := range v.Legend {
					legend[level] = label
				}
			}
			answers[id] = coredecisions.ScoreAnswer{
				Score:         v.Score,
				Legend:        legend,
				Probabilities: v.Probabilities,
				Confidence:    v.Confidence,
			}
		case RefusalAnswer:
			// Handled in Invoke: any refusal fails the whole request.
		}
	}
	core := coredecisions.ClassifierResponse{
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
		// A reported total wins; core completes it otherwise
		// (classifier.py:361-367 then _complete_usage).
		total := input + output
		if response.Usage.TotalTokens != nil && *response.Usage.TotalTokens != 0 {
			total = *response.Usage.TotalTokens
		}
		core.Usage = &messages.UsageMetadata{
			InputTokens:  input,
			OutputTokens: output,
			TotalTokens:  total,
		}
		if d := response.Usage.InputTokensDetails; d != nil {
			core.Usage.InputTokenDetails = &messages.InputTokenDetails{
				CacheRead:     d.CachedTokens,
				CacheCreation: d.CacheWriteTokens,
			}
		}
		if d := response.Usage.OutputTokensDetails; d != nil && d.ReasoningTokens > 0 {
			core.Usage.OutputTokenDetails = &messages.OutputTokenDetails{
				Reasoning: d.ReasoningTokens,
			}
		}
	}
	if len(failures) > 0 {
		names := append([]string(nil), slices.Sorted(maps.Keys(failures))...)
		return core, fmt.Errorf("openai decisions: invalid answers for %s", strings.Join(names, ", "))
	}
	return core, nil
}

// compile-time proof the adapter satisfies the core interface.
var _ coredecisions.DecisionModel = (*clientAdapter)(nil)
