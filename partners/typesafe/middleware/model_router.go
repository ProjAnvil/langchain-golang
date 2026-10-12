package middleware

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/projanvil/langchain-golang/core/messages"
	agentmiddleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
	"github.com/projanvil/langchain-golang/partners/typesafe"
)

const (
	routeQuestionID = "model_route"
	routeStateKey   = "model_route"
)

// ModelChoice pairs a routable model with the criterion for selecting it
// (model_router.py:41-48). Model is either a model instance or a
// "provider:model" string resolved through chatmodels.InitChatModel +
// chatmodels.Resolve at construction (the same chain model_fallback uses).
// Criteria is any JSON content describing the tasks suited to the model.
type ModelChoice struct {
	Model    any
	Criteria any
}

// ModelRouterMiddleware classifies the latest human message once per agent
// run with a TypeSafe Choice question and routes every model call in the
// run to the selected model (experimental; #40659).
type ModelRouterMiddleware struct {
	// Choices maps route labels to their model and criterion; at least one
	// entry is required.
	Choices map[string]ModelChoice
	// Instructions is the routing question (required, any JSON content).
	Instructions any
	// Classifier classifies the routing question; nil builds one from the
	// environment at construction.
	Classifier *typesafe.Classifier

	// models holds the resolved model per route label (string models are
	// resolved once, mirroring Python's init_chat_model in __init__).
	models map[string]any
}

// ModelRouterOption configures NewModelRouterMiddleware.
type ModelRouterOption func(*ModelRouterMiddleware)

// WithRouterClassifier injects a classifier, overriding env-based
// construction.
func WithRouterClassifier(classifier *typesafe.Classifier) ModelRouterOption {
	return func(m *ModelRouterMiddleware) { m.Classifier = classifier }
}

// NewModelRouterMiddleware validates the configuration, resolves string
// model specs, and builds the middleware.
func NewModelRouterMiddleware(choices map[string]ModelChoice, instructions any, opts ...ModelRouterOption) (*ModelRouterMiddleware, error) {
	m := &ModelRouterMiddleware{Choices: choices, Instructions: instructions}
	for _, opt := range opts {
		opt(m)
	}
	if len(m.Choices) == 0 {
		return nil, fmt.Errorf("typesafe: ModelRouterMiddleware requires at least one model choice")
	}
	if m.Instructions == nil {
		return nil, fmt.Errorf("typesafe: ModelRouterMiddleware requires instructions")
	}
	models := make(map[string]any, len(m.Choices))
	for route, choice := range m.Choices {
		model := choice.Model
		if spec, ok := model.(string); ok {
			// Resolve "provider:model" strings through the
			// InitChatModel+Resolve chain (model_router.py:146-151; Go
			// precedent model_fallback.go:37-43).
			parsed, err := chatmodels.InitChatModel(spec)
			if err != nil {
				return nil, err
			}
			resolved, err := chatmodels.Resolve(parsed)
			if err != nil {
				return nil, err
			}
			model = resolved
		}
		models[route] = model
	}
	m.models = models
	if m.Classifier == nil {
		classifier, err := typesafe.NewClassifier()
		if err != nil {
			return nil, err
		}
		m.Classifier = classifier
	}
	return m, nil
}

// BeforeAgent implements the agents.BeforeAgentHook surface: it classifies
// the latest human message and stores the routing answer in state.
//
// Unlike the OpenAI ModelRouter (which writes a nil route and keeps the
// agent's own model), a missing human message or a missing answer is an
// error here — Python's next() raises StopIteration and response.choices
// raises KeyError (model_router.py:155-174), which Go surfaces explicitly
// (DIVERGENCES #3).
func (m *ModelRouterMiddleware) BeforeAgent(ctx context.Context, state map[string]any) (map[string]any, error) {
	human, ok := latestHumanMessage(state)
	if !ok {
		return nil, fmt.Errorf("typesafe: ModelRouterMiddleware requires a human message to route")
	}
	response, err := m.Classifier.Invoke(ctx, typesafe.Request{
		State: human,
		Questions: map[string]typesafe.Question{
			routeQuestionID: typesafe.Choice{
				Instructions: m.Instructions,
				Criteria:     routingCriteria(m.Choices),
			},
		},
	})
	if err != nil {
		return nil, err
	}
	answer, ok := response.Choices()[routeQuestionID]
	if !ok {
		return nil, fmt.Errorf("typesafe: ModelRouterMiddleware response is missing the %q answer", routeQuestionID)
	}
	// The answer is stored as a JSON-native map (Python stores the pydantic
	// ChoiceAnswer; DIVERGENCES #4) so durable serde stays safe.
	return map[string]any{
		routeStateKey: map[string]any{
			"type":          "choice",
			"choice":        answer.Choice,
			"probabilities": answer.Probabilities,
			"confidence":    answer.Confidence,
		},
	}, nil
}

// WrapModelCall implements the agents.WrapModelCallHook surface: it routes
// the call to the model selected by BeforeAgent.
func (m *ModelRouterMiddleware) WrapModelCall(ctx context.Context, request agentmiddleware.ModelRequest, handler agentmiddleware.ModelHandler) (agentmiddleware.ModelResponse, error) {
	route, ok := request.State[routeStateKey].(map[string]any)
	if !ok {
		return agentmiddleware.ModelResponse{}, fmt.Errorf("typesafe: ModelRouterMiddleware state is missing the %q routing answer", routeStateKey)
	}
	choice, ok := route["choice"].(string)
	if !ok {
		return agentmiddleware.ModelResponse{}, fmt.Errorf("typesafe: ModelRouterMiddleware routing answer has no choice label")
	}
	model, ok := m.models[choice]
	if !ok {
		return agentmiddleware.ModelResponse{}, fmt.Errorf("typesafe: ModelRouterMiddleware has no model for route %q", choice)
	}
	next, err := request.Override(agentmiddleware.WithModel(model))
	if err != nil {
		return agentmiddleware.ModelResponse{}, err
	}
	return handler(ctx, next)
}

// StateSchema implements agentmiddleware.StateSchemaContributor, declaring
// the "model_route" state key (model_router.py:73-76).
func (m *ModelRouterMiddleware) StateSchema() []agentmiddleware.StateField {
	return []agentmiddleware.StateField{{Name: routeStateKey}}
}

// TracePolicy implements agentmiddleware.TracePolicyProvider with
// omit-payload inputs (model_router.py:134): traced inputs inside this
// wrap_model_call layer are replaced by an omitted marker so classified
// message content never reaches traces.
func (m *ModelRouterMiddleware) TracePolicy() agentmiddleware.TracePolicyConfig {
	return agentmiddleware.TracePolicyConfig{
		ProcessInputs: func(any) any { return map[string]any{"omitted": true} },
	}
}

// latestHumanMessage scans state messages newest-first for a human message.
func latestHumanMessage(state map[string]any) (messages.Message, bool) {
	msgs, ok := state["messages"].([]messages.Message)
	if !ok {
		return messages.Message{}, false
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleHuman {
			return msgs[i], true
		}
	}
	return messages.Message{}, false
}

// routingCriteria builds the ordered choice criteria from the configured
// routes (labels sorted for deterministic wire output; Python preserves
// the caller's dict order — DIVERGENCES #8). Routes with nil criteria are
// omitted from the wire object via the Choice marshal rules.
func routingCriteria(choices map[string]ModelChoice) []typesafe.ChoiceCriterion {
	criteria := make([]typesafe.ChoiceCriterion, 0, len(choices))
	for _, route := range slices.Sorted(maps.Keys(choices)) {
		criteria = append(criteria, typesafe.ChoiceCriterion{
			Label:       route,
			Description: choices[route].Criteria,
		})
	}
	return criteria
}
