// Model-routing middleware driven by a decision model: the provider-neutral
// port of Python langchain 1.5's ModelRouterMiddleware (upstream #41766,
// beta). Any core decisions.DecisionModel implementation (TypeSafe's Jev
// classifier, OpenAI's decisions API, or a hand-rolled one) can drive it;
// the TypeSafe- and OpenAI-specific variants live in their partner packages.

package middleware

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/projanvil/langchain-golang/core/decisions"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

const (
	routerQuestionID = "model_route"
	routeStateKey    = "model_route"
)

// ModelChoice pairs a routable model with the criterion for selecting it
// (model_router.py ModelChoice). Model is either a model instance or a
// "provider:model" string resolved through chatmodels.InitChatModel at
// construction. Criteria is any JSON content describing the tasks suited
// to the model.
type ModelChoice struct {
	Model    any
	Criteria any
}

// ModelRouterMiddleware selects an agent's model with a decision model's
// Choice answer (Python langchain 1.5, beta; #41766). The middleware
// classifies the latest human message once before an agent run, stores the
// complete routing answer in agent state, and uses its selected label for
// every model call in the run. Keeping the complete answer makes
// probabilities and confidence available in state and traces. Classifier
// failures propagate and terminate the run rather than silently selecting
// a different model.
type ModelRouterMiddleware struct {
	// Choices maps route labels to their model and criterion; at least one
	// entry is required.
	Choices map[string]ModelChoice
	// Instructions is the routing question (required, any JSON content).
	Instructions any
	// Classifier classifies the routing question (required).
	Classifier decisions.DecisionModel

	// models holds the resolved model per route label (string models are
	// resolved once, mirroring Python's init_chat_model in __init__).
	models map[string]any
}

// NewModelRouterMiddleware validates the configuration, resolves string
// model specs, and builds the middleware.
func NewModelRouterMiddleware(
	classifier decisions.DecisionModel,
	choices map[string]ModelChoice,
	instructions any,
) (*ModelRouterMiddleware, error) {
	warnBetaMiddleware("ModelRouterMiddleware")
	if classifier == nil {
		return nil, fmt.Errorf("middleware: ModelRouterMiddleware requires a classifier")
	}
	if len(choices) == 0 {
		return nil, fmt.Errorf("middleware: ModelRouterMiddleware requires at least one model choice")
	}
	if instructions == nil {
		return nil, fmt.Errorf("middleware: ModelRouterMiddleware requires instructions")
	}
	models := make(map[string]any, len(choices))
	for route, choice := range choices {
		model := choice.Model
		if spec, ok := model.(string); ok {
			// Resolve "provider:model" strings through the
			// InitChatModel chain (model_router.py:133-135).
			parsed, err := chatmodels.InitChatModel(spec)
			if err != nil {
				return nil, err
			}
			model = parsed
		}
		models[route] = model
	}
	return &ModelRouterMiddleware{
		Choices:      choices,
		Instructions: instructions,
		Classifier:   classifier,
		models:       models,
	}, nil
}

// BeforeAgent classifies the latest task and stores the complete routing
// answer. A missing human message or a missing answer is an error —
// Python's ValueError on the missing human message and KeyError on the
// missing answer (model_router.py:148-162) surface explicitly here.
func (m *ModelRouterMiddleware) BeforeAgent(ctx context.Context, state map[string]any) (map[string]any, error) {
	human, ok := latestHumanMessage(state)
	if !ok {
		return nil, fmt.Errorf("middleware: ModelRouterMiddleware requires a human message to route")
	}
	response, err := m.Classifier.Invoke(ctx, decisions.Request{
		State: human,
		Questions: map[string]decisions.Question{
			routerQuestionID: decisions.Choice{
				Instructions: m.Instructions,
				Criteria:     routerCriteria(m.Choices),
			},
		},
	})
	if err != nil {
		return nil, err
	}
	answer, ok := response.Choices()[routerQuestionID]
	if !ok {
		return nil, fmt.Errorf("middleware: ModelRouterMiddleware response is missing the %q answer", routerQuestionID)
	}
	// The answer is stored as a JSON-native map (Python stores the pydantic
	// ChoiceAnswer) so durable serde stays safe.
	return map[string]any{
		routeStateKey: map[string]any{
			"type":          "choice",
			"choice":        answer.Choice,
			"probabilities": answer.Probabilities,
			"confidence":    answer.Confidence,
		},
	}, nil
}

// WrapModelCall routes the call to the model selected by BeforeAgent.
func (m *ModelRouterMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	route, ok := request.State[routeStateKey].(map[string]any)
	if !ok {
		return ModelResponse{}, fmt.Errorf("middleware: ModelRouterMiddleware state is missing the %q routing answer", routeStateKey)
	}
	choice, ok := route["choice"].(string)
	if !ok {
		return ModelResponse{}, fmt.Errorf("middleware: ModelRouterMiddleware routing answer has no choice label")
	}
	model, ok := m.models[choice]
	if !ok {
		return ModelResponse{}, fmt.Errorf("middleware: ModelRouterMiddleware has no model for route %q", choice)
	}
	next, err := request.Override(WithModel(model))
	if err != nil {
		return ModelResponse{}, err
	}
	return handler(ctx, next)
}

// StateSchema declares the "model_route" state key (model_router.py:73-76).
func (m *ModelRouterMiddleware) StateSchema() []StateField {
	return []StateField{{Name: routeStateKey}}
}

// TracePolicy omits traced inputs inside this middleware's layers
// (model_router.py:119): classified message content never reaches traces.
func (m *ModelRouterMiddleware) TracePolicy() TracePolicyConfig {
	return TracePolicyConfig{
		ProcessInputs: func(any) any { return map[string]any{"omitted": true} },
	}
}

// latestHumanMessage scans state messages newest-first for a human message.
func latestHumanMessage(state map[string]any) (messages.Message, bool) {
	msgs, ok := state["messages"].([]messages.Message)
	if !ok {
		return messages.Message{}, false
	}
	for _, msg := range slices.Backward(msgs) {
		if msg.Role == messages.RoleHuman {
			return msg, true
		}
	}
	return messages.Message{}, false
}

// routerCriteria builds the Choice criteria from the configured routes.
// Route labels with nil criteria keep a nil description (the wire object
// preserves the explicit null).
func routerCriteria(choices map[string]ModelChoice) map[string]decisions.Content {
	criteria := make(map[string]decisions.Content, len(choices))
	for route := range maps.Keys(choices) {
		criteria[route] = choices[route].Criteria
	}
	return criteria
}
