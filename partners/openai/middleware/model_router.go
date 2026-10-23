package middleware

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
	"github.com/projanvil/langchain-golang/partners/openai/decisions"
)

// routerQuestionName is the choice question whose answer is stored in agent
// state under "model_route" (Python _QUESTION_NAME, model_router.py:41).
const routerQuestionName = "model_route"

// ModelChoice is a model available to the router and the criterion for
// selecting it (Python ModelChoice, model_router.py:44).
type ModelChoice struct {
	// Model is either a model string accepted by
	// chatmodels.InitChatModel+Resolve (e.g. "openai:gpt-4o-mini") or a
	// chat model instance used as-is.
	Model any
	// Criteria describes the tasks suited to the model.
	Criteria string
}

// ModelRouter selects an agent's model with an OpenAI Decisions Choice
// question, mirroring Python's OpenAIModelRouterMiddleware (PR #41125).
// BeforeAgent classifies the latest human message once per run and stores
// the routing answer (a plain JSON-native map, so checkpointers can persist
// it without custom types) under the "model_route" state key; WrapModelCall
// then overrides every model call in the run with the selected model. When
// no route is selected — the model refused or the state has no human
// message — model_route is set to nil and the agent's own model is used.
// API errors propagate and terminate the run.
type ModelRouter struct {
	// Choices maps route names to model choices. Empty is a construction
	// error.
	Choices map[string]ModelChoice
	// Instructions is the routing question. Blank is a construction error
	// (there is no default, unlike AutoMode).
	Instructions string
	// Model is the Decisions model: a model name string (resolved via
	// decisions.NewClient) or an injected *decisions.Client.
	Model any

	once       sync.Once
	resolved   *modelRouterResolved
	resolveErr error
}

type modelRouterResolved struct {
	question decisions.Choice
	models   map[string]any
	client   *decisions.Client
}

// resolve lazily normalizes the router's configuration (Python performs the
// equivalent validation and init_chat_model resolution in __init__).
func (r *ModelRouter) resolve() (*modelRouterResolved, error) {
	r.once.Do(func() {
		if len(r.Choices) == 0 {
			r.resolveErr = fmt.Errorf("`choices` must contain at least one model choice") //nolint:staticcheck // ST1005: Python-aligned message
			return
		}
		if strings.TrimSpace(r.Instructions) == "" {
			r.resolveErr = fmt.Errorf("`instructions` must not be empty") //nolint:staticcheck // ST1005: Python-aligned message
			return
		}
		// Stable wire order: Go map iteration is randomized while Python
		// preserves insertion order (DIVERGENCES #8).
		routes := slices.Sorted(maps.Keys(r.Choices))
		options := make([]decisions.ChoiceOption, 0, len(routes))
		models := make(map[string]any, len(routes))
		for _, route := range routes {
			choice := r.Choices[route]
			// Python maps route -> criteria; an empty criteria string is
			// still an explicit (empty) description on the wire.
			options = append(options, decisions.ChoiceOption{Value: route, Description: new(choice.Criteria)})
			model, err := routerModel(choice.Model)
			if err != nil {
				r.resolveErr = err
				return
			}
			models[route] = model
		}
		client, err := decisionsClientFor(r.Model)
		if err != nil {
			r.resolveErr = err
			return
		}
		r.resolved = &modelRouterResolved{
			question: decisions.Choice{Instructions: r.Instructions, Choices: options},
			models:   models,
			client:   client,
		}
	})
	return r.resolved, r.resolveErr
}

// routerModel resolves a ModelChoice.Model value: strings go through the
// chatmodels.InitChatModel + Resolve chain (the model_fallback.go:37-43
// precedent); anything else is used as the model directly.
func routerModel(model any) (any, error) {
	spec, ok := model.(string)
	if !ok {
		return model, nil
	}
	parsed, err := chatmodels.InitChatModel(spec)
	if err != nil {
		return nil, fmt.Errorf("openai model router: could not resolve model %q: %w", spec, err)
	}
	resolved, err := chatmodels.Resolve(parsed)
	if err != nil {
		return nil, fmt.Errorf("openai model router: could not resolve model %q: %w", spec, err)
	}
	return resolved, nil
}

// BeforeAgent implements the before-agent hook (Python before_agent,
// model_router.py:177): classify the latest task and store the routing
// answer.
func (r *ModelRouter) BeforeAgent(ctx context.Context, state map[string]any) (map[string]any, error) {
	resolved, err := r.resolve()
	if err != nil {
		return nil, err
	}
	requests := routerDecisionRequests(state, resolved)
	if len(requests) == 0 {
		return map[string]any{"model_route": nil}, nil
	}
	response, err := decideWithTextFallback(ctx, resolved.client, requests)
	if err != nil {
		return nil, err
	}
	return map[string]any{"model_route": routerRouteValue(response)}, nil
}

// WrapModelCall implements the wrap-model-call hook (Python
// wrap_model_call, model_router.py:199): route the call to the selected
// model.
func (r *ModelRouter) WrapModelCall(
	ctx context.Context,
	request middleware.ModelRequest,
	handler middleware.ModelHandler,
) (middleware.ModelResponse, error) {
	routed, err := r.routedRequest(request)
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	return handler(ctx, routed)
}

// routedRequest returns request with the model overridden by the stored
// route, or unchanged when no route is set (Python _routed,
// model_router.py:237).
func (r *ModelRouter) routedRequest(request middleware.ModelRequest) (middleware.ModelRequest, error) {
	routeValue, present := request.State["model_route"]
	if !present || routeValue == nil {
		return request, nil
	}
	route, ok := routeValue.(map[string]any)
	if !ok {
		return request, fmt.Errorf("openai model router: model_route state must be a map, got %T", routeValue)
	}
	choiceName, _ := route["choice"].(string)
	resolved, err := r.resolve()
	if err != nil {
		return request, err
	}
	model, ok := resolved.models[choiceName]
	if !ok {
		return request, fmt.Errorf("openai model router: unknown model route %q", choiceName)
	}
	next, err := request.Override(middleware.WithModel(model))
	if err != nil {
		return request, err
	}
	return next, nil
}

// StateSchema declares the "model_route" state key (Python's state_schema
// contribution, model_router.py:135).
func (r *ModelRouter) StateSchema() []middleware.StateField {
	return []middleware.StateField{{Name: routerQuestionName}}
}

// TracePolicy excludes user messages from middleware traces (Python
// TracePolicy(process_inputs=omit_payload), model_router.py:136). The
// wrap-model-call layer is where this port installs payload policies, so
// the hook is live for this middleware (unlike AutoMode — DIVERGENCES #9).
func (r *ModelRouter) TracePolicy() middleware.TracePolicyConfig {
	return middleware.TracePolicyConfig{ProcessInputs: omitPayload}
}

// omitPayload drops a payload entirely (the Go analogue of Python's
// omit_payload helper, matching langgraph/graph.OmitPayload without adding
// a langgraph dependency to this package).
func omitPayload(any) any { return nil }

// routerDecisionRequests returns the requests to try in order: with base64
// images (if any), then text only (Python _decision_requests,
// model_router.py:218). Only the latest human message is classified.
func routerDecisionRequests(state map[string]any, resolved *modelRouterResolved) []decisions.Request {
	msgs, _ := state["messages"].([]messages.Message)
	var latest *messages.Message
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleHuman {
			latest = &msgs[i]
			break
		}
	}
	if latest == nil {
		return nil
	}
	textOnly := routableMessage(*latest, false)
	withImages := routableMessage(*latest, true)
	var inputs []messages.Message
	if reflect.DeepEqual(withImages.ContentBlocks, textOnly.ContentBlocks) {
		inputs = []messages.Message{textOnly}
	} else {
		inputs = []messages.Message{withImages, textOnly}
	}
	requests := make([]decisions.Request, len(inputs))
	for i, input := range inputs {
		requests[i] = decisions.Request{
			Input:     input,
			Questions: map[string]decisions.Question{routerQuestionName: resolved.question},
		}
	}
	return requests
}

// routableMessage keeps text (and base64 images if requested) and replaces
// every other block with a placeholder (Python _routable,
// model_router.py:244). Unlike AutoMode's with-images variant, ONLY base64
// images survive here — hosted-URL images are placeholdered too (see
// DIVERGENCES.md #13).
func routableMessage(msg messages.Message, keepImages bool) messages.Message {
	if len(msg.ContentBlocks) == 0 {
		return msg
	}
	blocks := make([]messages.ContentBlock, 0, len(msg.ContentBlocks))
	changed := false
	for _, block := range msg.ContentBlocks {
		keep := false
		if _, isText := block.(messages.TextBlock); isText {
			keep = true
		} else if keepImages && decisions.IsBase64ImageBlock(block) {
			keep = true
		}
		if keep {
			blocks = append(blocks, block)
			continue
		}
		blocks = append(blocks, messages.TextBlock{Text: fmt.Sprintf("[%s omitted]", routerBlockLabel(block))})
		changed = true
	}
	if !changed {
		return msg
	}
	msg.ContentBlocks = blocks
	return msg
}

// routerBlockLabel names the placeholder after the block type, defaulting
// to "content" for blocks without one (Python block.get('type', 'content')).
func routerBlockLabel(block messages.ContentBlock) string {
	label := block.BlockType()
	if label == "" {
		return "content"
	}
	return label
}

// routerRouteValue converts the choice answer into the plain map stored in
// state (Python _route, model_router.py:270): nil when the model refused or
// produced no answer.
func routerRouteValue(response decisions.Response) any {
	answer, ok := response.Choices()[routerQuestionName]
	if !ok {
		return nil
	}
	probabilities := make(map[string]float64, len(answer.Probabilities))
	maps.Copy(probabilities, answer.Probabilities)
	return map[string]any{
		"choice":        routerChoiceString(answer.Choice),
		"probabilities": probabilities,
		"confidence":    answer.Confidence,
	}
}

// routerChoiceString stringifies the selected value (Python str(choice);
// bools render lowercase in Go — a deliberate divergence, only observable
// for bool-valued routes).
func routerChoiceString(choice any) string {
	switch v := choice.(type) {
	case string:
		return v
	case bool:
		return strconv.FormatBool(v)
	default:
		return fmt.Sprint(v)
	}
}
