package middleware

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/decisions"
	"github.com/projanvil/langchain-golang/core/messages"
	// Register the openai provider so "openai:..." model strings resolve
	// through the chatmodels InitChatModel+Resolve chain.
	_ "github.com/projanvil/langchain-golang/partners/openai"
)

// fakeDecisionModel stands in for any decisions.DecisionModel
// implementation; it returns canned responses and records requests.
type fakeDecisionModel struct {
	response decisions.Response
	err      error
	requests []decisions.Request
}

func (f *fakeDecisionModel) Invoke(_ context.Context, request decisions.Request) (decisions.Response, error) {
	f.requests = append(f.requests, request)
	return f.response, f.err
}


func routerChoices() (fast, powerful any) {
	return fakeModel{label: "fast"}, fakeModel{label: "powerful"}
}

func TestGenericModelRouterRoutesAndStoresAnswer(t *testing.T) {
	fast, powerful := routerChoices()
	classifier := &fakeDecisionModel{response: decisions.Response{
		Model: "fake",
		Answers: map[string]decisions.Answer{
			routerQuestionID: decisions.ChoiceAnswer{
				Choice:        "powerful",
				Probabilities: map[string]float64{"fast": 0.3, "powerful": 0.7},
				Confidence:    0.7,
			},
		},
	}}
	router, err := NewModelRouterMiddleware(classifier, map[string]ModelChoice{
		"fast":     {Model: fast, Criteria: "Simple, well-scoped tasks."},
		"powerful": {Model: powerful, Criteria: "Complex tasks."},
	}, "Choose the least costly model suited to the task.")
	if err != nil {
		t.Fatalf("construct: %v", err)
	}

	state := map[string]any{"messages": []messages.Message{messages.Human("refactor this module")}}
	update, err := router.BeforeAgent(t.Context(), state)
	if err != nil {
		t.Fatalf("before agent: %v", err)
	}
	route, ok := update[routeStateKey].(map[string]any)
	if !ok || route["choice"] != "powerful" || route["confidence"] != 0.7 {
		t.Fatalf("routing answer state: %#v", update)
	}

	// The classifier saw a Choice question carrying the route criteria.
	question, ok := classifier.requests[0].Questions[routerQuestionID].(decisions.Choice)
	if !ok {
		t.Fatalf("question type: %#v", classifier.requests[0].Questions[routerQuestionID])
	}
	if question.Criteria["powerful"] != "Complex tasks." {
		t.Fatalf("criteria: %#v", question.Criteria)
	}
	if _, ok := classifier.requests[0].State.(messages.Message); !ok {
		t.Fatalf("state must be the latest human message: %#v", classifier.requests[0].State)
	}

	// WrapModelCall overrides the model to the selected route.
	var seen any
	response, err := router.WrapModelCall(t.Context(), modelRequestForRoute(t, update), func(_ context.Context, request ModelRequest) (ModelResponse, error) {
		seen = request.Model
		return ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if seen != powerful {
		t.Fatalf("routed model: %#v", seen)
	}
	if len(response.Result) != 1 || response.Result[0].Content != "ok" {
		t.Fatalf("response: %#v", response)
	}
}

func TestGenericModelRouterNoHumanMessage(t *testing.T) {
	router, err := NewModelRouterMiddleware(&fakeDecisionModel{}, map[string]ModelChoice{
		"fast": {Model: fakeModel{label: "fast"}},
	}, "route")
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if _, err := router.BeforeAgent(t.Context(), map[string]any{"messages": []messages.Message{messages.AI("no human")}}); err == nil ||
		!errors.Is(err, err) || err.Error() == "" {
		t.Fatalf("expected no-human-message error, got %v", err)
	}
}

func TestGenericModelRouterMissingAnswer(t *testing.T) {
	router, err := NewModelRouterMiddleware(&fakeDecisionModel{response: decisions.Response{
		Answers: map[string]decisions.Answer{},
	}}, map[string]ModelChoice{
		"fast": {Model: fakeModel{label: "fast"}},
	}, "route")
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	state := map[string]any{"messages": []messages.Message{messages.Human("go")}}
	if _, err := router.BeforeAgent(t.Context(), state); err == nil {
		t.Fatal("expected missing-answer error")
	}
}

func TestGenericModelRouterWrapErrors(t *testing.T) {
	router, err := NewModelRouterMiddleware(&fakeDecisionModel{}, map[string]ModelChoice{
		"fast": {Model: fakeModel{label: "fast"}},
	}, "route")
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if _, err := router.WrapModelCall(t.Context(), ModelRequest{State: map[string]any{}}, func(_ context.Context, _ ModelRequest) (ModelResponse, error) {
		t.Fatal("handler must not run without a routing answer")
		return ModelResponse{}, nil
	}); err == nil {
		t.Fatal("expected missing-route error")
	}
}

func TestGenericModelRouterStringModelResolution(t *testing.T) {
	router, err := NewModelRouterMiddleware(&fakeDecisionModel{}, map[string]ModelChoice{
		"fast": {Model: "openai:gpt-4o-mini", Criteria: "simple"},
	}, "route")
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if router.models["fast"] == nil {
		t.Fatal("string model must resolve at construction")
	}
}

func TestGenericModelRouterSchemasAndPolicy(t *testing.T) {
	router, err := NewModelRouterMiddleware(&fakeDecisionModel{}, map[string]ModelChoice{
		"fast": {Model: fakeModel{label: "fast"}},
	}, "route")
	if err != nil {
		t.Fatalf("construct: %v", err)
	}
	if fields := router.StateSchema(); len(fields) != 1 || fields[0].Name != routeStateKey {
		t.Fatalf("state schema: %#v", fields)
	}
	policy := router.TracePolicy()
	if policy.ProcessInputs == nil {
		t.Fatal("process-inputs policy must be set")
	}
	if got := policy.ProcessInputs("secret"); !reflect.DeepEqual(got, map[string]any{"omitted": true}) {
		t.Fatalf("process inputs: %#v", got)
	}
}

func TestNewGenericModelRouterValidation(t *testing.T) {
	if _, err := NewModelRouterMiddleware(nil, map[string]ModelChoice{"fast": {}}, "route"); err == nil {
		t.Fatal("nil classifier must fail")
	}
	if _, err := NewModelRouterMiddleware(&fakeDecisionModel{}, nil, "route"); err == nil {
		t.Fatal("empty choices must fail")
	}
	if _, err := NewModelRouterMiddleware(&fakeDecisionModel{}, map[string]ModelChoice{"fast": {}}, nil); err == nil {
		t.Fatal("nil instructions must fail")
	}
}

// fakeModel stands in for an injected model instance; ModelChoice.Model is
// any, so no ChatModel implementation is required.
type fakeModel struct{ label string }

// modelRequestForRoute builds a WrapModelCall request carrying the routing
// answer BeforeAgent returned.
func modelRequestForRoute(t *testing.T, update map[string]any) ModelRequest {
	t.Helper()
	route, ok := update[routeStateKey].(map[string]any)
	if !ok {
		t.Fatalf("update missing routing answer: %#v", update)
	}
	return ModelRequest{State: map[string]any{routeStateKey: route}}
}
