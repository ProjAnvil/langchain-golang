package middleware

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents"
	agentmiddleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"

	// Register the openai provider so "openai:..." model strings resolve
	// through the chatmodels InitChatModel+Resolve chain the same way
	// langchain/agents blank-imports it.
	_ "github.com/projanvil/langchain-golang/partners/openai"
)

// fakeModel stands in for an injected model instance; ModelChoice.Model is
// any, so no ChatModel implementation is required.
type fakeModel struct{ label string }

// Compile-time proof that CreateAgent's duck-typed hook discovery finds the
// middleware (agents/*Hook plus the middleware contributor interfaces).
var (
	_ agents.WrapToolCallHook                = (*AutoModeMiddleware)(nil)
	_ agents.BeforeAgentHook                 = (*ModelRouterMiddleware)(nil)
	_ agents.WrapModelCallHook               = (*ModelRouterMiddleware)(nil)
	_ agentmiddleware.StateSchemaContributor = (*ModelRouterMiddleware)(nil)
	_ agentmiddleware.TracePolicyProvider    = (*ModelRouterMiddleware)(nil)
)

const routeInstructions = "Choose the least costly model suited to the task."

func routeAnswers(choice string) string {
	return fmt.Sprintf(
		`{"model_route":{"type":"choice","choice":%q,"probabilities":{"fast":0.3,"powerful":0.7},"confidence":0.7}}`,
		choice)
}

func TestModelRouterRoutesAndStateShape(t *testing.T) {
	server, body := classifierServer(t, routeAnswers("powerful"))
	classifier := testClassifier(server)

	fast, powerful := fakeModel{label: "fast"}, fakeModel{label: "powerful"}
	router, err := NewModelRouterMiddleware(
		map[string]ModelChoice{
			"fast":     {Model: fast, Criteria: "Simple, well-scoped tasks."},
			"powerful": {Model: powerful, Criteria: "Complex tasks requiring deeper reasoning."},
		},
		routeInstructions,
		WithRouterClassifier(classifier),
	)
	if err != nil {
		t.Fatalf("NewModelRouterMiddleware: %v", err)
	}

	state := map[string]any{
		"messages": []messages.Message{
			messages.AI("hello"),
			messages.Human("refactor the whole codebase"),
			messages.AI("are you sure?"),
		},
	}
	update, err := router.BeforeAgent(t.Context(), state)
	if err != nil {
		t.Fatalf("BeforeAgent: %v", err)
	}

	route, ok := update["model_route"].(map[string]any)
	if !ok {
		t.Fatalf("model_route update shape: %#v", update["model_route"])
	}
	wantRoute := map[string]any{
		"type":          "choice",
		"choice":        "powerful",
		"probabilities": map[string]float64{"fast": 0.3, "powerful": 0.7},
		"confidence":    0.7,
	}
	if !reflect.DeepEqual(route, wantRoute) {
		t.Errorf("model_route mismatch:\n got %#v\nwant %#v", route, wantRoute)
	}

	// The classifier sees the latest human message serialized as one
	// role/content dict, plus the sorted choice criteria.
	raw := *body
	if !strings.Contains(raw, `"state":{"content":"refactor the whole codebase","role":"user"}`) {
		t.Errorf("state shape mismatch: %s", raw)
	}
	if !strings.Contains(raw, `"criteria":{"fast":"Simple, well-scoped tasks.","powerful":"Complex tasks requiring deeper reasoning."}`) {
		t.Errorf("criteria shape mismatch: %s", raw)
	}
	if !strings.Contains(raw, fmt.Sprintf(`"instructions":%q`, routeInstructions)) {
		t.Errorf("instructions missing: %s", raw)
	}

	// WrapModelCall swaps in the selected model instance.
	var received any
	handler := func(_ context.Context, request agentmiddleware.ModelRequest) (agentmiddleware.ModelResponse, error) {
		received = request.Model
		return agentmiddleware.ModelResponse{Result: []messages.Message{messages.AI("done")}}, nil
	}
	response, err := router.WrapModelCall(t.Context(), agentmiddleware.ModelRequest{
		Model: fast,
		State: map[string]any{"model_route": route},
	}, handler)
	if err != nil {
		t.Fatalf("WrapModelCall: %v", err)
	}
	if received != any(powerful) {
		t.Errorf("handler model = %#v, want the powerful instance", received)
	}
	if len(response.Result) != 1 || response.Result[0].Content != "done" {
		t.Errorf("handler response = %#v", response)
	}
}

func TestModelRouterNoHumanMessage(t *testing.T) {
	server, _ := classifierServer(t, routeAnswers("fast"))
	router, err := NewModelRouterMiddleware(
		map[string]ModelChoice{"fast": {Model: fakeModel{label: "fast"}, Criteria: "c"}},
		routeInstructions,
		WithRouterClassifier(testClassifier(server)),
	)
	if err != nil {
		t.Fatalf("NewModelRouterMiddleware: %v", err)
	}

	// DIVERGENCES #3: Python's next() raises StopIteration; Go surfaces an
	// explicit error (the openai ModelRouter writes a nil route instead).
	if _, err := router.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{messages.AI("no human here")},
	}); err == nil {
		t.Fatal("expected error when no human message exists")
	}
	if _, err := router.BeforeAgent(t.Context(), map[string]any{}); err == nil {
		t.Fatal("expected error when state has no messages")
	}
}

func TestModelRouterMissingAnswer(t *testing.T) {
	server, _ := classifierServer(t, `{"unrelated":{"type":"noul","noul":0.5}}`)
	router, err := NewModelRouterMiddleware(
		map[string]ModelChoice{"fast": {Model: fakeModel{label: "fast"}, Criteria: "c"}},
		routeInstructions,
		WithRouterClassifier(testClassifier(server)),
	)
	if err != nil {
		t.Fatalf("NewModelRouterMiddleware: %v", err)
	}
	if _, err := router.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{messages.Human("hi")},
	}); err == nil {
		t.Fatal("expected error for missing model_route answer")
	}
}

func TestModelRouterWrapErrors(t *testing.T) {
	server, _ := classifierServer(t, routeAnswers("fast"))
	router, err := NewModelRouterMiddleware(
		map[string]ModelChoice{"fast": {Model: fakeModel{label: "fast"}, Criteria: "c"}},
		routeInstructions,
		WithRouterClassifier(testClassifier(server)),
	)
	if err != nil {
		t.Fatalf("NewModelRouterMiddleware: %v", err)
	}
	handler := func(_ context.Context, _ agentmiddleware.ModelRequest) (agentmiddleware.ModelResponse, error) {
		t.Error("handler must not run for a bad route")
		return agentmiddleware.ModelResponse{}, nil
	}

	// Missing model_route state.
	if _, err := router.WrapModelCall(t.Context(), agentmiddleware.ModelRequest{}, handler); err == nil {
		t.Error("expected error for missing model_route state")
	}

	// Unknown route name.
	if _, err := router.WrapModelCall(t.Context(), agentmiddleware.ModelRequest{
		State: map[string]any{"model_route": map[string]any{"type": "choice", "choice": "nope"}},
	}, handler); err == nil {
		t.Error("expected error for unknown route name")
	}

	// Route value without a choice string.
	if _, err := router.WrapModelCall(t.Context(), agentmiddleware.ModelRequest{
		State: map[string]any{"model_route": map[string]any{"type": "choice"}},
	}, handler); err == nil {
		t.Error("expected error for route without choice")
	}
}

func TestModelRouterStringModelResolution(t *testing.T) {
	server, _ := classifierServer(t, routeAnswers("fast"))
	router, err := NewModelRouterMiddleware(
		map[string]ModelChoice{
			"fast": {Model: "openai:gpt-4o-mini", Criteria: "c"},
		},
		routeInstructions,
		WithRouterClassifier(testClassifier(server)),
	)
	if err != nil {
		t.Fatalf("NewModelRouterMiddleware: %v", err)
	}

	var received any
	handler := func(_ context.Context, request agentmiddleware.ModelRequest) (agentmiddleware.ModelResponse, error) {
		received = request.Model
		return agentmiddleware.ModelResponse{}, nil
	}
	_, err = router.WrapModelCall(t.Context(), agentmiddleware.ModelRequest{
		State: map[string]any{"model_route": map[string]any{"type": "choice", "choice": "fast"}},
	}, handler)
	if err != nil {
		t.Fatalf("WrapModelCall: %v", err)
	}
	if received == nil {
		t.Fatal("handler received nil model")
	}
	if _, isString := received.(string); isString {
		t.Fatalf("model string was not resolved through InitChatModel+Resolve: %v", received)
	}
	if model, ok := received.(language.ChatModel); !ok {
		t.Fatalf("resolved model does not implement language.ChatModel: %T", received)
	} else {
		_ = model
	}
}

func TestModelRouterStateSchema(t *testing.T) {
	server, _ := classifierServer(t, routeAnswers("fast"))
	router, err := NewModelRouterMiddleware(
		map[string]ModelChoice{"fast": {Model: fakeModel{label: "fast"}, Criteria: "c"}},
		routeInstructions,
		WithRouterClassifier(testClassifier(server)),
	)
	if err != nil {
		t.Fatalf("NewModelRouterMiddleware: %v", err)
	}
	fields := router.StateSchema()
	if len(fields) != 1 || fields[0].Name != "model_route" {
		t.Errorf("state schema = %#v", fields)
	}
}

func TestModelRouterTracePolicy(t *testing.T) {
	server, _ := classifierServer(t, routeAnswers("fast"))
	router, err := NewModelRouterMiddleware(
		map[string]ModelChoice{"fast": {Model: fakeModel{label: "fast"}, Criteria: "c"}},
		routeInstructions,
		WithRouterClassifier(testClassifier(server)),
	)
	if err != nil {
		t.Fatalf("NewModelRouterMiddleware: %v", err)
	}
	policy := router.TracePolicy()
	if policy.ProcessInputs == nil {
		t.Fatal("ProcessInputs policy missing")
	}
	payload := map[string]any{"messages": []messages.Message{messages.Human("secret")}}
	omitted := policy.ProcessInputs(payload)
	if reflect.DeepEqual(omitted, payload) {
		t.Error("process_inputs policy must replace the payload")
	}
}

func TestNewModelRouterValidation(t *testing.T) {
	server, _ := classifierServer(t, routeAnswers("fast"))
	if _, err := NewModelRouterMiddleware(nil, "i", WithRouterClassifier(testClassifier(server))); err == nil {
		t.Error("expected error for empty choices")
	}
	if _, err := NewModelRouterMiddleware(
		map[string]ModelChoice{"fast": {Model: fakeModel{label: "f"}, Criteria: "c"}},
		nil,
		WithRouterClassifier(testClassifier(server)),
	); err == nil {
		t.Error("expected error for nil instructions")
	}

	// An unresolvable model string fails construction.
	if _, err := NewModelRouterMiddleware(
		map[string]ModelChoice{"fast": {Model: "nosuchprovider:model-x", Criteria: "c"}},
		"i",
		WithRouterClassifier(testClassifier(server)),
	); err == nil {
		t.Error("expected error for unresolvable model string")
	}

	t.Setenv("TYPESAFE_API_KEY", "")
	if _, err := NewModelRouterMiddleware(
		map[string]ModelChoice{"fast": {Model: fakeModel{label: "f"}, Criteria: "c"}},
		"i",
	); err == nil {
		t.Error("expected error when the classifier cannot be built from env")
	}
}
