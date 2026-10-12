package middleware

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	// Registers the "openai" chatmodels factory for the string-model test.
	_ "github.com/projanvil/langchain-golang/partners/openai"
)

func choiceResponse(choice string) string {
	return fmt.Sprintf(`{"model": "gpt-6-luna", "answers": [{"type": "choice", "name": "model_route", "choice": %q,
		"probabilities": [{"value": "fast", "probability": 0.9}, {"value": "powerful", "probability": 0.1}],
		"confidence": 0.9}]}`, choice)
}

func routerChoices(fast, powerful any) map[string]ModelChoice {
	return map[string]ModelChoice{
		"fast":    {Model: fast, Criteria: "Simple, well-scoped tasks."},
		"powerful": {Model: powerful, Criteria: "Complex tasks requiring deeper reasoning."},
	}
}

func TestModelRouterRoutesToSelectedModel(t *testing.T) {
	rec := &decisionRecorder{responses: []stubResponse{{http.StatusOK, choiceResponse("fast")}}}
	server := newDecisionsServer(t, rec)
	fast := language.NewFakeChatModel()
	powerful := language.NewFakeChatModel()
	mw := &ModelRouter{
		Choices:      routerChoices(fast, powerful),
		Instructions: "Choose the least costly model suited to the task.",
		Model:        testClient(server),
	}

	update, err := mw.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{messages.System("sys"), messages.Human("write a haiku")},
	})
	if err != nil {
		t.Fatalf("BeforeAgent() error = %v", err)
	}
	route, ok := update["model_route"].(map[string]any)
	if !ok {
		t.Fatalf("model_route = %#v, want a JSON-native map", update["model_route"])
	}
	if route["choice"] != "fast" {
		t.Fatalf("choice = %v", route["choice"])
	}
	probabilities, ok := route["probabilities"].(map[string]float64)
	if !ok {
		t.Fatalf("probabilities = %#v, want map[string]float64", route["probabilities"])
	}
	if probabilities["fast"] != 0.9 || probabilities["powerful"] != 0.1 {
		t.Fatalf("probabilities = %#v", probabilities)
	}
	if route["confidence"] != 0.9 {
		t.Fatalf("confidence = %v", route["confidence"])
	}

	var receivedModel any
	handler := func(_ context.Context, request middleware.ModelRequest) (middleware.ModelResponse, error) {
		receivedModel = request.Model
		return middleware.ModelResponse{Result: []messages.Message{messages.AI("done")}}, nil
	}
	original := powerful
	_, err = mw.WrapModelCall(t.Context(), middleware.ModelRequest{
		Model: original,
		State: map[string]any{"model_route": route},
		Tools: []any{},
	}, handler)
	if err != nil {
		t.Fatalf("WrapModelCall() error = %v", err)
	}
	if receivedModel != any(fast) {
		t.Fatalf("handler model = %#v, want the fast model", receivedModel)
	}
}

func TestModelRouterWire(t *testing.T) {
	rec := &decisionRecorder{responses: []stubResponse{{http.StatusOK, choiceResponse("fast")}}}
	server := newDecisionsServer(t, rec)
	fast := language.NewFakeChatModel()
	mw := &ModelRouter{
		Choices:      routerChoices(fast, language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	_, err := mw.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{messages.Human("hello")},
	})
	if err != nil {
		t.Fatalf("BeforeAgent() error = %v", err)
	}

	// The latest human message goes natively (single HumanMessage input).
	input, ok := rec.bodies[0]["input"].([]any)
	if !ok || len(input) != 1 {
		t.Fatalf("input = %#v, want one native user message", rec.bodies[0]["input"])
	}
	user := input[0].(map[string]any)
	if user["role"] != "user" || user["content"] != "hello" {
		t.Fatalf("input message = %#v", user)
	}

	questions := rec.bodies[0]["questions"].([]any)
	question := questions[0].(map[string]any)
	if question["type"] != "choice" || question["name"] != "model_route" {
		t.Fatalf("question = %#v", question)
	}
	if question["instructions"] != "Which model?" {
		t.Fatalf("instructions = %v", question["instructions"])
	}
	choices, ok := question["choices"].([]any)
	if !ok || len(choices) != 2 {
		t.Fatalf("choices = %#v", question["choices"])
	}
	first := choices[0].(map[string]any)
	if first["value"] != "fast" || first["description"] != "Simple, well-scoped tasks." {
		t.Fatalf("choices[0] = %#v", first)
	}
}

func TestModelRouterNoHumanSetsNilRoute(t *testing.T) {
	rec := &decisionRecorder{}
	server := newDecisionsServer(t, rec)
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	update, err := mw.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{messages.System("sys"), messages.AI("no human here")},
	})
	if err != nil {
		t.Fatalf("BeforeAgent() error = %v", err)
	}
	value, present := update["model_route"]
	if !present || value != nil {
		t.Fatalf("model_route = %#v (present=%v), want an explicit nil", value, present)
	}
	if rec.requests != 0 {
		t.Fatalf("requests = %d, want 0", rec.requests)
	}
}

func TestModelRouterRefusalSetsNilRoute(t *testing.T) {
	rec := &decisionRecorder{responses: []stubResponse{{http.StatusOK, `{"model": "m", "answers": [{"type": "refusal", "name": "model_route"}]}`}}}
	server := newDecisionsServer(t, rec)
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	update, err := mw.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{messages.Human("hi")},
	})
	if err != nil {
		t.Fatalf("BeforeAgent() error = %v", err)
	}
	if update["model_route"] != nil {
		t.Fatalf("model_route = %#v, want nil for a refusal", update["model_route"])
	}
}

func TestModelRouterNilRoutePassesModelThrough(t *testing.T) {
	server := newDecisionsServer(t, &decisionRecorder{})
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	original := language.NewFakeChatModel()
	var receivedModel any
	handler := func(_ context.Context, request middleware.ModelRequest) (middleware.ModelResponse, error) {
		receivedModel = request.Model
		return middleware.ModelResponse{}, nil
	}
	_, err := mw.WrapModelCall(t.Context(), middleware.ModelRequest{
		Model: original,
		State: map[string]any{"messages": []messages.Message{}},
		Tools: []any{},
	}, handler)
	if err != nil {
		t.Fatalf("WrapModelCall() error = %v", err)
	}
	if receivedModel != any(original) {
		t.Fatalf("handler model = %#v, want the original", receivedModel)
	}
}

func TestModelRouterUnknownRouteErrors(t *testing.T) {
	server := newDecisionsServer(t, &decisionRecorder{})
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	_, err := mw.WrapModelCall(t.Context(), middleware.ModelRequest{
		Model: language.NewFakeChatModel(),
		State: map[string]any{"model_route": map[string]any{"choice": "nonexistent"}},
		Tools: []any{},
	}, func(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
		t.Fatal("handler must not run for an unknown route")
		return middleware.ModelResponse{}, nil
	})
	if err == nil {
		t.Fatal("WrapModelCall() expected an error for an unknown route")
	}
	if !strings.Contains(err.Error(), "nonexistent") {
		t.Fatalf("WrapModelCall() error = %v", err)
	}
}

func TestModelRouterBase64ImageKeptWithDegrade(t *testing.T) {
	withImage := humanWithBlocks(
		messages.TextBlock{Text: "describe"},
		messages.ImageBlock{Base64: "pic", MimeType: "image/png"},
	)
	rec := &decisionRecorder{responses: []stubResponse{
		{http.StatusBadRequest, `{"error": "nope"}`},
		{http.StatusOK, choiceResponse("powerful")},
	}}
	server := newDecisionsServer(t, rec)
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	update, err := mw.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{messages.System("sys"), withImage},
	})
	if err != nil {
		t.Fatalf("BeforeAgent() error = %v", err)
	}
	if rec.requests != 2 {
		t.Fatalf("requests = %d, want 2 (with base64 image, then text only)", rec.requests)
	}
	// Native input: the base64 image rides as an input_image part.
	firstJSON := fullRequestJSON(t, rec.bodies[0])
	if !strings.Contains(firstJSON, "data:image/png;base64,pic") {
		t.Fatalf("first request = %s, want the base64 image retained natively", firstJSON)
	}
	secondJSON := fullRequestJSON(t, rec.bodies[1])
	if strings.Contains(secondJSON, "data:image") {
		t.Fatalf("second request = %s, want the image placeholdered", secondJSON)
	}
	if !strings.Contains(secondJSON, "[image omitted]") {
		t.Fatalf("second request = %s, want an [image omitted] placeholder", secondJSON)
	}
	route := update["model_route"].(map[string]any)
	if route["choice"] != "powerful" {
		t.Fatalf("choice = %v", route["choice"])
	}
}

func TestModelRouterHostedImagePlaceholderSingleRequest(t *testing.T) {
	hostedOnly := humanWithBlocks(
		messages.TextBlock{Text: "look"},
		messages.ImageBlock{URL: "https://example.com/pic.png"},
	)
	rec := &decisionRecorder{responses: []stubResponse{{http.StatusOK, choiceResponse("fast")}}}
	server := newDecisionsServer(t, rec)
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	_, err := mw.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{hostedOnly},
	})
	if err != nil {
		t.Fatalf("BeforeAgent() error = %v", err)
	}
	// Hosted URLs are not base64: both variants placeholder them, so only one
	// request is sent (contrast with AutoMode, which keeps hosted images).
	if rec.requests != 1 {
		t.Fatalf("requests = %d, want 1", rec.requests)
	}
	bodyJSON := fullRequestJSON(t, rec.bodies[0])
	if !strings.Contains(bodyJSON, "[image omitted]") {
		t.Fatalf("request = %s, want an [image omitted] placeholder", bodyJSON)
	}
}

func TestModelRouterNonMediaBlocksPlaceholdered(t *testing.T) {
	withAudio := humanWithBlocks(
		messages.TextBlock{Text: "listen"},
		messages.AudioBlock{Base64: "wav"},
	)
	rec := &decisionRecorder{responses: []stubResponse{{http.StatusOK, choiceResponse("fast")}}}
	server := newDecisionsServer(t, rec)
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	_, err := mw.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{withAudio},
	})
	if err != nil {
		t.Fatalf("BeforeAgent() error = %v", err)
	}
	if rec.requests != 1 {
		t.Fatalf("requests = %d, want 1 (audio placeholdered in both variants)", rec.requests)
	}
	bodyJSON := fullRequestJSON(t, rec.bodies[0])
	if !strings.Contains(bodyJSON, "[audio omitted]") {
		t.Fatalf("request = %s, want an [audio omitted] placeholder", bodyJSON)
	}
}

func TestModelRouterStringModelResolved(t *testing.T) {
	rec := &decisionRecorder{responses: []stubResponse{{http.StatusOK, choiceResponse("fast")}}}
	server := newDecisionsServer(t, rec)
	mw := &ModelRouter{
		Choices: map[string]ModelChoice{
			"fast":    {Model: "openai:gpt-4o-mini", Criteria: "Simple tasks."},
			"powerful": {Model: "openai:gpt-4o", Criteria: "Complex tasks."},
		},
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	update, err := mw.BeforeAgent(t.Context(), map[string]any{
		"messages": []messages.Message{messages.Human("hi")},
	})
	if err != nil {
		t.Fatalf("BeforeAgent() error = %v", err)
	}
	route := update["model_route"].(map[string]any)

	var receivedModel any
	_, err = mw.WrapModelCall(t.Context(), middleware.ModelRequest{
		Model: language.NewFakeChatModel(),
		State: map[string]any{"model_route": route},
		Tools: []any{},
	}, func(_ context.Context, request middleware.ModelRequest) (middleware.ModelResponse, error) {
		receivedModel = request.Model
		return middleware.ModelResponse{}, nil
	})
	if err != nil {
		t.Fatalf("WrapModelCall() error = %v", err)
	}
	model, ok := receivedModel.(language.ChatModel)
	if !ok {
		t.Fatalf("resolved model = %#v, want a language.ChatModel", receivedModel)
	}
	if lt, ok := model.(interface{ LLMType() string }); ok && lt.LLMType() != "openai-chat" {
		t.Fatalf("resolved model LLMType = %q, want openai-chat", lt.LLMType())
	}
}

func TestModelRouterTracePolicyAndStateSchema(t *testing.T) {
	server := newDecisionsServer(t, &decisionRecorder{})
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	cfg := mw.TracePolicy()
	if cfg.ProcessInputs == nil {
		t.Fatal("TracePolicy().ProcessInputs must be set (omit-payload)")
	}
	if got := cfg.ProcessInputs("secret input"); got != nil {
		t.Fatalf("ProcessInputs = %#v, want nil (payload omitted)", got)
	}
	if cfg.ProcessOutputs != nil {
		t.Fatal("ProcessOutputs must be nil")
	}

	schema := mw.StateSchema()
	if len(schema) != 1 || schema[0].Name != "model_route" {
		t.Fatalf("StateSchema() = %#v, want [model_route]", schema)
	}
}

func TestModelRouterValidation(t *testing.T) {
	server := newDecisionsServer(t, &decisionRecorder{})
	tests := []struct {
		name string
		mw   *ModelRouter
		want string
	}{
		{
			name: "empty choices",
			mw:   &ModelRouter{Instructions: "i", Model: testClient(server)},
			want: "`choices` must contain at least one model choice",
		},
		{
			name: "blank instructions",
			mw: &ModelRouter{
				Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
				Instructions: "   ",
				Model:        testClient(server),
			},
			want: "`instructions` must not be empty",
		},
		{
			name: "missing model",
			mw: &ModelRouter{
				Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
				Instructions: "i",
			},
			want: "model",
		},
		{
			name: "bad model string",
			mw: &ModelRouter{
				Choices:      map[string]ModelChoice{"fast": {Model: "no-such-provider:x", Criteria: "c"}},
				Instructions: "i",
				Model:        testClient(server),
			},
			want: "provider",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.mw.BeforeAgent(t.Context(), map[string]any{
				"messages": []messages.Message{messages.Human("hi")},
			})
			if err == nil {
				t.Fatal("BeforeAgent() expected a validation error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("BeforeAgent() error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

// Compile-time interface checks against the hook surfaces the agent uses.
var _ interface {
	BeforeAgent(context.Context, map[string]any) (map[string]any, error)
} = (*ModelRouter)(nil)

func TestModelRouterRouteValueHelpers(t *testing.T) {
	if got := routerChoiceString(true); got != "true" {
		t.Fatalf("routerChoiceString(true) = %q", got)
	}
	if got := routerChoiceString(false); got != "false" {
		t.Fatalf("routerChoiceString(false) = %q", got)
	}
	if got := routerChoiceString(7); got != "7" {
		t.Fatalf("routerChoiceString(7) = %q", got)
	}
	if got := routerBlockLabel(messages.NonStandardContentBlock{}); got != "content" {
		t.Fatalf("routerBlockLabel(empty) = %q", got)
	}
	if got := routerBlockLabel(messages.AudioBlock{}); got != "audio" {
		t.Fatalf("routerBlockLabel(audio) = %q", got)
	}
}

func TestModelRouterBadRouteStateShape(t *testing.T) {
	server := newDecisionsServer(t, &decisionRecorder{})
	mw := &ModelRouter{
		Choices:      routerChoices(language.NewFakeChatModel(), language.NewFakeChatModel()),
		Instructions: "Which model?",
		Model:        testClient(server),
	}
	_, err := mw.WrapModelCall(t.Context(), middleware.ModelRequest{
		Model: language.NewFakeChatModel(),
		State: map[string]any{"model_route": "fast"}, // not a map
		Tools: []any{},
	}, func(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
		t.Fatal("handler must not run for a malformed route")
		return middleware.ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "must be a map") {
		t.Fatalf("WrapModelCall() error = %v, want a malformed-route error", err)
	}
}
