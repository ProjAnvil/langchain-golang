package typesafe

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/decisions"
)

// TestQuestionToTypesafeMapping pins the core → TypeSafe question mapping:
// Noul/Choice/Score convert, Box errors.
func TestQuestionToTypesafeMapping(t *testing.T) {
	noul, err := questionToTypesafe(decisions.Noul{Instructions: "urgent?"})
	if err != nil || noul.(Noul).Instructions != "urgent?" {
		t.Fatalf("noul: %#v %v", noul, err)
	}
	choice, err := questionToTypesafe(decisions.Choice{
		Instructions: "route",
		Criteria:     map[string]decisions.Content{"billing": "payments", "other": nil},
	})
	if err != nil {
		t.Fatalf("choice: %v", err)
	}
	typed := choice.(Choice)
	if len(typed.Criteria) != 2 || typed.Criteria[0].Label != "billing" || typed.Criteria[0].Description != "payments" {
		t.Fatalf("choice criteria: %#v", typed.Criteria)
	}
	score, err := questionToTypesafe(decisions.Score{Instructions: "rate", Criteria: []decisions.Content{"low", "high"}})
	if err != nil || len(score.(Score).Criteria) != 2 {
		t.Fatalf("score: %#v %v", score, err)
	}
	if _, err := questionToTypesafe(decisions.Box{Instructions: "find the dog"}); err == nil {
		t.Fatal("box must error on TypeSafe")
	}
}

// TestTypesafeResponseToCore pins the TypeSafe → core response mapping,
// including usage completion.
func TestTypesafeResponseToCore(t *testing.T) {
	input, output := 3, 4
	core := responseToCore(Response{
		Model: "jev",
		Answers: map[string]Answer{
			"q": ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"billing": 0.9}, Confidence: 0.9},
		},
		Usage:     Usage{InputTokens: &input, OutputTokens: &output},
		RequestID: "req_1",
	})
	answer, ok := core.Answers["q"].(decisions.ChoiceAnswer)
	if !ok || answer.Choice != "billing" || answer.Confidence != 0.9 {
		t.Fatalf("answers: %#v", core.Answers)
	}
	if core.Usage == nil || core.Usage.TotalTokens != 7 {
		t.Fatalf("usage: %#v", core.Usage)
	}
	if core.RequestID != "req_1" {
		t.Fatalf("request id: %q", core.RequestID)
	}
}

// TestAdapterEmitsDecisionRunEvents ports the BaseDecisionModel run
// semantics of upstream test_base (#41765): the classification surfaces as
// a chat-model run with ls_provider/ls_model_name metadata and token usage
// on the end event; a handler failure never fails the classification.
func TestAdapterEmitsDecisionRunEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "jev-latest",
			"answers": {"q": {"type": "noul", "noul": 0.5}},
			"usage": {"input_tokens": 42}
		}`))
	}))
	defer server.Close()
	classifier := testClassifier(t, server.URL, server.Client())
	adapter := classifier.AsDecisionModel()

	recorder := callbacks.NewRecorder()
	ctx := callbacks.ContextWithManager(t.Context(), callbacks.NewManager(recorder))
	response, err := adapter.Invoke(ctx, decisions.Request{
		State: "hello",
		Questions: map[string]decisions.Question{
			"q": decisions.Noul{Instructions: "urgent?"},
		},
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if response.Usage == nil || response.Usage.InputTokens != 42 || response.Usage.TotalTokens != 42 {
		t.Fatalf("usage: %#v", response.Usage)
	}
	var start, end bool
	for _, event := range recorder.Events() {
		switch event.Kind {
		case callbacks.EventChatModelStart:
			start = true
			if event.Metadata["ls_provider"] != "typesafe" ||
				event.Metadata["ls_model_name"] != classifier.Model ||
				event.Metadata["ls_model_type"] != "chat" {
				t.Fatalf("start metadata: %#v", event.Metadata)
			}
		case callbacks.EventChatModelEnd:
			end = true
			usage, _ := event.Metadata["usage_metadata"].(map[string]any)
			if usage["input_tokens"] != 42 || usage["total_tokens"] != 42 {
				t.Fatalf("end usage metadata: %#v", event.Metadata)
			}
		}
	}
	if !start || !end {
		t.Fatalf("run events: start=%v end=%v", start, end)
	}

	// A handler that rejects every event must not fail the classification
	// (upstream: tracing failures never fail a successful request).
	failing := callbacks.NewManager(&rejectingHandler{})
	response, err = adapter.Invoke(callbacks.ContextWithManager(t.Context(), failing), decisions.Request{
		State:     "hello",
		Questions: map[string]decisions.Question{"q": decisions.Noul{Instructions: "urgent?"}},
	})
	if err != nil || response.Usage == nil {
		t.Fatalf("tracing failure must not fail classification: %v %#v", err, response.Usage)
	}
}

// rejectingHandler rejects every event, standing in for a broken tracer.
type rejectingHandler struct{}

func (*rejectingHandler) HandleEvent(_ context.Context, _ callbacks.Event) error {
	return errors.New("tracer down")
}
