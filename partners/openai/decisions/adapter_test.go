package decisions

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	coredecisions "github.com/projanvil/langchain-golang/core/decisions"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// TestQuestionToOpenAIMapping pins the core → Decisions API question
// mapping: Noul→Predicate, Choice, Score convert; Box and non-string
// instructions/score levels error.
func TestQuestionToOpenAIMapping(t *testing.T) {
	predicate, err := questionToOpenAI(coredecisions.Noul{Instructions: "risky?"})
	if err != nil || predicate.(Predicate).Instructions != "risky?" {
		t.Fatalf("noul/predicate: %#v %v", predicate, err)
	}
	choice, err := questionToOpenAI(coredecisions.Choice{
		Instructions: "route",
		Criteria:     map[string]coredecisions.Content{"billing": "payments", "other": nil},
	})
	if err != nil {
		t.Fatalf("choice: %v", err)
	}
	typed := choice.(Choice)
	if len(typed.Choices) != 2 || typed.Choices[0].Value != "billing" || typed.Choices[0].Description == nil {
		t.Fatalf("choice options: %#v", typed.Choices)
	}
	score, err := questionToOpenAI(coredecisions.Score{Instructions: "rate", Criteria: []coredecisions.Content{"low", "high"}})
	if err != nil || len(score.(Score).Levels) != 2 {
		t.Fatalf("score: %#v %v", score, err)
	}
	if _, err := questionToOpenAI(coredecisions.Box{Instructions: "find"}); err == nil {
		t.Fatal("box must error on the Decisions API")
	}
	if _, err := questionToOpenAI(coredecisions.Noul{Instructions: map[string]any{"k": "v"}}); err == nil {
		t.Fatal("non-string instructions must error")
	}
	if _, err := questionToOpenAI(coredecisions.Score{Instructions: "r", Criteria: []coredecisions.Content{map[string]any{}, "high"}}); err == nil {
		t.Fatal("non-string score level must error")
	}
}

// TestOpenAIResponseToCore pins the Decisions API → core response mapping:
// Predicate→Noul, string choices keep their label, bool choices and
// refusals are dropped, and usage completes.
func TestOpenAIResponseToCore(t *testing.T) {
	input := 5
	core := responseToCore(Response{
		Model: "gpt-test",
		Answers: map[string]Answer{
			"risk":  PredicateAnswer{Probability: 0.75},
			"route": ChoiceAnswer{Choice: "billing", Probabilities: map[string]float64{"billing": 1.0}, Confidence: 1.0},
			"bool":  ChoiceAnswer{Choice: true},
			"decl":  RefusalAnswer{},
		},
		Usage:     Usage{InputTokens: &input},
		RequestID: "req_2",
	})
	if noul := core.Answers["risk"].(coredecisions.NoulAnswer); noul.Noul != 0.75 {
		t.Fatalf("predicate mapping: %#v", noul)
	}
	if choice := core.Answers["route"].(coredecisions.ChoiceAnswer); choice.Choice != "billing" {
		t.Fatalf("choice mapping: %#v", choice)
	}
	if _, has := core.Answers["bool"]; has {
		t.Fatal("bool choice must be dropped")
	}
	if _, has := core.Answers["decl"]; has {
		t.Fatal("refusal answer must be dropped")
	}
	if core.Usage == nil || core.Usage.TotalTokens != 5 {
		t.Fatalf("usage: %#v", core.Usage)
	}
}

// TestOpenAIAdapterEmitsDecisionRunEvents ports the BaseDecisionModel run
// semantics (#41765) for the OpenAI adapter: ls_provider/ls_model_name on
// the chat-model start event, usage on the end event.
func TestOpenAIAdapterEmitsDecisionRunEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "gpt-6-luna",
			"answers": [{"name": "q", "type": "predicate", "probability": 0.5}],
			"usage": {"input_tokens": 7, "output_tokens": 1}
		}`))
	}))
	defer server.Close()
	client := &Client{Model: "gpt-6-luna", APIKey: "k", BaseURL: server.URL}
	adapter := client.AsDecisionModel()

	recorder := callbacks.NewRecorder()
	if _, err := adapter.Invoke(t.Context(), coredecisions.Request{
		State:     "hello",
		Questions: map[string]coredecisions.Question{"q": coredecisions.Noul{Instructions: "risky?"}},
	}, runnables.WithCallbacks(callbacks.NewManager(recorder))); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	var start, end bool
	for _, event := range recorder.Events() {
		switch event.Kind {
		case callbacks.EventChatModelStart:
			start = true
			if event.Metadata["ls_provider"] != "openai" || event.Metadata["ls_model_name"] != "gpt-6-luna" {
				t.Fatalf("start metadata: %#v", event.Metadata)
			}
		case callbacks.EventChatModelEnd:
			end = true
			usage, _ := event.Metadata["usage_metadata"].(map[string]any)
			if usage["total_tokens"] != 8 {
				t.Fatalf("end usage metadata: %#v", event.Metadata)
			}
		}
	}
	if !start || !end {
		t.Fatalf("run events: start=%v end=%v", start, end)
	}
}

// TestOpenAIAdapterErrorPaths covers the adapter's failure branches: an
// invalid request fails before any event, an unsupported question (Box)
// fails per-question, and a client failure surfaces an error event before
// the error return.
func TestOpenAIAdapterErrorPaths(t *testing.T) {
	adapter := (&Client{Model: "gpt-6-luna", APIKey: "k", BaseURL: "http://unused"}).AsDecisionModel()

	// Invalid request: no questions.
	if _, err := adapter.Invoke(t.Context(), coredecisions.Request{State: "s"}); err == nil {
		t.Fatal("invalid request must fail")
	}

	// Unsupported question type: Box has no Decisions counterpart.
	if _, err := adapter.Invoke(t.Context(), coredecisions.Request{
		State:     "s",
		Questions: map[string]coredecisions.Question{"q": coredecisions.Box{Instructions: "find"}},
	}); err == nil {
		t.Fatal("box question must fail")
	}

	// Client failure: an error event is emitted, then the error returns.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error": {"message": "boom", "code": "server_error"}}`))
	}))
	defer server.Close()
	failing := (&Client{Model: "gpt-6-luna", APIKey: "k", BaseURL: server.URL}).AsDecisionModel()
	recorder := callbacks.NewRecorder()
	_, err := failing.Invoke(t.Context(), coredecisions.Request{
		State:     "s",
		Questions: map[string]coredecisions.Question{"q": coredecisions.Noul{Instructions: "r?"}},
	}, runnables.WithCallbacks(callbacks.NewManager(recorder)))
	if err == nil {
		t.Fatal("client failure must propagate")
	}
	sawStart, sawError := false, false
	for _, event := range recorder.Events() {
		switch event.Kind {
		case callbacks.EventChatModelStart:
			sawStart = true
		case callbacks.EventChatModelError:
			sawError = true
		}
	}
	if !sawStart || !sawError {
		t.Fatalf("run events on failure: start=%v error=%v", sawStart, sawError)
	}
}

// TestOpenAIAdapterScoreMapError covers the non-string score level failure
// through the adapter.
func TestOpenAIAdapterScoreMapError(t *testing.T) {
	adapter := (&Client{Model: "m", APIKey: "k", BaseURL: "http://unused"}).AsDecisionModel()
	if _, err := adapter.Invoke(t.Context(), coredecisions.Request{
		State: "s",
		Questions: map[string]coredecisions.Question{"q": coredecisions.Score{
			Instructions: "r", Criteria: []coredecisions.Content{123, "high"},
		}},
	}); err == nil {
		t.Fatal("non-string score level must fail")
	}
}

// TestOpenAIAdapterScoreAnswerMapping covers the ScoreAnswer branch of
// responseToCore (level-keyed legend and distribution) through a full
// client round trip.
func TestOpenAIAdapterScoreAnswerMapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "gpt-6-luna",
			"answers": [{
				"name": "q",
				"type": "score",
				"score": 1.2,
				"levels": ["low", "high"],
				"probabilities": [
					{"value": 0, "label": "low", "probability": 0.3},
					{"value": 1, "label": "high", "probability": 0.7}
				],
				"confidence": 0.6
			}]
		}`))
	}))
	defer server.Close()
	adapter := (&Client{Model: "gpt-6-luna", APIKey: "k", BaseURL: server.URL}).AsDecisionModel()
	response, err := adapter.Invoke(t.Context(), coredecisions.Request{
		State:     "s",
		Questions: map[string]coredecisions.Question{"q": coredecisions.Score{Instructions: "r", Criteria: []coredecisions.Content{"low", "high"}}},
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	score, ok := response.Answers["q"].(coredecisions.ScoreAnswer)
	if !ok {
		t.Fatalf("answer type: %#v", response.Answers["q"])
	}
	if score.Score != 1.2 || score.Legend[0] != "low" || score.Legend[1] != "high" ||
		score.Probabilities[1] != 0.7 || score.Confidence != 0.6 {
		t.Fatalf("score mapping: %#v", score)
	}
}
