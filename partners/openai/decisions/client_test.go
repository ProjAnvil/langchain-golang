package decisions

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

func TestNewClientRequiresAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_API_BASE", "")
	t.Setenv("OPENAI_BASE_URL", "")
	_, err := NewClient("gpt-6-luna")
	if err == nil {
		t.Fatal("NewClient() expected an error when OPENAI_API_KEY is missing")
	}
	want := "OpenAI API key is required. Pass an API key or set `OPENAI_API_KEY`."
	if err.Error() != want {
		t.Fatalf("NewClient() error = %q, want %q", err.Error(), want)
	}
}

func TestNewClientRequiresModel(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test")
	_, err := NewClient("")
	if err == nil {
		t.Fatal("NewClient() expected an error for an empty model")
	}
}

func TestNewClientBaseURLResolution(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("OPENAI_API_BASE", "")
	t.Setenv("OPENAI_BASE_URL", "")
	client, err := NewClient("gpt-6-luna")
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if client.BaseURL != "https://api.openai.com/v1" {
		t.Fatalf("BaseURL = %q, want the default", client.BaseURL)
	}

	t.Setenv("OPENAI_BASE_URL", "https://base.example.com/v1")
	client, err = NewClient("gpt-6-luna")
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if client.BaseURL != "https://base.example.com/v1" {
		t.Fatalf("BaseURL = %q, want OPENAI_BASE_URL fallback", client.BaseURL)
	}

	t.Setenv("OPENAI_API_BASE", "https://api-base.example.com/v1")
	client, err = NewClient("gpt-6-luna")
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if client.BaseURL != "https://api-base.example.com/v1" {
		t.Fatalf("BaseURL = %q, want OPENAI_API_BASE to win over OPENAI_BASE_URL", client.BaseURL)
	}
}

// goldenResponse mixes predicate/choice/score/refusal answers, one answer
// with an unknown type (skipped), usage counts, and a request ID header.
const goldenResponse = `{
  "model": "gpt-6-luna",
  "answers": [
    {"type": "predicate", "name": "urgent", "probability": 0.87},
    {"type": "choice", "name": "department", "choice": "billing",
     "probabilities": [
       {"value": "billing", "probability": 0.75},
       {"value": true, "probability": 0.05},
       {"value": "technical", "probability": 0.2}
     ],
     "confidence": 0.75},
    {"type": "score", "name": "frustration", "score": 1.1,
     "probabilities": [
       {"value": 0, "label": "Calm", "probability": 0.4},
       {"value": 1, "label": "Angry", "probability": 0.6}
     ],
     "confidence": 0.9},
    {"type": "refusal", "name": "blocked_q", "reason": "safety"},
    {"type": "mystery", "name": "future"}
  ],
  "usage": {"input_tokens": 12, "output_tokens": 3}
}`

func goldenQuestions() map[string]Question {
	return map[string]Question{
		"zeta": Predicate{Instructions: "Is this urgent?"},
		"alpha": Choice{
			Instructions: "Which department?",
			Choices:      []ChoiceOption{{Value: "billing", Description: ptr("Payments.")}, {Value: "technical"}},
		},
		"mid": Score{
			Instructions: "How frustrated?",
			Levels:       []ScoreLevel{L("Calm"), L("Angry")},
		},
	}
}

func TestInvokeGolden(t *testing.T) {
	var gotRequest *http.Request
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRequest = r.Clone(r.Context())
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("x-request-id", "req_123")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(goldenResponse))
	}))
	defer server.Close()

	client := &Client{
		Model:      "gpt-6-luna",
		APIKey:     "sk-test",
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	}
	resp, err := client.Invoke(t.Context(), Request{Input: "help me", Questions: goldenQuestions()})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	// Request wire assertions.
	if gotRequest.Method != http.MethodPost {
		t.Fatalf("method = %s, want POST", gotRequest.Method)
	}
	if gotRequest.URL.Path != "/decisions" {
		t.Fatalf("path = %s, want /decisions", gotRequest.URL.Path)
	}
	if auth := gotRequest.Header.Get("Authorization"); auth != "Bearer sk-test" {
		t.Fatalf("Authorization = %q", auth)
	}
	if ct := gotRequest.Header.Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type = %q", ct)
	}
	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	if body["model"] != "gpt-6-luna" {
		t.Fatalf("model = %v", body["model"])
	}
	if body["input"] != "help me" {
		t.Fatalf("input = %v", body["input"])
	}
	questions, ok := body["questions"].([]any)
	if !ok {
		t.Fatalf("questions = %#v", body["questions"])
	}
	// Questions must be sorted by name (Go map iteration is randomized;
	// Python preserves insertion order — DIVERGENCES #8).
	var names []string
	for _, q := range questions {
		names = append(names, q.(map[string]any)["name"].(string))
	}
	if strings.Join(names, ",") != "alpha,mid,zeta" {
		t.Fatalf("question order = %v, want [alpha mid zeta]", names)
	}
	first := questions[0].(map[string]any)
	if first["type"] != "choice" {
		t.Fatalf("questions[0] = %#v", first)
	}

	// Response parsing assertions.
	if resp.Model != "gpt-6-luna" {
		t.Fatalf("Model = %q", resp.Model)
	}
	if resp.RequestID != "req_123" {
		t.Fatalf("RequestID = %q", resp.RequestID)
	}
	if len(resp.Answers) != 4 {
		t.Fatalf("Answers = %#v, want the mystery type skipped", resp.Answers)
	}
	if _, present := resp.Answers["future"]; present {
		t.Fatal("unknown answer type must be skipped")
	}
	if got := resp.Predicates()["urgent"]; got.Probability != 0.87 {
		t.Fatalf("predicates[urgent] = %#v", got)
	}
	choice := resp.Choices()["department"]
	if choice.Choice != "billing" || choice.Confidence != 0.75 {
		t.Fatalf("choices[department] = %#v", choice)
	}
	wantProbs := map[string]float64{"billing": 0.75, "true": 0.05, "technical": 0.2}
	for k, v := range wantProbs {
		if got, ok := choice.Probabilities[k]; !ok || got != v {
			t.Fatalf("choices[department].Probabilities[%q] = %v (ok=%v), want %v", k, got, ok, v)
		}
	}
	score := resp.Scores()["frustration"]
	if score.Score != 1.1 || score.Confidence != 0.9 {
		t.Fatalf("scores[frustration] = %#v", score)
	}
	if score.Legend[0] != "Calm" || score.Legend[1] != "Angry" {
		t.Fatalf("scores[frustration].Legend = %#v", score.Legend)
	}
	if score.Probabilities[0] != 0.4 || score.Probabilities[1] != 0.6 {
		t.Fatalf("scores[frustration].Probabilities = %#v", score.Probabilities)
	}
	refusal := resp.Refusals()["blocked_q"]
	if refusal.Extra["reason"] != "safety" {
		t.Fatalf("refusals[blocked_q] = %#v", refusal)
	}
	if resp.Usage.InputTokens == nil || *resp.Usage.InputTokens != 12 {
		t.Fatalf("Usage.InputTokens = %v", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens == nil || *resp.Usage.OutputTokens != 3 {
		t.Fatalf("Usage.OutputTokens = %v", resp.Usage.OutputTokens)
	}
}

func TestInvokeDefaultHTTPClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model": "m", "answers": [{"type": "predicate", "name": "q", "probability": 0.2}]}`))
	}))
	defer server.Close()

	// A nil HTTPClient falls back to &http.Client{}.
	client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL}
	resp, err := client.Invoke(t.Context(), Request{
		Input:     "x",
		Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp.Predicates()["q"].Probability != 0.2 {
		t.Fatalf("predicates[q] = %#v", resp.Predicates()["q"])
	}
}

func TestInvokeUsageDefaults(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model": "m", "answers": [{"type": "predicate", "name": "q", "probability": 0.1}]}`))
	}))
	defer server.Close()

	client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client()}
	resp, err := client.Invoke(t.Context(), Request{
		Input:     "x",
		Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp.Usage.InputTokens != nil || resp.Usage.OutputTokens != nil {
		t.Fatalf("Usage = %#v, want nil token counts when unreported", resp.Usage)
	}
}

func TestInvokeHTTPErrorTable(t *testing.T) {
	tests := []struct {
		name         string
		status       int
		body         string
		wantSentinel error
	}{
		{"400", 400, `{"error": {"message": "bad request"}}`, lcerrors.ErrModelInvalidRequest},
		{"401", 401, `{"error": {"message": "bad key"}}`, lcerrors.ErrModelAuth},
		{"429", 429, `{"error": {"message": "slow down"}}`, lcerrors.ErrModelRateLimit},
		{"500", 500, `{"error": {"message": "boom"}}`, lcerrors.ErrModelServer},
		{"400 context overflow", 400, `{"error": {"code": "context_length_exceeded"}}`, lcerrors.ErrModelContextOverflow},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client()}
			_, err := client.Invoke(t.Context(), Request{
				Input:     "x",
				Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
			})
			if err == nil {
				t.Fatal("Invoke() expected an error")
			}
			if !errors.Is(err, tt.wantSentinel) {
				t.Fatalf("Invoke() error = %v, want sentinel %v", err, tt.wantSentinel)
			}
			var providerErr *lcerrors.ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("Invoke() error = %v, want a *ProviderError", err)
			}
			if providerErr.Provider != "openai" || providerErr.Endpoint != "/decisions" || providerErr.StatusCode != tt.status {
				t.Fatalf("ProviderError = %+v", providerErr)
			}
		})
	}
}

func TestInvokeInvalidResponse(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"not json", `not json at all`},
		{"missing model", `{"answers": []}`},
		{"missing answers", `{"model": "m"}`},
		{"answer missing name", `{"model": "m", "answers": [{"type": "predicate", "probability": 0.5}]}`},
		{"predicate missing probability", `{"model": "m", "answers": [{"type": "predicate", "name": "q"}]}`},
		{"choice missing probabilities", `{"model": "m", "answers": [{"type": "choice", "name": "q", "choice": "a", "confidence": 0.5}]}`},
		{"score malformed levels", `{"model": "m", "answers": [{"type": "score", "name": "q", "score": 0.5, "probabilities": "nope", "confidence": 0.5}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client()}
			_, err := client.Invoke(t.Context(), Request{
				Input:     "x",
				Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
			})
			if err == nil {
				t.Fatal("Invoke() expected an error")
			}
			if !errors.Is(err, lcerrors.ErrSchemaValidation) {
				t.Fatalf("Invoke() error = %v, want ErrSchemaValidation", err)
			}
			if !strings.Contains(err.Error(), "Invalid response from the OpenAI Decisions API.") {
				t.Fatalf("Invoke() error = %v, want the verbatim message", err)
			}
		})
	}
}

func TestInvokeInvalidResponseNormalizationEdges(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"choice not a string or bool", `{"model": "m", "answers": [{"type": "choice", "name": "q", "choice": 5, "probabilities": [], "confidence": 0.5}]}`},
		{"choice missing confidence", `{"model": "m", "answers": [{"type": "choice", "name": "q", "choice": "a", "probabilities": []}]}`},
		{"choice probability entry not object", `{"model": "m", "answers": [{"type": "choice", "name": "q", "choice": "a", "probabilities": ["x"], "confidence": 0.5}]}`},
		{"choice probability entry without value", `{"model": "m", "answers": [{"type": "choice", "name": "q", "choice": "a", "probabilities": [{"probability": 0.5}], "confidence": 0.5}]}`},
		{"score missing score", `{"model": "m", "answers": [{"type": "score", "name": "q", "probabilities": [], "confidence": 0.5}]}`},
		{"score level without label", `{"model": "m", "answers": [{"type": "score", "name": "q", "score": 0.5, "probabilities": [{"value": 0, "probability": 1}], "confidence": 0.5}]}`},
		{"score level non-integer index", `{"model": "m", "answers": [{"type": "score", "name": "q", "score": 0.5, "probabilities": [{"value": 0.5, "label": "a", "probability": 1}], "confidence": 0.5}]}`},
		{"score level value not a number", `{"model": "m", "answers": [{"type": "score", "name": "q", "score": 0.5, "probabilities": [{"value": "0", "label": "a", "probability": 1}], "confidence": 0.5}]}`},
		{"probability above one", `{"model": "m", "answers": [{"type": "predicate", "name": "q", "probability": 1.5}]}`},
		{"choice confidence below zero", `{"model": "m", "answers": [{"type": "choice", "name": "q", "choice": "a", "probabilities": [], "confidence": -0.1}]}`},
		{"score confidence below zero", `{"model": "m", "answers": [{"type": "score", "name": "q", "score": 0.5, "probabilities": [], "confidence": -0.1}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client()}
			_, err := client.Invoke(t.Context(), Request{
				Input:     "x",
				Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
			})
			if err == nil || !errors.Is(err, lcerrors.ErrSchemaValidation) {
				t.Fatalf("Invoke() error = %v, want ErrSchemaValidation", err)
			}
		})
	}
}

func TestInvokeNumberKeyedProbabilities(t *testing.T) {
	// Numeric choice values stringify through fmt.Sprint.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model": "m", "answers": [{"type": "choice", "name": "q", "choice": "a",
			"probabilities": [{"value": 7, "probability": 0.3}, {"value": "a", "probability": 0.7}], "confidence": 0.7}]}`))
	}))
	defer server.Close()

	client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client()}
	resp, err := client.Invoke(t.Context(), Request{
		Input:     "x",
		Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := resp.Choices()["q"].Probabilities["7"]; got != 0.3 {
		t.Fatalf("Probabilities[7] = %v", got)
	}
}

func TestInvokeBadURL(t *testing.T) {
	client := &Client{Model: "m", APIKey: "k", BaseURL: "http://bad url with spaces"}
	_, err := client.Invoke(t.Context(), Request{
		Input:     "x",
		Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
	})
	if err == nil || !strings.Contains(err.Error(), "bad request URL") {
		t.Fatalf("Invoke() error = %v, want a bad-URL error", err)
	}
}

func TestInvokeTransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	server.Close() // closed immediately: connection refused

	client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client()}
	_, err := client.Invoke(t.Context(), Request{
		Input:     "x",
		Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
	})
	if err == nil {
		t.Fatal("Invoke() expected a transport error")
	}
	if !errors.Is(err, lcerrors.ErrModelConnection) {
		t.Fatalf("Invoke() error = %v, want ErrModelConnection", err)
	}
}

func TestInvokeTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		_, _ = w.Write([]byte(`{"model": "m", "answers": []}`))
	}))
	defer server.Close()

	client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client(), Timeout: time.Millisecond}
	_, err := client.Invoke(t.Context(), Request{
		Input:     "x",
		Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
	})
	if err == nil {
		t.Fatal("Invoke() expected a timeout error")
	}
	if !errors.Is(err, lcerrors.ErrTimeout) {
		t.Fatalf("Invoke() error = %v, want ErrTimeout", err)
	}
}

func TestInvokeContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		_, _ = w.Write([]byte(`{"model": "m", "answers": []}`))
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client()}
	_, err := client.Invoke(ctx, Request{
		Input:     "x",
		Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
	})
	if err == nil {
		t.Fatal("Invoke() expected an error for a canceled context")
	}
}

func TestInvokeQuestionValidation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()

	tests := []struct {
		name     string
		question Question
	}{
		{"blank predicate instructions", Predicate{Instructions: ""}},
		{"blank choice instructions", Choice{Instructions: "", Choices: []ChoiceOption{{Value: "v"}}}},
		{"choice without options", Choice{Instructions: "i"}},
		{"blank score instructions", Score{Instructions: "", Levels: []ScoreLevel{L("a"), L("b")}}},
		{"score with one level", Score{Instructions: "i", Levels: []ScoreLevel{L("a")}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &Client{Model: "m", APIKey: "k", BaseURL: server.URL, HTTPClient: server.Client()}
			_, err := client.Invoke(t.Context(), Request{
				Input:     "x",
				Questions: map[string]Question{"q": tt.question},
			})
			if err == nil {
				t.Fatal("Invoke() expected a validation error")
			}
			if !errors.Is(err, lcerrors.ErrInvalidInput) {
				t.Fatalf("Invoke() error = %v, want ErrInvalidInput", err)
			}
		})
	}
}

func TestInvokeInputErrorPropagates(t *testing.T) {
	client := &Client{Model: "m", APIKey: "k", BaseURL: "http://unused.example.com"}
	human := messages.Message{
		Role:          messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{messages.AudioBlock{Base64: "x"}},
	}
	_, err := client.Invoke(t.Context(), Request{
		Input:     []any{human, "mix"},
		Questions: map[string]Question{"q": Predicate{Instructions: "i"}},
	})
	if err == nil || !errorsIsInvalidInput(err) {
		t.Fatalf("Invoke() error = %v, want ErrInvalidInput", err)
	}
}
