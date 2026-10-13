package agents

// End-to-end composition tests: the Decisions/TypeSafe middleware wired into
// a real CreateAgent runtime (model node + tool node + middleware discovery),
// complementing the unit-level middleware tests in the partner packages.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	openaidecisions "github.com/projanvil/langchain-golang/partners/openai/decisions"
	openaimiddleware "github.com/projanvil/langchain-golang/partners/openai/middleware"
	"github.com/projanvil/langchain-golang/partners/typesafe"
	typesafemiddleware "github.com/projanvil/langchain-golang/partners/typesafe/middleware"
)

// decisionsServer answers every POST with the supplied answer entries
// (openai decisions wire shape: answers is a list keyed by name).
func decisionsServer(t *testing.T, entries ...map[string]any) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode decisions request: %v", err)
		}
		if _, ok := payload["questions"]; !ok {
			t.Error("decisions request missing questions")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "gpt-6-luna",
			"answers": entries,
			"usage":   map[string]any{"input_tokens": 3, "output_tokens": 2},
		})
	}))
	t.Cleanup(server.Close)
	return server
}

// typeSafeServer answers every POST with the supplied answers object
// (typesafe wire shape: answers is an object keyed by question id).
func typeSafeServer(t *testing.T, answers map[string]any) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/systemone") {
			t.Errorf("unexpected typesafe path %q", r.URL.Path)
		}
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode typesafe request: %v", err)
		}
		if _, ok := payload["questions"]; !ok {
			t.Error("typesafe request missing questions")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "jev-latest",
			"answers": answers,
			"usage":   map[string]any{"input_tokens": 3, "output_tokens": 2},
		})
	}))
	t.Cleanup(server.Close)
	return server
}

func decisionTestClient(server *httptest.Server) *openaidecisions.Client {
	return &openaidecisions.Client{
		Model:      "gpt-6-luna",
		APIKey:     "test-key",
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	}
}

func classifierForTest(server *httptest.Server) *typesafe.Classifier {
	return &typesafe.Classifier{
		Model:      "jev-latest",
		APIKey:     "test-key",
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	}
}

func riskyToolCall(name string) messages.Message {
	return messages.Message{
		Role: messages.RoleAI,
		ToolCalls: []messages.ToolCall{
			{ID: "call_1", Name: name, Args: map[string]any{"tool_input": "hi"}},
		},
	}
}

func TestCreateAgentOpenAIAutoModeBlocksRiskyToolEndToEnd(t *testing.T) {
	server := decisionsServer(t, map[string]any{
		"name": "is_risky", "type": "predicate", "probability": 0.91,
	})
	model := &sequenceModel{responses: []messages.Message{
		riskyToolCall("echo"),
		messages.AI("recovered"),
	}}
	auto := &openaimiddleware.AutoMode{
		Tools: []any{"echo"},
		Model: decisionTestClient(server),
	}
	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)}, WithAgentMiddleware(auto))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	var blocked strings.Builder
	for _, m := range out {
		blocked.WriteString(m.Content)
		blocked.WriteString(" ")
	}
	if strings.Contains(blocked.String(), "echo:") {
		t.Errorf("echo tool should not have executed, output: %q", blocked.String())
	}
	want := "The tool call `echo` was blocked because it was classified as risky (probability: 0.91). The tool was not executed."
	if !strings.Contains(blocked.String(), want) {
		t.Errorf("blocked notice missing, output: %q", blocked.String())
	}
	if !strings.Contains(blocked.String(), "recovered") {
		t.Errorf("model should still finish after the block, output: %q", blocked.String())
	}
}

func TestCreateAgentOpenAIModelRouterSwapsModelEndToEnd(t *testing.T) {
	server := decisionsServer(t, map[string]any{
		"name":         "model_route",
		"type":         "choice",
		"choice":       "routed",
		"probabilities": []map[string]any{{"value": "routed", "probability": 1.0}},
		"confidence":   1.0,
	})
	base := &sequenceModel{responses: []messages.Message{messages.AI("base-model-answer")}}
	routed := &sequenceModel{responses: []messages.Message{messages.AI("routed-model-answer")}}
	router := &openaimiddleware.ModelRouter{
		Choices: map[string]openaimiddleware.ModelChoice{
			"routed": {Model: routed, Criteria: "Everything."},
		},
		Instructions: "Pick the model.",
		Model:        decisionTestClient(server),
	}
	agent, err := CreateAgent(base, nil, WithAgentMiddleware(router))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) == 0 || !strings.Contains(out[len(out)-1].Content, "routed-model-answer") {
		t.Errorf("routed model was not used, final: %#v", out)
	}
	if len(base.invocations) != 0 {
		t.Errorf("base model must not be invoked when a route is set, got %d calls", len(base.invocations))
	}
}

func TestCreateAgentTypeSafeAutoModeBlocksRiskyToolEndToEnd(t *testing.T) {
	server := typeSafeServer(t, map[string]any{
		"is_risky": map[string]any{"type": "noul", "noul": 0.9},
	})
	model := &sequenceModel{responses: []messages.Message{
		riskyToolCall("echo"),
		messages.AI("recovered"),
	}}
	auto := &typesafemiddleware.AutoModeMiddleware{
		Tools:      []any{"echo"},
		Classifier: classifierForTest(server),
	}
	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)}, WithAgentMiddleware(auto))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	var all strings.Builder
	for _, m := range out {
		all.WriteString(m.Content)
		all.WriteString(" ")
	}
	if strings.Contains(all.String(), "echo:") {
		t.Errorf("echo tool should not have executed, output: %q", all.String())
	}
	want := "The tool call `echo` was blocked because it was classified as risky (probability: 0.90). The tool was not executed."
	if !strings.Contains(all.String(), want) {
		t.Errorf("blocked notice missing, output: %q", all.String())
	}
}

func TestCreateAgentTypeSafeModelRouterSwapsModelEndToEnd(t *testing.T) {
	server := typeSafeServer(t, map[string]any{
		"model_route": map[string]any{
			"type":         "choice",
			"choice":       "routed",
			"probabilities": map[string]any{"routed": 1.0},
			"confidence":   1.0,
		},
	})
	base := &sequenceModel{responses: []messages.Message{messages.AI("base-model-answer")}}
	routed := &sequenceModel{responses: []messages.Message{messages.AI("routed-model-answer")}}
	router, err := typesafemiddleware.NewModelRouterMiddleware(
		map[string]typesafemiddleware.ModelChoice{
			"routed": {Model: routed, Criteria: "Everything."},
		},
		"Pick the model.",
		typesafemiddleware.WithRouterClassifier(classifierForTest(server)),
	)
	if err != nil {
		t.Fatalf("new router: %v", err)
	}
	agent, err := CreateAgent(base, nil, WithAgentMiddleware(router))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) == 0 || !strings.Contains(out[len(out)-1].Content, "routed-model-answer") {
		t.Errorf("routed model was not used, final: %#v", out)
	}
	if len(base.invocations) != 0 {
		t.Errorf("base model must not be invoked when a route is set, got %d calls", len(base.invocations))
	}
}
