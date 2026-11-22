package agents

// End-to-end composition tests: the Decisions/TypeSafe middleware wired into
// a real CreateAgent runtime (model node + tool node + middleware discovery),
// complementing the unit-level middleware tests in the partner packages.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/decisions"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
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
		"name":          "model_route",
		"type":          "choice",
		"choice":        "routed",
		"probabilities": []map[string]any{{"value": "routed", "probability": 1.0}},
		"confidence":    1.0,
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
			"type":          "choice",
			"choice":        "routed",
			"probabilities": map[string]any{"routed": 1.0},
			"confidence":    1.0,
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

// stubDecisionModel answers every Invoke with canned answers (core
// decisions.DecisionModel), for the generic middleware end-to-end tests.
type stubDecisionModel struct {
	answers map[string]decisions.Answer
	err     error
}

func (s *stubDecisionModel) Invoke(_ context.Context, _ decisions.Request, _ ...runnables.Option) (decisions.Response, error) {
	return decisions.Response{Model: "stub", Answers: s.answers}, s.err
}

// TestCreateAgentGenericModelRouterRoutesEndToEnd ports the upstream
// test_agent_routes_using_latest_human_message: the router classifies the
// latest human message, the selected model answers, and the full routing
// answer lands in final state (#41766).
func TestCreateAgentGenericModelRouterRoutesEndToEnd(t *testing.T) {
	defaultModel := &sequenceModel{responses: []messages.Message{messages.AI("default")}}
	fast := &sequenceModel{responses: []messages.Message{messages.AI("fast response")}}
	classifier := &stubDecisionModel{answers: map[string]decisions.Answer{
		"model_route": decisions.ChoiceAnswer{
			Choice:        "fast",
			Probabilities: map[string]float64{"fast": 0.9, "powerful": 0.1},
			Confidence:    0.9,
		},
	}}
	router, err := agentsmw.NewModelRouterMiddleware(classifier, map[string]agentsmw.ModelChoice{
		"fast":     {Model: fast, Criteria: "Simple tasks."},
		"powerful": {Model: &sequenceModel{responses: []messages.Message{messages.AI("powerful response")}}, Criteria: "Complex tasks."},
	}, "Choose the least costly model suited to the task.")
	if err != nil {
		t.Fatalf("router: %v", err)
	}
	agent, err := CreateAgent(defaultModel, nil, WithAgentMiddleware(router))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{
		messages.Human("earlier task"),
		messages.AI("ready"),
		messages.Human("update the readme"),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	msgs, _ := values["messages"].([]messages.Message)
	if len(msgs) == 0 || msgs[len(msgs)-1].Content != "fast response" {
		t.Fatalf("final message: %#v", msgs)
	}
	if len(fast.invocations) != 1 || len(defaultModel.invocations) != 0 {
		t.Fatalf("routing: fast=%d default=%d", len(fast.invocations), len(defaultModel.invocations))
	}
	route, ok := values["model_route"].(map[string]any)
	if !ok || route["choice"] != "fast" || route["confidence"] != 0.9 {
		t.Fatalf("model_route state: %#v", values["model_route"])
	}
}

// TestCreateAgentGenericAutoModeEndToEnd ports the upstream
// test_agent_executes_or_blocks_calls_by_probability: probability 0.2
// executes the tool, probability 0.5 blocks with an error tool message
// (#41766).
func TestCreateAgentGenericAutoModeEndToEnd(t *testing.T) {
	cases := []struct {
		probability float64
		wantBlocked bool
	}{
		{probability: 0.2, wantBlocked: false},
		{probability: 0.5, wantBlocked: true},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("probability_%.1f", tc.probability), func(t *testing.T) {
			var toolRuns int
			model := &sequenceModel{responses: []messages.Message{
				{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
				messages.AI("done"),
			}}
			classifier := &stubDecisionModel{answers: map[string]decisions.Answer{
				agentsmw.AutoModeQuestionID: decisions.NoulAnswer{Noul: tc.probability},
			}}
			auto, err := agentsmw.NewAutoModeMiddleware(classifier, []any{"echo"})
			if err != nil {
				t.Fatalf("auto mode: %v", err)
			}
			agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)}, WithAgentMiddleware(auto))
			if err != nil {
				t.Fatalf("create agent: %v", err)
			}
			values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("go")})
			if err != nil {
				t.Fatalf("invoke: %v", err)
			}
			msgs, _ := values["messages"].([]messages.Message)
			var toolMsg *messages.Message
			for i := range msgs {
				if msgs[i].Role == messages.RoleTool {
					toolMsg = &msgs[i]
				}
			}
			if toolMsg == nil {
				t.Fatal("no tool message")
			}
			if tc.wantBlocked {
				if toolRuns != 0 || toolMsg.AdditionalKwargs["status"] != "error" {
					t.Fatalf("blocked: runs=%d msg=%#v", toolRuns, toolMsg)
				}
				if !strings.Contains(toolMsg.Content, "was blocked because it was classified as risky") {
					t.Fatalf("blocked content: %q", toolMsg.Content)
				}
			} else if toolRuns != 1 || toolMsg.AdditionalKwargs["status"] == "error" {
				t.Fatalf("allowed: runs=%d msg=%#v", toolRuns, toolMsg)
			}
		})
	}
}
