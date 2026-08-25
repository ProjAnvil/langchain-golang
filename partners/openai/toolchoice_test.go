package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// Mirrors libs/partners/openai/tests/unit_tests/chat_models/test_base.py
// ::test_bind_tools_tool_choice (tool_choice parametrization) plus the
// Responses-API flattening in chat_models/base.py (_get_request_payload).

const (
	toolChoiceResponsesBody = `{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`
	toolChoiceChatBody      = `{"id":"chatcmpl-1","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`
)

// toolChoiceServer captures the decoded request body and replies with body.
func toolChoiceServer(t *testing.T, body string) (*httptest.Server, *map[string]any) {
	t.Helper()
	got := map[string]any{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server, &got
}

func toolChoiceBoundModel(t *testing.T, baseURL string, chatCompletions bool) ChatModel {
	t.Helper()
	tool, err := coretools.FromFunc("GenerateUsername", "Get a username.", func(ctx context.Context, args struct{ Name string }) (coretools.Result, error) {
		return coretools.Result{Content: args.Name}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	model := NewChatModel(
		modelconfig.WithBaseURL(baseURL),
		modelconfig.WithModel("gpt-test"),
	)
	if chatCompletions {
		model = model.WithChatCompletions()
	}
	bound, err := model.BindTools([]coretools.Tool{tool})
	if err != nil {
		t.Fatal(err)
	}
	return bound.(ChatModel)
}

func TestToolChoiceStringModesResponsesAPI(t *testing.T) {
	// Python parametrizes "any" (mapped to "required" by bind_tools), "none",
	// "auto", "required". ToolChoiceRequired() covers the "any"/True mapping.
	cases := []struct {
		name   string
		choice ToolChoice
		want   string
	}{
		{"auto", ToolChoiceAuto(), "auto"},
		{"none", ToolChoiceNone(), "none"},
		{"required (python any/True)", ToolChoiceRequired(), "required"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server, got := toolChoiceServer(t, toolChoiceResponsesBody)
			model := toolChoiceBoundModel(t, server.URL, false).WithToolChoice(tc.choice)
			if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			if (*got)["tool_choice"] != tc.want {
				t.Fatalf("tool_choice = %v, want %q", (*got)["tool_choice"], tc.want)
			}
		})
	}
}

func TestToolChoiceStringModesChatCompletions(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceChatBody)
	model := toolChoiceBoundModel(t, server.URL, true).WithToolChoice(ToolChoiceAuto())
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if (*got)["tool_choice"] != "auto" {
		t.Fatalf("tool_choice = %v, want auto", (*got)["tool_choice"])
	}
}

func TestToolChoiceFunctionResponsesAPIFlattens(t *testing.T) {
	// base.py:4331-4341: {"type":"function","function":{"name":X}} ->
	// {"type":"function","name":X} on the Responses API.
	server, got := toolChoiceServer(t, toolChoiceResponsesBody)
	model := toolChoiceBoundModel(t, server.URL, false).WithToolChoice(ToolChoiceFunction("MakeASandwich"))
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	choice, ok := (*got)["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %v, want object", (*got)["tool_choice"])
	}
	if choice["type"] != "function" || choice["name"] != "MakeASandwich" {
		t.Fatalf("tool_choice = %v, want {type:function name:MakeASandwich}", choice)
	}
	if _, nested := choice["function"]; nested {
		t.Fatalf("Chat Completions nesting leaked into Responses API: %v", choice)
	}
}

func TestToolChoiceFunctionChatCompletionsNested(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceChatBody)
	model := toolChoiceBoundModel(t, server.URL, true).WithToolChoice(ToolChoiceFunction("MakeASandwich"))
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	choice, ok := (*got)["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %v, want object", (*got)["tool_choice"])
	}
	fn, ok := choice["function"].(map[string]any)
	if !ok || choice["type"] != "function" || fn["name"] != "MakeASandwich" {
		t.Fatalf("tool_choice = %v, want {type:function function:{name:MakeASandwich}}", choice)
	}
}

func TestToolChoiceRawPassthrough(t *testing.T) {
	// Python dict form passes through (e.g. WellKnownTools {"type":"file_search"}).
	raw := map[string]any{"type": "file_search"}
	for _, chatCompletions := range []bool{false, true} {
		body := toolChoiceResponsesBody
		if chatCompletions {
			body = toolChoiceChatBody
		}
		server, got := toolChoiceServer(t, body)
		model := toolChoiceBoundModel(t, server.URL, chatCompletions).WithToolChoice(ToolChoiceRaw(raw))
		if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
			t.Fatalf("Invoke: %v", err)
		}
		choice, ok := (*got)["tool_choice"].(map[string]any)
		if !ok || choice["type"] != "file_search" {
			t.Fatalf("chatCompletions=%v tool_choice = %v, want {type:file_search}", chatCompletions, (*got)["tool_choice"])
		}
	}
}

func TestToolChoiceOmittedByDefault(t *testing.T) {
	// Python False/None: no tool_choice key in the payload.
	server, got := toolChoiceServer(t, toolChoiceResponsesBody)
	model := toolChoiceBoundModel(t, server.URL, false)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if _, present := (*got)["tool_choice"]; present {
		t.Fatalf("tool_choice must be omitted when unset, got %v", (*got)["tool_choice"])
	}
}
