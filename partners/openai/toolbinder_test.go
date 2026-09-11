package openai

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// toolBinderBoundModel binds one tool via BindToolsWithOptions against the
// given test server, for asserting the serialized HTTP tool_choice.
func toolBinderBoundModel(t *testing.T, baseURL string, chatCompletions bool, choice language.ToolChoice) ChatModel {
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
	bound, err := model.BindToolsWithOptions([]coretools.Tool{tool}, language.BindToolsOptions{ToolChoice: choice})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	return bound.(ChatModel)
}

// TestBindToolsWithOptionsMapsAnyToRequired proves the language.ToolChoice
// surface maps onto the existing provider serialization: "any" (the agent's
// ToolStrategy forcing value) becomes OpenAI's "required" on BOTH APIs,
// mirroring Python bind_tools' "any"/True → "required".
func TestBindToolsWithOptionsMapsAnyToRequired(t *testing.T) {
	for _, tc := range []struct {
		name            string
		chatCompletions bool
		body            string
	}{
		{"responses api", false, toolChoiceResponsesBody},
		{"chat completions api", true, toolChoiceChatBody},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, got := toolChoiceServer(t, tc.body)
			model := toolBinderBoundModel(t, server.URL, tc.chatCompletions, language.ToolChoiceAny)
			if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			if (*got)["tool_choice"] != "required" {
				t.Fatalf("tool_choice = %v, want \"required\"", (*got)["tool_choice"])
			}
		})
	}
}

// TestBindToolsWithOptionsStringModes covers auto/none passthrough.
func TestBindToolsWithOptionsStringModes(t *testing.T) {
	for _, mode := range []language.ToolChoice{language.ToolChoiceAuto, language.ToolChoiceNone} {
		server, got := toolChoiceServer(t, toolChoiceResponsesBody)
		model := toolBinderBoundModel(t, server.URL, false, mode)
		if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
			t.Fatalf("Invoke: %v", err)
		}
		if (*got)["tool_choice"] != string(mode) {
			t.Fatalf("tool_choice = %v, want %q", (*got)["tool_choice"], string(mode))
		}
	}
}

// TestBindToolsWithOptionsNamedToolFlattens: a named tool (non-reserved
// string) becomes {"type":"function","name":X} on the Responses API and the
// nested Chat-Completions form there.
func TestBindToolsWithOptionsNamedToolFlattens(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceResponsesBody)
	model := toolBinderBoundModel(t, server.URL, false, language.ToolChoice("GenerateUsername"))
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	choice, ok := (*got)["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %v, want object", (*got)["tool_choice"])
	}
	if choice["type"] != "function" || choice["name"] != "GenerateUsername" {
		t.Fatalf("tool_choice = %v, want {type:function name:GenerateUsername}", choice)
	}
	if _, nested := choice["function"]; nested {
		t.Fatalf("Chat Completions nesting leaked into Responses API: %v", choice)
	}
}

// TestBindToolsWithOptionsZeroOmitsToolChoice: zero-value options keep the
// BindTools contract — no tool_choice key in the payload.
func TestBindToolsWithOptionsZeroOmitsToolChoice(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceResponsesBody)
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	)
	bound, err := model.BindToolsWithOptions(nil, language.BindToolsOptions{})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	if _, err := bound.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if _, present := (*got)["tool_choice"]; present {
		t.Fatalf("tool_choice must be omitted for zero-value options, got %v", (*got)["tool_choice"])
	}
}
