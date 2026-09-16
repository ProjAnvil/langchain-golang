package openai

import (
	"context"
	"fmt"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// Mirrors Python's BaseChatOpenAI default_params parallel_tool_calls
// (chat_models/base.py:1340-1350 exclude_if_none family): nil omits the
// field, a set value serializes on both the Responses and Chat Completions
// paths.

func parallelToolCallsModel(t *testing.T, baseURL string, chatCompletions bool, choice language.ToolChoice, parallel *bool) ChatModel {
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
	bound, err := model.BindToolsWithOptions([]coretools.Tool{tool}, language.BindToolsOptions{ToolChoice: choice, ParallelToolCalls: parallel})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	return bound.(ChatModel)
}

func TestParallelToolCallsSerializedBothAPIs(t *testing.T) {
	for _, tc := range []struct {
		name            string
		chatCompletions bool
		body            string
	}{
		{"responses", false, toolChoiceResponsesBody},
		{"chat completions", true, toolChoiceChatBody},
	} {
		for _, want := range []bool{false, true} {
			t.Run(tc.name+"/"+fmt.Sprint(want), func(t *testing.T) {
				server, got := toolChoiceServer(t, tc.body)
				value := want
				model := parallelToolCallsModel(t, server.URL, tc.chatCompletions, "", &value)
				if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
					t.Fatalf("Invoke: %v", err)
				}
				if (*got)["parallel_tool_calls"] != want {
					t.Fatalf("parallel_tool_calls = %v, want %v", (*got)["parallel_tool_calls"], want)
				}
			})
		}
	}
}

func TestParallelToolCallsOmittedWhenNil(t *testing.T) {
	for _, tc := range []struct {
		name            string
		chatCompletions bool
		body            string
	}{
		{"responses", false, toolChoiceResponsesBody},
		{"chat completions", true, toolChoiceChatBody},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, got := toolChoiceServer(t, tc.body)
			model := parallelToolCallsModel(t, server.URL, tc.chatCompletions, "", nil)
			if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			if _, present := (*got)["parallel_tool_calls"]; present {
				t.Fatalf("parallel_tool_calls must be omitted when nil, got %v", (*got)["parallel_tool_calls"])
			}
		})
	}
}

func TestParallelToolCallsCombinedWithNamedTool(t *testing.T) {
	// The other edge of the combination matrix: a named-tool choice and the
	// parallel flag must coexist (CC nests the function, Responses flattens).
	for _, tc := range []struct {
		name            string
		chatCompletions bool
		body            string
	}{
		{"responses", false, toolChoiceResponsesBody},
		{"chat completions", true, toolChoiceChatBody},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, got := toolChoiceServer(t, tc.body)
			no := false
			model := parallelToolCallsModel(t, server.URL, tc.chatCompletions, "GenerateUsername", &no)
			if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			choice, ok := (*got)["tool_choice"].(map[string]any)
			if !ok {
				t.Fatalf("tool_choice = %v, want object", (*got)["tool_choice"])
			}
			if choice["type"] != "function" {
				t.Fatalf("tool_choice type = %v, want function", choice["type"])
			}
			if (*got)["parallel_tool_calls"] != false {
				t.Fatalf("parallel_tool_calls = %v, want false", (*got)["parallel_tool_calls"])
			}
		})
	}
}

func TestParallelToolCallsCombinedWithToolChoice(t *testing.T) {
	// Combination matrix from the M0a spec: tool_choice and
	// parallel_tool_calls must coexist in one payload on both API paths
	// ("any" flattens to "required").
	for _, tc := range []struct {
		name            string
		chatCompletions bool
		body            string
	}{
		{"responses", false, toolChoiceResponsesBody},
		{"chat completions", true, toolChoiceChatBody},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, got := toolChoiceServer(t, tc.body)
			no := false
			model := parallelToolCallsModel(t, server.URL, tc.chatCompletions, language.ToolChoiceAny, &no)
			if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			if (*got)["tool_choice"] != "required" {
				t.Fatalf("tool_choice = %v, want required (any→required)", (*got)["tool_choice"])
			}
			if (*got)["parallel_tool_calls"] != false {
				t.Fatalf("parallel_tool_calls = %v, want false", (*got)["parallel_tool_calls"])
			}
		})
	}
}
