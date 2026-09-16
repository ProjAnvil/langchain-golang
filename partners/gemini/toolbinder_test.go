package gemini

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

func binderTestTools(t *testing.T) []tools.Tool {
	t.Helper()
	magic, err := tools.NewFunc("magic_function", "Apply a magic function.",
		schema.Object(map[string]schema.Schema{
			"input": schema.Integer("the input number"),
		}, "input"),
		func(context.Context, map[string]any) (tools.Result, error) {
			return tools.Result{Content: "ok"}, nil
		})
	if err != nil {
		t.Fatalf("new magic tool: %v", err)
	}
	weather, err := tools.NewFunc("get_weather", "Get the weather.",
		schema.Object(map[string]schema.Schema{
			"location": schema.String("the location"),
		}, "location"),
		func(context.Context, map[string]any) (tools.Result, error) {
			return tools.Result{Content: "ok"}, nil
		})
	if err != nil {
		t.Fatalf("new weather tool: %v", err)
	}
	return []tools.Tool{magic, weather}
}

// captureRequestModel wires a model against a server capturing the request
// body and answering with a minimal candidate.
func captureRequestModel(t *testing.T, captured *map[string]any) ChatModel {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		*captured = body
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`))
	}))
	t.Cleanup(server.Close)
	return NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
}

// toolConfigOf extracts the wire toolConfig.functionCallingConfig.
func toolConfigOf(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	toolConfig, _ := body["toolConfig"].(map[string]any)
	if toolConfig == nil {
		return nil
	}
	config, _ := toolConfig["functionCallingConfig"].(map[string]any)
	return config
}

// TestBindToolsDeclaresFunctions verifies bound tools serialize as one Tool
// with N typed function declarations (Python's _convert_to_genai_tools).
func TestBindToolsDeclaresFunctions(t *testing.T) {
	var body map[string]any
	model := captureRequestModel(t, &body)
	bound, err := model.BindTools(binderTestTools(t))
	if err != nil {
		t.Fatalf("BindTools: %v", err)
	}
	if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	toolsList, _ := body["tools"].([]any)
	if len(toolsList) != 1 {
		t.Fatalf("expected a single Tool, got %v", toolsList)
	}
	declarations, _ := toolsList[0].(map[string]any)["functionDeclarations"].([]any)
	if len(declarations) != 2 {
		t.Fatalf("declarations: %v", declarations)
	}
	first := declarations[0].(map[string]any)
	if first["name"] != "magic_function" || first["description"] != "Apply a magic function." {
		t.Errorf("declaration: %v", first)
	}
	parameters, _ := first["parameters"].(map[string]any)
	if parameters["type"] != "OBJECT" {
		t.Errorf("parameters type: %v (genai enums are uppercase on the wire)", parameters)
	}
	if required, _ := parameters["required"].([]any); len(required) != 1 || required[0] != "input" {
		t.Errorf("parameters required: %v", parameters["required"])
	}
	properties, _ := parameters["properties"].(map[string]any)
	input, _ := properties["input"].(map[string]any)
	if input["type"] != "INTEGER" {
		t.Errorf("input property: %v", input)
	}
	if toolConfigOf(t, body) != nil {
		t.Errorf("unexpected toolConfig without a choice: %v", body["toolConfig"])
	}
}

// TestBindToolsToolChoiceModes pins the Python _tool_choice_to_tool_config
// mapping (_function_utils.py:700-770).
func TestBindToolsToolChoiceModes(t *testing.T) {
	cases := []struct {
		name      string
		choice    language.ToolChoice
		wantMode  string
		wantNames []string
	}{
		{"auto", language.ToolChoiceAuto, "AUTO", nil},
		{"any pins all names", language.ToolChoiceAny, "ANY", []string{"magic_function", "get_weather"}},
		{"named tool", language.ToolChoice("magic_function"), "ANY", []string{"magic_function"}},
		{"none", language.ToolChoiceNone, "NONE", nil},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			var body map[string]any
			model := captureRequestModel(t, &body)
			var binder language.ToolBinder = model
			bound, err := binder.BindToolsWithOptions(binderTestTools(t), language.BindToolsOptions{
				ToolChoice: testCase.choice,
			})
			if err != nil {
				t.Fatalf("BindToolsWithOptions: %v", err)
			}
			if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
				t.Fatalf("invoke: %v", err)
			}
			config := toolConfigOf(t, body)
			if config == nil {
				t.Fatalf("toolConfig missing: %v", body)
			}
			if config["mode"] != testCase.wantMode {
				t.Errorf("mode: got %v want %v", config["mode"], testCase.wantMode)
			}
			names, _ := config["allowedFunctionNames"].([]any)
			switch {
			case testCase.wantNames == nil:
				if len(names) != 0 {
					t.Errorf("allowedFunctionNames: %v want none", names)
				}
			default:
				if len(names) != len(testCase.wantNames) {
					t.Fatalf("allowedFunctionNames: %v want %v", names, testCase.wantNames)
				}
				for i, want := range testCase.wantNames {
					if names[i] != want {
						t.Errorf("allowedFunctionNames[%d]: got %v want %v", i, names[i], want)
					}
				}
			}
		})
	}
}

// TestBindToolsZeroOptionsKeepsToolConfig verifies a zero-value
// BindToolsOptions call keeps any prior WithToolConfig (the ToolBinder
// contract: BindTools == BindToolsWithOptions(tools, {})).
func TestBindToolsZeroOptionsKeepsToolConfig(t *testing.T) {
	var body map[string]any
	model := captureRequestModel(t, &body)
	configured := model.WithToolConfig(toolConfigFromCore(language.ToolChoiceAny, binderTestTools(t)))

	var binder language.ToolBinder = configured
	bound, err := binder.BindToolsWithOptions(binderTestTools(t), language.BindToolsOptions{})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	config := toolConfigOf(t, body)
	if config == nil || config["mode"] != "ANY" {
		t.Fatalf("toolConfig not retained: %v", config)
	}
}

// TestBindToolsParallelToolCallsIgnored documents that the Gemini API has no
// parallel_tool_calls payload field: the option is accepted and dropped.
func TestBindToolsParallelToolCallsIgnored(t *testing.T) {
	var body map[string]any
	model := captureRequestModel(t, &body)
	var binder language.ToolBinder = model
	disabled := false
	bound, err := binder.BindToolsWithOptions(binderTestTools(t), language.BindToolsOptions{
		ParallelToolCalls: &disabled,
	})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if config := toolConfigOf(t, body); config != nil {
		t.Errorf("ParallelToolCalls must be dropped, not synthesized into toolConfig: %v", config)
	}
}
