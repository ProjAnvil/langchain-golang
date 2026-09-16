package anthropic

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// bindWithToolChoice binds one tool via BindToolsWithOptions against a test
// server, returning the captured request body for assertions.
func bindWithToolChoice(t *testing.T, choice language.ToolChoice) map[string]any {
	t.Helper()
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	tool, err := tools.NewFunc(
		"get_weather",
		"gets weather",
		schema.Object(map[string]schema.Schema{
			"location": schema.String("location"),
		}, "location"),
		func(_ context.Context, _ map[string]any) (tools.Result, error) {
			return tools.Result{Content: "sunny"}, nil
		},
	)
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
	)
	bound, err := model.BindToolsWithOptions([]tools.Tool{tool}, language.BindToolsOptions{ToolChoice: choice})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	return request
}

// TestBindToolsWithOptionsAny mirrors Python's anthropic bind_tools
// tool_choice conversion: "any" → {"type":"any"} (agent ToolStrategy forcing).
func TestBindToolsWithOptionsAny(t *testing.T) {
	request := bindWithToolChoice(t, language.ToolChoiceAny)
	choice, ok := request["tool_choice"].(map[string]any)
	if !ok || choice["type"] != "any" {
		t.Fatalf("tool_choice: %+v", request["tool_choice"])
	}
}

// TestBindToolsWithOptionsAuto: "auto" → {"type":"auto"}.
func TestBindToolsWithOptionsAuto(t *testing.T) {
	request := bindWithToolChoice(t, language.ToolChoiceAuto)
	choice, ok := request["tool_choice"].(map[string]any)
	if !ok || choice["type"] != "auto" {
		t.Fatalf("tool_choice: %+v", request["tool_choice"])
	}
}

// TestBindToolsWithOptionsNamedTool: a named tool → {"type":"tool","name":X}.
func TestBindToolsWithOptionsNamedTool(t *testing.T) {
	request := bindWithToolChoice(t, language.ToolChoice("get_weather"))
	choice, ok := request["tool_choice"].(map[string]any)
	if !ok || choice["type"] != "tool" || choice["name"] != "get_weather" {
		t.Fatalf("tool_choice: %+v", request["tool_choice"])
	}
}

// TestBindToolsWithOptionsZeroOmitsToolChoice: zero-value options keep the
// BindTools contract — no tool_choice key.
func TestBindToolsWithOptionsZeroOmitsToolChoice(t *testing.T) {
	request := bindWithToolChoice(t, "")
	if _, present := request["tool_choice"]; present {
		t.Fatalf("tool_choice must be omitted for zero-value options, got %v", request["tool_choice"])
	}
}

// TestBindToolsWithOptionsNoneUnsupported: the Messages API has no "none"
// tool_choice; the adapter must fail loudly instead of silently misrouting.
func TestBindToolsWithOptionsNoneUnsupported(t *testing.T) {
	model := NewChatModel()
	_, err := model.BindToolsWithOptions(nil, language.BindToolsOptions{ToolChoice: language.ToolChoiceNone})
	if err == nil {
		t.Fatal("expected unsupported tool_choice error for \"none\"")
	}
	if !strings.Contains(err.Error(), "none") {
		t.Fatalf("error should mention the unsupported mode: %v", err)
	}
}
