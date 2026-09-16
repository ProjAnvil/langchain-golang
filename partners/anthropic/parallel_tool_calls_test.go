package anthropic

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// bindWithParallel mirrors bindWithToolChoice (toolbinder_test.go) plus the
// parallel flag. Anthropic expresses parallel_tool_calls as
// tool_choice.disable_parallel_tool_use (inverted), synthesizing a default
// {"type":"auto"} tool_choice when none is set — mirroring
// langchain-anthropic's payload assembly.

func bindWithParallel(t *testing.T, choice language.ToolChoice, parallel *bool) map[string]any {
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
	bound, err := model.BindToolsWithOptions([]tools.Tool{tool}, language.BindToolsOptions{ToolChoice: choice, ParallelToolCalls: parallel})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	return request
}

func TestParallelToolCallsSynthesizesDisableFlag(t *testing.T) {
	yes, no := true, false
	cases := []struct {
		name       string
		choice     language.ToolChoice
		parallel   *bool
		wantChoice map[string]any
	}{
		{"disable with auto choice", language.ToolChoiceAuto, &no,
			map[string]any{"type": "auto", "disable_parallel_tool_use": true}},
		{"disable without choice", "", &no,
			map[string]any{"type": "auto", "disable_parallel_tool_use": true}},
		{"enable without choice", "", &yes,
			map[string]any{"type": "auto", "disable_parallel_tool_use": false}},
		{"disable with named tool", "get_weather", &no,
			map[string]any{"type": "tool", "name": "get_weather", "disable_parallel_tool_use": true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := bindWithParallel(t, tc.choice, tc.parallel)
			got, ok := request["tool_choice"].(map[string]any)
			if !ok {
				t.Fatalf("tool_choice = %v, want object %v", request["tool_choice"], tc.wantChoice)
			}
			for k, v := range tc.wantChoice {
				if got[k] != v {
					t.Fatalf("tool_choice[%q] = %v, want %v (full: %v)", k, got[k], v, got)
				}
			}
		})
	}
}

func TestParallelToolCallsNilLeavesToolChoiceUntouched(t *testing.T) {
	request := bindWithParallel(t, language.ToolChoiceAuto, nil)
	got, ok := request["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %v, want object", request["tool_choice"])
	}
	if _, present := got["disable_parallel_tool_use"]; present {
		t.Fatalf("disable_parallel_tool_use must be absent when parallel unset: %v", got)
	}
}
