package anthropic

// Effort / reasoning_effort constructor option (Python #41536 superset,
// langchain-anthropic chat_models.py). Python declares reasoning_effort with
// alias "effort"; Go exposes it as WithEffort. Behavior mirrored:
//
//   - payload: the resolved level is emitted as output_config.effort
//     (chat_models.py:1970-1998);
//   - validation: the level must be one of the Literal values; an illegal level
//     errors with pydantic's exact text (chat_models.py:1523; Python raises at
//     construction, Go surfaces the same text at request-build since WithEffort
//     cannot fail);
//   - profile gate: when no explicit thinking is set and the model's profile
//     declares xhigh support, thinking defaults to adaptive+summarized
//     (chat_models.py:2005-2010);
//   - conflict: claude-opus-5* rejects disabled thinking at xhigh/max effort
//     (chat_models.py:1851-1863);
//   - getter: Effort() exposes the resolved level (Python's effort property
//     alias, chat_models.py:1611-1613).
//
// Models come from the materialized modelprofiles bundle: claude-opus-5 /
// claude-opus-4-7 carry xhigh; claude-opus-4-5-20251101 / claude-opus-4-6 do
// not; claude-sonnet-4-5 carries no effort keys at all.

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// effortModel builds a ChatModel against a request-capturing test server.
func effortModel(t *testing.T, model string, request *map[string]any) ChatModel {
	t.Helper()
	server := newTestServer(t, request)
	t.Cleanup(server.Close)
	return NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel(model),
	)
}

func TestEffortEmitsOutputConfig(t *testing.T) {
	var request map[string]any
	model := effortModel(t, "claude-opus-4-5-20251101", &request).WithEffort("high")
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	outputConfig, ok := request["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("missing output_config: %+v", request)
	}
	if outputConfig["effort"] != "high" {
		t.Fatalf("output_config.effort = %v, want high", outputConfig["effort"])
	}
	// Opus 4.5 has no xhigh level, so no adaptive-thinking default fires
	// (Python test_reasoning_effort_older_model_does_not_default_thinking).
	if _, ok := request["thinking"]; ok {
		t.Fatalf("thinking present, want absent: %+v", request)
	}
}

func TestEffortAllLevelsReachOutputConfig(t *testing.T) {
	// The Literal validation is model-independent (chat_models.py:1523): every
	// level is accepted regardless of the model's declared levels, which are
	// not enforced client-side.
	for _, level := range []string{"low", "medium", "high", "xhigh", "max"} {
		t.Run(level, func(t *testing.T) {
			var request map[string]any
			model := effortModel(t, "claude-opus-4-6", &request).WithEffort(level)
			if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
				t.Fatalf("invoke: %v", err)
			}
			outputConfig, ok := request["output_config"].(map[string]any)
			if !ok {
				t.Fatalf("missing output_config: %+v", request)
			}
			if outputConfig["effort"] != level {
				t.Fatalf("output_config.effort = %v, want %v", outputConfig["effort"], level)
			}
		})
	}
}

func TestEffortDefaultsAdaptiveThinking(t *testing.T) {
	// Python test_reasoning_effort_defaults_adaptive_thinking: on an
	// xhigh-capable model, effort pairs with adaptive thinking so the model
	// actually reasons at the requested level.
	var request map[string]any
	model := effortModel(t, "claude-opus-5", &request).WithEffort("low")
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	thinking, ok := request["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("missing thinking default: %+v", request)
	}
	if thinking["type"] != "adaptive" || thinking["display"] != "summarized" {
		t.Fatalf("thinking default = %v, want {type:adaptive display:summarized}", thinking)
	}
	outputConfig, ok := request["output_config"].(map[string]any)
	if !ok || outputConfig["effort"] != "low" {
		t.Fatalf("output_config = %v, want effort=low", request["output_config"])
	}
}

func TestEffortAdaptiveThinkingEnforcesSamplingConstraints(t *testing.T) {
	// Thinking present — including the effort-injected default — forces
	// temperature 1 and drops top_p/top_k, matching the explicit WithThinking
	// behavior (Anthropic rejects sampling params alongside thinking).
	var request map[string]any
	model := NewChatModel(
		modelconfig.WithBaseURL(newTestServer(t, &request).URL),
		modelconfig.WithModel("claude-opus-4-7"),
		modelconfig.WithTemperature(0.2),
		WithTopP(0.75),
		WithTopK(64),
	).WithEffort("high")
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if request["temperature"] != float64(1) {
		t.Fatalf("temperature = %v, want 1", request["temperature"])
	}
	if _, ok := request["top_p"]; ok {
		t.Fatalf("top_p should be omitted with thinking: %+v", request)
	}
	if _, ok := request["top_k"]; ok {
		t.Fatalf("top_k should be omitted with thinking: %+v", request)
	}
}

func TestEffortPreservesExplicitThinking(t *testing.T) {
	// Python test_reasoning_effort_preserves_explicit_construction_time_thinking:
	// an explicit thinking map is never clobbered by the adaptive default, even
	// on an xhigh-capable model.
	var request map[string]any
	model := effortModel(t, "claude-opus-4-7", &request).
		WithThinking(map[string]any{"type": "enabled", "budget_tokens": 10_000}).
		WithEffort("high")
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	thinking, ok := request["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("missing thinking: %+v", request)
	}
	if thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(10_000) {
		t.Fatalf("thinking = %v, want enabled budget 10000", thinking)
	}
	outputConfig, ok := request["output_config"].(map[string]any)
	if !ok || outputConfig["effort"] != "high" {
		t.Fatalf("output_config = %v, want effort=high", request["output_config"])
	}
}

func TestEffortInvalidLevelRejected(t *testing.T) {
	// Python rejects illegal levels at construction via pydantic's literal_error
	// (chat_models.py:1523); Go surfaces the same message at request-build.
	var request map[string]any
	model := effortModel(t, "claude-opus-4-5-20251101", &request).WithEffort("invalid")
	_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")})
	if err == nil {
		t.Fatal("invoke succeeded, want error")
	}
	const want = "Input should be 'max', 'xhigh', 'high', 'medium' or 'low'"
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("error = %q, want it to contain %q", err.Error(), want)
	}
}

func TestEffortOpus5RejectsDisabledThinkingAtHighEffort(t *testing.T) {
	// Python test_opus_5_rejects_disabled_thinking_at_high_effort: Opus 5
	// permits disabled thinking only through `high` effort.
	for _, effort := range []string{"xhigh", "max"} {
		t.Run(effort, func(t *testing.T) {
			var request map[string]any
			model := effortModel(t, "claude-opus-5", &request).
				WithThinking(map[string]any{"type": "disabled"}).
				WithEffort(effort)
			_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")})
			if err == nil {
				t.Fatal("invoke succeeded, want error")
			}
			for _, want := range []string{
				"`thinking={\"type\": \"disabled\"}` is not supported for claude-opus-5",
				"`output_config.effort='" + effort + "'`",
				"set effort to `high` or below",
			} {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error = %q, want it to contain %q", err.Error(), want)
				}
			}
		})
	}
}

func TestEffortOpus5AllowsDisabledThinkingAtHighEffort(t *testing.T) {
	// Python test_opus_5_allows_disabled_thinking_at_high_effort.
	var request map[string]any
	model := effortModel(t, "claude-opus-5", &request).
		WithThinking(map[string]any{"type": "disabled"}).
		WithEffort("high")
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	thinking, ok := request["thinking"].(map[string]any)
	if !ok || thinking["type"] != "disabled" {
		t.Fatalf("thinking = %v, want disabled", request["thinking"])
	}
	outputConfig, ok := request["output_config"].(map[string]any)
	if !ok || outputConfig["effort"] != "high" {
		t.Fatalf("output_config = %v, want effort=high", request["output_config"])
	}
}

func TestEffortGetter(t *testing.T) {
	// Python's `effort` property is a pure alias of `reasoning_effort`
	// (chat_models.py:1611-1613): it returns the explicitly configured level
	// or None. Go's Effort() returns "" when unset.
	if got := NewChatModel(modelconfig.WithModel("claude-opus-5")).Effort(); got != "" {
		t.Fatalf("Effort() = %q, want empty", got)
	}
	if got := NewChatModel(modelconfig.WithModel("claude-opus-5")).WithEffort("medium").Effort(); got != "medium" {
		t.Fatalf("Effort() = %q, want medium", got)
	}
}

func TestNoEffortOmitsOutputConfig(t *testing.T) {
	// Python never resolves the profile's reasoning_effort_default client-side
	// (the value documents API behavior only); without an explicit effort no
	// output_config is emitted even though the model's profile carries a
	// default.
	var request map[string]any
	model := effortModel(t, "claude-opus-4-7", &request)
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if _, ok := request["output_config"]; ok {
		t.Fatalf("output_config present without effort, want absent: %+v", request)
	}
}
