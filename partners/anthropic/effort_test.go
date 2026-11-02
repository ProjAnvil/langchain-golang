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

// --- New-model thinking validations (chat_models.py:1812-1849): the model
// set is hardcoded startswith prefixes, not profile lookups —
// claude-opus-5*, claude-fable-5*, claude-sonnet-5-5*. ---

// TestEffortRejectsManualEnabledThinkingOnNewModels mirrors Python
// test_opus_5_rejects_manual_thinking_at_all_effort_levels and
// test_fable_5_1_rejects_manual_thinking (test_chat_models.py:5054,5177):
// opus-5* / fable-5* / sonnet-5-5* reject thinking={"type":"enabled"} at
// every effort level and with no effort at all.
func TestEffortRejectsManualEnabledThinkingOnNewModels(t *testing.T) {
	for _, modelID := range []string{"claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"} {
		for _, effort := range []string{"", "low", "medium", "high", "xhigh", "max"} {
			t.Run(modelID+"/"+effort, func(t *testing.T) {
				var request map[string]any
				model := effortModel(t, modelID, &request).
					WithThinking(map[string]any{"type": "enabled", "budget_tokens": 1_024})
				if effort != "" {
					model = model.WithEffort(effort)
				}
				_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")})
				if err == nil {
					t.Fatal("invoke succeeded, want error")
				}
				for _, want := range []string{
					"`thinking={\"type\": \"enabled\", \"budget_tokens\": ...}` is not supported for " + modelID,
					"use adaptive thinking and `output_config.effort` instead",
				} {
					if !strings.Contains(err.Error(), want) {
						t.Fatalf("error %q missing %q", err.Error(), want)
					}
				}
			})
		}
	}
}

// TestEffortFableSonnet55RejectDisabledThinking mirrors Python's fable/sonnet
// disabled-thinking rejections (chat_models.py:1832-1837; Python cases
// test_fable_5_1_rejects_unsupported_parameters and test_sonnet55
// test_invalid_configuration): fable-5* and sonnet-5-5* cannot disable
// thinking — adaptive thinking is their only mode, so the remedy is to omit
// the parameter entirely.
func TestEffortFableSonnet55RejectDisabledThinking(t *testing.T) {
	for _, modelID := range []string{"claude-fable-5-1", "claude-sonnet-5-5"} {
		t.Run(modelID, func(t *testing.T) {
			var request map[string]any
			model := effortModel(t, modelID, &request).
				WithThinking(map[string]any{"type": "disabled"})
			_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")})
			if err == nil {
				t.Fatal("invoke succeeded, want error")
			}
			for _, want := range []string{
				"`thinking={\"type\": \"disabled\"}` is not supported for " + modelID,
				"omit `thinking` to use adaptive thinking",
			} {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error %q missing %q", err.Error(), want)
				}
			}
		})
	}
}

// TestEffortOpus5DisabledStillAllowedWithoutEffort guards the reconciled rule
// set: opus-5* rejects disabled thinking only at xhigh/max effort
// (chat_models.py:1851-1863); with no effort set, disabled stays legal — the
// unconditional disabled rejection belongs to fable-5*/sonnet-5-5* only.
func TestEffortOpus5DisabledStillAllowedWithoutEffort(t *testing.T) {
	var request map[string]any
	model := effortModel(t, "claude-opus-5", &request).
		WithThinking(map[string]any{"type": "disabled"})
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if thinking, _ := request["thinking"].(map[string]any); thinking["type"] != "disabled" {
		t.Fatalf("thinking = %v, want disabled", request["thinking"])
	}
}

// TestEffortNonPinnedThinkingModesAllowedOnNewModels guards the positive
// scope: only enabled/disabled are pinned; adaptive and between_tools stay
// valid on all three families (Python test_between_tools).
func TestEffortNonPinnedThinkingModesAllowedOnNewModels(t *testing.T) {
	for _, modelID := range []string{"claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"} {
		for _, thinkingType := range []string{"adaptive", "between_tools"} {
			t.Run(modelID+"/"+thinkingType, func(t *testing.T) {
				var request map[string]any
				model := effortModel(t, modelID, &request).
					WithThinking(map[string]any{"type": thinkingType})
				if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
					t.Fatalf("invoke: %v", err)
				}
				thinking, _ := request["thinking"].(map[string]any)
				if thinking["type"] != thinkingType {
					t.Fatalf("thinking = %v, want type=%s", thinking, thinkingType)
				}
			})
		}
	}
}

// --- Constructor output_config dict (chat_models.py:1495, 1970-1998): merged
// ahead of the effort-derived key; WithEffort's level wins over a constructor
// effort key. Keys are not validated client-side (Python's field is a plain
// dict); documented keys are effort / format / task_budget. ---

func TestOutputConfigEmitsConstructorDict(t *testing.T) {
	// Python test_output_config_top_level_field (test_chat_models.py:5761):
	// the constructor dict reaches the wire alongside the effort default.
	var request map[string]any
	model := effortModel(t, "claude-opus-4-5-20251101", &request).WithOutputConfig(map[string]any{
		"effort":      "low",
		"task_budget": map[string]any{"type": "tokens", "total": 50_000},
	})
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	outputConfig, ok := request["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("missing output_config: %+v", request)
	}
	if outputConfig["effort"] != "low" {
		t.Fatalf("output_config.effort = %v, want low", outputConfig["effort"])
	}
	budget, _ := outputConfig["task_budget"].(map[string]any)
	if budget["type"] != "tokens" || budget["total"] != float64(50_000) {
		t.Fatalf("output_config.task_budget = %v, want tokens/50000", outputConfig["task_budget"])
	}
}

func TestOutputConfigEffortFromWithEffortWins(t *testing.T) {
	// Python's merge applies reasoning_effort on top of the constructor dict
	// (chat_models.py:1973-1978), so WithEffort overrides a constructor
	// effort key while other keys survive.
	var request map[string]any
	model := effortModel(t, "claude-opus-4-5-20251101", &request).
		WithOutputConfig(map[string]any{
			"effort":      "low",
			"task_budget": map[string]any{"type": "tokens", "total": 50_000},
		}).
		WithEffort("high")
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	outputConfig, ok := request["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("missing output_config: %+v", request)
	}
	if outputConfig["effort"] != "high" {
		t.Fatalf("output_config.effort = %v, want high (WithEffort wins)", outputConfig["effort"])
	}
	if _, present := outputConfig["task_budget"]; !present {
		t.Fatalf("task_budget should survive the merge: %+v", outputConfig)
	}
}

func TestOutputConfigAloneDoesNotDefaultAdaptiveThinking(t *testing.T) {
	// Python's adaptive default keys on reasoning_effort_applied
	// (chat_models.py:1976-1979, 2005-2010), which only an explicit
	// reasoning_effort/effort sets — a constructor output_config effort does
	// not trigger it even on an xhigh-capable model.
	var request map[string]any
	model := effortModel(t, "claude-opus-5", &request).
		WithOutputConfig(map[string]any{"effort": "high"})
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if _, present := request["thinking"]; present {
		t.Fatalf("constructor output_config must not default thinking: %+v", request["thinking"])
	}
	outputConfig, _ := request["output_config"].(map[string]any)
	if outputConfig["effort"] != "high" {
		t.Fatalf("output_config.effort = %v, want high", request["output_config"])
	}
}

func TestOutputConfigTaskBudgetAutoAppendsBeta(t *testing.T) {
	// Python test_task_budget_auto_appends_beta (test_chat_models.py:5799):
	// a task_budget in the resolved output_config requires its beta flag and
	// beta-endpoint routing.
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-sonnet-4-5"),
		).WithOutputConfig(map[string]any{
			"task_budget": map[string]any{"type": "tokens", "total": 128_000},
		})
	}, []messages.Message{messages.Human("run the errand")})

	const taskBudgetsBeta = "task-budgets-2026-03-13"
	if captured.beta != taskBudgetsBeta {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, taskBudgetsBeta)
	}
	if captured.path != "/messages?beta=true" {
		t.Fatalf("request path: got %q want /messages?beta=true", captured.path)
	}
}

func TestOutputConfigTaskBudgetBetaNotDuplicated(t *testing.T) {
	// Python test_task_budget_beta_not_duplicated (test_chat_models.py:5810).
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-sonnet-4-5"),
			WithBetaHeaders("task-budgets-2026-03-13"),
		).WithOutputConfig(map[string]any{
			"task_budget": map[string]any{"type": "tokens", "total": 128_000},
		})
	}, []messages.Message{messages.Human("run the errand")})

	if got := strings.Count(captured.beta, "task-budgets-2026-03-13"); got != 1 {
		t.Fatalf("anthropic-beta header should contain the beta exactly once, got %q", captured.beta)
	}
}

func TestOutputConfigNoTaskBudgetNoBeta(t *testing.T) {
	// Python test_no_task_budget_no_beta (test_chat_models.py:5821): an
	// output_config without task_budget triggers no beta.
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-sonnet-4-5"),
		).WithOutputConfig(map[string]any{"effort": "high"})
	}, []messages.Message{messages.Human("run the errand")})

	if strings.Contains(captured.beta, "task-budgets") {
		t.Fatalf("task-budgets beta should not be appended, got %q", captured.beta)
	}
}

func TestOutputConfigTaskBudgetPrecedesThinkingDisplayBeta(t *testing.T) {
	// Python appends the task-budget beta (chat_models.py:2120) before the
	// thinking-display beta (:2130); the combined header preserves that order.
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-sonnet-4-5"),
		).
			WithOutputConfig(map[string]any{
				"task_budget": map[string]any{"type": "tokens", "total": 128_000},
			}).
			WithThinking(map[string]any{"type": "adaptive", "display": "updates"})
	}, []messages.Message{messages.Human("run the errand")})

	want := "task-budgets-2026-03-13," + testThinkingDisplayBeta
	if captured.beta != want {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, want)
	}
}

func TestOutputConfigEffortFeedsOpus5Conflict(t *testing.T) {
	// Python's _assert_valid_model_configuration resolves the constructor
	// output_config effort (chat_models.py:1800-1805) before the opus-5
	// disabled-thinking conflict (test_chat_models.py:5133,5149): a
	// constructor effort of xhigh/max rejects disabled thinking with the
	// resolved level in the message, and high stays legal.
	t.Run("rejects at max", func(t *testing.T) {
		var request map[string]any
		model := effortModel(t, "claude-opus-5", &request).
			WithThinking(map[string]any{"type": "disabled"}).
			WithOutputConfig(map[string]any{"effort": "max"})
		_, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")})
		if err == nil {
			t.Fatal("invoke succeeded, want error")
		}
		for _, want := range []string{
			"is not supported for claude-opus-5",
			"`output_config.effort='max'`",
			"set effort to `high` or below",
		} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("error %q missing %q", err.Error(), want)
			}
		}
	})
	t.Run("allows at high", func(t *testing.T) {
		var request map[string]any
		model := effortModel(t, "claude-opus-5", &request).
			WithThinking(map[string]any{"type": "disabled"}).
			WithOutputConfig(map[string]any{"effort": "high"})
		if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("think")}); err != nil {
			t.Fatalf("invoke: %v", err)
		}
		thinking, _ := request["thinking"].(map[string]any)
		if thinking["type"] != "disabled" {
			t.Fatalf("thinking = %v, want disabled", request["thinking"])
		}
	})
}
