package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// Mirrors the sampling-parameter fields of Python BaseChatOpenAI
// (chat_models/base.py:753-781, 963) flowing into the Chat Completions payload
// via _default_params' exclude_if_none map (chat_models/base.py:1340-1350).

func TestChatCompletionsSamplingParamsInPayload(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-sampling",
			"model":"gpt-test",
			"choices":[{"message":{"role":"assistant","content":"ok"}}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions().
		WithTopP(0.9).
		WithStop("END", "STOP").
		WithSeed(42).
		WithPresencePenalty(0.5).
		WithFrequencyPenalty(0.25).
		WithLogitBias(map[int]int{50256: -100}).
		WithN(2).
		WithLogprobs(true).
		WithTopLogprobs(5)
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if gotBody["top_p"] != 0.9 {
		t.Fatalf("top_p = %v, want 0.9", gotBody["top_p"])
	}
	stop, ok := gotBody["stop"].([]any)
	if !ok || len(stop) != 2 || stop[0] != "END" || stop[1] != "STOP" {
		t.Fatalf("stop = %#v, want [END STOP]", gotBody["stop"])
	}
	if gotBody["seed"] != float64(42) {
		t.Fatalf("seed = %v, want 42", gotBody["seed"])
	}
	if gotBody["presence_penalty"] != 0.5 {
		t.Fatalf("presence_penalty = %v, want 0.5", gotBody["presence_penalty"])
	}
	if gotBody["frequency_penalty"] != 0.25 {
		t.Fatalf("frequency_penalty = %v, want 0.25", gotBody["frequency_penalty"])
	}
	bias, ok := gotBody["logit_bias"].(map[string]any)
	if !ok || bias["50256"] != float64(-100) {
		t.Fatalf("logit_bias = %#v, want {\"50256\": -100}", gotBody["logit_bias"])
	}
	if gotBody["n"] != float64(2) {
		t.Fatalf("n = %v, want 2", gotBody["n"])
	}
	if gotBody["logprobs"] != true {
		t.Fatalf("logprobs = %v, want true", gotBody["logprobs"])
	}
	if gotBody["top_logprobs"] != float64(5) {
		t.Fatalf("top_logprobs = %v, want 5", gotBody["top_logprobs"])
	}
}

func TestChatCompletionsSamplingParamsOmittedByDefault(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-plain",
			"model":"gpt-test",
			"choices":[{"message":{"role":"assistant","content":"ok"}}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	for _, key := range []string{
		"top_p", "stop", "seed", "presence_penalty", "frequency_penalty",
		"logit_bias", "n", "logprobs", "top_logprobs",
	} {
		if _, ok := gotBody[key]; ok {
			t.Fatalf("payload unexpectedly carries %q: %#v", key, gotBody[key])
		}
	}
}

// Responses API: top_p is forwarded (chat_models/base.py:1343), but Python
// drops stop before the Responses call ("Responses API has no `stop`
// parameter", chat_models/base.py:4284-4286), and the remaining Chat
// Completions-only sampling knobs must not leak into the Responses payload.
func TestResponsesSamplingParams(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"resp_sampling",
			"model":"gpt-test",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).
		WithTopP(0.8).
		WithStop("END").
		WithSeed(7).
		WithPresencePenalty(0.1)
	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if gotBody["top_p"] != 0.8 {
		t.Fatalf("top_p = %v, want 0.8", gotBody["top_p"])
	}
	for _, key := range []string{"stop", "seed", "presence_penalty"} {
		if _, ok := gotBody[key]; ok {
			t.Fatalf("Responses payload unexpectedly carries %q: %#v", key, gotBody[key])
		}
	}
}
