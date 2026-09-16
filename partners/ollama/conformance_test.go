package ollama

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/standardtests"
)

// newChatTestServer returns an httptest server that answers both non-streaming
// and streaming /api/chat requests with deterministic content and usage.
func newChatTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Stream {
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"ok"},"done":false}`+"\n")
			_, _ = fmt.Fprint(w, `{"model":"llama3","created_at":"t2","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":2,"eval_count":1}`+"\n")
			return
		}
		_, _ = w.Write([]byte(`{"model":"llama3","created_at":"t1","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop","prompt_eval_count":2,"eval_count":1}`))
	}))
}

func TestStandardChatModelBasics(t *testing.T) {
	standardtests.RunChatModelBasics(
		t,
		func(t testing.TB) language.ChatModel {
			server := newChatTestServer(t.(*testing.T))
			t.Cleanup(server.Close)
			model := NewChatModel(modelconfig.WithBaseURL(server.URL))
			return model
		},
		standardtests.ChatModelCapabilities{
			ToolCalling:      true,
			UsageMetadata:    true,
			Streaming:        true,
			ImageInputs:      true,
			StructuredOutput: true,
		},
	)
}

func TestStandardEmbeddingsBasics(t *testing.T) {
	standardtests.RunEmbeddingsBasics(
		t,
		func(t testing.TB) embeddings.Embeddings {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req embedRequest
				_ = json.NewDecoder(r.Body).Decode(&req)
				embeddings := make([][]float64, len(req.Input))
				for i := range req.Input {
					embeddings[i] = []float64{0.1, 0.2, 0.3}
				}
				resp, _ := json.Marshal(map[string]any{
					"model":      "nomic-embed-text",
					"embeddings": embeddings,
				})
				_, _ = w.Write(resp)
			}))
			t.Cleanup(server.Close)
			return NewEmbeddings(modelconfig.WithBaseURL(server.URL))
		},
	)
}

// --- Layered chat-model suite wiring ---
//
// The fake below is request-aware: it inspects each /api/chat payload (tools,
// format, message images) and answers with provider-shaped tool_calls, JSON,
// or ndjson frames accordingly. Serialization gaps inside the adapter surface
// as suite failures instead of silently passing.

// conformanceToolArgs returns canned arguments for the standard-suite tools.
func conformanceToolArgs(name string) (any, bool) {
	switch name {
	case "magic_function":
		return map[string]any{"input": 3}, true
	case "magic_function_no_args":
		return map[string]any{}, true
	case "get_weather":
		return map[string]any{"location": "San Francisco"}, true
	case "Joke":
		return map[string]any{
			"setup":     "Why do programmers prefer dark mode?",
			"punchline": "Because light attracts bugs.",
		}, true
	case "my_adder_tool":
		return map[string]any{"a": 1, "b": 2}, true
	default:
		return nil, false
	}
}

const conformanceJokeJSON = `{"setup":"Why do programmers prefer dark mode?","punchline":"Because light attracts bugs."}`

type conformanceChatPlan struct {
	toolName   string
	toolArgs   any
	structured bool
	imageSeen  bool
}

func planConformanceChat(t *testing.T, body []byte) conformanceChatPlan {
	t.Helper()
	var req chatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Errorf("conformance fake: decode request: %v", err)
		return conformanceChatPlan{}
	}
	plan := conformanceChatPlan{}
	for _, message := range req.Messages {
		for _, image := range message.Images {
			if image == "" {
				t.Errorf("conformance fake: empty image payload in request")
				continue
			}
			plan.imageSeen = true
		}
	}
	if len(req.Tools) > 0 {
		plan.toolName = req.Tools[0].Function.Name
		args, ok := conformanceToolArgs(plan.toolName)
		if !ok {
			t.Errorf("conformance fake: unexpected bound tool %q", plan.toolName)
		}
		plan.toolArgs = args
	}
	if format, ok := req.Format.(map[string]any); ok && format != nil {
		plan.structured = true
		if _, hasProperties := format["properties"]; !hasProperties {
			t.Errorf("conformance fake: structured-output format is not a JSON schema object: %v", format)
		}
	}
	return plan
}

func writeConformanceChat(w http.ResponseWriter, plan conformanceChatPlan) {
	switch {
	case plan.toolName != "":
		toolCalls := map[string]any{
			"tool_calls": []any{map[string]any{
				"function": map[string]any{"name": plan.toolName, "arguments": plan.toolArgs},
			}},
		}
		writeConformanceChatLine(w, plan, toolCalls, true)
	case plan.structured:
		writeConformanceChatLine(w, plan, map[string]any{"content": conformanceJokeJSON}, true)
	case plan.imageSeen:
		writeConformanceChatLine(w, plan, map[string]any{"content": "A small diagram."}, true)
	default:
		writeConformanceChatLine(w, plan, map[string]any{"content": "ok"}, true)
	}
}

func writeConformanceChatLine(
	w http.ResponseWriter,
	plan conformanceChatPlan,
	overrides map[string]any,
	done bool,
) {
	message := map[string]any{"role": "assistant", "content": ""}
	for key, value := range overrides {
		message[key] = value
	}
	line := map[string]any{
		"model":      "llama3",
		"created_at": "t1",
		"message":    message,
		"done":       done,
	}
	if done {
		line["done_reason"] = "stop"
		line["prompt_eval_count"] = 2
		line["eval_count"] = 1
	}
	data, _ := json.Marshal(line)
	_, _ = w.Write(append(data, '\n'))
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func writeConformanceChatStream(w http.ResponseWriter, plan conformanceChatPlan) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	switch {
	case plan.toolName != "":
		// Ollama streams complete tool calls on the terminating frame.
		writeConformanceChatLine(w, plan, map[string]any{
			"tool_calls": []any{map[string]any{
				"function": map[string]any{"name": plan.toolName, "arguments": plan.toolArgs},
			}},
		}, true)
	case plan.structured:
		writeConformanceChatLine(w, plan, map[string]any{"content": conformanceJokeJSON}, true)
	default:
		writeConformanceChatLine(w, plan, map[string]any{"content": "Hello"}, false)
		writeConformanceChatLine(w, plan, map[string]any{"content": " world"}, false)
		writeConformanceChatLine(w, plan, nil, true)
	}
}

func newConformanceChatServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		plan := planConformanceChat(t, body)
		var probe chatRequest
		_ = json.Unmarshal(body, &probe)
		if probe.Stream {
			writeConformanceChatStream(w, plan)
			return
		}
		writeConformanceChat(w, plan)
	}))
}

// TestStandardChatModelSuites wires the layered standardtests suites against
// the /api/chat fake transport. Capability flags mirror the adapter's
// surface: tool choice, audio inputs, and streaming-chunk usage are not
// declared (no BindToolsWithOptions; images only; usage arrives on the
// terminal frame's metadata, not on a message chunk), so those subtests skip
// with a recorded reason.
func TestStandardChatModelSuites(t *testing.T) {
	standardtests.RunChatModelStandardSuites(
		t,
		func(t testing.TB) language.ChatModel {
			server := newConformanceChatServer(t.(*testing.T))
			t.Cleanup(server.Close)
			return NewChatModel(modelconfig.WithBaseURL(server.URL))
		},
		standardtests.ChatModelCapabilities{
			ToolCalling:      true,
			StructuredOutput: true,
			ImageInputs:      true,
			ImageURLs:        true,
			UsageMetadata:    true,
			Streaming:        true,
		},
		standardtests.StructuredOutputSuiteHooks{},
	)
}
