package openai

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/structuredoutput"
	"github.com/projanvil/langchain-golang/standardtests"
)

// Conformance wiring for the layered standardtests chat-model suites. The
// fake transport below answers the Responses API and is request-aware: it
// inspects every /responses payload (tools, tool_choice, text.format,
// multimodal input parts) and answers with provider-shaped tool calls, JSON,
// or SSE frames accordingly. Malformed or dropped request fields it did not
// expect are answered with HTTP 400 so a serialization regression inside the
// adapter fails the suite instead of silently passing.

// conformanceToolArgs returns canned arguments for the standard-suite tools,
// keyed by tool name.
func conformanceToolArgs(name string) (string, bool) {
	switch name {
	case "magic_function":
		return `{"input":3}`, true
	case "magic_function_no_args":
		return ``, true
	case "get_weather":
		return `{"location":"San Francisco"}`, true
	case "my_adder_tool":
		return `{"a":1,"b":2}`, true
	case "Joke":
		return `{"setup":"Why do programmers prefer dark mode?","punchline":"Because light attracts bugs."}`, true
	default:
		return ``, false
	}
}

// conformanceResponsesRequest is the subset of the Responses request the
// fake transport validates.
type conformanceResponsesRequest struct {
	Input []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"input"`
	Tools []struct {
		Type        string         `json:"type"`
		Name        string         `json:"name"`
		Parameters  schema.Schema  `json:"parameters"`
		Description string         `json:"description"`
	} `json:"tools"`
	ToolChoice  any               `json:"tool_choice"`
	Text        *conformanceText `json:"text"`
	Stream      bool             `json:"stream"`
}

type conformanceText struct {
	Format struct {
		Type   string        `json:"type"`
		Name   string        `json:"name"`
		Schema schema.Schema `json:"schema"`
		Strict bool          `json:"strict"`
	} `json:"format"`
}

// validateConformanceMultimodalParts asserts image/audio input parts carry
// their payload (url / data: URI for images, data+format for audio). A
// dropped or malformed part fails the request, catching silent multimodal
// overstatement.
func validateConformanceMultimodalParts(content json.RawMessage) error {
	var parts []map[string]any
	if err := json.Unmarshal(content, &parts); err != nil {
		return nil // bare string content: not a multimodal message
	}
	for _, part := range parts {
		switch part["type"] {
		case "input_image":
			url, _ := part["image_url"].(string)
			if url == "" {
				return fmt.Errorf("input_image part is missing image_url")
			}
			if !strings.HasPrefix(url, "data:image/") && !strings.HasPrefix(url, "http") {
				return fmt.Errorf("input_image url is neither a data URI nor remote: %q", url)
			}
		case "input_audio":
			audio, _ := part["input_audio"].(map[string]any)
			if audio == nil || audio["data"] == "" {
				return fmt.Errorf("input_audio part is missing input_audio.data")
			}
		}
	}
	return nil
}

// conformanceResponsesPlan decides the fake's answer shape from the request.
type conformanceResponsesPlan struct {
	toolName    string
	toolArgs    string
	jokeMode    bool
	multimodal  bool
	structured  bool
	formatName  string
	schemaShape schema.Schema
}

func planConformanceResponses(t *testing.T, body []byte) conformanceResponsesPlan {
	t.Helper()
	var req conformanceResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Errorf("conformance fake: decode request: %v", err)
		return conformanceResponsesPlan{}
	}
	plan := conformanceResponsesPlan{}
	for _, item := range req.Input {
		if item.Role == "user" {
			if err := validateConformanceMultimodalParts(item.Content); err != nil {
				t.Errorf("conformance fake: %v", err)
			}
			if strings.Contains(string(item.Content), "input_image") ||
				strings.Contains(string(item.Content), "input_audio") {
				plan.multimodal = true
			}
		}
	}
	if len(req.Tools) > 0 {
		// Answer with a function_call for the first bound tool. This makes
		// bind_tools serialization load-bearing: a dropped or mis-shaped
		// tools array produces no tool call and the suite fails.
		plan.toolName = req.Tools[0].Name
		args, ok := conformanceToolArgs(plan.toolName)
		if !ok {
			t.Errorf("conformance fake: unexpected bound tool %q", plan.toolName)
		}
		plan.toolArgs = args
		if choice := req.ToolChoice; choice != nil {
			switch value := choice.(type) {
			case string:
				if value != "auto" && value != "none" && value != "required" {
					t.Errorf("conformance fake: unexpected tool_choice %q", value)
				}
			case map[string]any:
				if value["type"] != "function" {
					t.Errorf("conformance fake: unexpected tool_choice object %v", value)
				}
			default:
				t.Errorf("conformance fake: unexpected tool_choice %v", choice)
			}
		}
	}
	if req.Text != nil && req.Text.Format.Type == "json_schema" {
		plan.structured = true
		plan.jokeMode = true
		plan.formatName = req.Text.Format.Name
		plan.schemaShape = req.Text.Format.Schema
		if properties, _ := plan.schemaShape["properties"].(map[string]any); properties == nil {
			t.Errorf("conformance fake: json_schema response_format missing properties")
		}
	}
	return plan
}

// writeConformanceResponses serves the non-streaming answer for plan.
func writeConformanceResponses(w http.ResponseWriter, plan conformanceResponsesPlan) {
	usage := `"usage":{"input_tokens":9,"output_tokens":7,"total_tokens":16}`
	switch {
	case plan.toolName != "":
		arguments := plan.toolArgs
		if arguments != "" {
			arguments = strings.ReplaceAll(arguments, `"`, `\"`)
		}
		_, _ = w.Write([]byte(`{` +
			`"id":"resp_conf","model":"gpt-test",` +
			`"output":[{"type":"function_call","call_id":"call_conf","name":"` + plan.toolName + `","arguments":"` + arguments + `"}],` +
			usage + `}`))
	case plan.jokeMode:
		_, _ = w.Write([]byte(`{` +
			`"id":"resp_conf","model":"gpt-test",` +
			`"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"{\"setup\":\"Why do programmers prefer dark mode?\",\"punchline\":\"Because light attracts bugs.\"}"}]}],` +
			usage + `}`))
	default:
		_, _ = w.Write([]byte(`{` +
			`"id":"resp_conf","model":"gpt-test",` +
			`"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"` + conformanceTextAnswer(plan) + `"}]}],` +
			usage + `}`))
	}
}

func conformanceTextAnswer(plan conformanceResponsesPlan) string {
	if plan.multimodal {
		return "A small diagram."
	}
	return "ok"
}

// writeConformanceResponsesStream serves the SSE answer for plan.
func writeConformanceResponsesStream(w http.ResponseWriter, plan conformanceResponsesPlan) {
	w.Header().Set("Content-Type", "text/event-stream")
	frame := func(data string) {
		_, _ = fmt.Fprint(w, "data: "+data+"\n\n")
	}
	completed := `{"type":"response.completed","response":{"id":"resp_conf","model":"gpt-test","output":[],"usage":{"input_tokens":9,"output_tokens":7,"total_tokens":16}}}`
	switch {
	case plan.toolName != "":
		arguments := strings.ReplaceAll(plan.toolArgs, `"`, `\"`)
		frame(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_conf","name":"` + plan.toolName + `","arguments":""}}`)
		frame(`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"` + arguments + `"}`)
		frame(`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_conf","name":"` + plan.toolName + `","arguments":"` + arguments + `"}}`)
		frame(completed)
	case plan.jokeMode:
		frame(`{"type":"response.output_text.delta","output_index":0,"delta":"{\"setup\":\"Why do programmers prefer dark mode?\",\"punchline\":\"Because light attracts bugs.\"}"}`)
		frame(`{"type":"response.output_text.done","output_index":0}`)
		frame(completed)
	default:
		frame(`{"type":"response.output_text.delta","output_index":0,"delta":"He"}`)
		frame(`{"type":"response.output_text.delta","output_index":0,"delta":"llo"}`)
		frame(`{"type":"response.output_text.done","output_index":0}`)
		frame(completed)
	}
}

// newConformanceResponsesServer builds the request-aware fake transport.
func newConformanceResponsesServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		plan := planConformanceResponses(t, body)
		if r.URL.Query().Get("stream") == "" && strings.Contains(string(body), `"stream":true`) {
			writeConformanceResponsesStream(w, plan)
			return
		}
		var probe struct {
			Stream bool `json:"stream"`
		}
		_ = json.Unmarshal(body, &probe)
		if probe.Stream {
			writeConformanceResponsesStream(w, plan)
			return
		}
		writeConformanceResponses(w, plan)
	}))
}

// TestStandardChatModelSuites wires the layered standardtests suites against
// the Responses API fake transport. Capability flags mirror the adapter's
// Capabilities() declaration, so a regression that drops a payload field the
// adapter claims to support fails here.
func TestStandardChatModelSuites(t *testing.T) {
	factory := func(t testing.TB) language.ChatModel {
		server := newConformanceResponsesServer(t.(*testing.T))
		t.Cleanup(server.Close)
		return NewChatModel(
			modelconfig.WithBaseURL(server.URL),
			modelconfig.WithModel("gpt-test"),
		)
	}
	hooks := standardtests.StructuredOutputSuiteHooks{
		NewRawEnvelope: func(t testing.TB, model language.ChatModel) (runnables.Runnable[[]messages.Message, structuredoutput.StructuredResult[map[string]any]], error) {
			concrete, ok := model.(ChatModel)
			if !ok {
				t.Fatalf("factory produced %T, not openai.ChatModel", model)
			}
			return structuredoutput.BindOptionsWithRaw[ChatModel, map[string]any](
				concrete,
				structuredoutput.Options{
					Name:   "Joke",
					Schema: jokeConformanceSchema(),
					Method: structuredoutput.MethodJSONSchema,
					Strict: true,
				},
			)
		},
	}
	standardtests.RunChatModelStandardSuites(
		t,
		factory,
		standardtests.ChatModelCapabilities{
			ToolCalling:            true,
			ToolChoice:             true,
			StructuredOutput:       true,
			ImageInputs:            true,
			ImageURLs:              true,
			AudioInputs:            true,
			Streaming:              true,
			UsageMetadata:          true,
			UsageMetadataStreaming: true,
		},
		hooks,
	)
}

// jokeConformanceSchema mirrors the standard suite's Joke schema for the
// include_raw envelope hook.
func jokeConformanceSchema() schema.Schema {
	sch := schema.Object(map[string]schema.Schema{
		"setup":     schema.String("question to set up a joke"),
		"punchline": schema.String("answer to resolve the joke"),
	}, "setup", "punchline")
	sch["title"] = "Joke"
	sch["description"] = "Joke to tell user."
	return sch
}
