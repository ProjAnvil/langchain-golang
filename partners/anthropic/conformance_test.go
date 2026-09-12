package anthropic

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/standardtests"
)

// Conformance wiring for the layered standardtests chat-model suites. The
// fake transport answers the Messages API and is request-aware: it inspects
// every /messages payload (tools, tool_choice, image source blocks) and
// answers with provider-shaped tool_use blocks, JSON, or SSE frames
// accordingly. Malformed or dropped request fields are reported with
// t.Errorf so a serialization regression inside the adapter fails the suite
// instead of silently passing.

// conformanceToolArgs returns canned input for the standard-suite tools.
func conformanceToolArgs(name string) (map[string]any, bool) {
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

type conformanceMessagesRequest struct {
	Messages []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	Tools []struct {
		Name        string        `json:"name"`
		InputSchema schema.Schema `json:"input_schema"`
	} `json:"tools"`
	ToolChoice map[string]any `json:"tool_choice"`
	Stream     bool           `json:"stream"`
}

// validateConformanceContent asserts image content blocks carry a
// well-formed source (base64 data with a media type, or a url), catching
// silent multimodal overstatement.
func validateConformanceContent(t *testing.T, raw json.RawMessage) {
	t.Helper()
	var blocks []map[string]any
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return // string content
	}
	for _, block := range blocks {
		if block["type"] != "image" {
			continue
		}
		source, _ := block["source"].(map[string]any)
		if source == nil {
			t.Errorf("conformance fake: image content block missing source: %v", block)
			continue
		}
		switch source["type"] {
		case "base64":
			if source["data"] == "" || source["media_type"] == "" {
				t.Errorf("conformance fake: base64 image source missing data/media_type: %v", source)
			}
		case "url":
			if source["url"] == "" {
				t.Errorf("conformance fake: url image source missing url: %v", source)
			}
		default:
			t.Errorf("conformance fake: unexpected image source type %v", source["type"])
		}
	}
}

// writeConformanceMessages serves the non-streaming answer.
func writeConformanceMessages(w http.ResponseWriter, req conformanceMessagesRequest) {
	usage := `"usage":{"input_tokens":4,"output_tokens":2}`
	if len(req.Tools) > 0 {
		name := req.Tools[0].Name
		args, ok := conformanceToolArgs(name)
		if !ok {
			args = map[string]any{}
		}
		input, _ := json.Marshal(args)
		_, _ = w.Write([]byte(`{` +
			`"id":"msg_conf","type":"message","role":"assistant","model":"claude-test",` +
			`"content":[{"type":"tool_use","id":"toolu_conf","name":"` + name + `","input":` + string(input) + `}],` +
			`"stop_reason":"tool_use",` + usage + `}`))
		return
	}
	_, _ = w.Write([]byte(`{` +
		`"id":"msg_conf","type":"message","role":"assistant","model":"claude-test",` +
		`"content":[{"type":"text","text":"A small diagram."}],` +
		`"stop_reason":"end_turn",` + usage + `}`))
}

// writeConformanceMessagesStream serves the SSE answer.
func writeConformanceMessagesStream(w http.ResponseWriter, req conformanceMessagesRequest) {
	w.Header().Set("Content-Type", "text/event-stream")
	frame := func(event string, data string) {
		_, _ = fmt.Fprint(w, "event: "+event+"\ndata: "+data+"\n\n")
	}
	messageStart := `{"type":"message_start","message":{"id":"msg_conf","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":4,"output_tokens":0}}}`
	frame("message_start", messageStart)
	if len(req.Tools) > 0 {
		name := req.Tools[0].Name
		args, ok := conformanceToolArgs(name)
		if !ok {
			args = map[string]any{}
		}
		input, _ := json.Marshal(args)
		inputString := string(input)
		frame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_conf","name":"`+name+`"}}`)
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":`+strconv.Quote(inputString)+`}}`)
		frame("content_block_stop", `{"type":"content_block_stop","index":0}`)
		frame("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}`)
	} else {
		frame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`)
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}`)
		frame("content_block_stop", `{"type":"content_block_stop","index":0}`)
		frame("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`)
	}
	frame("message_stop", `{"type":"message_stop"}`)
}

func newConformanceMessagesServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/messages") {
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		var req conformanceMessagesRequest
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "decode body: "+err.Error(), http.StatusBadRequest)
			return
		}
		for _, message := range req.Messages {
			validateConformanceContent(t, message.Content)
		}
		if len(req.Tools) > 0 && req.ToolChoice != nil {
			switch req.ToolChoice["type"] {
			case "auto", "any", "tool":
				// Valid Messages API tool_choice shapes.
			default:
				t.Errorf("conformance fake: unexpected tool_choice %v", req.ToolChoice)
			}
		}
		if req.Stream {
			writeConformanceMessagesStream(w, req)
			return
		}
		writeConformanceMessages(w, req)
	}))
}

// TestStandardChatModelSuites wires the layered standardtests suites against
// the Messages API fake transport. Capability flags mirror the adapter's
// surface: structured output is declared because the adapter implements
// language.StructuredCaller (function-calling method); audio inputs and
// streaming-chunk usage are not declared (the Messages API stream surfaces
// usage on the terminal event, not on message chunks), so those subtests
// skip with a recorded reason.
func TestStandardChatModelSuites(t *testing.T) {
	standardtests.RunChatModelStandardSuites(
		t,
		func(t testing.TB) language.ChatModel {
			server := newConformanceMessagesServer(t.(*testing.T))
			t.Cleanup(server.Close)
			return NewChatModel(
				modelconfig.WithBaseURL(server.URL),
				modelconfig.WithModel("claude-test"),
			)
		},
		standardtests.ChatModelCapabilities{
			ToolCalling:      true,
			ToolChoice:       true,
			StructuredOutput: true,
			ImageInputs:      true,
			ImageURLs:        true,
			UsageMetadata:    true,
			// message_delta yields a terminal usage-only chunk
			// (mirroring Python _stream, chat_models.py:1702-1725).
			UsageMetadataStreaming: true,
			Streaming:              true,
		},
		standardtests.StructuredOutputSuiteHooks{},
	)
}
