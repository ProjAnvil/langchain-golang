package gemini

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
	"github.com/projanvil/langchain-golang/core/structuredoutput"
	"github.com/projanvil/langchain-golang/standardtests"
)

// Conformance wiring for the layered standardtests chat-model suites. The
// fake transport answers the Gemini API and is request-aware: it inspects
// every generateContent payload (tools, toolConfig.functionCallingConfig,
// inline_data/file_data parts, generationConfig.responseJsonSchema) and
// answers with provider-shaped candidates or SSE frames accordingly.
// Malformed or dropped request fields are reported with t.Errorf so a
// serialization regression inside the adapter fails the suite instead of
// silently passing.

// conformanceToolArgs returns canned args for the standard-suite tools.
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

type conformanceGenerateRequest struct {
	Contents []struct {
		Role  string `json:"role"`
		Parts []struct {
			Text         string `json:"text"`
			FunctionCall *struct {
				Name string         `json:"name"`
				Args map[string]any `json:"args"`
			} `json:"functionCall"`
			InlineData *struct {
				MimeType string `json:"mimeType"`
				Data     string `json:"data"`
			} `json:"inlineData"`
			FileData *struct {
				FileURI  string `json:"fileUri"`
				MimeType string `json:"mimeType"`
			} `json:"fileData"`
		} `json:"parts"`
	} `json:"contents"`
	Tools []struct {
		FunctionDeclarations []struct {
			Name       string          `json:"name"`
			Parameters json.RawMessage `json:"parameters"`
		} `json:"functionDeclarations"`
	} `json:"tools"`
	ToolConfig *struct {
		FunctionCallingConfig struct {
			Mode                 string   `json:"mode"`
			AllowedFunctionNames []string `json:"allowedFunctionNames"`
		} `json:"functionCallingConfig"`
	} `json:"toolConfig"`
	GenerationConfig *struct {
		ResponseMIMEType   string          `json:"responseMimeType"`
		ResponseJSONSchema json.RawMessage `json:"responseJsonSchema"`
	} `json:"generationConfig"`
}

// validateConformanceContents asserts media parts are well-formed
// (inline_data carries data + mime type; file_data carries a URI), catching
// silent multimodal overstatement.
func validateConformanceContents(t *testing.T, req conformanceGenerateRequest) {
	t.Helper()
	for _, content := range req.Contents {
		for _, part := range content.Parts {
			if part.InlineData != nil && (part.InlineData.Data == "" || part.InlineData.MimeType == "") {
				t.Errorf("conformance fake: inlineData part missing data/mimeType: %+v", part.InlineData)
			}
			if part.FileData != nil && part.FileData.FileURI == "" {
				t.Errorf("conformance fake: fileData part missing fileUri: %+v", part.FileData)
			}
		}
	}
}

// conformancePickToolName picks the tool the fake will call: the forced name
// when allowedFunctionNames pins one, else the first declared tool.
func conformancePickToolName(req conformanceGenerateRequest) string {
	var first string
	for _, tool := range req.Tools {
		if len(tool.FunctionDeclarations) > 0 {
			first = tool.FunctionDeclarations[0].Name
			break
		}
	}
	if req.ToolConfig != nil && len(req.ToolConfig.FunctionCallingConfig.AllowedFunctionNames) > 0 {
		if allowed := req.ToolConfig.FunctionCallingConfig.AllowedFunctionNames[0]; allowed != "" {
			return allowed
		}
	}
	return first
}

func conformanceArgsFor(t *testing.T, req conformanceGenerateRequest) map[string]any {
	t.Helper()
	if args, ok := conformanceToolArgs(conformancePickToolName(req)); ok {
		return args
	}
	return map[string]any{}
}

const conformanceUsage = `"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6}`

// conformanceFunctionCallFrame renders one SSE data frame carrying a complete
// function call (Gemini sends complete calls per chunk) and terminal usage.
func conformanceFunctionCallFrame(name string, args map[string]any) string {
	encoded, _ := json.Marshal(args)
	return fmt.Sprintf(
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":%q,"args":%s}}]},"finishReason":"STOP"}],%s}`,
		name, encoded, conformanceUsage)
}

// conformanceJokeJSON is the JSON text the fake returns for structured
// output requests, matching the suite's Joke schema.
var conformanceJokeJSON = mustJSON("{" +
	"\"setup\":\"Why do programmers prefer dark mode?\"," +
	"\"punchline\":\"Because light attracts bugs.\"" +
	"}")

func mustJSON(v string) string {
	encoded, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(encoded)
}

func newConformanceGeminiServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		var req conformanceGenerateRequest
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "decode body: "+err.Error(), http.StatusBadRequest)
			return
		}
		validateConformanceContents(t, req)
		structured := req.GenerationConfig != nil && req.GenerationConfig.ResponseJSONSchema != nil

		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			writeFrame := func(data string) {
				_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
			}
			switch {
			case len(req.Tools) > 0:
				writeFrame(conformanceFunctionCallFrame(conformancePickToolName(req), conformanceArgsFor(t, req)))
			case structured:
				writeFrame(fmt.Sprintf(
					`{"candidates":[{"content":{"parts":[{"text":%s}]},"finishReason":"STOP"}],%s}`,
					conformanceJokeJSON, conformanceUsage))
			default:
				writeFrame(fmt.Sprintf(`{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}],%s}`, conformanceUsage))
				writeFrame(fmt.Sprintf(`{"candidates":[{"content":{"parts":[{"text":" there"}]},"finishReason":"STOP"}],%s}`, conformanceUsage))
			}
			return
		}

		switch {
		case len(req.Tools) > 0:
			name := conformancePickToolName(req)
			encoded, _ := json.Marshal(conformanceArgsFor(t, req))
			_, _ = w.Write([]byte(fmt.Sprintf(
				`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":%q,"args":%s}}]},"finishReason":"STOP"}],%s}`,
				name, encoded, conformanceUsage)))
		case structured:
			_, _ = w.Write([]byte(fmt.Sprintf(
				`{"candidates":[{"content":{"role":"model","parts":[{"text":%s}]},"finishReason":"STOP"}],%s}`,
				conformanceJokeJSON, conformanceUsage)))
		default:
			_, _ = w.Write([]byte(fmt.Sprintf(
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"A small diagram."}]},"finishReason":"STOP"}],%s}`,
				conformanceUsage)))
		}
	}))
}

// TestStandardChatModelSuites wires the layered standardtests suites against
// the Gemini API fake transport. Capability flags mirror the adapter's
// Capabilities() declaration: usage metadata on invoke and on the terminal
// stream chunk, image/url/audio inputs, structured output via
// generationConfig.responseJsonSchema, and tool choice through
// toolConfig.functionCallingConfig.
func TestStandardChatModelSuites(t *testing.T) {
	factory := func(t testing.TB) language.ChatModel {
		server := newConformanceGeminiServer(t.(*testing.T))
		t.Cleanup(server.Close)
		return NewChatModel(
			modelconfig.WithBaseURL(server.URL),
			modelconfig.WithModel("gemini-test"),
			modelconfig.WithAPIKey("test-key"),
		)
	}
	hooks := standardtests.StructuredOutputSuiteHooks{
		NewRawEnvelope: func(
			t testing.TB,
			model language.ChatModel,
		) (runnables.Runnable[[]messages.Message, structuredoutput.StructuredResult[map[string]any]], error) {
			concrete, ok := model.(ChatModel)
			if !ok {
				t.Fatalf("factory produced %T, not gemini.ChatModel", model)
			}
			return structuredoutput.BindOptionsWithRaw[ChatModel, map[string]any](
				concrete,
				structuredoutput.Options{
					Name:   "Joke",
					Schema: jokeSchema(),
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
			UsageMetadata:          true,
			UsageMetadataStreaming: true,
			Streaming:              true,
		},
		hooks,
	)
}
