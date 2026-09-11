package standardtests

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/structuredoutput"
	"github.com/projanvil/langchain-golang/core/tools"
)

// Layered chat-model conformance suites mirroring the feature areas of
// Python's ChatModelIntegrationTests
// (libs/standard-tests/langchain_tests/integration_tests/chat_models.py).
//
// RunChatModelBasics covers the always-required surface (invoke/batch/stream
// basics). The suites in this file go deeper per declared capability:
//
//   - RunChatModelToolCallingSuite        test_tool_calling family
//   - RunChatModelToolChoiceSuite         test_tool_choice
//   - RunChatModelStructuredOutputSuite   test_structured_output family
//   - RunChatModelMultimodalInputSuite    test_image_inputs / test_audio_inputs
//   - RunChatModelStreamingSuite          test_stream / usage streaming / cancel
//
// Contract for every suite: a capability flag that is NOT declared skips the
// corresponding checks with a recorded reason (mirroring Python's pytest.skip
// on the has_* properties); a capability that IS declared but not honored by
// the model FAILS the suite. This is what surfaces capability overstatement.
//
// The suites are offline: they drive language.ChatModel only, so partners run
// them against wire-level fake transports whose handlers validate request
// payload shapes (both sides of the contract — serialization and parsing).

// chatSuiteSkip records a prerequisite-capability skip as its own subtest.
// Suites call this instead of t.Skip on the caller's t so a skipping suite
// does not abort sibling suites when several are composed under one test
// (RunChatModelStandardSuites).
func chatSuiteSkip(t *testing.T, suite string, reason string) {
	t.Helper()
	t.Run(suite, func(t *testing.T) {
		t.Skip(reason)
	})
}

// RunChatModelStandardSuites runs every layered chat-model suite in one call.
// Each suite skips itself when its prerequisite capabilities are not declared.
// hooks is optional (zero value omits the include_raw envelope check).
func RunChatModelStandardSuites(
	t *testing.T,
	factory ChatModelFactory,
	capabilities ChatModelCapabilities,
	hooks StructuredOutputSuiteHooks,
) {
	t.Helper()
	RunChatModelToolCallingSuite(t, factory, capabilities)
	RunChatModelToolChoiceSuite(t, factory, capabilities)
	RunChatModelStructuredOutputSuite(t, factory, capabilities, hooks)
	RunChatModelMultimodalInputSuite(t, factory, capabilities)
	RunChatModelStreamingSuite(t, factory, capabilities)
}

// --- Shared fixtures (mirror Python's module-level tools and schemas) ---

// chatSuiteJokeSchema mirrors Python's Joke pydantic model used by the
// structured-output tests: {"setup": str, "punchline": str}, both required.
func chatSuiteJokeSchema() schema.Schema {
	sch := schema.Object(map[string]schema.Schema{
		"setup":     schema.String("question to set up a joke"),
		"punchline": schema.String("answer to resolve the joke"),
	}, "setup", "punchline")
	sch["title"] = "Joke"
	sch["description"] = "Joke to tell user."
	return sch
}

func chatSuiteJokeArgs() map[string]any {
	return map[string]any{
		"setup":     "Why do programmers prefer dark mode?",
		"punchline": "Because light attracts bugs.",
	}
}

func chatSuiteJokeJSON() string {
	data, _ := json.Marshal(chatSuiteJokeArgs())
	return string(data)
}

func chatSuiteNewTool(
	t *testing.T,
	name string,
	description string,
	argsSchema schema.Schema,
) tools.Tool {
	t.Helper()
	tool, err := tools.NewFunc(name, description, argsSchema,
		func(context.Context, map[string]any) (tools.Result, error) {
			return tools.Result{Content: "ok"}, nil
		})
	requireNoErr(t, "NewFunc "+name, err)
	return tool
}

// chatSuiteMagicFunctionTool mirrors Python's magic_function(_input: int).
func chatSuiteMagicFunctionTool(t *testing.T) tools.Tool {
	t.Helper()
	return chatSuiteNewTool(t,
		"magic_function",
		"Apply a magic function to an input.",
		schema.Object(map[string]schema.Schema{
			"input": schema.Integer("the input number"),
		}, "input"),
	)
}

// chatSuiteMagicFunctionNoArgsTool mirrors magic_function_no_args().
func chatSuiteMagicFunctionNoArgsTool(t *testing.T) tools.Tool {
	t.Helper()
	return chatSuiteNewTool(t,
		"magic_function_no_args",
		"Calculate a magic function.",
		schema.Object(map[string]schema.Schema{}),
	)
}

// chatSuiteGetWeatherTool mirrors get_weather(location: str).
func chatSuiteGetWeatherTool(t *testing.T) tools.Tool {
	t.Helper()
	return chatSuiteNewTool(t,
		"get_weather",
		"Get the weather at a location.",
		schema.Object(map[string]schema.Schema{
			"location": schema.String("the location to look up"),
		}, "location"),
	)
}

// chatSuiteAdderTool mirrors the my_adder_tool(a: int, b: int) fixture.
func chatSuiteAdderTool(t *testing.T) tools.Tool {
	t.Helper()
	return chatSuiteNewTool(t,
		"my_adder_tool",
		"Add two integers.",
		schema.Object(map[string]schema.Schema{
			"a": schema.Integer("first addend"),
			"b": schema.Integer("second addend"),
		}, "a", "b"),
	)
}

// chatSuiteJokeTool adapts the Joke schema as a tool for the
// function_calling structured-output method.
func chatSuiteJokeTool(t *testing.T) tools.Tool {
	t.Helper()
	return chatSuiteNewTool(t, "Joke", "Joke to tell user.", chatSuiteJokeSchema())
}

// chatSuiteBindTools binds tools, applying ToolChoiceAny when the model
// declares tool choice (mirroring Python's
// `tool_choice_value = None if not self.has_tool_choice else "any"`).
func chatSuiteBindTools(
	t *testing.T,
	model language.ChatModel,
	force bool,
	boundTools []tools.Tool,
) language.ChatModel {
	t.Helper()
	if force {
		binder, ok := model.(language.ToolBinder)
		if !ok {
			t.Fatalf("model declares tool choice but does not implement language.ToolBinder")
		}
		bound, err := binder.BindToolsWithOptions(boundTools, language.BindToolsOptions{
			ToolChoice: language.ToolChoiceAny,
		})
		requireNoErr(t, "BindToolsWithOptions(tool_choice=any)", err)
		return bound
	}
	bound, err := model.BindTools(boundTools)
	requireNoErr(t, "BindTools", err)
	return bound
}

// --- Assertion helpers ---

// requireSingleToolCall mirrors Python's _validate_tool_call_message: exactly
// one tool call with the expected name and a non-empty ID on an AI message.
func requireSingleToolCall(t *testing.T, what string, message messages.Message, wantName string) messages.ToolCall {
	t.Helper()
	if message.Role != messages.RoleAI {
		t.Fatalf("%s: role: got %q want %q", what, message.Role, messages.RoleAI)
	}
	if len(message.ToolCalls) != 1 {
		t.Fatalf("%s: tool calls: got %d want 1 (calls: %+v)", what, len(message.ToolCalls), message.ToolCalls)
	}
	call := message.ToolCalls[0]
	if call.Name != wantName {
		t.Fatalf("%s: tool call name: got %q want %q", what, call.Name, wantName)
	}
	if call.ID == "" {
		t.Fatalf("%s: tool call %q has an empty id", what, wantName)
	}
	return call
}

// requireToolCallArgNumber asserts args[key] is the number want. Tool-call
// arguments travel as JSON, so numbers may arrive as float64, int, or
// json.Number depending on the parsing path.
func requireToolCallArgNumber(t *testing.T, what string, args map[string]any, key string, want float64) {
	t.Helper()
	value, ok := args[key]
	if !ok {
		t.Fatalf("%s: tool call args missing %q (args: %#v)", what, key, args)
	}
	switch n := value.(type) {
	case float64:
		if n != want {
			t.Fatalf("%s: tool call args[%q]: got %v want %v", what, key, n, want)
		}
	case int:
		if float64(n) != want {
			t.Fatalf("%s: tool call args[%q]: got %v want %v", what, key, n, want)
		}
	case int64:
		if float64(n) != want {
			t.Fatalf("%s: tool call args[%q]: got %v want %v", what, key, n, want)
		}
	case json.Number:
		parsed, err := n.Float64()
		if err != nil || parsed != want {
			t.Fatalf("%s: tool call args[%q]: got %v want %v", what, key, n.String(), want)
		}
	default:
		t.Fatalf("%s: tool call args[%q]: got %#v want the number %v", what, key, value, want)
	}
}

// requireJokeJSON decodes text as JSON and validates the Joke schema shape:
// non-empty string setup and punchline keys.
func requireJokeJSON(t *testing.T, what string, text string) map[string]any {
	t.Helper()
	var decoded map[string]any
	if err := json.Unmarshal([]byte(text), &decoded); err != nil {
		t.Fatalf("%s: response is not valid JSON: %v (text: %q)", what, err, text)
	}
	for _, key := range []string{"setup", "punchline"} {
		value, ok := decoded[key].(string)
		if !ok {
			t.Fatalf("%s: %q is not a string: %#v", what, key, decoded[key])
		}
		if value == "" {
			t.Fatalf("%s: %q is empty", what, key)
		}
	}
	return decoded
}

// --- Tool-calling suite ---

// RunChatModelToolCallingSuite verifies tool-calling behavior for models that
// declare ToolCalling. It mirrors the sync subset of Python's test_tool_calling
// family: bound tools must produce well-formed tool_calls on invoke, tool
// message histories must round-trip, zero-argument tools must work, an agent
// loop (call -> tool result -> answer) must complete, and — when Streaming is
// also declared — streamed tool calls must surface as complete tool calls on
// at least one chunk (the Go port's replacement for Python's chunk
// aggregation: providers assemble argument deltas before exposing them).
func RunChatModelToolCallingSuite(
	t *testing.T,
	factory ChatModelFactory,
	capabilities ChatModelCapabilities,
) {
	t.Helper()
	if !capabilities.ToolCalling {
		chatSuiteSkip(t, "tool calling", "model does not declare tool calling support (ChatModelCapabilities.ToolCalling)")
		return
	}
	query := "What is the value of magic_function(3)? Use the tool."

	t.Run("invoke with bound tools produces tool calls", func(t *testing.T) { // test_tool_calling
		model := chatSuiteBindTools(t, factory(t), capabilities.ToolChoice,
			[]tools.Tool{chatSuiteMagicFunctionTool(t)})
		response, err := model.Invoke(context.Background(), []messages.Message{
			messages.Human(query),
		})
		requireNoErr(t, "invoke with tools", err)
		call := requireSingleToolCall(t, "tool call response", response, "magic_function")
		requireToolCallArgNumber(t, "tool call response", call.Args, "input", 3)
	})

	t.Run("tool message histories string content", func(t *testing.T) { // test_tool_message_histories_string_content
		model := chatSuiteBindTools(t, factory(t), false,
			[]tools.Tool{chatSuiteAdderTool(t)})
		aiTurn := messages.AI("")
		aiTurn.ToolCalls = []messages.ToolCall{{
			Name: "my_adder_tool",
			Args: map[string]any{"a": 1, "b": 2},
			ID:   "abc123",
		}}
		response, err := model.Invoke(context.Background(), []messages.Message{
			messages.Human("What is 1 + 2"),
			aiTurn,
			messages.Tool("abc123", `{"result": 3}`),
		})
		requireNoErr(t, "invoke with tool history", err)
		if response.Role != messages.RoleAI {
			t.Fatalf("history response role: got %q want %q", response.Role, messages.RoleAI)
		}
	})

	t.Run("tool calling with no arguments", func(t *testing.T) { // test_tool_calling_with_no_arguments
		model := chatSuiteBindTools(t, factory(t), capabilities.ToolChoice,
			[]tools.Tool{chatSuiteMagicFunctionNoArgsTool(t)})
		response, err := model.Invoke(context.Background(), []messages.Message{
			messages.Human("What is the value of magic_function_no_args()? Use the tool."),
		})
		requireNoErr(t, "invoke with no-args tool", err)
		call := requireSingleToolCall(t, "no-args tool response", response, "magic_function_no_args")
		if len(call.Args) != 0 {
			t.Fatalf("no-args tool call args: got %#v want none", call.Args)
		}
	})

	t.Run("agent loop", func(t *testing.T) { // test_agent_loop
		model := chatSuiteBindTools(t, factory(t), false,
			[]tools.Tool{chatSuiteGetWeatherTool(t)})
		input := []messages.Message{
			messages.Human("What is the weather in San Francisco, CA?"),
		}
		toolCallMessage, err := model.Invoke(context.Background(), input)
		requireNoErr(t, "agent loop first invoke", err)
		if toolCallMessage.Role != messages.RoleAI || len(toolCallMessage.ToolCalls) == 0 {
			t.Fatalf("agent loop first invoke: expected an AI message with tool calls, got role=%q calls=%d",
				toolCallMessage.Role, len(toolCallMessage.ToolCalls))
		}
		call := toolCallMessage.ToolCalls[0]
		if call.ID == "" {
			t.Fatalf("agent loop tool call has an empty id")
		}
		response, err := model.Invoke(context.Background(), []messages.Message{
			input[0],
			toolCallMessage,
			messages.Tool(call.ID, "It's sunny."),
		})
		requireNoErr(t, "agent loop second invoke", err)
		if response.Role != messages.RoleAI {
			t.Fatalf("agent loop second invoke role: got %q want %q", response.Role, messages.RoleAI)
		}
	})

	if capabilities.Streaming {
		t.Run("streaming tool calls", func(t *testing.T) { // test_tool_calling (stream half)
			model := chatSuiteBindTools(t, factory(t), capabilities.ToolChoice,
				[]tools.Tool{chatSuiteMagicFunctionTool(t)})
			stream, err := model.Stream(context.Background(), []messages.Message{
				messages.Human(query),
			})
			requireNoErr(t, "stream with tools", err)
			defer stream.Close()

			var chunks int
			var complete *messages.ToolCall
			for {
				chunk, ok, err := stream.Next(context.Background())
				if err != nil {
					t.Fatalf("stream next: %v", err)
				}
				if !ok {
					break
				}
				chunks++
				if chunk.Role != messages.RoleAI {
					t.Fatalf("stream chunk role: got %q want %q", chunk.Role, messages.RoleAI)
				}
				for _, call := range chunk.ToolCalls {
					if call.ID != "" && call.Name == "magic_function" {
						if complete == nil {
							copied := call
							complete = &copied
						}
					}
				}
			}
			if chunks == 0 {
				t.Fatalf("tool-call stream yielded no chunks")
			}
			if complete == nil {
				t.Fatalf("expected a stream chunk carrying the complete magic_function tool call (id, name, parsed args)")
			}
			requireToolCallArgNumber(t, "streamed tool call", complete.Args, "input", 3)
		})
	}
}

// --- Tool-choice suite ---

// RunChatModelToolChoiceSuite verifies forced tool calling for models that
// declare ToolChoice, mirroring Python's test_tool_choice: tool_choice "any"
// must force at least one call and a named tool_choice must force that tool.
// It also pins the ToolBinder contract: a declared ToolChoice capability
// without a language.ToolBinder implementation is a capability overstatement
// and fails the suite.
func RunChatModelToolChoiceSuite(
	t *testing.T,
	factory ChatModelFactory,
	capabilities ChatModelCapabilities,
) {
	t.Helper()
	if !capabilities.ToolCalling || !capabilities.ToolChoice {
		chatSuiteSkip(t, "tool choice", "model does not declare tool choice support (ChatModelCapabilities.ToolChoice)")
		return
	}

	t.Run("tool binder interface", func(t *testing.T) {
		if _, ok := factory(t).(language.ToolBinder); !ok {
			t.Fatalf("model declares ToolChoice but does not implement language.ToolBinder (BindToolsWithOptions)")
		}
	})

	bind := func(t *testing.T, choice language.ToolChoice) language.ChatModel {
		t.Helper()
		binder, ok := factory(t).(language.ToolBinder)
		if !ok {
			t.Fatalf("model declares ToolChoice but does not implement language.ToolBinder (BindToolsWithOptions)")
		}
		bound, err := binder.BindToolsWithOptions(
			[]tools.Tool{chatSuiteMagicFunctionTool(t), chatSuiteGetWeatherTool(t)},
			language.BindToolsOptions{ToolChoice: choice},
		)
		requireNoErr(t, fmt.Sprintf("BindToolsWithOptions(tool_choice=%q)", choice), err)
		return bound
	}

	t.Run("any forces a tool call", func(t *testing.T) {
		model := bind(t, language.ToolChoiceAny)
		response, err := model.Invoke(context.Background(), []messages.Message{
			messages.Human("Hello!"),
		})
		requireNoErr(t, "invoke with tool_choice=any", err)
		if response.Role != messages.RoleAI {
			t.Fatalf("response role: got %q want %q", response.Role, messages.RoleAI)
		}
		if len(response.ToolCalls) == 0 {
			t.Fatalf("tool_choice=any did not force a tool call (response: %+v)", response)
		}
	})

	t.Run("named tool is forced", func(t *testing.T) {
		model := bind(t, language.ToolChoice("magic_function"))
		response, err := model.Invoke(context.Background(), []messages.Message{
			messages.Human("Hello!"),
		})
		requireNoErr(t, "invoke with named tool_choice", err)
		if len(response.ToolCalls) == 0 {
			t.Fatalf("named tool_choice did not force a tool call (response: %+v)", response)
		}
		if response.ToolCalls[0].Name != "magic_function" {
			t.Fatalf("forced tool name: got %q want %q", response.ToolCalls[0].Name, "magic_function")
		}
	})
}

// --- Structured-output suite ---

// StructuredOutputSuiteHooks carries the one provider binding the
// language.ChatModel interface cannot express: an include_raw envelope
// builder (Python's with_structured_output(include_raw=True)). Partners wire
// it to structuredoutput.BindOptionsWithRaw with their concrete model type;
// the zero value skips the envelope check.
type StructuredOutputSuiteHooks struct {
	// NewRawEnvelope builds a runnable returning the include_raw envelope for
	// the Joke schema, given the model produced by the suite's factory.
	NewRawEnvelope func(
		t testing.TB,
		model language.ChatModel,
	) (runnables.Runnable[[]messages.Message, structuredoutput.StructuredResult[map[string]any]], error)
}

// RunChatModelStructuredOutputSuite verifies structured output for models
// that declare StructuredOutput, mirroring Python's test_structured_output
// (schema-conforming JSON on invoke), the function_calling method (bound
// schema tool whose call arguments conform), and — when hooks wire it — the
// include_raw envelope. A model that declares the capability but returns
// non-conforming output fails.
func RunChatModelStructuredOutputSuite(
	t *testing.T,
	factory ChatModelFactory,
	capabilities ChatModelCapabilities,
	hooks StructuredOutputSuiteHooks,
) {
	t.Helper()
	if !capabilities.StructuredOutput {
		chatSuiteSkip(t, "structured output", "model does not declare structured output support (ChatModelCapabilities.StructuredOutput)")
		return
	}
	jokePrompt := []messages.Message{messages.Human("Tell me a joke about cats.")}

	t.Run("invoke structured returns schema-shaped json", func(t *testing.T) { // test_structured_output
		model := factory(t)
		response, err := language.InvokeStructured(context.Background(), model, jokePrompt, chatSuiteJokeSchema())
		requireNoErr(t, "InvokeStructured", err)
		requireJokeJSON(t, "structured output", messages.Text(response))
	})

	if capabilities.ToolCalling {
		t.Run("function calling method", func(t *testing.T) { // test_structured_output (function_calling method)
			bound, err := factory(t).BindTools([]tools.Tool{chatSuiteJokeTool(t)})
			requireNoErr(t, "BindTools joke schema tool", err)
			response, err := bound.Invoke(context.Background(), jokePrompt)
			requireNoErr(t, "invoke with joke schema tool", err)
			call := requireSingleToolCall(t, "function calling response", response, "Joke")
			data, err := json.Marshal(call.Args)
			requireNoErr(t, "encode tool call args", err)
			requireJokeJSON(t, "function calling args", string(data))
		})
	}

	t.Run("include raw envelope", func(t *testing.T) { // with_structured_output(include_raw=True)
		if hooks.NewRawEnvelope == nil {
			t.Skip("no include_raw envelope hook wired for this model")
		}
		envelope, err := hooks.NewRawEnvelope(t, factory(t))
		requireNoErr(t, "build include_raw envelope runnable", err)
		result, err := envelope.Invoke(context.Background(), jokePrompt)
		requireNoErr(t, "invoke include_raw envelope", err)
		if result.ParsingError != nil {
			t.Fatalf("include_raw envelope parsing error: %v", result.ParsingError)
		}
		if result.Raw.Role != messages.RoleAI {
			t.Fatalf("include_raw raw message role: got %q want %q", result.Raw.Role, messages.RoleAI)
		}
		if result.Parsed["setup"] == "" || result.Parsed["punchline"] == "" {
			t.Fatalf("include_raw parsed joke incomplete: %#v", result.Parsed)
		}
	})
}

// --- Multimodal-input suite ---

// Standard fixtures mirroring Python's image/audio downloads, kept inline so
// the suite stays offline: a 1x1 transparent PNG and a minimal WAV header.
const (
	chatSuiteTestImageURL = "https://raw.githubusercontent.com/langchain-ai/docs/4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png"
	chatSuiteTinyPNGB64   = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
	chatSuiteTinyWAVB64   = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
)

// RunChatModelMultimodalInputSuite verifies multimodal inputs for models
// that declare them, mirroring Python's test_image_inputs and
// test_audio_inputs: a declared ImageInputs/ImageURLs/AudioInputs capability
// must accept the corresponding content-block message without error. This is
// the suite that catches capability overstatement — a model that declares
// image inputs but fails (or rejects) block-bearing messages fails here.
// Partner wiring should pair it with wire-level payload assertions so silent
// drops are also caught.
func RunChatModelMultimodalInputSuite(
	t *testing.T,
	factory ChatModelFactory,
	capabilities ChatModelCapabilities,
) {
	t.Helper()
	invokeWithBlocks := func(t *testing.T, blocks []messages.ContentBlock) {
		t.Helper()
		model := factory(t)
		message := messages.Human("")
		message = message.WithContentBlocks(blocks)
		response, err := model.Invoke(context.Background(), []messages.Message{message})
		requireNoErr(t, "invoke with multimodal blocks", err)
		if response.Role != messages.RoleAI {
			t.Fatalf("multimodal response role: got %q want %q", response.Role, messages.RoleAI)
		}
	}
	prompt := messages.TextBlock{Text: "Give a concise description of this input."}

	t.Run("image base64 input", func(t *testing.T) { // test_image_inputs (standard LangChain format)
		if !capabilities.ImageInputs {
			t.Skip("model does not declare image input support (ChatModelCapabilities.ImageInputs)")
		}
		invokeWithBlocks(t, []messages.ContentBlock{
			prompt,
			messages.ImageBlock{Base64: chatSuiteTinyPNGB64, MimeType: "image/png"},
		})
	})

	t.Run("image url input", func(t *testing.T) { // test_image_inputs (url form)
		if !capabilities.ImageURLs {
			t.Skip("model does not declare image url input support (ChatModelCapabilities.ImageURLs)")
		}
		invokeWithBlocks(t, []messages.ContentBlock{
			prompt,
			messages.ImageBlock{URL: chatSuiteTestImageURL},
		})
	})

	t.Run("audio base64 input", func(t *testing.T) { // test_audio_inputs
		if !capabilities.AudioInputs {
			t.Skip("model does not declare audio input support (ChatModelCapabilities.AudioInputs)")
		}
		invokeWithBlocks(t, []messages.ContentBlock{
			prompt,
			messages.AudioBlock{Base64: chatSuiteTinyWAVB64, MimeType: "audio/wav"},
		})
	})
}

// --- Streaming suite ---

// RunChatModelStreamingSuite verifies streaming behavior for models that
// declare Streaming, mirroring Python's test_stream (chunk sequence with
// non-empty aggregated text), test_usage_metadata_streaming (usage carried on
// stream chunks, with input tokens on at most one chunk to guard against
// overcounting), and a Go-port addition: a canceled context must terminate
// the stream (providers must not hang on cancellation).
func RunChatModelStreamingSuite(
	t *testing.T,
	factory ChatModelFactory,
	capabilities ChatModelCapabilities,
) {
	t.Helper()
	if !capabilities.Streaming {
		chatSuiteSkip(t, "streaming", "model does not declare streaming support (ChatModelCapabilities.Streaming)")
		return
	}

	t.Run("chunk sequence", func(t *testing.T) { // test_stream
		model := factory(t)
		stream, err := model.Stream(context.Background(), []messages.Message{
			messages.Human("Write a two-sentence story about a cat."),
		})
		requireNoErr(t, "stream", err)
		defer stream.Close()

		var chunks []messages.Message
		var text strings.Builder
		for {
			chunk, ok, err := stream.Next(context.Background())
			if err != nil {
				t.Fatalf("stream next: %v", err)
			}
			if !ok {
				break
			}
			if chunk.Role != messages.RoleAI {
				t.Fatalf("stream chunk role: got %q want %q", chunk.Role, messages.RoleAI)
			}
			chunks = append(chunks, chunk)
			text.WriteString(messages.Text(chunk))
		}
		if len(chunks) == 0 {
			t.Fatalf("stream yielded no chunks")
		}
		if strings.TrimSpace(text.String()) == "" {
			t.Fatalf("aggregated stream text is empty")
		}
	})

	if capabilities.UsageMetadataStreaming {
		t.Run("usage metadata on stream chunks", func(t *testing.T) { // test_usage_metadata_streaming
			model := factory(t)
			stream, err := model.Stream(context.Background(), []messages.Message{
				messages.Human("Write me 2 haikus. Only include the haikus."),
			})
			requireNoErr(t, "stream", err)
			defer stream.Close()

			var sawUsage bool
			var inputTokenChunks int
			for {
				chunk, ok, err := stream.Next(context.Background())
				if err != nil {
					t.Fatalf("stream next: %v", err)
				}
				if !ok {
					break
				}
				usage := chunk.UsageMetadata
				if usage.InputTokens > 0 || usage.OutputTokens > 0 || usage.TotalTokens > 0 {
					sawUsage = true
				}
				if usage.InputTokens > 0 {
					inputTokenChunks++
				}
			}
			if !sawUsage {
				t.Fatalf("model declares streaming usage metadata but no chunk carried usage")
			}
			if inputTokenChunks > 1 {
				t.Fatalf("input tokens set on %d chunks; at most one chunk may report them (overcounting bug)",
					inputTokenChunks)
			}
		})
	}

	t.Run("cancellation terminates the stream", func(t *testing.T) {
		model := factory(t)
		ctx, cancel := context.WithCancel(context.Background())
		stream, err := model.Stream(ctx, []messages.Message{
			messages.Human("stream then get canceled"),
		})
		if err != nil {
			cancel()
			t.Fatalf("stream: %v", err)
		}
		defer stream.Close()
		cancel()

		terminated := make(chan struct{})
		go func() {
			defer close(terminated)
			for {
				_, ok, err := stream.Next(ctx)
				if err != nil || !ok {
					return
				}
			}
		}()
		select {
		case <-terminated:
		case <-time.After(10 * time.Second):
			t.Fatalf("stream did not terminate within 10s after context cancellation")
		}
	})
}
