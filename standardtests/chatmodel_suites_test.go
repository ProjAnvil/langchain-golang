package standardtests

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/structuredoutput"
)

// The layered suites must (1) pass against a well-behaved fake model, and
// (2) fail against models that claim capabilities they do not honor. The
// factories below select canned responses per subtest name because each
// suite subtest builds a fresh model through the factory.

func fakeToolCallMessage(name string, args map[string]any) messages.Message {
	message := messages.AI("")
	message.ToolCalls = []messages.ToolCall{{ID: "call_" + name, Name: name, Args: args}}
	return message
}

func fakeStreamingCaps() ChatModelCapabilities {
	return ChatModelCapabilities{
		ToolCalling:            true,
		ToolChoice:             true,
		StructuredOutput:       true,
		ImageInputs:            true,
		ImageURLs:              true,
		AudioInputs:            true,
		Streaming:              true,
		UsageMetadata:          true,
		UsageMetadataStreaming: true,
	}
}

// fakeChatModelFactory builds FakeChatModels whose responses and stream
// chunks are selected by the leaf subtest name.
func fakeChatModelFactory(
	selectScenario func(leaf string) ([]messages.Message, []messages.Message),
) ChatModelFactory {
	return func(t testing.TB) language.ChatModel {
		t.Helper()
		parts := strings.Split(t.Name(), "/")
		leaf := parts[len(parts)-1]
		responses, chunks := selectScenario(leaf)
		fakeCaps := language.ChatModelCapabilities{
			ToolCalling:      true,
			StructuredOutput: true,
			Streaming:        true,
			UsageMetadata:    true,
		}
		options := []language.FakeChatModelOption{language.WithCapabilities(fakeCaps)}
		if responses != nil {
			options = append(options, language.WithResponses(responses...))
		}
		if chunks != nil {
			options = append(options, language.WithStreamChunks(chunks...))
		}
		return language.NewFakeChatModel(options...)
	}
}

func TestRunChatModelToolCallingSuiteWithFakeModel(t *testing.T) {
	RunChatModelToolCallingSuite(t, fakeChatModelFactory(func(leaf string) ([]messages.Message, []messages.Message) {
		switch {
		case strings.Contains(leaf, "no_arguments"):
			return []messages.Message{fakeToolCallMessage("magic_function_no_args", nil)}, nil
		case strings.Contains(leaf, "agent_loop"):
			return []messages.Message{
				fakeToolCallMessage("get_weather", map[string]any{"location": "San Francisco"}),
				messages.AI("It's sunny."),
			}, nil
		case strings.Contains(leaf, "streaming_tool_calls"):
			return nil, []messages.Message{
				fakeToolCallMessage("magic_function", map[string]any{"input": 3}),
			}
		default:
			// "invoke with bound tools" and "tool message histories".
			return []messages.Message{
				fakeToolCallMessage("magic_function", map[string]any{"input": 3}),
				messages.AI("The sum is 3."),
			}, nil
		}
	}), fakeStreamingCaps())
}

func TestRunChatModelToolChoiceSuiteWithFakeModel(t *testing.T) {
	RunChatModelToolChoiceSuite(t, fakeChatModelFactory(func(leaf string) ([]messages.Message, []messages.Message) {
		return []messages.Message{
			fakeToolCallMessage("magic_function", map[string]any{"input": 3}),
		}, nil
	}), fakeStreamingCaps())
}

// fakeStructuredChatModel wraps FakeChatModel with the WithStructuredOutput
// constructor required by structuredoutput.JSONSchemaModel, so the suite's
// include_raw envelope hook can be exercised offline.
type fakeStructuredChatModel struct {
	*language.FakeChatModel
}

func (m fakeStructuredChatModel) WithStructuredOutput(
	string,
	schema.Schema,
	bool,
) fakeStructuredChatModel {
	// The fake is pre-configured to answer with Joke-shaped JSON; the
	// provider-native configuration is a no-op.
	return m
}

func TestRunChatModelStructuredOutputSuiteWithFakeModel(t *testing.T) {
	hooks := StructuredOutputSuiteHooks{
		NewRawEnvelope: func(t testing.TB, model language.ChatModel) (runnables.Runnable[[]messages.Message, structuredoutput.StructuredResult[map[string]any]], error) {
			t.Helper()
			concrete, ok := model.(fakeStructuredChatModel)
			if !ok {
				t.Fatalf("factory produced %T, not fakeStructuredChatModel", model)
			}
			return structuredoutput.BindOptionsWithRaw[fakeStructuredChatModel, map[string]any](
				concrete,
				structuredoutput.Options{
					Schema: chatSuiteJokeSchema(),
					Method: structuredoutput.MethodJSONSchema,
				},
			)
		},
	}

	factory := func(t testing.TB) language.ChatModel {
		t.Helper()
		fake := fakeChatModelFactory(func(leaf string) ([]messages.Message, []messages.Message) {
			switch {
			case strings.Contains(leaf, "function_calling"):
				return []messages.Message{fakeToolCallMessage("Joke", chatSuiteJokeArgs())}, nil
			default:
				return []messages.Message{messages.AI(chatSuiteJokeJSON())}, nil
			}
		})(t)
		return fakeStructuredChatModel{FakeChatModel: fake.(*language.FakeChatModel)}
	}

	RunChatModelStructuredOutputSuite(t, factory, fakeStreamingCaps(), hooks)
}

func TestRunChatModelMultimodalInputSuiteWithFakeModel(t *testing.T) {
	RunChatModelMultimodalInputSuite(t, fakeChatModelFactory(func(leaf string) ([]messages.Message, []messages.Message) {
		return []messages.Message{messages.AI("A small diagram.")}, nil
	}), fakeStreamingCaps())
}

func TestRunChatModelStreamingSuiteWithFakeModel(t *testing.T) {
	RunChatModelStreamingSuite(t, fakeChatModelFactory(func(leaf string) ([]messages.Message, []messages.Message) {
		final := messages.AI(" world.")
		final.UsageMetadata = messages.UsageMetadata{InputTokens: 5, OutputTokens: 2, TotalTokens: 7}
		return nil, []messages.Message{messages.AI("Hello"), final}
	}), fakeStreamingCaps())
}

func TestRunChatModelStandardSuitesWithFakeModel(t *testing.T) {
	// The convenience entry point composes every suite; the scenario table
	// merges the per-suite tables above.
	factory := fakeChatModelFactory(func(leaf string) ([]messages.Message, []messages.Message) {
		switch {
		case strings.Contains(leaf, "no_arguments"):
			return []messages.Message{fakeToolCallMessage("magic_function_no_args", nil)}, nil
		case strings.Contains(leaf, "agent_loop"):
			return []messages.Message{
				fakeToolCallMessage("get_weather", map[string]any{"location": "San Francisco"}),
				messages.AI("It's sunny."),
			}, nil
		case strings.Contains(leaf, "streaming_tool_calls"):
			return nil, []messages.Message{
				fakeToolCallMessage("magic_function", map[string]any{"input": 3}),
			}
		case strings.Contains(leaf, "function_calling"):
			return []messages.Message{fakeToolCallMessage("Joke", chatSuiteJokeArgs())}, nil
		case strings.Contains(leaf, "invoke_structured"):
			return []messages.Message{messages.AI(chatSuiteJokeJSON())}, nil
		case strings.Contains(leaf, "chunk_sequence"), strings.Contains(leaf, "usage_metadata"), strings.Contains(leaf, "cancellation"):
			final := messages.AI(" world.")
			final.UsageMetadata = messages.UsageMetadata{InputTokens: 5, OutputTokens: 2, TotalTokens: 7}
			return nil, []messages.Message{messages.AI("Hello"), final}
		default:
			return []messages.Message{
				fakeToolCallMessage("magic_function", map[string]any{"input": 3}),
				messages.AI(chatSuiteJokeJSON()),
			}, nil
		}
	})
	RunChatModelStandardSuites(t, factory, fakeStreamingCaps(), StructuredOutputSuiteHooks{})
}

func TestRunChatModelSuitesSkipUndeclaredCapabilities(t *testing.T) {
	// A model that declares nothing beyond basics must skip (not fail) every
	// feature suite.
	basicFactory := fakeChatModelFactory(func(leaf string) ([]messages.Message, []messages.Message) {
		return []messages.Message{messages.AI("ok")}, []messages.Message{messages.AI("ok")}
	})
	noFeatures := ChatModelCapabilities{Streaming: false}
	RunChatModelStandardSuites(t, basicFactory, noFeatures, StructuredOutputSuiteHooks{})
}

// blockRejectingModel claims multimodal support but errors on any message
// carrying content blocks — the canonical capability overstatement.
type blockRejectingModel struct {
	stubChatModel
}

func (m blockRejectingModel) Invoke(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (messages.Message, error) {
	for _, message := range input {
		if len(message.ContentBlocks) > 0 {
			return messages.Message{}, errConformanceStub
		}
	}
	return m.stubChatModel.Invoke(ctx, input, opts...)
}

func TestRunChatModelSuitesFailures(t *testing.T) {
	factory := func(model language.ChatModel) ChatModelFactory {
		return func(t testing.TB) language.ChatModel {
			t.Helper()
			return model
		}
	}

	expectConformanceFailure(t, "tool calling claimed but no tool calls", func(t *testing.T) {
		RunChatModelToolCallingSuite(t, factory(stubChatModel{}), fakeStreamingCaps())
	})
	expectConformanceFailure(t, "tool choice claimed without ToolBinder", func(t *testing.T) {
		RunChatModelToolChoiceSuite(t, factory(stubChatModel{}), fakeStreamingCaps())
	})
	expectConformanceFailure(t, "structured output claimed but non-json response", func(t *testing.T) {
		RunChatModelStructuredOutputSuite(t, factory(stubChatModel{}), fakeStreamingCaps(), StructuredOutputSuiteHooks{})
	})
	expectConformanceFailure(t, "image inputs claimed but blocks rejected", func(t *testing.T) {
		RunChatModelMultimodalInputSuite(t, factory(blockRejectingModel{}), fakeStreamingCaps())
	})
	expectConformanceFailure(t, "streaming claimed but no chunks", func(t *testing.T) {
		RunChatModelStreamingSuite(t, factory(stubChatModel{streamChunks: []messages.Message{}}), fakeStreamingCaps())
	})
	expectConformanceFailure(t, "streaming usage claimed but absent", func(t *testing.T) {
		RunChatModelStreamingSuite(t, factory(stubChatModel{
			streamChunks: []messages.Message{messages.AI("text without usage")},
		}), fakeStreamingCaps())
	})
}
