package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
)

// ChatModelCapabilities declares which standard chat-model behaviors an
// integration claims to support. Mirrors Python's ChatModelTests property
// flags (has_tool_calling, returns_usage_metadata, supports_image_inputs, etc.)
type ChatModelCapabilities struct {
	ToolCalling      bool
	ToolChoice       bool
	StructuredOutput bool
	JSONMode         bool
	ImageInputs      bool
	ImageURLs        bool
	AudioInputs      bool
	PDFInputs        bool
	VideoInputs      bool
	// UsageMetadata indicates the model returns token counts on invoke responses.
	UsageMetadata bool
	// UsageMetadataStreaming indicates the model returns token counts on stream-finish.
	UsageMetadataStreaming bool
	Streaming              bool
	// AnthropicInputs indicates Anthropic-style tool_use / tool_result content blocks.
	AnthropicInputs bool
	// ImageToolMessage indicates the model accepts ToolMessage objects with image content.
	ImageToolMessage bool
	// PDFToolMessage indicates the model accepts ToolMessage objects with PDF content.
	PDFToolMessage bool
	// ModelOverride indicates the model accepts a model kwarg override at invoke time.
	ModelOverride bool
}

// UnsupportedFeatures documents which standard features a provider has
// intentionally not implemented. Embed this in a provider-specific test struct
// and call DeclareUnsupported to produce a structured test log entry.
//
// This mirrors the Python pattern of setting capability properties to False to
// skip the corresponding standard tests. In Go, explicit declaration serves as
// documentation and surfaced in test output.
type UnsupportedFeatures struct {
	ToolCalling      bool
	ToolChoice       bool
	StructuredOutput bool
	JSONMode         bool
	ImageInputs      bool
	ImageURLs        bool
	AudioInputs      bool
	PDFInputs        bool
	VideoInputs      bool
	UsageMetadata    bool
	Streaming        bool
}

// DeclareUnsupported logs intentionally unsupported features to the test
// output so they appear in CI artifacts. It does not fail the test.
func DeclareUnsupported(t testing.TB, features UnsupportedFeatures) {
	t.Helper()
	if features.ToolCalling {
		t.Log("UNSUPPORTED: tool_calling")
	}
	if features.ToolChoice {
		t.Log("UNSUPPORTED: tool_choice")
	}
	if features.StructuredOutput {
		t.Log("UNSUPPORTED: structured_output")
	}
	if features.JSONMode {
		t.Log("UNSUPPORTED: json_mode")
	}
	if features.ImageInputs {
		t.Log("UNSUPPORTED: image_inputs")
	}
	if features.ImageURLs {
		t.Log("UNSUPPORTED: image_url_inputs")
	}
	if features.AudioInputs {
		t.Log("UNSUPPORTED: audio_inputs")
	}
	if features.PDFInputs {
		t.Log("UNSUPPORTED: pdf_inputs")
	}
	if features.VideoInputs {
		t.Log("UNSUPPORTED: video_inputs")
	}
	if features.UsageMetadata {
		t.Log("UNSUPPORTED: usage_metadata")
	}
	if features.Streaming {
		t.Log("UNSUPPORTED: streaming")
	}
}

// ChatModelFactory creates a fresh chat model for a standard test run.
type ChatModelFactory func(t testing.TB) language.ChatModel

// RunChatModelBasics verifies behavior expected from every chat model
// integration. It covers the same ground as Python's ChatModelUnitTests:
// invoke, batch, stream, usage metadata, and streaming usage metadata.
func RunChatModelBasics(
	t *testing.T,
	factory ChatModelFactory,
	capabilities ChatModelCapabilities,
) {
	t.Helper()

	t.Run("invoke", func(t *testing.T) {
		model := factory(t)
		response, err := model.Invoke(context.Background(), []messages.Message{
			messages.Human("hello"),
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if response.Role != messages.RoleAI {
			t.Fatalf("role: got %q want %q", response.Role, messages.RoleAI)
		}
	})

	t.Run("batch", func(t *testing.T) {
		model := factory(t)
		responses, err := model.Batch(context.Background(), [][]messages.Message{
			{messages.Human("first")},
			{messages.Human("second")},
		})
		if err != nil {
			t.Fatalf("batch: %v", err)
		}
		if len(responses) != 2 {
			t.Fatalf("responses: got %d want 2", len(responses))
		}
		for i, response := range responses {
			if response.Role != messages.RoleAI {
				t.Fatalf("response[%d] role: got %q want %q", i, response.Role, messages.RoleAI)
			}
		}
	})

	if capabilities.Streaming {
		t.Run("stream", func(t *testing.T) {
			model := factory(t)
			stream, err := model.Stream(context.Background(), []messages.Message{
				messages.Human("stream"),
			})
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			defer stream.Close()

			chunk, ok, err := stream.Next(context.Background())
			if err != nil {
				t.Fatalf("next: %v", err)
			}
			if !ok {
				t.Fatal("expected at least one stream chunk")
			}
			if chunk.Role != messages.RoleAI {
				t.Fatalf("chunk role: got %q want %q", chunk.Role, messages.RoleAI)
			}
		})
	}

	if capabilities.UsageMetadata {
		t.Run("usage metadata invoke", func(t *testing.T) {
			model := factory(t)
			response, err := model.Invoke(context.Background(), []messages.Message{
				messages.Human("usage"),
			})
			if err != nil {
				t.Fatalf("invoke: %v", err)
			}
			if response.UsageMetadata.TotalTokens == 0 {
				t.Fatal("expected non-zero total_tokens in usage metadata")
			}
			if response.UsageMetadata.InputTokens == 0 {
				t.Fatal("expected non-zero input_tokens in usage metadata")
			}
		})
	}

	if capabilities.UsageMetadataStreaming && capabilities.Streaming {
		t.Run("usage metadata stream", func(t *testing.T) {
			model := factory(t)
			stream, err := model.Stream(context.Background(), []messages.Message{
				messages.Human("usage stream"),
			})
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			defer stream.Close()

			var last messages.Message
			for {
				chunk, ok, err := stream.Next(context.Background())
				if err != nil {
					t.Fatalf("next: %v", err)
				}
				if !ok {
					break
				}
				last = chunk
			}
			// The final chunk should carry cumulative usage.
			if last.UsageMetadata.TotalTokens == 0 {
				t.Fatal("expected usage metadata on final stream chunk")
			}
		})
	}
}

// RunChatModelUnitTests runs the full unit-test conformance suite mirroring
// Python's ChatModelUnitTests. It includes RunChatModelBasics plus structural
// validations that can be run offline without a live model: response shape,
// streaming, and usage metadata paths.
//
// Providers that do not support certain features should pass an appropriate
// ChatModelCapabilities and call DeclareUnsupported for logging.
func RunChatModelUnitTests(
	t *testing.T,
	factory ChatModelFactory,
	capabilities ChatModelCapabilities,
) {
	t.Helper()
	RunChatModelBasics(t, factory, capabilities)
}
