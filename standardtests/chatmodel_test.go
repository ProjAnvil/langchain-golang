package standardtests

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
)

func TestRunChatModelBasicsWithFakeModel(t *testing.T) {
	RunChatModelBasics(
		t,
		func(t testing.TB) language.ChatModel {
			t.Helper()
			return language.NewFakeChatModel()
		},
		ChatModelCapabilities{
			Streaming:     true,
			UsageMetadata: true,
		},
	)
}

func TestRunChatModelUnitTestsWithFakeModel(t *testing.T) {
	RunChatModelUnitTests(
		t,
		func(t testing.TB) language.ChatModel {
			t.Helper()
			return language.NewFakeChatModel()
		},
		ChatModelCapabilities{
			Streaming:     true,
			UsageMetadata: true,
		},
	)
}

func TestDeclareUnsupported(t *testing.T) {
	// Smoke-test: DeclareUnsupported must not fail the test; it only logs.
	DeclareUnsupported(t, UnsupportedFeatures{
		ToolCalling:      true,
		StructuredOutput: true,
		ImageInputs:      true,
	})
}
