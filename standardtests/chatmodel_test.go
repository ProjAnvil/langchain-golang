package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
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

func TestRunChatModelBasicsWithStreamingUsageMetadata(t *testing.T) {
	RunChatModelBasics(
		t,
		func(t testing.TB) language.ChatModel {
			t.Helper()
			return language.NewFakeChatModel()
		},
		ChatModelCapabilities{
			Streaming:              true,
			UsageMetadata:          true,
			UsageMetadataStreaming: true,
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
		ToolChoice:       true,
		StructuredOutput: true,
		JSONMode:         true,
		ImageInputs:      true,
		ImageURLs:        true,
		AudioInputs:      true,
		PDFInputs:        true,
		VideoInputs:      true,
		UsageMetadata:    true,
		Streaming:        true,
	})
}

// stubChatModel is a chat model whose behavior is configured per scenario.
type stubChatModel struct {
	invokeMessage messages.Message
	invokeErr     error
	batchMessages []messages.Message
	batchErr      error
	streamChunks  []messages.Message
	streamErr     error
	streamNextErr error
}

func goodStubMessage() messages.Message {
	message := messages.AI("stub response")
	message.UsageMetadata = messages.UsageMetadata{
		InputTokens:  2,
		OutputTokens: 1,
		TotalTokens:  3,
	}
	return message
}

func (m stubChatModel) Invoke(
	context.Context,
	[]messages.Message,
	...runnables.Option,
) (messages.Message, error) {
	if m.invokeErr != nil {
		return messages.Message{}, m.invokeErr
	}
	if m.invokeMessage.Role == "" {
		return goodStubMessage(), nil
	}
	return m.invokeMessage, nil
}

func (m stubChatModel) Batch(
	context.Context,
	[][]messages.Message,
	...runnables.Option,
) ([]messages.Message, error) {
	if m.batchErr != nil {
		return nil, m.batchErr
	}
	if m.batchMessages == nil {
		return []messages.Message{goodStubMessage(), goodStubMessage()}, nil
	}
	return m.batchMessages, nil
}

func (m stubChatModel) Stream(
	context.Context,
	[]messages.Message,
	...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	if m.streamErr != nil {
		return nil, m.streamErr
	}
	chunks := m.streamChunks
	if chunks == nil {
		chunks = []messages.Message{goodStubMessage()}
	}
	return &stubMessageStream{chunks: chunks, err: m.streamNextErr}, nil
}

func (m stubChatModel) InputSchema() schema.Schema  { return schema.Schema{} }
func (m stubChatModel) OutputSchema() schema.Schema { return schema.Schema{} }

func (m stubChatModel) BindTools([]tools.Tool) (language.ChatModel, error) {
	return m, nil
}

func (m stubChatModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{Streaming: true, UsageMetadata: true}
}

type stubMessageStream struct {
	chunks []messages.Message
	err    error
	index  int
}

func (s *stubMessageStream) Next(context.Context) (messages.Message, bool, error) {
	if s.err != nil {
		return messages.Message{}, false, s.err
	}
	if s.index >= len(s.chunks) {
		return messages.Message{}, false, nil
	}
	chunk := s.chunks[s.index]
	s.index++
	return chunk, true, nil
}

func (s *stubMessageStream) Close() error { return nil }

func TestRunChatModelBasicsFailures(t *testing.T) {
	factory := func(model stubChatModel) ChatModelFactory {
		return func(t testing.TB) language.ChatModel {
			t.Helper()
			return model
		}
	}
	capabilities := ChatModelCapabilities{
		Streaming:              true,
		UsageMetadata:          true,
		UsageMetadataStreaming: true,
	}

	expectConformanceFailure(t, "invoke errors", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{invokeErr: errConformanceStub}), capabilities)
	})
	expectConformanceFailure(t, "invoke returns non-AI role", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{
			invokeMessage: messages.Human("not an AI response"),
		}), capabilities)
	})
	expectConformanceFailure(t, "batch errors", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{batchErr: errConformanceStub}), capabilities)
	})
	expectConformanceFailure(t, "batch returns too few responses", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{
			batchMessages: []messages.Message{goodStubMessage()},
		}), capabilities)
	})
	expectConformanceFailure(t, "batch returns non-AI role", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{
			batchMessages: []messages.Message{goodStubMessage(), messages.Human("bad")},
		}), capabilities)
	})
	expectConformanceFailure(t, "stream errors", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{streamErr: errConformanceStub}), capabilities)
	})
	expectConformanceFailure(t, "stream yields no chunks", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{
			streamChunks: []messages.Message{},
		}), capabilities)
	})
	expectConformanceFailure(t, "stream next errors", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{streamNextErr: errConformanceStub}), capabilities)
	})
	expectConformanceFailure(t, "stream chunk has non-AI role and no usage", func(t *testing.T) {
		RunChatModelBasics(t, factory(stubChatModel{
			streamChunks: []messages.Message{messages.Human("bad chunk")},
		}), capabilities)
	})
	expectConformanceFailure(t, "usage metadata missing total tokens", func(t *testing.T) {
		message := messages.AI("stub response")
		message.UsageMetadata = messages.UsageMetadata{InputTokens: 2}
		RunChatModelBasics(t, factory(stubChatModel{invokeMessage: message}), capabilities)
	})
	expectConformanceFailure(t, "usage metadata missing input tokens", func(t *testing.T) {
		message := messages.AI("stub response")
		message.UsageMetadata = messages.UsageMetadata{TotalTokens: 2}
		RunChatModelBasics(t, factory(stubChatModel{invokeMessage: message}), capabilities)
	})
}
